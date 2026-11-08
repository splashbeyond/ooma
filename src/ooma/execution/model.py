"""Execution model for candle-based backtests (no book history). Conservative on purpose.

  entry / horizon exit : taker. price = reference +- (half-spread + sqrt-law impact), taker fee
  stop exit            : taker stop-market. as above + extra stop slippage; gaps fill at the open (done by caller)
  target exit          : resting maker LIMIT. fills only if the bar trades THROUGH the target by >= 1 tick;
                         a bar that merely touches it fills with probability `touch_fill_prob`
                         (deterministic per trade, so reruns match); size capped by bar participation
  funding              : hourly rate x notional, longs pay when positive
  liquidation          : HL formula, isolated margin; trades whose stop sits beyond liquidation are rejected
Live paper trading should use the real book (execution/book.py) and queue sim (execution/resting.py).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from .fees import FeeSchedule
from .rules import MarketRules


@dataclass
class ExecutionModel:
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    half_spread_bps: dict = field(default_factory=lambda: {"BTC": 0.15, "ETH": 0.25, "SOL": 0.6, "HYPE": 1.0})
    default_half_spread_bps: float = 1.5
    impact_coeff: float = 0.5            # impact_bps = coeff * bar_sigma_bps * sqrt(notional / bar_notional_volume)
    stop_extra_bps: float = 3.0
    touch_fill_prob: float = 0.35
    participation_cap: float = 0.10      # max share of a bar's volume we may take
    leverage: float = 3.0
    rules: dict = field(default_factory=lambda: {
        "BTC": MarketRules(5, 40), "ETH": MarketRules(4, 25), "SOL": MarketRules(2, 20), "HYPE": MarketRules(2, 10)})
    default_rules: MarketRules = field(default_factory=lambda: MarketRules(2, 10))

    def mrules(self, market: str) -> MarketRules:
        return self.rules.get(market, self.default_rules)

    # ---- prices ----
    def impact_bps(self, notional: float, bar_notional_volume: float, bar_sigma_bps: float) -> float:
        if bar_notional_volume <= 0:
            return 25.0
        return self.impact_coeff * bar_sigma_bps * math.sqrt(max(notional, 0) / bar_notional_volume)

    def taker_px(self, market, side: int, ref_px: float, notional, bar_notional_volume, bar_sigma_bps, extra_bps=0.0):
        """side +1 = buy (pay up), -1 = sell. Returns (px, slip_bps)."""
        hs = self.half_spread_bps.get(market, self.default_half_spread_bps)
        slip = hs + self.impact_bps(notional, bar_notional_volume, bar_sigma_bps) + extra_bps
        return self.mrules(market).round_price(ref_px * (1 + side * slip / 1e4), "up" if side > 0 else "down"), slip

    # ---- resting target ----
    def target_fill(self, market, side: int, target: float, bar_high: float, bar_low: float,
                    bar_notional_volume: float, notional: float, key: str) -> float:
        """Fraction (0..1) of a resting exit at `target` that fills in this bar. side = position side (+1 long)."""
        tick = self.mrules(market).tick(target)
        through = (bar_high >= target + tick) if side > 0 else (bar_low <= target - tick)
        touched = (bar_high >= target) if side > 0 else (bar_low <= target)
        if not touched:
            return 0.0
        if not through:
            u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if u >= self.touch_fill_prob:
                return 0.0
        cap = self.participation_cap * bar_notional_volume
        return 1.0 if notional <= 0 else min(1.0, cap / notional)

    # ---- funding & liquidation ----
    @staticmethod
    def funding_cash(side: int, notional: float, funding_bps_per_hour: float, hours: float) -> float:
        """Cash flow to the position (+ received, - paid). Positive funding: longs pay shorts."""
        return -side * notional * funding_bps_per_hour / 1e4 * hours

    def liq_price(self, market: str, side: int, entry: float, leverage: float | None = None) -> float:
        """HL isolated-margin liquidation price: p - s * (margin - mm)/size / (1 - l*s), l = 1/(2*maxLev)."""
        lev = leverage or self.leverage
        l = 1 / (2 * self.mrules(market).max_leverage)
        margin_per_unit = entry / lev
        mm_per_unit = entry * l
        return entry - side * (margin_per_unit - mm_per_unit) / (1 - l * side)

    def sigma_bps(self, high: float, low: float, close: float) -> float:
        return (high - low) / close * 1e4 * 0.6     # crude per-bar vol proxy from the bar range (Parkinson-ish)


def from_config(cfg: dict | None) -> ExecutionModel:
    c = dict(cfg or {})
    fees = FeeSchedule(**c.pop("fees", {}))
    rules = {k: MarketRules(**v) for k, v in c.pop("rules", {}).items()}
    kw = {"fees": fees, **c}
    if rules:
        kw["rules"] = rules
    return ExecutionModel(**kw)
