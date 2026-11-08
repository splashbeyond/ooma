"""Replay one Setup over future candles. Pure, conservative, shared by backtest and paper checks.

Rules: market entry at the next bar's open (+slippage); if stop and target are both inside one bar,
the stop wins; stops that gap fill at the open; horizon exit at last bar's close.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from ..types import Setup, Trade


def simulate(setup: Setup, future: pl.DataFrame, *, risk_usd: float = 100.0,
             fee_bps: float = 4.5, slip_bps: float = 2.0, source: str = "backtest", model=None) -> Trade | None:
    """`future` = bars opening at or after setup.as_of. Returns None if the setup is invalid or never fills.
    With `model` (execution.ExecutionModel) costs are realistic; without it, flat fee/slippage (legacy)."""
    if model is not None:
        return _simulate_real(setup, future, risk_usd, source, model)
    sign = 1 if setup.direction == "long" else -1
    slip = slip_bps / 1e4
    rows = future.filter(pl.col("t") >= setup.as_of).sort("t").to_dicts()
    if not rows:
        return None
    risk_px = abs(setup.entry - setup.stop)
    if risk_px <= 0 or (setup.entry - setup.stop) * sign <= 0:
        return None
    entry = rows[0]["open"] * (1 + sign * slip)
    if (entry - setup.stop) * sign <= 0:
        return None                                   # gapped through the stop before entry
    size = risk_usd / risk_px
    deadline = setup.as_of + timedelta(hours=setup.horizon_hours)
    best = worst = entry
    exit_px = exit_t = reason = None
    target = setup.targets[0] if setup.targets else None

    window = [r for r in rows if r["t"] < deadline]
    for r in window:
        best = max(best, r["high"]) if sign == 1 else min(best, r["low"])
        worst = min(worst, r["low"]) if sign == 1 else max(worst, r["high"])
        stopped = r["low"] <= setup.stop if sign == 1 else r["high"] >= setup.stop
        hit = target is not None and (r["high"] >= target if sign == 1 else r["low"] <= target)
        if stopped:
            gap = r["open"] <= setup.stop if sign == 1 else r["open"] >= setup.stop
            px = r["open"] if gap else setup.stop
            exit_px, reason = px * (1 - sign * slip), "stop"
        elif hit:
            exit_px, reason = target, "target"
        if reason:
            exit_t = r["t"]
            break
    if reason is None:
        if len(window) == len(rows):                   # data ends before the horizon: still open
            return Trade(**_base(setup, rows[0]["t"], entry, size, best, worst, source))
        last = window[-1]
        exit_px, exit_t, reason = last["close"] * (1 - sign * slip), last["t"], "horizon"

    fees = fee_bps / 1e4 * size * (entry + exit_px)
    t = _base(setup, rows[0]["t"], entry, size, best, worst, source)
    t.update(fees=fees, closed_at=exit_t, exit_fill=exit_px, exit_reason=reason)
    return Trade(**t)


def _base(setup, opened, entry, size, best, worst, source) -> dict:
    return dict(id=f"{setup.id}-t", setup_id=setup.id, strategy=setup.strategy,
                strategy_version=setup.strategy_version, market=setup.market,
                direction=setup.direction, opened_at=opened, entry_fill=entry, stop=setup.stop,
                targets=setup.targets, size=size, best_price=best, worst_price=worst, source=source)


def _simulate_real(setup: Setup, future: pl.DataFrame, risk_usd: float, source: str, m) -> Trade | None:
    from ..adapters.hyperliquid import INTERVAL_MS
    sign = 1 if setup.direction == "long" else -1
    rows = future.filter(pl.col("t") >= setup.as_of).sort("t").to_dicts()
    if not rows:
        return None
    risk_px = abs(setup.entry - setup.stop)
    if risk_px <= 0 or (setup.entry - setup.stop) * sign <= 0:
        return None
    rules, mk = m.mrules(setup.market), setup.market
    hours = INTERVAL_MS[setup.timeframe] / 3.6e6
    r0 = rows[0]
    size = rules.round_size(risk_usd / risk_px)
    if not rules.valid_order(r0["open"], size):
        return None                                           # below exchange minimum or rounds to zero
    sig = lambda r: m.sigma_bps(r["high"], r["low"], r["close"])
    nv = lambda r: r["volume"] * r["close"]
    entry, _ = m.taker_px(mk, sign, r0["open"], size * r0["open"], nv(r0), sig(r0))
    if (entry - setup.stop) * sign <= 0:
        return None
    liq = m.liq_price(mk, sign, entry)
    if (setup.stop - liq) * sign <= 0:
        return None                                           # stop sits beyond liquidation: not a valid trade
    notional = size * entry
    fees = m.fees.fee(notional, "taker")
    deadline = setup.as_of + timedelta(hours=setup.horizon_hours)
    window = [r for r in rows if r["t"] < deadline]
    target = setup.targets[0] if setup.targets else None
    best = worst = entry
    funding = 0.0
    exit_px = exit_t = reason = None
    role = "taker"
    for r in window:
        best = max(best, r["high"]) if sign == 1 else min(best, r["low"])
        worst = min(worst, r["low"]) if sign == 1 else max(worst, r["high"])
        funding += m.funding_cash(sign, notional, r.get("funding") or 0.0, hours)
        liquidated = r["low"] <= liq if sign == 1 else r["high"] >= liq
        stopped = r["low"] <= setup.stop if sign == 1 else r["high"] >= setup.stop
        if liquidated or stopped:
            trig = liq if liquidated else setup.stop
            gap = r["open"] <= trig if sign == 1 else r["open"] >= trig
            ref = r["open"] if gap else trig
            exit_px, _ = m.taker_px(mk, -sign, ref, notional, nv(r), sig(r), extra_bps=m.stop_extra_bps)
            reason = "liquidated" if liquidated else "stop"
        elif target is not None and m.target_fill(mk, sign, target, r["high"], r["low"], nv(r), notional,
                                                  f"{setup.id}|{r['t'].isoformat()}") >= 1.0:
            exit_px, role, reason = rules.round_price(target), "maker", "target"
        if reason:
            exit_t = r["t"]
            break
    if reason is None:
        if len(window) == len(rows):                          # data ends before the horizon: still open
            return Trade(**{**_base(setup, r0["t"], entry, size, best, worst, source), "fees": fees, "funding": funding})
        last = window[-1]
        exit_px, _ = m.taker_px(mk, -sign, last["close"], notional, nv(last), sig(last))
        exit_t, reason = last["t"], "horizon"
    fees += m.fees.fee(size * exit_px, role)
    t = _base(setup, r0["t"], entry, size, best, worst, source)
    t.update(fees=fees, funding=funding, closed_at=exit_t, exit_fill=exit_px, exit_reason=reason)
    return Trade(**t)
