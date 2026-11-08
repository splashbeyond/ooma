"""Replay one Setup over future candles. Pure, conservative, shared by backtest and paper checks.

Rules: market entry at the next bar's open (+slippage); if stop and target are both inside one bar,
the stop wins; stops that gap fill at the open; horizon exit at last bar's close.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from ..types import Setup, Trade


def simulate(setup: Setup, future: pl.DataFrame, *, risk_usd: float = 100.0,
             fee_bps: float = 4.5, slip_bps: float = 2.0, source: str = "backtest") -> Trade | None:
    """`future` = bars opening at or after setup.as_of. Returns None if the setup is invalid or never fills."""
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
