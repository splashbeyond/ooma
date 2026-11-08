"""Trade grading. Two grades per trade: outcome (R) and process (did it follow the plan).

A -1R stop-out with clean execution is a D outcome but an A process. Learning keys off
process + aggregate statistics, never off a single result.
"""
from __future__ import annotations

from ..types import Grade, Setup, Trade


def _letter_from_r(r: float) -> str:
    return "A" if r >= 2 else "B" if r >= 1 else "C" if r >= 0 else "D" if r >= -1.1 else "F"


def _letter_from_score(s: float) -> str:
    return "A" if s >= 0.9 else "B" if s >= 0.75 else "C" if s >= 0.5 else "D" if s >= 0.25 else "F"


def grade(setup: Setup, trade: Trade, price_at_horizon: float | None = None) -> Grade:
    if trade.closed_at is None or trade.exit_fill is None:
        raise ValueError("cannot grade an open trade")
    sign = 1 if trade.direction == "long" else -1
    risk_px = abs(setup.entry - setup.stop)
    risk_usd = trade.size * risk_px

    gross = (trade.exit_fill - trade.entry_fill) * sign * trade.size
    r = (gross - trade.fees + trade.funding) / risk_usd
    mfe_r = (trade.best_price - trade.entry_fill) * sign / risk_px
    mae_r = (trade.entry_fill - trade.worst_price) * sign / risk_px
    entry_slip = (trade.entry_fill - setup.entry) * sign / risk_px          # + = worse
    stop_slip = ((trade.stop - trade.exit_fill) * sign / risk_px
                 if trade.exit_reason == "stop" else 0.0)                    # + = worse

    score = 1.0
    if entry_slip > 0.2:
        score -= 0.3
    if stop_slip > 0.25:
        score -= 0.3
    if trade.exit_reason == "manual":
        score -= 0.4
    score = max(0.0, score)

    correct = None if price_at_horizon is None else (price_at_horizon - setup.entry) * sign > 0
    return Grade(
        trade_id=trade.id, strategy=trade.strategy, strategy_version=trade.strategy_version,
        r_multiple=r, outcome_grade=_letter_from_r(r), process_score=score,
        process_grade=_letter_from_score(score), mfe_r=mfe_r, mae_r=mae_r,
        entry_slippage_r=entry_slip, stop_slippage_r=stop_slip, direction_correct=correct)
