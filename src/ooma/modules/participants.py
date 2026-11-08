"""Participants module: what skilled and weak (directional, non-hedger) wallets hold in this market.
Reads only the latest cohort snapshot at or before `as_of`."""
from __future__ import annotations

from datetime import datetime

import polars as pl

from ..types import Findings, Signal


def run(market: str, timeframe: str, as_of: datetime, cohort: pl.DataFrame | None = None, **_) -> Findings:
    base = dict(module="participants", market=market, timeframe=timeframe, as_of=as_of)
    if cohort is None or cohort.height == 0:
        return Findings(**base, data_quality="partial", notes="no wallet snapshots yet")
    past = cohort.filter(pl.col("t") <= as_of).sort("t")
    if past.height == 0:
        return Findings(**base, data_quality="partial", notes="no snapshot before as_of")
    r = past.row(-1, named=True)
    if (as_of - r["t"]).total_seconds() > 3 * 3600:
        return Findings(**base, data_quality="stale", notes="latest wallet snapshot older than 3h")
    skill, weak = r.get("skill_net"), r.get("weak_net")
    sig = [Signal(name="skill_net", value=skill if skill is not None else 0.0, detail="skilled wallets net/gross, -1 short .. +1 long"),
           Signal(name="weak_net", value=weak if weak is not None else 0.0, detail="weak wallets net/gross (contrarian)")]
    score = (skill or 0.0) - 0.5 * (weak or 0.0)                  # follow the skilled, lean against the weak
    bias = "bullish" if score > 0.25 else "bearish" if score < -0.25 else "neutral"
    return Findings(**base, bias=bias, strength=min(1.0, abs(score)), signals=sig,
                    notes=f"skilled net {skill if skill is not None else 'n/a'}, weak net {weak if weak is not None else 'n/a'}")
