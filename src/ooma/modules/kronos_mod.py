"""Kronos-small sampled forecasts: P(up), median path, quantiles, vol. CPU. Verify predictor API in repo first (milestone 3)."""
from __future__ import annotations

from datetime import datetime

from ..types import Findings


def run(market: str, timeframe: str, as_of: datetime, **data) -> Findings:
    raise NotImplementedError("kronos_mod")
