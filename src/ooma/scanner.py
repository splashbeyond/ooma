"""Live scan: for each market x timeframe, build context, run every active strategy, return setups.
The same code path the backtester uses, fed live closed candles."""
from __future__ import annotations

from datetime import datetime

from .strategies.base import ScanContext, all_strategies
from .types import Setup


def scan(adapter, markets: list[str], timeframes: list[str], as_of: datetime, cfg: dict) -> list[Setup]:
    raise NotImplementedError("wire adapter.candles -> modules -> strategies after milestone 2")
