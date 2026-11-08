"""Price action, code only. Thresholds come from config.yaml (price_action section).

Milestone 2. Build bottom-up, each function unit-tested on synthetic candles.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from ..types import Findings, Level


def ema(close: pl.Series, period: int) -> pl.Series: raise NotImplementedError
def sma(close: pl.Series, period: int) -> pl.Series: raise NotImplementedError
def atr(candles: pl.DataFrame, period: int = 14) -> pl.Series: raise NotImplementedError
def pivots(candles: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """Confirmed swing highs/lows. A pivot is only known `lookback` bars after it forms."""
    raise NotImplementedError
def trend_state(pivot_df: pl.DataFrame) -> str:
    """'up' (HH+HL), 'down' (LH+LL) or 'range'."""
    raise NotImplementedError
def cluster_levels(pivot_df: pl.DataFrame, atr_now: float, k: float) -> list[Level]:
    raise NotImplementedError
def detect_range(candles: pl.DataFrame, cfg: dict) -> dict | None:
    raise NotImplementedError
def detect_breakout(candles: pl.DataFrame, levels: list[Level], cfg: dict) -> dict | None:
    raise NotImplementedError


def run(market: str, timeframe: str, as_of: datetime, candles: pl.DataFrame, cfg: dict) -> Findings:
    raise NotImplementedError
