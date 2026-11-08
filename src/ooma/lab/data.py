"""Candle loader with a parquet cache so the lab does not re-download history for every candidate."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ..adapters.hyperliquid import INTERVAL_MS


class CachedLoader:
    def __init__(self, adapter, root: str = "data/candles"):
        self.adapter, self.root = adapter, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: dict[tuple, pl.DataFrame] = {}

    def prefetch(self, market: str, tf: str, start: datetime, end: datetime):
        path = self.root / f"{market}_{tf}.parquet"
        df = pl.read_parquet(path) if path.exists() else None
        if df is None or df["t"].min() > start or df["t"].max() < end - timedelta(milliseconds=INTERVAL_MS[tf] * 2):
            df = self.adapter.candles(market, tf, start, end)
            df.write_parquet(path)
        self._mem[(market, tf)] = df

    def __call__(self, market: str, tf: str, start: datetime, end: datetime) -> pl.DataFrame:
        df = self._mem[(market, tf)]
        return df.filter((pl.col("t") >= start) & (pl.col("t") < end))
