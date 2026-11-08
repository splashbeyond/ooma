"""Candle loader with a parquet cache. Frames are enriched (zones, funding, BTC-relative) once."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ..adapters.hyperliquid import INTERVAL_MS
from .enrich import enrich

REF = "BTC"


class CachedLoader:
    def __init__(self, adapter, root: str = "data/candles"):
        self.adapter, self.root = adapter, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: dict[tuple, pl.DataFrame] = {}
        self._raw: dict[tuple, pl.DataFrame] = {}

    def _raw_candles(self, market, tf, start, end):
        k = (market, tf)
        if k not in self._raw:
            self._raw[k] = self.adapter.candles(market, tf, start, end)
        return self._raw[k]

    def prefetch(self, market: str, tf: str, start: datetime, end: datetime):
        path = self.root / f"{market}_{tf}_x.parquet"
        df = pl.read_parquet(path) if path.exists() else None
        if df is None or df["t"].min() > start or df["t"].max() < end - timedelta(milliseconds=INTERVAL_MS[tf] * 2):
            c = self._raw_candles(market, tf, start, end)
            ref = c if market == REF else self._raw_candles(REF, tf, start, end)
            df = enrich(c, tf, self.adapter.funding_history(market, start, end), ref)
            df.write_parquet(path)
        self._mem[(market, tf)] = df

    def __call__(self, market: str, tf: str, start: datetime, end: datetime) -> pl.DataFrame:
        return self._mem[(market, tf)].filter((pl.col("t") >= start) & (pl.col("t") < end))
