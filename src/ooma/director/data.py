"""Data for scripts: enriched candle history (cached daily) and live windows. Columns match what the prompt promises."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..adapters.hyperliquid import INTERVAL_MS
from ..lab.enrich import enrich

REF = "BTC"
COLS = ["open", "high", "low", "close", "volume", "funding", "funding_z", "btc_ret_1", "btc_ret_6", "btc_ret_24",
        "rel_btc_6", "rel_btc_24", "sup_dist", "res_dist", "sup_touches", "res_touches"]
DESCRIPTION = """Columns in `bars` (numpy float arrays, last element = latest CLOSED bar, at most 500 bars of history):
  open, high, low, close, volume
  funding        funding rate in basis points per hour (positive = longs pay shorts); NaN if unknown
  funding_z      72-bar z-score of funding
  btc_ret_1/6/24 BTC return over the last 1/6/24 bars of this timeframe
  rel_btc_6/24   this market's return minus BTC's over 6/24 bars (relative strength)
  sup_dist/res_dist   distance in ATRs from close to the nearest support/resistance zone (clustered swing pivots), NaN if none
  sup_touches/res_touches   number of swing pivots forming that zone (more = stronger)
  t              bar open time in epoch milliseconds
Helper functions: from ooma_lib import sma, ema, rsi, atr, rolling_max, rolling_min, zscore, pct_change (all causal, NaN-padded)."""


class DataHub:
    def __init__(self, adapter, root: str = "data/candles", markets=("BTC", "ETH", "SOL", "HYPE"),
                 timeframes=("1h", "4h"), days: int = 540, max_age_h: float = 24):
        self.a, self.root, self.markets, self.tfs, self.days, self.max_age = adapter, Path(root), list(markets), timeframes, days, max_age_h
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: dict = {}

    def history(self, market: str, tf: str) -> pl.DataFrame:
        k = (market, tf)
        path = self.root / f"{market}_{tf}_dir.parquet"
        if k not in self._mem:
            fresh = path.exists() and (time.time() - path.stat().st_mtime) < self.max_age * 3600
            if fresh:
                self._mem[k] = pl.read_parquet(path)
            else:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=self.days)
                c = self.a.candles(market, tf, start, end)
                ref = c if market == REF else self.a.candles(REF, tf, start, end)
                df = enrich(c, tf, self.a.funding_history(market, start, end), ref)
                df.write_parquet(path)
                self._mem[k] = df
        return self._mem[k]

    def live(self, market: str, tf: str, bars: int = 700) -> pl.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=INTERVAL_MS[tf] * bars)
        c = self.a.candles(market, tf, start, end)
        ref = c if market == REF else self.a.candles(REF, tf, start, end)
        return enrich(c, tf, self.a.funding_history(market, start, end), ref)


def to_arrays(df: pl.DataFrame) -> dict:
    out = {"t": df["t"].dt.epoch("ms").cast(pl.Float64).to_numpy()}
    for c in COLS:
        out[c] = (df[c].cast(pl.Float64).fill_null(float("nan")).fill_nan(float("nan")).to_numpy()
                  if c in df.columns else np.full(df.height, np.nan))
    return out
