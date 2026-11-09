"""Data for scripts: the liquid Hyperliquid universe, enriched candle history (cached), live windows, and
cross-sectional features that only exist when you look at the whole market at once."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..adapters.hyperliquid import INTERVAL_MS
from ..lab.enrich import enrich

REF = "BTC"
COLS = ["open", "high", "low", "close", "volume", "trades", "tradesx_24", "funding", "funding_z", "btc_ret_1", "btc_ret_6",
        "btc_ret_24", "rel_btc_6", "rel_btc_24", "sup_dist", "res_dist", "sup_touches", "res_touches",
        "mkt_ret_6", "mkt_ret_24", "xs_rank_ret_24", "xs_rank_funding", "xs_rank_vol"]
DESCRIPTION = """Columns in `bars` (numpy float arrays, last element = latest CLOSED bar, at most 500 bars of history):
  open, high, low, close, volume
  trades         number of trades in the bar;  tradesx_24 = trades / its 24-bar average (activity surge)
  funding        funding rate in basis points per hour (positive = longs pay shorts); NaN if unknown
  funding_z      72-bar z-score of this coin's funding
  btc_ret_1/6/24 BTC return over the last 1/6/24 bars;  rel_btc_6/24 = this coin's return minus BTC's
  sup_dist/res_dist   ATR distance from close to the nearest support/resistance zone (swing-pivot clusters), NaN if none
  sup_touches/res_touches   swing pivots forming that zone (more = stronger)
  CROSS-SECTIONAL (this coin versus the whole liquid Hyperliquid universe at the same moment):
  mkt_ret_6/24   average 6/24-bar return across all coins (the market's direction)
  xs_rank_ret_24 this coin's 24-bar return rank among all coins, 0 = weakest .. 1 = strongest
  xs_rank_funding  its funding rank among all coins, 0 = lowest .. 1 = highest (crowded longs)
  xs_rank_vol    its recent volatility rank, 0 = calmest .. 1 = wildest
  t              bar open time in epoch milliseconds
Helper functions: from ooma_lib import sma, ema, rsi, atr, rolling_max, rolling_min, zscore, pct_change (all causal, NaN-padded)."""


MIN_BARS = {"1h": 1000, "4h": 1000, "1d": 400}      # 540 days of daily candles is only ~540 bars


def day_cost(day_volume_usd: float) -> float:
    """Round-trip cost fraction by liquidity tier. Thin coins cost more than the flat 13 bps; do not flatter them."""
    return 0.0013 if day_volume_usd >= 50e6 else 0.0020 if day_volume_usd >= 10e6 else 0.0030


def add_cross_section(frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Attach market-wide features. Causal: every value at bar t uses only data up to t. frames: market -> enriched df."""
    parts = []
    for m, df in frames.items():
        c = pl.col("close")
        parts.append(df.select("t", pl.lit(m).alias("_m"), (c / c.shift(6) - 1).alias("_r6"), (c / c.shift(24) - 1).alias("_r24"),
                               pl.col("funding").alias("_f"),
                               (c / c.shift(1) - 1).rolling_std(24).alias("_v")))
    big = pl.concat(parts)
    n = pl.col("_m").count().over("t")
    big = big.with_columns(
        pl.col("_r6").mean().over("t").alias("mkt_ret_6"), pl.col("_r24").mean().over("t").alias("mkt_ret_24"),
        ((pl.col("_r24").rank().over("t") - 1) / (pl.col("_r24").count().over("t") - 1)).alias("xs_rank_ret_24"),
        ((pl.col("_f").rank().over("t") - 1) / (pl.col("_f").count().over("t") - 1)).alias("xs_rank_funding"),
        ((pl.col("_v").rank().over("t") - 1) / (pl.col("_v").count().over("t") - 1)).alias("xs_rank_vol"))
    out = {}
    for m, df in frames.items():
        x = big.filter(pl.col("_m") == m).select("t", "mkt_ret_6", "mkt_ret_24", "xs_rank_ret_24", "xs_rank_funding", "xs_rank_vol")
        out[m] = df.join(x, on="t", how="left")
    return out


class DataHub:
    def __init__(self, adapter, root: str = "data/candles", markets=None, timeframes=("1h", "4h"), days: int = 540,
                 max_age_h: float = 24, top_n: int = 17, min_day_volume: float = 5e6):
        self.a, self.root, self.fixed, self.tfs, self.days, self.max_age = adapter, Path(root), markets, timeframes, days, max_age_h
        self.top_n, self.min_vol = top_n, min_day_volume
        self.root.mkdir(parents=True, exist_ok=True)
        self._hist: dict = {}
        self._live: dict = {}
        self._uni: tuple | None = None
        self.on_progress = None                      # called while downloading so a watchdog knows we are alive

    # ---------- universe ----------
    def universe_info(self) -> dict:
        """market -> 24h notional volume, for the most liquid perps. Refreshed daily, cached on disk."""
        path = self.root / "universe.json"
        if self._uni and time.time() - self._uni[0] < self.max_age * 3600:
            return self._uni[1]
        if path.exists() and time.time() - path.stat().st_mtime < self.max_age * 3600:
            info = json.loads(path.read_text())
        else:
            if self.fixed:
                ctx = {c.market: c.day_notional_volume for c in self.a.all_contexts()}
                info = {m: ctx.get(m, 0.0) for m in self.fixed}
            else:
                cs = [c for c in self.a.all_contexts() if c.day_notional_volume >= self.min_vol]
                cs.sort(key=lambda c: -c.day_notional_volume)
                info = {c.market: c.day_notional_volume for c in cs[: self.top_n]}
            info.setdefault(REF, 1e9)
            path.write_text(json.dumps(info))
        self._uni = (time.time(), info)
        return info

    def universe(self) -> list[str]:
        return list(self.universe_info())

    # ---------- history ----------
    def _cached(self, name: str, build):
        path = self.root / name
        if path.exists() and time.time() - path.stat().st_mtime < self.max_age * 3600:
            return pl.read_parquet(path)
        df = build()
        df.write_parquet(path)
        return df

    def _market_frame(self, market: str, tf: str) -> pl.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.days)
        candles = lambda m: self._cached(f"{m}_{tf}_raw.parquet", lambda: self.a.candles(m, tf, start, end))
        funding = self._cached(f"{market}_funding.parquet", lambda: self.a.funding_history(market, start, end))
        return self._cached(f"{market}_{tf}_dir2.parquet", lambda: enrich(candles(market), tf, funding, candles(REF)))

    def history(self, market: str, tf: str) -> pl.DataFrame:
        if tf not in self._hist:
            frames, errors = {}, []
            for m in self.universe():
                self.on_progress and self.on_progress()
                try:
                    df = self._market_frame(m, tf)
                    if df.height >= MIN_BARS.get(tf, 1000):    # newly listed coins lack history for a fair test
                        frames[m] = df
                except Exception as e:                         # noqa: BLE001  one bad coin must not break the universe
                    errors.append(f"{m}: {e!r}"[:120])
                    continue
            if not frames:
                raise RuntimeError(f"no usable history for {tf} (min {MIN_BARS.get(tf, 1000)} bars); errors: {errors[:3]}")
            self._hist[tf] = add_cross_section(frames)
        return self._hist[tf][market]

    def history_markets(self, tf: str) -> list[str]:
        self.history(REF, tf)
        return list(self._hist[tf])

    # ---------- live ----------
    def live(self, market: str, tf: str, bars: int = 700) -> pl.DataFrame:
        """Newest closed bars for `market`, with cross-sectional features computed against the whole universe (cached ~5 min)."""
        c = self._live.get(tf)
        if c is None or time.time() - c[0] > 300:
            end = datetime.now(timezone.utc)
            start = end - timedelta(milliseconds=INTERVAL_MS[tf] * bars)
            ref = self.a.candles(REF, tf, start, end)
            frames = {}
            for m in self.universe():
                self.on_progress and self.on_progress()
                try:
                    cn = ref if m == REF else self.a.candles(m, tf, start, end)
                    frames[m] = enrich(cn, tf, self.a.funding_history(m, start, end), ref)
                except Exception:                              # noqa: BLE001
                    continue
            c = (time.time(), add_cross_section(frames))
            self._live[tf] = c
        return c[1][market]


def to_arrays(df: pl.DataFrame) -> dict:
    out = {"t": df["t"].dt.epoch("ms").cast(pl.Float64).to_numpy()}
    for col in COLS:
        out[col] = (df[col].cast(pl.Float64).fill_null(float("nan")).fill_nan(float("nan")).to_numpy()
                    if col in df.columns else np.full(df.height, np.nan))
    return out
