"""Adds context columns to candles so strategy specs can use more than price: S/R zones,
funding, and cross-market (BTC) strength. All causal. Columns match spec.EXTRA."""
from __future__ import annotations

import polars as pl

from ..adapters.hyperliquid import INTERVAL_MS
from ..modules.zones import zone_columns


def enrich(candles: pl.DataFrame, tf: str, funding: pl.DataFrame | None = None,
           ref: pl.DataFrame | None = None, zone_cfg: dict | None = None) -> pl.DataFrame:
    zc = {"L": 5, "k": 0.5, "window": 300, **(zone_cfg or {})}
    c, h, l = pl.col("close"), pl.col("high"), pl.col("low")
    pc = c.shift(1)
    df = candles.sort("t").with_columns(
        pl.max_horizontal(h - l, (h - pc).abs(), (l - pc).abs()).rolling_mean(14).alias("_atr"),
        (pl.col("t") + pl.duration(milliseconds=INTERVAL_MS[tf])).alias("_ct"))
    if "trades" in df.columns:
        df = df.with_columns((pl.col("trades") / pl.col("trades").rolling_mean(24)).alias("tradesx_24"))
    z = zone_columns(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
                     df["_atr"].to_numpy(), **zc)
    df = df.with_columns([pl.Series(k, v) for k, v in z.items()]).with_columns(
        [pl.col(k).fill_nan(None) for k in z])
    if funding is not None and funding.height:
        f = funding.select((pl.col("t") + pl.duration(minutes=1)).alias("_cf"),
                           (pl.col("funding_rate") * 1e4).alias("funding")).sort("_cf")
        df = df.join_asof(f, left_on="_ct", right_on="_cf", strategy="backward").drop("_cf")
        df = df.with_columns(((pl.col("funding") - pl.col("funding").rolling_mean(72))
                              / pl.col("funding").rolling_std(72)).alias("funding_z"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding"),
                             pl.lit(None, dtype=pl.Float64).alias("funding_z"))
    if ref is not None and ref.height:
        rc = pl.col("close")
        rf = ref.sort("t").select("t", *[(rc / rc.shift(n) - 1).alias(f"btc_ret_{n}") for n in (1, 6, 24)])
        df = df.join(rf, on="t", how="left")
    else:
        df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(f"btc_ret_{n}") for n in (1, 6, 24)])
    df = df.with_columns([(c / c.shift(n) - 1 - pl.col(f"btc_ret_{n}")).alias(f"rel_btc_{n}") for n in (6, 24)])
    inf_clean = [pl.when(pl.col(x).is_infinite()).then(None).otherwise(pl.col(x)).alias(x)
                 for x in ("funding_z",)]
    return df.with_columns(inf_clean).drop("_atr", "_ct")
