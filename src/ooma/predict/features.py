"""Measurables. Every feature is causal: row i uses only bars <= i, so replay == live."""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from ..modules.zones import zone_columns

BAR_MS = 3_600_000   # the loop runs on 1h bars
FEATURES = ["ret_1", "ret_4", "ret_24", "rsi_14", "z_20", "z_50", "volx_20", "atr_pct",
            "ema_dist_20", "ema_dist_100", "hh_prox_24", "ll_prox_24", "range_pos_48", "rv_ratio",
            "body_pct", "wick_up", "wick_dn", "hour", "dow", "funding", "funding_z",
            "btc_ret_1", "btc_ret_4", "sup_dist", "res_dist", "sup_touches", "res_touches",
            "skill_net", "weak_net", "skill_chg_4h"]


def feature_frame(candles: pl.DataFrame, funding: pl.DataFrame | None = None,
                  ref: pl.DataFrame | None = None, cohort: pl.DataFrame | None = None) -> pl.DataFrame:
    """Columns: t (bar open), ct (bar close = when the info is known), close, FEATURES.
    funding: t, funding_rate (hourly). ref: reference-market candles (BTC) for cross-market features."""
    c, h, l, o, v = (pl.col(x) for x in ("close", "high", "low", "open", "volume"))
    pc = c.shift(1)
    tr = pl.max_horizontal(h - l, (h - pc).abs(), (l - pc).abs())
    r1 = c / pc - 1
    d = c.diff()
    gain = d.clip(lower_bound=0).ewm_mean(alpha=1 / 14, adjust=False)
    loss = (-d).clip(lower_bound=0).ewm_mean(alpha=1 / 14, adjust=False)
    ct = pl.col("t") + pl.duration(milliseconds=BAR_MS)
    df = candles.sort("t").with_columns(
        ct.alias("ct"), r1.alias("ret_1"), (c / c.shift(4) - 1).alias("ret_4"),
        (c / c.shift(24) - 1).alias("ret_24"), (100 - 100 / (1 + gain / loss)).alias("rsi_14"),
        ((c - c.rolling_mean(20)) / c.rolling_std(20)).alias("z_20"),
        ((c - c.rolling_mean(50)) / c.rolling_std(50)).alias("z_50"),
        (v / v.rolling_mean(20)).alias("volx_20"), (tr.rolling_mean(14) / c).alias("atr_pct"),
        (c / c.ewm_mean(span=20, adjust=False) - 1).alias("ema_dist_20"),
        (c / c.ewm_mean(span=100, adjust=False) - 1).alias("ema_dist_100"),
        (c / h.rolling_max(24) - 1).alias("hh_prox_24"), (c / l.rolling_min(24) - 1).alias("ll_prox_24"),
        ((c - l.rolling_min(48)) / (h.rolling_max(48) - l.rolling_min(48))).alias("range_pos_48"),
        (r1.rolling_std(6) / r1.rolling_std(48)).alias("rv_ratio"),
        ((c - o) / o).alias("body_pct"), ((h - pl.max_horizontal(o, c)) / c).alias("wick_up"),
        ((pl.min_horizontal(o, c) - l) / c).alias("wick_dn"),
        ct.dt.hour().cast(pl.Float64).alias("hour"), ct.dt.weekday().cast(pl.Float64).alias("dow"))
    if funding is not None and funding.height:
        f = funding.select((pl.col("t") + pl.duration(minutes=1)).alias("ct_f"),
                           (pl.col("funding_rate") * 1e4).alias("funding")).sort("ct_f")
        df = df.join_asof(f, left_on="ct", right_on="ct_f", strategy="backward").drop("ct_f")
        fz = (pl.col("funding") - pl.col("funding").rolling_mean(72)) / pl.col("funding").rolling_std(72)
        df = df.with_columns(fz.alias("funding_z"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding"),
                             pl.lit(None, dtype=pl.Float64).alias("funding_z"))
    if ref is not None and ref.height:
        rc = pl.col("close")
        rf = ref.sort("t").select("t", (rc / rc.shift(1) - 1).alias("btc_ret_1"), (rc / rc.shift(4) - 1).alias("btc_ret_4"))
        df = df.join(rf, on="t", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("btc_ret_1"), pl.lit(None, dtype=pl.Float64).alias("btc_ret_4"))
    if cohort is not None and cohort.height:        # wallet snapshots exist only from the first live snapshot on
        co = cohort.select("t", "skill_net", "weak_net", "skill_chg_4h").rename({"t": "_ts"}).sort("_ts")
        df = df.join_asof(co, left_on="ct", right_on="_ts", strategy="backward",
                          tolerance=timedelta(hours=3)).drop("_ts")
    else:
        df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in ("skill_net", "weak_net", "skill_chg_4h")])
    atr = pl.max_horizontal(pl.col("high") - pl.col("low"), (pl.col("high") - pl.col("close").shift(1)).abs(),
                            (pl.col("low") - pl.col("close").shift(1)).abs()).rolling_mean(14)
    z = zone_columns(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
                     df.select(atr.alias("a"))["a"].to_numpy())
    df = df.with_columns([pl.Series(k, v) for k, v in z.items()])
    clean = [pl.when(pl.col(f).is_infinite()).then(None).otherwise(pl.col(f)).fill_nan(None).alias(f) for f in FEATURES]
    return df.with_columns(clean).select("t", "ct", "close", *FEATURES)
