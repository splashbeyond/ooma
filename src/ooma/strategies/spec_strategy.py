"""Runs a StrategySpec as a Strategy. Features are causal, so batch and bar-by-bar give identical setups."""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from ..adapters.hyperliquid import INTERVAL_MS
from ..spec import EXTRA, PARAM, Condition, StrategySpec
from ..types import Setup
from .base import ScanContext, Strategy


def _expr(name: str) -> pl.Expr:
    if name in ("open", "high", "low", "close", "volume") or name in EXTRA:
        return pl.col(name)
    kind, n = PARAM.match(name).group(1), int(PARAM.match(name).group(2))
    c, h, l, v = pl.col("close"), pl.col("high"), pl.col("low"), pl.col("volume")
    if kind == "ema": return c.ewm_mean(span=n, adjust=False)
    if kind == "sma": return c.rolling_mean(n)
    if kind == "atr":
        pc = c.shift(1)
        return pl.max_horizontal(h - l, (h - pc).abs(), (l - pc).abs()).rolling_mean(n)
    if kind == "rsi":
        d = c.diff()
        g = d.clip(lower_bound=0).ewm_mean(alpha=1 / n, adjust=False)
        ls = (-d).clip(lower_bound=0).ewm_mean(alpha=1 / n, adjust=False)
        return 100 - 100 / (1 + g / ls)
    if kind == "ret": return c / c.shift(n) - 1
    if kind == "z": return (c - c.rolling_mean(n)) / c.rolling_std(n)
    if kind == "hh": return h.shift(1).rolling_max(n)
    if kind == "ll": return l.shift(1).rolling_min(n)
    if kind == "volx": return v / v.rolling_mean(n)
    raise ValueError(name)


def _operand(x) -> pl.Expr:
    return pl.lit(x) if isinstance(x, (int, float)) else _expr(x)


def _prev(x) -> pl.Expr:
    return pl.lit(x) if isinstance(x, (int, float)) else _expr(x).shift(1)


def _cond(c: Condition) -> pl.Expr:
    a, b = _operand(c.lhs), _operand(c.rhs)
    if c.op == ">": return a > b
    if c.op == "<": return a < b
    pa, pb = _prev(c.lhs), _prev(c.rhs)
    return ((a > b) & (pa <= pb)) if c.op == "cross_above" else ((a < b) & (pa >= pb))


def _all(conds: list[Condition]) -> pl.Expr:
    if not conds:
        return pl.lit(False)
    e = _cond(conds[0])
    for c in conds[1:]:
        e = e & _cond(c)
    return e.fill_null(False)


class SpecStrategy(Strategy):
    status = "active"

    def __init__(self, spec: StrategySpec, version: int = 1):
        self.spec, self.name, self.version = spec, spec.name, version
        self.description = spec.hypothesis

    def scan_batch(self, candles: pl.DataFrame, market: str, timeframe: str) -> dict:
        """as_of -> setups, for every closed bar. Used by the backtester for speed."""
        s = self.spec
        if market not in s.markets or timeframe != s.timeframe:
            return {}
        step = timedelta(milliseconds=INTERVAL_MS[timeframe])
        used = {x for c in s.long_when + s.short_when for x in (c.lhs, c.rhs) if isinstance(x, str)} & EXTRA
        if used - set(candles.columns):
            return {}                                   # needs enriched candles (lab/enrich.py)
        d = candles.with_columns(_all(s.long_when).alias("_L"), _all(s.short_when).alias("_S"),
                                 _expr("atr_14").alias("_atr"))
        out: dict = {}
        for r in d.filter((pl.col("_L") ^ pl.col("_S")) & pl.col("_atr").is_not_null()).to_dicts():
            sign = 1 if r["_L"] else -1
            entry, risk = r["close"], s.stop_atr * r["_atr"]
            as_of = r["t"] + step
            out[as_of] = [Setup(
                id=f"{self.name}-{market}-{as_of.isoformat()}", strategy=self.name,
                strategy_version=self.version, market=market, timeframe=timeframe, as_of=as_of,
                direction="long" if sign == 1 else "short", entry=entry, stop=entry - sign * risk,
                targets=[entry + sign * s.target_r * risk],
                horizon_hours=max(1, int(s.horizon_bars * step.total_seconds() / 3600)),
                rationale=s.hypothesis, features={"atr_14": r["_atr"], "close": entry})]
        return out

    def scan(self, ctx: ScanContext) -> list[Setup]:
        return self.scan_batch(ctx.candles, ctx.market, ctx.timeframe).get(ctx.as_of, [])
