"""Judging candidates. The only thing that counts: net R after fees and slippage on unseen data."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from ..history.backtest import Backtester, Loader
from ..journal.stats import Stats, p_value_positive, summarize
from ..spec import StrategySpec
from ..strategies.spec_strategy import SpecStrategy


@dataclass
class Gates:
    min_train_n: int = 30
    min_train_avg_r: float = 0.10
    top_k_to_holdout: int = 3
    min_holdout_n: int = 15
    max_holdout_p: float = 0.05     # Bonferroni-adjusted by number of holdout looks in the batch


@dataclass
class Evaluation:
    stage: str
    stats: Stats
    p: float | None = None

    def dump(self) -> dict:
        return {"stage": self.stage, **self.stats.model_dump(), "p": self.p}


def window_stats(spec: StrategySpec, loader: Loader, start: datetime, end: datetime,
                 warmup: int, risk_usd: float, stage: str) -> tuple[Evaluation, list[float]]:
    res = Backtester(loader, warmup_bars=warmup, risk_usd=risk_usd).run(
        [SpecStrategy(spec)], spec.markets, spec.timeframe, start, end)
    rs = [g.r_multiple for g in res.grades]
    return Evaluation(stage, summarize(rs), p_value_positive(rs)), rs


def split(start: datetime, end: datetime, holdout_frac: float) -> datetime:
    return start + (end - start) * (1 - holdout_frac)


def train_ok(ev: Evaluation, g: Gates) -> bool:
    s = ev.stats
    return s.n >= g.min_train_n and (s.avg_r or 0) >= g.min_train_avg_r


def rank_score(ev: Evaluation) -> float:
    return (ev.stats.avg_r or 0) * math.sqrt(ev.stats.n)     # t-stat-like: rewards edge AND sample size


def holdout_ok(ev: Evaluation, g: Gates, n_looks: int) -> bool:
    s = ev.stats
    return (s.n >= g.min_holdout_n and (s.avg_r or 0) > 0
            and ev.p is not None and min(1.0, ev.p * max(1, n_looks)) <= g.max_holdout_p)
