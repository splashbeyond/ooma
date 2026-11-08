"""Honest statistics: always report n and a confidence interval; flag n < 30."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

MIN_N = 30


class Stats(BaseModel):
    n: int
    hit_rate: float | None = None
    avg_r: float | None = None
    ci_low: float | None = None      # bootstrap 95% CI on avg_r
    ci_high: float | None = None
    sufficient: bool = False         # False when n < MIN_N: do not act on this


def summarize(rs: list[float], n_boot: int = 2000, seed: int = 0) -> Stats:
    n = len(rs)
    if n == 0:
        return Stats(n=0)
    a = np.asarray(rs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return Stats(n=n, hit_rate=float((a > 0).mean()), avg_r=float(a.mean()),
                 ci_low=float(lo), ci_high=float(hi), sufficient=n >= MIN_N)


def p_value_positive(rs: list[float], n_boot: int = 4000, seed: int = 0) -> float:
    """One-sided bootstrap p-value for 'true mean R <= 0'. Small = real edge is likely. Floor 1/n_boot."""
    if not rs:
        return 1.0
    a = np.asarray(rs, dtype=float)
    means = np.random.default_rng(seed).choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    return float(max((means <= 0).mean(), 1 / n_boot))
