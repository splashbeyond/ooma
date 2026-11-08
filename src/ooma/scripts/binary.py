"""The binary judge. A script answers two yes/no questions per closed bar:
    direction: will the price be higher after HORIZON_BARS?   act: should we take this trade?
Trades enter at the NEXT bar's open (the signal bar's close is not tradable) and exit at the close of bar i+H.

Graded on NON-OVERLAPPING acted trades only (so counts are independent), against the best constant guess
(always-up or always-down over the same bars), with an exact one-sided binomial test, and on net return after costs.
Code only: the LLM never grades its own work and never sees holdout results.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

COST = 0.0013          # round-trip fees + slippage as a fraction (HL taker both ways + spread), matches execution defaults


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p), exact, computed in log space."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    p = min(max(p, 1e-12), 1 - 1e-12)
    lp, lq = math.log(p), math.log(1 - p)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs))


@dataclass
class BinaryResult:
    n_bars: int = 0
    n_acts: int = 0              # all acted bars (overlapping)
    n: int = 0                   # independent acted trades (non-overlapping): the sample size that counts
    hits: int = 0
    hit_rate: float | None = None
    chance: float | None = None  # best constant guess over the same bars
    p: float = 1.0               # P(hits this good | no skill)
    net_mean: float | None = None    # mean net return per trade after costs
    net_t: float | None = None
    up_rate: float | None = None

    def dump(self):
        return asdict(self)


def judge(direction: np.ndarray, act: np.ndarray, open_: np.ndarray, close: np.ndarray, H: int,
          lo: int, hi: int, cost: float = COST) -> BinaryResult:
    """Decisions at bar indices [lo, hi) (decision i uses data through bar i). `direction`/`act` are indexed by bar."""
    n = len(close)
    idx = [i for i in range(lo, min(hi, n - H - 0)) if i + H < n and i + 1 < n]
    r = BinaryResult(n_bars=len(idx))
    if not idx:
        return r
    i_arr = np.array(idx)
    ret = close[i_arr + H] / open_[i_arr + 1] - 1.0
    up = ret > 0
    r.up_rate = float(up.mean())
    acted = act[i_arr]
    r.n_acts = int(acted.sum())
    chosen, last = [], -10 ** 9
    for j, i in enumerate(i_arr):
        if acted[j] and i >= last + H:
            chosen.append(j)
            last = i
    if not chosen:
        return r
    c = np.array(chosen)
    d = direction[i_arr][c]
    correct = d == up[c]
    net = np.where(d, ret[c], -ret[c]) - cost
    r.n, r.hits = len(c), int(correct.sum())
    r.hit_rate = r.hits / r.n
    r.chance = float(max(up.mean(), 1 - up.mean()))
    r.p = binom_sf(r.hits, r.n, r.chance)
    r.net_mean = float(net.mean())
    if r.n >= 3 and net.std(ddof=1) > 0:
        r.net_t = float(net.mean() / (net.std(ddof=1) / math.sqrt(r.n)))
    return r


def pool(results: list[BinaryResult]) -> BinaryResult:
    """Combine per-market results (independent series) into one verdict."""
    rs = [r for r in results if r.n > 0]
    out = BinaryResult(n_bars=sum(r.n_bars for r in results), n_acts=sum(r.n_acts for r in results))
    if not rs:
        return out
    out.n, out.hits = sum(r.n for r in rs), sum(r.hits for r in rs)
    out.hit_rate = out.hits / out.n
    out.chance = sum(r.chance * r.n for r in rs) / out.n
    out.p = binom_sf(out.hits, out.n, out.chance)
    out.net_mean = sum(r.net_mean * r.n for r in rs) / out.n
    ts = [r.net_t for r in rs if r.net_t is not None]
    out.net_t = float(sum(ts) / math.sqrt(len(ts))) if ts else None      # Stouffer-style combination
    out.up_rate = sum(r.up_rate * r.n_bars for r in rs) / max(1, sum(r.n_bars for r in rs))
    return out


@dataclass
class Gates:
    min_train_n: int = 30
    min_lift: float = 0.02           # hit rate must beat chance by this much in train
    min_holdout_n: int = 20
    max_holdout_p: float = 0.05      # Bonferroni-adjusted by holdout looks in the session
    fwd_min_n: int = 30
    fwd_max_p: float = 0.05          # adjusted by number of scripts incubating at once


def train_ok(r: BinaryResult, g: Gates) -> bool:
    return r.n >= g.min_train_n and (r.hit_rate or 0) - (r.chance or 1) >= g.min_lift and (r.net_mean or -1) > 0


def holdout_ok(r: BinaryResult, g: Gates, looks: int) -> bool:
    return (r.n >= g.min_holdout_n and (r.hit_rate or 0) > (r.chance or 1) and (r.net_mean or -1) > 0
            and min(1.0, r.p * max(1, looks)) <= g.max_holdout_p)


def forward_verdict(r: BinaryResult, g: Gates, k_incubating: int) -> str:
    """'promote' | 'retire' | 'wait' from FORWARD (live paper) calls only."""
    if r.n < g.fwd_min_n:
        return "wait"
    if (r.net_mean or -1) > 0 and (r.hit_rate or 0) > (r.chance or 1) and min(1.0, r.p * max(1, k_incubating)) <= g.fwd_max_p:
        return "promote"
    if (r.net_mean or 0) < 0 and (r.net_t or 0) < -1.64:
        return "retire"
    return "wait"
