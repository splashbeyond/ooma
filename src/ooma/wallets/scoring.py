"""Skill = statistically credible, net-of-fees, DIRECTIONAL profit. Walk-forward: a score at time T may use only fills before T.

A wallet is 'skilled' only if ALL hold: directional (not MM/hedger/carry/vault), enough closes and days,
daily-PnL t-stat survives a Bonferroni correction for the number of wallets screened, positive net, and the
profit is not a few lucky trades. 'weak' = significantly negative. Everything else is 'neutral' and ignored.
Selecting on past PnL still overfits: persistence_report() measures whether skill carries into the NEXT period.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from .profile import DEFAULTS, Profile, prepare, profile_wallet


def pvalue_pos(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def status(p: Profile, n_screened: int, cfg: dict | None = None) -> str:
    c = {**DEFAULTS, **(cfg or {})}
    if p.kind != "directional":
        return f"excluded:{p.kind}"
    if p.n_closes < c["min_closes"] or p.active_days < c["min_active_days"]:
        return "neutral"
    k = max(1, n_screened)
    p_good, p_bad = min(1.0, pvalue_pos(p.daily_t) * k), min(1.0, pvalue_pos(-p.daily_t) * k)
    if p.daily_t > 0 and p.net_pnl > 0 and p_good <= c["alpha"]:
        return "skilled" if not any("concentrated" in f for f in p.flags) else "neutral"
    if p.daily_t < 0 and p.net_pnl < 0 and p_bad <= c["alpha"]:
        return "weak"
    return "neutral"


def score_as_of(address: str, fills: list[dict], as_of_ms: int, **kw) -> Profile:
    """Profile using only fills strictly before as_of (no look-ahead)."""
    df = prepare(fills)
    return profile_wallet(address, df.filter(pl.col("time") < as_of_ms), **kw)


def persistence_report(fill_sets: dict[str, pl.DataFrame], min_days: int = 20) -> dict:
    """Does first-half skill predict second-half profit? Spearman correlation across wallets between the first
    half's daily t-stat and the second half's net bps per $ traded, plus top-quintile vs the rest."""
    xs, ys = [], []
    for addr, df in fill_sets.items():
        if df.height < 40:
            continue
        mid = (df["time"].min() + df["time"].max()) // 2
        a, b = df.filter(pl.col("time") < mid), df.filter(pl.col("time") >= mid)
        pa, pb = profile_wallet(addr, a, cfg=dict(min_fills=20, min_days=min_days // 2)), \
            profile_wallet(addr, b, cfg=dict(min_fills=20, min_days=min_days // 2))
        if pa.kind == "sparse" or pb.kind == "sparse" or pb.notional <= 0:
            continue
        xs.append(pa.daily_t); ys.append(pb.pnl_bps)
    n = len(xs)
    if n < 10:
        return {"n": n, "spearman": None, "note": "too few wallets"}
    rx, ry = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    q = max(1, n // 5)
    order = np.argsort(xs)[::-1]
    top, rest = np.array(ys)[order[:q]], np.array(ys)[order[q:]]
    return {"n": n, "spearman": rho, "z": rho * math.sqrt(n - 1),
            "top_quintile_bps": float(top.mean()), "rest_bps": float(rest.mean())}
