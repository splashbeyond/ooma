"""Spawning new hypotheses. Every offspring is judged only on outcomes AFTER its birth, so
selecting parents from past results never contaminates the offspring's own score."""
from __future__ import annotations

import random

import numpy as np

from .features import FEATURES
from .models import Predictor, RuleStump


def _values(buf, feat) -> np.ndarray:
    return np.array([r[0][feat] for r in buf if r[0].get(feat) is not None], dtype=float)


def fresh_stump(buf, H, rng: random.Random, born: str) -> RuleStump | None:
    feat = rng.choice(FEATURES)
    v = _values(buf, feat)
    if len(v) < 100 or v.std() == 0:
        return None
    return RuleStump([(feat, ">", float(np.quantile(v, rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))))],
                     H, born, parent=None)


def jitter(p: RuleStump, buf, rng: random.Random, born: str) -> RuleStump | None:
    conds = []
    for f, op, thr in p.conds:
        v = _values(buf, f)
        if len(v) < 100:
            return None
        q = float((v < thr).mean()) + rng.choice([-0.1, -0.05, 0.05, 0.1])
        conds.append((f, op, float(np.quantile(v, min(0.95, max(0.05, q))))))
    return RuleStump(conds, p.horizon, born, parent=p.id)


def conjoin(a: RuleStump, b: RuleStump, born: str) -> RuleStump | None:
    feats = {c[0] for c in a.conds}
    if len(a.conds) + len(b.conds) > 3 or any(c[0] in feats for c in b.conds):
        return None
    return RuleStump(a.conds + b.conds, a.horizon, born, parent=f"{a.id}+{b.id}")


def initial_population(buf, H, born: str) -> list[Predictor]:
    out = []
    for feat in FEATURES:
        v = _values(buf, feat)
        if len(v) < 100 or v.std() == 0:
            continue
        for q in (0.2, 0.4, 0.6, 0.8):
            out.append(RuleStump([(feat, ">", float(np.quantile(v, q)))], H, born))
    return out


def offspring(promising: list[RuleStump], buf, H, rng: random.Random, n: int, born: str) -> list[Predictor]:
    out = []
    for i in range(n * 3):
        if len(out) >= n:
            break
        r = rng.random()
        if promising and r < 0.35:
            c = jitter(rng.choice(promising), buf, rng, born)
        elif len(promising) >= 2 and r < 0.65:
            c = conjoin(*rng.sample(promising, 2), born)
        else:
            c = fresh_stump(buf, H, rng, born)
        if c is not None:
            out.append(c)
    return out
