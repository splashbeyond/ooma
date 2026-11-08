"""Predictors: anything that maps measurables to P(price up in H hours). All learn online from outcomes."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from .features import FEATURES


def clip_p(p: float) -> float:
    return min(0.99, max(0.01, p))


def logloss(p: float, y: bool) -> float:
    p = clip_p(p)
    return -math.log(p if y else 1 - p)


@dataclass
class PredStats:
    """Running score vs the base-rate baseline, measured only on predictions made BEFORE the outcome."""
    n: int = 0
    sum_d: float = 0.0       # sum of (baseline logloss - my logloss): positive = better than baseline
    sum_d2: float = 0.0
    hits: int = 0
    sum_ll: float = 0.0
    n_edge: int = 0          # predictions confident enough to trade
    sum_edge: float = 0.0    # sum of signed return minus costs for those

    def update(self, p, y, ll, ll_base, ret, cost, margin):
        d = ll_base - ll
        self.n += 1; self.sum_d += d; self.sum_d2 += d * d; self.sum_ll += ll
        self.hits += int((p > 0.5) == y)
        if p > 0.5 + margin:
            self.n_edge += 1; self.sum_edge += ret - cost
        elif p < 0.5 - margin:
            self.n_edge += 1; self.sum_edge += -ret - cost

    def n_eff(self, H: int) -> float:
        return self.n / max(1, H)               # H-bar labels sampled every bar overlap: n/H is honest

    def mean_d(self) -> float:
        return self.sum_d / self.n if self.n else 0.0

    def z(self, H: int) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean_d()
        var = max(self.sum_d2 / self.n - m * m, 1e-12)
        return m / math.sqrt(var / max(self.n_eff(H), 1e-9))

    def pval(self, H: int) -> float:             # one-sided: better than baseline?
        return 0.5 * math.erfc(self.z(H) / math.sqrt(2))

    def edge(self) -> float | None:
        return self.sum_edge / self.n_edge if self.n_edge else None

    def hit_rate(self) -> float | None:
        return self.hits / self.n if self.n else None


class Predictor:
    kind = "base"

    def __init__(self, id: str, horizon: int, born: str = "", parent: str | None = None):
        self.id, self.horizon, self.born, self.parent = id, horizon, born, parent
        self.stats = PredStats()
        self.status = "live"            # live | trusted | retired
        self.note_id: str | None = None
        self.w = 1.0                    # ensemble weight

    def predict(self, f: dict) -> float | None: raise NotImplementedError
    def update(self, f: dict, y: bool) -> None: raise NotImplementedError
    def params(self) -> dict: return {}
    def state(self) -> dict: return {}
    def load_state(self, s: dict) -> None: pass

    def to_row(self) -> dict:
        return dict(id=self.id, horizon=self.horizon, kind=self.kind, params=json.dumps(self.params()),
                    state=json.dumps(self.state()), stats=json.dumps(asdict(self.stats)), status=self.status,
                    note_id=self.note_id, born=self.born, parent=self.parent, w=self.w)

    @staticmethod
    def from_row(r: dict) -> "Predictor":
        params = json.loads(r["params"])
        cls = {"stump": RuleStump, "logit": OnlineLogit, "baseline": Baseline, "ensemble": EnsembleStat}[r["kind"]]
        p = cls._build(r["id"], r["horizon"], params, r["born"], r["parent"])
        p.load_state(json.loads(r["state"]))
        p.stats = PredStats(**json.loads(r["stats"]))
        p.status, p.note_id, p.w = r["status"], r["note_id"], r["w"]
        return p


class Baseline(Predictor):
    """The bar to beat: the recent base rate of 'up'. Everything is scored against this."""
    kind = "baseline"

    def __init__(self, horizon, born="", parent=None):
        super().__init__(f"baseline@{horizon}", horizon, born, parent)
        self.up, self.n = 0.0, 0.0
        self.w = 20.0

    @classmethod
    def _build(cls, id, H, params, born, parent): return cls(H, born, parent)
    def predict(self, f): return (self.up + 1) / (self.n + 2)
    def update(self, f, y): self.up = self.up * 0.9995 + y; self.n = self.n * 0.9995 + 1
    def state(self): return {"up": self.up, "n": self.n}
    def load_state(self, s): self.up, self.n = s["up"], s["n"]


class EnsembleStat(Predictor):
    """Holds the stats of the weighted ensemble so it is judged like any other predictor."""
    kind = "ensemble"

    def __init__(self, horizon, born="", parent=None):
        super().__init__(f"ensemble@{horizon}", horizon, born, parent)

    @classmethod
    def _build(cls, id, H, params, born, parent): return cls(H, born, parent)
    def predict(self, f): return None
    def update(self, f, y): pass


class RuleStump(Predictor):
    """If ALL conditions hold -> learn P(up) of that region; else learn P(up) of the rest. Shrunk toward 0.5."""
    kind = "stump"
    PRIOR = 20

    def __init__(self, conds: list, horizon: int, born="", parent=None):
        self.conds = [(f, op, float(t)) for f, op, t in conds]
        cid = "&".join(f"{f}{op}{t:.4g}" for f, op, t in self.conds)
        super().__init__(f"stump[{cid}]@{horizon}", horizon, born, parent)
        self.cells = {1: [0, 0], 0: [0, 0]}      # [ups, n]

    @classmethod
    def _build(cls, id, H, params, born, parent): return cls(params["conds"], H, born, parent)
    def params(self): return {"conds": self.conds}
    def state(self): return {"cells": {str(k): v for k, v in self.cells.items()}}
    def load_state(self, s): self.cells = {int(k): v for k, v in s["cells"].items()}

    def holds(self, f):
        for feat, op, thr in self.conds:
            x = f.get(feat)
            if x is None:
                return None
            if (x > thr) != (op == ">"):
                return False
        return True

    def predict(self, f):
        h = self.holds(f)
        if h is None:
            return None
        up, n = self.cells[int(h)]
        return (up + 0.5 * self.PRIOR) / (n + self.PRIOR)

    def update(self, f, y):
        h = self.holds(f)
        if h is not None:
            c = self.cells[int(h)]; c[0] += int(y); c[1] += 1


class OnlineLogit(Predictor):
    """Logistic regression on standardised features, one SGD step per resolved outcome."""
    kind = "logit"

    def __init__(self, horizon, born="", parent=None, lr=0.01, l2=1e-3):
        super().__init__(f"logit@{horizon}", horizon, born, parent)
        k = len(FEATURES)
        self.lr, self.l2, self.n = lr, l2, 0
        self.mean, self.m2, self.cnt = [0.0] * k, [0.0] * k, [0] * k
        self.wt, self.b = [0.0] * k, 0.0

    @classmethod
    def _build(cls, id, H, params, born, parent): return cls(H, born, parent, params.get("lr", 0.01), params.get("l2", 1e-3))
    def params(self): return {"lr": self.lr, "l2": self.l2}
    def state(self): return dict(n=self.n, mean=self.mean, m2=self.m2, cnt=self.cnt, wt=self.wt, b=self.b)

    def load_state(self, s):
        self.n, self.mean, self.m2, self.cnt, self.wt, self.b = s["n"], s["mean"], s["m2"], s["cnt"], s["wt"], s["b"]

    def _z(self, f):
        out = []
        for i, name in enumerate(FEATURES):
            x = f.get(name)
            if x is None or self.cnt[i] < 30:
                out.append(0.0); continue
            sd = math.sqrt(self.m2[i] / self.cnt[i]) or 1.0
            out.append(max(-5.0, min(5.0, (x - self.mean[i]) / sd)))
        return out

    def _p(self, z):
        s = self.b + sum(w * x for w, x in zip(self.wt, z))
        return 1 / (1 + math.exp(-max(-30, min(30, s))))

    def predict(self, f):
        return None if self.n < 200 else self._p(self._z(f))

    def update(self, f, y):
        for i, name in enumerate(FEATURES):                      # Welford running mean/variance
            x = f.get(name)
            if x is not None:
                self.cnt[i] += 1
                dlt = x - self.mean[i]; self.mean[i] += dlt / self.cnt[i]
                self.m2[i] += dlt * (x - self.mean[i])
        z = self._z(f)
        g = self._p(z) - float(y)
        self.wt = [w - self.lr * (g * x + self.l2 * w) for w, x in zip(self.wt, z)]
        self.b -= self.lr * g
        self.n += 1
