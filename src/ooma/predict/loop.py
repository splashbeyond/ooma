"""The recursive learning loop.

Every closed hour, for each market and horizon H, the world is a binary question:
    "Will the close H hours from now be above the close now?"
Many predictors answer with a probability from measurable features. When the outcome arrives:
  1. every prediction is scored against the base-rate baseline (log loss, hits, net edge after costs)
  2. every live predictor learns from the outcome (online update)
  3. ensemble weights shift toward what worked (Hedge with a floor)
  4. predictors with a statistically real edge (Bonferroni-adjusted, overlap-corrected) are marked trusted
     and written to the notebook; ones that are clearly worse than baseline are retired
  5. new hypotheses are spawned from what is working (jitter / conjunction / fresh) and from the notebook
The notebook and trusted set feed the analyst and the strategy lab (see ARCHITECTURE.md).
"""
from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

from .evolve import initial_population, offspring
from .features import BAR_MS, FEATURES
from .models import Baseline, EnsembleStat, OnlineLogit, Predictor, RuleStump, clip_p, logloss
from .store import PredictStore

DEFAULTS = dict(horizons=[1, 4, 24], cost=0.0013, margin=0.03, eta=0.05, w_floor=0.02,
                min_eff=30, alpha=0.05, cap=500, spawn_per_day=25, retire_z=-1.64, retire_eff=60,
                maint_every=24, buffer=4000, calib=500)


@dataclass
class Item:
    sid: str
    market: str
    ct: datetime
    H: int
    price0: float
    f: dict
    preds: dict
    base_p: float
    ens_p: float
    in_db: bool = False


@dataclass
class MaintReport:
    trusted: list = field(default_factory=list)
    demoted: list = field(default_factory=list)
    retired: int = 0
    spawned: int = 0


class RecursiveLoop:
    def __init__(self, store: PredictStore, notes=None, cfg: dict | None = None, seed: int = 0):
        self.s, self.notes = store, notes
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.rng = random.Random(seed)
        self.H = self.cfg["horizons"]
        self.preds: dict[int, dict[str, Predictor]] = {h: {} for h in self.H}
        self.pending: dict[tuple, deque] = {}
        self.buffer: dict[int, deque] = {h: deque(maxlen=self.cfg["buffer"]) for h in self.H}
        self.last_ct: dict[str, datetime] = {}
        self.bars = 0
        self.history: list[MaintReport] = []
        self._ev: dict[str, list[int]] = {}
        self._persist = False
        self.load()

    # ---------- persistence ----------
    def load(self):
        for r in self.s.load_predictors():
            p = Predictor.from_row(r)
            if p.horizon in self.preds:
                self.preds[p.horizon][p.id] = p
        self.last_ct = {k: datetime.fromisoformat(v) for k, v in self.s.get_meta("last_ct", {}).items()}
        self.bars = self.s.get_meta("bars", 0)
        st = self.s.get_meta("rng")
        if st:
            self.rng.setstate((st[0], tuple(st[1]), st[2]))
        for h, rows in self.s.get_meta("buffer", {}).items():
            self.buffer[int(h)].extend((f, bool(y)) for f, y in rows)
        for r in self.s.pending():
            it = Item(r["id"], r["market"], datetime.fromisoformat(r["ct"]), r["horizon"], r["price0"],
                      json.loads(r["features"]), json.loads(r["preds"]), r["base_p"], r["ens_p"], True)
            self.pending.setdefault((it.market, it.H), deque()).append(it)

    def save_predictors(self):
        self.s.save_predictors([p.to_row() for P in self.preds.values() for p in P.values()])

    def save(self):
        self._flush_evidence()
        self.save_predictors()
        self.s.set_meta("last_ct", {k: v.isoformat() for k, v in self.last_ct.items()})
        self.s.set_meta("bars", self.bars)
        v, internal, g = self.rng.getstate()
        self.s.set_meta("rng", [v, list(internal), g])
        self.s.set_meta("buffer", {str(h): [[f, y] for f, y in b] for h, b in self.buffer.items()})
        for q in self.pending.values():                       # unresolved predictions survive a restart
            for it in q:
                if not it.in_db:
                    self.s.add_snapshot(it.sid, it.market, it.ct, it.H, it.price0, it.f, it.preds, it.base_p, it.ens_p)
                    it.in_db = True

    # ---------- setup ----------
    def _add(self, p: Predictor, warm: bool = True):
        if p.id in self.preds[p.horizon]:
            return False
        if warm:
            for f, y in self.buffer[p.horizon]:
                p.update(f, y)
        self.preds[p.horizon][p.id] = p
        return True

    def warm_start(self, frames: dict[str, pl.DataFrame], born: str):
        """Fit on the first `calib` bars (labels known), spawn the starting population from them, then walk forward."""
        if any(len(P) for P in self.preds.values()):
            return
        for H in self.H:
            self._add(Baseline(H, born), warm=False)
            self._add(EnsembleStat(H, born), warm=False)
        n = self.cfg["calib"]
        for m, fr in frames.items():
            rows = fr.head(n).to_dicts()
            for i, r in enumerate(rows):
                for H in self.H:
                    if i + H < len(rows):
                        self.buffer[H].append(({k: r[k] for k in FEATURES}, rows[i + H]["close"] > r["close"]))
            if rows:
                self.last_ct[m] = rows[-1]["ct"]
        for H in self.H:
            for p in initial_population(self.buffer[H], H, born) + [OnlineLogit(H, born), ]:
                self._add(p)
            for f, y in self.buffer[H]:
                self.preds[H][f"baseline@{H}"].update(f, y)

    # ---------- the loop ----------
    def process_market(self, market: str, frame: pl.DataFrame, persist: bool = True) -> int:
        self._persist = persist
        last = self.last_ct.get(market)
        fr = frame.filter(pl.col("ct") > last) if last else frame
        n = 0
        for r in fr.drop_nulls("close").to_dicts():
            self._resolve(market, r["ct"], r["close"], persist)
            self._predict(market, r["ct"], r["close"], {k: r[k] for k in FEATURES}, persist)
            self.last_ct[market] = r["ct"]
            self.bars += 1; n += 1
            if self.bars % (self.cfg["maint_every"] * max(1, len(self.last_ct))) == 0:
                self.history.append(self.maintenance(r["ct"]))
        return n

    def _predict(self, market, ct, price, f, persist):
        for H in self.H:
            P = self.preds[H]
            preds = {}
            for pid, p in P.items():
                if p.status != "retired" and p.kind in ("stump", "logit", "baseline"):
                    x = p.predict(f)
                    if x is not None:
                        preds[pid] = x
            base_p = preds[f"baseline@{H}"]
            tw = sum(P[k].w for k in preds)
            ens_p = sum(P[k].w * v for k, v in preds.items()) / tw
            it = Item(f"{market}|{ct.isoformat()}|{H}", market, ct, H, price, f, preds, base_p, ens_p)
            if persist:
                self.s.add_snapshot(it.sid, market, ct, H, price, f, preds, base_p, ens_p)
                it.in_db = True
            self.pending.setdefault((market, H), deque()).append(it)

    def _resolve(self, market, ct, price, persist):
        for H in self.H:
            q = self.pending.get((market, H))
            while q and q[0].ct + timedelta(hours=H) <= ct:
                it = q.popleft()
                self._learn(it, price > it.price0, price / it.price0 - 1, ct)

    def _learn(self, it: Item, y: bool, ret: float, at: datetime):
        c, H, P = self.cfg, it.H, self.preds[it.H]
        ll_base = logloss(it.base_p, y)
        for pid, p in it.preds.items():
            pr = P.get(pid)
            if pr is None:
                continue
            ll = logloss(p, y)
            pr.stats.update(p, y, ll, ll_base, ret, c["cost"], c["margin"])
            pr.w *= math.exp(-c["eta"] * ll)
            if pr.note_id:
                self._ev.setdefault(pr.note_id, [0, 0])[0 if (p > 0.5) == y else 1] += 1
        P[f"ensemble@{H}"].stats.update(it.ens_p, y, logloss(it.ens_p, y), ll_base, ret, c["cost"], c["margin"])
        for pr in P.values():
            if pr.status != "retired" and pr.kind in ("stump", "logit", "baseline"):
                pr.update(it.f, y)
        ws = [p for p in P.values() if p.kind in ("stump", "logit", "baseline") and p.status != "retired"]
        mean_w = sum(p.w for p in ws) / len(ws)
        for p in ws:
            p.w = max(c["w_floor"], p.w / mean_w)
        self.buffer[H].append((it.f, y))
        if it.in_db:
            self.s.resolve_snapshot(it.sid, y, ret, at)

    # ---------- judgement, notes, evolution ----------
    def maintenance(self, ct: datetime) -> MaintReport:
        c, rep, born = self.cfg, MaintReport(), ct.isoformat()
        for H in self.H:
            P = self.preds[H]
            pool = [p for p in P.values() if p.kind in ("stump", "logit") and p.status != "retired"]
            K = max(1, len(pool))
            for p in pool:
                st, z, ne = p.stats, p.stats.z(H), p.stats.n_eff(H)
                padj = min(1.0, st.pval(H) * K)
                if p.status == "live" and ne >= c["min_eff"] and st.mean_d() > 0 and padj <= c["alpha"]:
                    p.status = "trusted"
                    self._note(p, H, z, ne, padj, promoted=True)
                    rep.trusted.append(p.id)
                elif p.status == "trusted" and z < 0.5:
                    p.status = "live"
                    self._note(p, H, z, ne, padj, promoted=False)
                    rep.demoted.append(p.id)
                if p.kind == "stump" and ne >= c["retire_eff"] and z < c["retire_z"]:
                    p.status = "retired"; rep.retired += 1
            alive = [p for p in P.values() if p.kind in ("stump", "logit") and p.status != "retired"]
            if len(alive) > c["cap"]:                          # over capacity: drop the worst proven losers
                worst = sorted((p for p in alive if p.kind == "stump" and p.stats.n_eff(H) >= 20),
                               key=lambda p: p.stats.z(H))[: len(alive) - c["cap"]]
                for p in worst:
                    p.status = "retired"; rep.retired += 1
            promising = sorted((p for p in alive if p.kind == "stump" and p.stats.n_eff(H) >= 20 and p.stats.z(H) > 1.0),
                               key=lambda p: p.id)             # sorted: spawn must not depend on DB row order
            room = max(0, c["cap"] - len([p for p in alive if p.status != "retired"]))
            for child in offspring(promising, self.buffer[H], H, self.rng, min(c["spawn_per_day"], room), born):
                rep.spawned += self._add(child)
        self._flush_evidence()
        if self._persist:
            self.save_predictors()
        return rep

    def _flush_evidence(self):
        if self.notes:
            for nid, (sup, ag) in self._ev.items():
                self.notes.add_counts(nid, sup, ag)
        self._ev = {}

    def _note(self, p: Predictor, H, z, ne, padj, promoted: bool):
        if not self.notes:
            return
        e = p.stats.edge()
        if promoted:
            n = self.notes.add(
                author="code", kind="lesson",
                text=f"{p.id}: P(up in {H}h) beats the base rate out-of-sample. skill {p.stats.mean_d():+.4f} nats/pred, "
                     f"z={z:.2f}, n_eff={ne:.0f}, adj p={padj:.3g}, hit {p.stats.hit_rate():.1%}, "
                     f"net edge after costs {('n/a' if e is None else f'{e:+.3%}')} over {p.stats.n_edge} confident calls.")
            self.notes.set_status(n.id, "supported")
            p.note_id = n.id
        elif p.note_id:
            self.notes.set_status(p.note_id, "refuted")
            self.notes.add(author="code", kind="observation", text=f"{p.id} lost its edge (z={z:.2f}, n_eff={ne:.0f}).")
            p.note_id = None
