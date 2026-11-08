"""The strategy lab. Mission: make more than you lose, found honestly.

idea -> train backtest -> top-K -> untouched holdout (multiple-testing adjusted) -> incubating
(forward paper trading) -> active (earned live evidence) -> retired (evidence turned against it).

Autonomy: the lab creates, tests, runs and retires strategies on paper by itself. It cannot touch the
risk firewall, config, or real money; those stay manual.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..journal.notes import NoteStore
from ..journal.stats import summarize
from ..journal.store import Journal
from ..spec import StrategyRecord, StrategySpec
from .evaluate import Gates, holdout_ok, rank_score, split, train_ok, window_stats
from .ideation import Ideator


@dataclass
class LabReport:
    tried: int = 0
    duplicates: int = 0
    rejected_train: int = 0
    holdout_tested: int = 0
    rejected_holdout: int = 0
    incubating: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)


class Lab:
    def __init__(self, journal: Journal, notes: NoteStore, loader, ideator: Ideator, *,
                 gates: Gates | None = None, holdout_frac=0.25, warmup=250, risk_usd=100.0, seed=0):
        self.j, self.notes, self.loader, self.ideator = journal, notes, loader, ideator
        self.g, self.holdout_frac, self.warmup, self.risk = gates or Gates(), holdout_frac, warmup, risk_usd
        self.rng = random.Random(seed)

    def _records(self) -> list[StrategyRecord]:
        return self.j.all("strategy_records")

    def _save(self, spec, stage, origin, evals, reason="", parent=None):
        self.j.put("strategy_records", spec.digest(), StrategyRecord(
            id=spec.digest(), spec=spec, stage=stage, created_at=datetime.now(timezone.utc),
            origin=origin, parent_id=parent, evaluations=[e.dump() for e in evals], reason=reason))

    def discover(self, start: datetime, end: datetime, markets: list[str], timeframes: list[str],
                 n_ideas: int = 30) -> LabReport:
        rep = LabReport()
        known = {r.id for r in self._records()}
        seeds = [r.spec for r in self._records() if r.stage in ("incubating", "active")]
        train_end = split(start, end, self.holdout_frac)

        survivors = []
        for spec, origin in self.ideator.propose(n_ideas, markets, timeframes, seeds, self.rng):
            if spec.digest() in known:
                rep.duplicates += 1
                continue
            known.add(spec.digest())
            rep.tried += 1
            ev, _ = window_stats(spec, self.loader, start, train_end, self.warmup, self.risk, "train")
            if train_ok(ev, self.g):
                survivors.append((spec, origin, ev))
            else:
                rep.rejected_train += 1
                self._save(spec, "rejected_train", origin, [ev], f"train n={ev.stats.n} avgR={ev.stats.avg_r}")

        survivors.sort(key=lambda x: rank_score(x[2]), reverse=True)
        top, rest = survivors[: self.g.top_k_to_holdout], survivors[self.g.top_k_to_holdout:]
        for spec, origin, ev in rest:                       # passed train but not looked at: holdout stays clean
            self._save(spec, "rejected_train", origin, [ev], "passed train, outranked; not holdout-tested")
        for spec, origin, tev in top:
            hev, _ = window_stats(spec, self.loader, train_end, end, self.warmup, self.risk, "holdout")
            rep.holdout_tested += 1
            if holdout_ok(hev, self.g, len(top)):
                rep.incubating.append(spec.name)
                self._save(spec, "incubating", origin, [tev, hev], "passed holdout")
                self.notes.add(author="code", kind="lesson", strategy=spec.name,
                               text=f"{spec.family} '{spec.name}' passed holdout: train {tev.stats.avg_r:.2f}R "
                                    f"(n={tev.stats.n}), holdout {hev.stats.avg_r:.2f}R (n={hev.stats.n}, p={hev.p:.3f}). "
                                    f"Hypothesis: {spec.hypothesis} Now forward-testing on paper.")
            else:
                rep.rejected_holdout += 1
                self._save(spec, "rejected_holdout", origin, [tev, hev], "failed holdout")
                self.notes.add(author="code", kind="lesson", strategy=spec.family,
                               text=f"Overfit: {spec.family} looked good in train ({tev.stats.avg_r:.2f}R, n={tev.stats.n}) "
                                    f"but holdout was {hev.stats.avg_r or 0:.2f}R (n={hev.stats.n}). "
                                    f"Params: stop {spec.stop_atr} ATR, target {spec.target_r}R.")
        fam = Counter(r.spec.family for r in self._records() if r.stage == "rejected_train")
        if fam:
            self.notes.add(author="code", kind="observation",
                           text=f"Train rejections so far by family: {dict(fam)}.")
        return rep

    def review_live(self, rep: LabReport | None = None) -> LabReport:
        """Forward paper evidence decides trust. No backtest can promote a strategy to active."""
        rep = rep or LabReport()
        paper = {t.id for t in self.j.all("trades") if t.source == "paper"}
        for r in self._records():
            if r.stage not in ("incubating", "active"):
                continue
            rs = [g.r_multiple for g in self.j.all("grades")
                  if g.strategy == r.spec.name and g.trade_id in paper]
            s = summarize(rs)
            if not s.sufficient:
                continue
            if r.stage == "incubating" and s.ci_low > 0:
                r.stage, r.reason = "active", f"live paper n={s.n} avgR={s.avg_r:.2f} CI low {s.ci_low:.2f}"
                rep.promoted.append(r.spec.name)
            elif s.ci_high < 0:
                r.stage, r.reason = "retired", f"live paper n={s.n} avgR={s.avg_r:.2f} CI high {s.ci_high:.2f}"
                rep.retired.append(r.spec.name)
            else:
                continue
            self.j.put("strategy_records", r.id, r)
            self.notes.add(author="code", kind="lesson", strategy=r.spec.name, text=f"{r.spec.name}: {r.reason} -> {r.stage}.")
        return rep


def live_strategies(journal: Journal):
    """Strategies the scanner/paper loop should run: everything incubating or active."""
    from ..strategies.spec_strategy import SpecStrategy
    return [SpecStrategy(r.spec) for r in journal.all("strategy_records") if r.stage in ("incubating", "active")]
