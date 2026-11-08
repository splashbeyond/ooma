"""The recursive learning loop. Runs nightly/weekly; every step writes to the notebook.

  1. GRADE      close-out new trades -> outcome + process grades            (implemented)
  2. MEASURE    stats per strategy@version with n and CI                    (implemented)
  3. REFLECT    write notes: code rules always, LLM post-mortems if enabled (rules implemented)
  4. TEST       turn open hypotheses into backtests; train vs holdout       (stub)
  5. PROPOSE    queue config/strategy changes with evidence                 (stub)
  6. APPROVE    human approves risk/config changes; strategies auto-advance on paper via lab/ (see ARCHITECTURE.md)

Guardrails: nothing self-modifies. Risk rules are out of reach. Findings with n < 30 are not acted on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..grading.grader import grade
from ..journal.notes import NoteStore
from ..journal.stats import Stats, summarize
from ..journal.store import Journal
from ..types import Grade, Note, Setup, Trade


class Reflector(Protocol):
    def reflect(self, setup: Setup, trade: Trade, g: Grade, context: list[Note]) -> list[Note]: ...


class RuleReflector:
    """Deterministic notes, no LLM. Flags bad process and notable patterns."""
    def __init__(self, notes: NoteStore):
        self.notes = notes

    def reflect(self, setup, trade, g, context):
        out = []
        if g.process_grade in ("D", "F"):
            out.append(self.notes.add(
                author="code", kind="observation", strategy=setup.strategy, market=setup.market,
                trade_ids=[trade.id],
                text=f"Poor execution on {setup.market} {setup.direction}: entry slip {g.entry_slippage_r:.2f}R, "
                     f"stop slip {g.stop_slippage_r:.2f}R, exit {trade.exit_reason}."))
        if g.outcome_grade in ("D", "F") and g.mfe_r >= 1.0:
            out.append(self.notes.add(
                author="code", kind="hypothesis", strategy=setup.strategy, market=setup.market,
                trade_ids=[trade.id],
                text=f"{setup.strategy}: trade reached +{g.mfe_r:.1f}R before stopping out. "
                     f"Hypothesis: partial profit or earlier trail would improve expectancy."))
        return out


@dataclass
class CycleReport:
    graded: int = 0
    stats: dict[str, Stats] = field(default_factory=dict)
    notes_written: int = 0
    skipped: dict[str, str] = field(default_factory=dict)   # step -> reason


class LearningCycle:
    def __init__(self, journal: Journal, notes: NoteStore, reflector: Reflector | None = None):
        self.j, self.notes = journal, notes
        self.reflector = reflector or RuleReflector(notes)

    def run(self) -> CycleReport:
        rep = CycleReport()
        for t in self.j.ungraded_closed_trades():
            setup = self.j.get("setups", t.setup_id)
            g = grade(setup, t)
            self.j.save_grade(g)
            rep.graded += 1
            rep.notes_written += len(self.reflector.reflect(setup, t, g, self.notes.context_for(t.strategy, t.market)))

        by: dict[str, list[float]] = {}
        for g in self.j.all("grades"):
            by.setdefault(f"{g.strategy}@v{g.strategy_version}", []).append(g.r_multiple)
        rep.stats = {k: summarize(v) for k, v in by.items()}

        for step in (self.test_hypotheses, self.propose_changes):
            try:
                step()
            except NotImplementedError:
                rep.skipped[step.__name__] = "not implemented yet"
        return rep

    def test_hypotheses(self):
        """For each open hypothesis note: encode as a param variant, backtest on train, confirm on holdout,
        then record_evidence / set_status. Needs Backtester + parametrised strategies."""
        raise NotImplementedError

    def propose_changes(self):
        """Create Proposal records (strategy, param diff, train/holdout stats). Human approves via CLI."""
        raise NotImplementedError
