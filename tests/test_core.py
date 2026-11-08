from datetime import datetime, timedelta, timezone

import polars as pl

from ooma.grading.grader import grade
from ooma.history.backtest import Backtester
from ooma.history.simulate import simulate
from ooma.journal.notes import NoteStore
from ooma.journal.stats import summarize
from ooma.journal.store import Journal
from ooma.learning.cycle import LearningCycle
from ooma.strategies.base import ScanContext, Strategy, all_strategies
from ooma.types import Setup

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=4)


def mk_setup(direction="long", entry=100.0, stop=98.0, targets=(106.0,), as_of=T0, id="s1", horizon=24):
    return Setup(id=id, strategy="x", strategy_version=1, market="ETH", timeframe="4h", as_of=as_of,
                 direction=direction, entry=entry, stop=stop, targets=list(targets), horizon_hours=horizon)


def bars(closes, spread=0.5, start=T0):
    return pl.DataFrame({
        "t": [start + H * i for i in range(len(closes))],
        "open": [closes[0]] + list(closes[:-1]), "high": [c + spread for c in closes],
        "low": [c - spread for c in closes], "close": list(map(float, closes)),
        "volume": [1.0] * len(closes), "trades": [1] * len(closes)})


def test_target_hit_is_good_trade():
    t = simulate(mk_setup(), bars([100, 101, 103, 107, 108]), slip_bps=0)
    assert t.exit_reason == "target"
    g = grade(mk_setup(), t)
    assert g.outcome_grade in ("A", "B") and g.process_grade == "A"


def test_stop_out_is_bad_outcome_good_process():
    t = simulate(mk_setup(), bars([100, 99, 97, 96, 95]), slip_bps=0)
    assert t.exit_reason == "stop"
    g = grade(mk_setup(), t)
    assert g.outcome_grade in ("D", "F") and g.process_grade == "A"
    assert g.r_multiple < -0.9


def test_stop_wins_when_both_inside_one_bar():
    f = pl.DataFrame({"t": [T0], "open": [100.0], "high": [107.0], "low": [97.0], "close": [100.0],
                      "volume": [1.0], "trades": [1]})
    t = simulate(mk_setup(horizon=4), f, slip_bps=0)
    assert t.exit_reason == "stop"


def test_gap_through_stop_fills_at_open_and_hurts_process():
    f = bars([100, 100])
    f = f.with_columns(pl.when(pl.col("t") == T0 + H).then(94.0).otherwise(pl.col("open")).alias("open"),
                       pl.when(pl.col("t") == T0 + H).then(93.0).otherwise(pl.col("low")).alias("low"))
    t = simulate(mk_setup(), f, slip_bps=0)
    g = grade(mk_setup(), t)
    assert t.exit_fill == 94.0 and g.stop_slippage_r > 1 and g.process_grade != "A"


def test_short_direction():
    t = simulate(mk_setup("short", 100, 102, (94,)), bars([100, 99, 96, 93, 92]), slip_bps=0)
    assert t.exit_reason == "target" and grade(mk_setup("short", 100, 102, (94,)), t).r_multiple > 1.5


def test_stats_flag_small_n_and_ci():
    s = summarize([1, -1, 2])
    assert s.n == 3 and not s.sufficient
    assert summarize([1.0] * 40).sufficient and summarize([]).n == 0


def test_notes_evidence_and_context():
    ns = NoteStore(Journal())
    a = ns.add(author="llm", kind="hypothesis", text="fade funding", strategy="funding_squeeze")
    ns.add(author="code", kind="lesson", text="refuted one", strategy="funding_squeeze")
    ns.set_status(ns.find(kind="lesson")[0].id, "refuted")
    ns.record_evidence(a.id, True, "t1"); ns.record_evidence(a.id, True, "t1"); ns.record_evidence(a.id, False, "t2")
    n = ns.find(kind="hypothesis")[0]
    assert (n.n_support, n.n_against) == (1, 1)
    ctx = ns.context_for("funding_squeeze", "ETH")
    assert [c.text for c in ctx] == ["fade funding"]


class EveryBarLong(Strategy):
    name, status = "every_bar_long", "active"

    def scan(self, ctx: ScanContext):
        c = ctx.candles["close"][-1]
        return [Setup(id=f"{ctx.market}-{ctx.as_of.isoformat()}", strategy=self.name, strategy_version=1,
                      market=ctx.market, timeframe=ctx.timeframe, as_of=ctx.as_of, direction="long",
                      entry=c, stop=c * 0.98, targets=[c * 1.02], horizon_hours=24)]


def test_backtest_replays_without_lookahead_and_grades():
    closes = [100 + (i % 7) * 0.5 + i * 0.05 for i in range(400)]
    data = bars(closes)
    seen = []

    class Spy(EveryBarLong):
        def scan(self, ctx):
            assert ctx.candles["t"][-1] + H == ctx.as_of      # last visible bar just closed
            seen.append(ctx.candles.height)
            return super().scan(ctx)

    bt = Backtester(lambda m, tf, s, e: data, warmup_bars=250)
    res = bt.run([Spy()], ["ETH"], "4h", T0, T0 + H * 400)
    assert res.grades and seen == sorted(seen)
    assert next(iter(res.stats_by_strategy())) == "every_bar_long@v1"


def test_learning_cycle_grades_and_writes_notes():
    j = Journal(); ns = NoteStore(j)
    s = mk_setup(); j.save_setup(s)
    t = simulate(s, bars([100, 103, 104, 97, 96, 95]), slip_bps=0)   # ran +1.5R then stopped
    j.save_trade(t)
    rep = LearningCycle(j, ns).run()
    assert rep.graded == 1 and "x@v1" in rep.stats and not rep.stats["x@v1"].sufficient
    assert "test_hypotheses" in rep.skipped
    assert LearningCycle(j, ns).run().graded == 0               # idempotent


def test_catalog_is_mapped_but_unimplemented():
    assert len(all_strategies(include_stubs=True)) >= 12
    assert all_strategies() == {}
