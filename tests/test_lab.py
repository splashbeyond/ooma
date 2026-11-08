import random
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from ooma.journal.notes import NoteStore
from ooma.journal.store import Journal
from ooma.lab.cycle import Lab, live_strategies
from ooma.lab.evaluate import Gates
from ooma.lab.ideation import TemplateIdeator, mutate
from ooma.spec import Condition, StrategySpec
from ooma.strategies.base import ScanContext
from ooma.strategies.spec_strategy import SpecStrategy

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
H4 = timedelta(hours=4)


def synth(n=1500, seed=1, phi=0.0, vol=0.01):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, vol)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, n)) * close
    return pl.DataFrame({
        "t": [T0 + H4 * i for i in range(n)], "open": open_,
        "high": np.maximum(open_, close) + wick, "low": np.minimum(open_, close) - wick,
        "close": close, "volume": 1000 * (1 + 3 * np.abs(r) / vol) * rng.uniform(0.7, 1.3, n),
        "trades": np.ones(n, dtype=int)})


def loader_for(df):
    return lambda m, tf, s, e: df.filter((pl.col("t") >= s) & (pl.col("t") < e))


def test_spec_rejects_unknown_feature_and_missing_stop_range():
    with pytest.raises(ValueError):
        Condition(lhs="os.system", op=">", rhs=1)
    with pytest.raises(ValueError):
        StrategySpec(family="x", hypothesis="h", timeframe="4h", markets=["ETH"],
                     long_when=[Condition(lhs="close", op=">", rhs="ema_20")], stop_atr=0)
    with pytest.raises(ValueError):
        StrategySpec(family="x", hypothesis="h", timeframe="4h", markets=["ETH"])   # no rules


def test_mutation_stays_valid_and_digest_stable():
    rng = random.Random(0)
    spec = TemplateIdeator().propose(1, ["ETH"], ["4h"], [], rng)[0][0]
    assert spec.digest() == spec.model_copy().digest()
    for _ in range(50):
        mutate(spec, rng)


def test_batch_equals_bar_by_bar_and_no_lookahead():
    df = synth(600, phi=0.2)
    spec = TemplateIdeator().propose(1, ["ETH"], ["4h"], [], random.Random(3))[0][0]
    spec = StrategySpec(family="mom", hypothesis="h", timeframe="4h", markets=["ETH"],
                        long_when=[Condition(lhs="ret_6", op=">", rhs=0.02)],
                        short_when=[Condition(lhs="ret_6", op="<", rhs=-0.02)])
    s = SpecStrategy(spec)
    batch = s.scan_batch(df, "ETH", "4h")
    assert batch
    for as_of, setups in list(batch.items())[:25]:
        i = df.filter(pl.col("t") == as_of - H4).row(0)  # bar that just closed
        hist = df.filter(pl.col("t") < as_of)             # nothing after as_of
        got = s.scan(ScanContext("ETH", "4h", as_of, hist))
        assert [x.id for x in got] == [x.id for x in setups]
        assert got[0].stop != got[0].entry and got[0].stop is not None


def test_random_walk_produces_no_trusted_strategies():
    """Pure noise: the lab must not hand out trust. This is the false-discovery check."""
    df = synth(1500, seed=7, phi=0.0)
    j = Journal(); lab = Lab(j, NoteStore(j), loader_for(df), TemplateIdeator(), seed=1, warmup=250)
    rep = lab.discover(T0 + H4 * 250, T0 + H4 * 1500, ["ETH"], ["4h"], n_ideas=40)
    assert rep.tried > 20
    assert rep.holdout_tested <= Gates().top_k_to_holdout
    assert rep.incubating == [] and live_strategies(j) == []


def test_real_edge_is_found_and_enters_incubation():
    """Strong trend persistence: momentum/trend families should survive train AND holdout."""
    df = synth(3000, seed=11, phi=0.35, vol=0.012)
    j = Journal(); lab = Lab(j, NoteStore(j), loader_for(df), TemplateIdeator(), seed=2, warmup=250)
    rep = lab.discover(T0 + H4 * 250, T0 + H4 * 3000, ["ETH"], ["4h"], n_ideas=60)
    stages = {r.stage for r in j.all("strategy_records")}
    assert rep.tried > 30 and "rejected_train" in stages
    assert rep.incubating, f"nothing survived: {rep}"
    assert live_strategies(j)


def test_incubating_needs_live_paper_evidence_to_be_promoted():
    from ooma.spec import StrategyRecord
    j = Journal(); ns = NoteStore(j); lab = Lab(j, ns, None, TemplateIdeator())
    spec = StrategySpec(family="m", hypothesis="h", timeframe="4h", markets=["ETH"],
                        long_when=[Condition(lhs="close", op=">", rhs="ema_20")])
    j.put("strategy_records", spec.digest(), StrategyRecord(id=spec.digest(), spec=spec, stage="incubating",
                                                            created_at=T0))
    assert lab.review_live().promoted == []                 # no evidence, no promotion
