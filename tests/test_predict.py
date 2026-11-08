from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import polars as pl

from ooma.journal.notes import NoteStore
from ooma.journal.store import Journal
from ooma.predict.features import FEATURES, feature_frame
from ooma.predict.loop import RecursiveLoop
from ooma.predict.models import OnlineLogit, PredStats, RuleStump, logloss
from ooma.predict.store import PredictStore

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
H1 = timedelta(hours=1)


def synth(n, seed, phi=0.0, vol=0.004):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, vol)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, n)) * close
    return pl.DataFrame({"t": [T0 + H1 * i for i in range(n)], "open": open_,
                         "high": np.maximum(open_, close) + wick, "low": np.minimum(open_, close) - wick,
                         "close": close, "volume": 1000 * (1 + 3 * np.abs(r) / vol) * rng.uniform(0.7, 1.3, n),
                         "trades": np.ones(n, dtype=int)})


def mk_loop(cfg=None, seed=0, db=":memory:"):
    j = Journal(db)
    return RecursiveLoop(PredictStore(j.con), NoteStore(j), cfg, seed=seed), j


def test_features_are_causal():
    df = synth(400, 1)
    full = feature_frame(df)
    cut = feature_frame(df.head(300))
    for name in FEATURES:
        a, b = full.head(300)[name].to_list(), cut[name].to_list()
        assert all((x is None and y is None) or abs(x - y) < 1e-9 for x, y in zip(a, b)), name


def test_stump_learns_a_planted_rule_and_stats_math():
    s = RuleStump([("ret_1", ">", 0.0)], 1)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x = rng.normal(); y = bool(rng.random() < (0.7 if x > 0 else 0.3))
        s.update({"ret_1": x}, y)
    assert s.predict({"ret_1": 1.0}) > 0.65 and s.predict({"ret_1": -1.0}) < 0.35
    assert s.predict({"ret_1": None}) is None
    st = PredStats()
    for _ in range(100):
        st.update(0.7, True, logloss(0.7, True), logloss(0.5, True), 0.01, 0.0013, 0.03)
    assert st.z(1) > 3 and st.n_eff(24) < st.n_eff(1)             # overlap correction shrinks n_eff


def test_logit_learns_direction():
    m = OnlineLogit(1); rng = np.random.default_rng(1)
    for _ in range(4000):
        x = rng.normal(); f = {"ret_1": x}
        m.update(f, bool(rng.random() < 1 / (1 + np.exp(-2 * x))))
    assert m.predict({"ret_1": 2.0}) > m.predict({"ret_1": -2.0}) + 0.3


def test_noise_earns_no_trust_planted_signal_does():
    cfg = dict(horizons=[1], cap=200, calib=300)
    noise, _ = mk_loop(cfg)
    df = feature_frame(synth(4000, 5, phi=0.0))
    noise.warm_start({"ETH": df}, "x"); noise.process_market("ETH", df, persist=False)
    assert [p for p in noise.preds[1].values() if p.status == "trusted"] == []
    assert abs(noise.preds[1]["ensemble@1"].stats.z(1)) < 3          # ensemble is not fooled either

    sig, j = mk_loop(cfg)
    df = feature_frame(synth(4000, 5, phi=0.3))
    sig.warm_start({"ETH": df}, "x"); sig.process_market("ETH", df, persist=False)
    trusted = [p for p in sig.preds[1].values() if p.status == "trusted"]
    assert trusted, "should find the autocorrelation edge"
    assert any(n.status == "supported" for n in NoteStore(j).find())  # and write it down
    assert sig.preds[1]["ensemble@1"].stats.mean_d() > 0


def test_no_prediction_sees_its_own_outcome():
    """A prediction made at bar t must exist before bar t+H is processed."""
    loop, _ = mk_loop(dict(horizons=[4], calib=300))
    df = feature_frame(synth(600, 2))
    loop.warm_start({"ETH": df}, "x")
    first = df.filter(pl.col("ct") > loop.last_ct["ETH"]).head(5)
    loop.process_market("ETH", first, persist=False)
    q = loop.pending[("ETH", 4)]
    assert len(q) == 4 and all(it.preds for it in q)                  # 5 made, 1 resolved (needs 4 bars after)
    assert loop.preds[4]["ensemble@4"].stats.n == 1


def test_restart_resumes_exactly(tmp_path):
    db = str(tmp_path / "o.duckdb")
    df = feature_frame(synth(1500, 9, phi=0.2))
    cfg = dict(horizons=[1, 4], calib=300)
    a, ja = mk_loop(cfg, db=db)
    a.warm_start({"ETH": df}, "x")
    a.process_market("ETH", df.head(1000), persist=True); a.save()
    ja.con.close()
    b, _ = mk_loop(cfg, db=db)                                        # "restart"
    assert b.last_ct["ETH"] == a.last_ct["ETH"] and b.bars == a.bars
    assert sum(len(q) for q in b.pending.values()) == sum(len(q) for q in a.pending.values())
    b.process_market("ETH", df, persist=True)                         # continues only with new bars

    c, _ = mk_loop(cfg)
    c.warm_start({"ETH": df}, "x"); c.process_market("ETH", df, persist=False)
    sb, sc = b.preds[1]["ensemble@1"].stats, c.preds[1]["ensemble@1"].stats
    assert sb.n == sc.n and abs(sb.sum_d - sc.sum_d) < 1e-6          # stop/start == one straight run


# ---------- does it actually LEARN over time? ----------
import copy


def _seg(loop, prev):
    s = loop.preds[1]["ensemble@1"].stats
    return (s.sum_d - prev.sum_d) / max(1, s.n - prev.n)


def _splice(n1, n2, phi1, phi2, seed):
    a, b = synth(n1, seed, phi=phi1), synth(n2, seed + 1, phi=phi2)
    k = a["close"][-1] / b["close"][0]
    b = b.with_columns((pl.col(c) * k for c in ("open", "high", "low", "close"))).with_columns(
        (pl.col("t") + (a["t"][-1] - b["t"][0] + H1)).alias("t"))
    return pl.concat([a, b])


def test_learning_curve_holds_and_beats_best_single_predictor_share():
    """Stable edge: late-period skill must stay near the best single rule's skill, not fade as the population grows."""
    loop, _ = mk_loop(dict(horizons=[1], cap=300, calib=300))
    df = feature_frame(synth(3600, 5, phi=0.3))
    loop.warm_start({"ETH": df}, "x")
    loop.process_market("ETH", df.head(1800), persist=False)
    mid = copy.deepcopy(loop.preds[1]["ensemble@1"].stats)
    loop.process_market("ETH", df, persist=False)
    late = _seg(loop, mid)
    best = max(p.stats.mean_d() for p in loop.preds[1].values() if p.kind == "stump")
    assert late > 0.01 and late > 0.5 * best          # the ensemble keeps most of the edge it found


def test_adapts_after_the_regime_flips():
    """Edge reverses mid-run: skill dips, then recovers and turns positive again in the new regime."""
    loop, _ = mk_loop(dict(horizons=[1], cap=300, calib=300))
    df = feature_frame(_splice(2400, 3000, 0.3, -0.3, 5))
    loop.warm_start({"ETH": df}, "x")
    loop.process_market("ETH", df.head(2400), persist=False)
    a = copy.deepcopy(loop.preds[1]["ensemble@1"].stats)
    loop.process_market("ETH", df.head(3300), persist=False)
    trough = _seg(loop, a)
    b = copy.deepcopy(loop.preds[1]["ensemble@1"].stats)
    loop.process_market("ETH", df, persist=False)
    recovered = _seg(loop, b)
    assert recovered > 0.005 and recovered > trough   # it re-learned the reversed pattern


def test_noise_does_not_fake_a_learning_curve():
    """No edge: the ensemble must stay at baseline skill (about 0), never drift to confident-and-wrong."""
    loop, _ = mk_loop(dict(horizons=[1], cap=300, calib=300))
    df = feature_frame(synth(3600, 21, phi=0.0))
    loop.warm_start({"ETH": df}, "x"); loop.process_market("ETH", df, persist=False)
    s = loop.preds[1]["ensemble@1"].stats
    assert abs(s.mean_d()) < 0.004 and s.z(1) < 3


def test_book_summary_depth_and_imbalance():
    from ooma.adapters.base import BookSnapshot
    from ooma.predict.store import book_summary
    b = BookSnapshot("X", T0, bids=[(99.95, 10.0), (99.7, 50.0), (99.0, 100.0)], asks=[(100.05, 5.0), (101.0, 100.0)])
    d = book_summary(b)
    assert d["spread_bps"] == pytest.approx(10.0, rel=1e-3)
    assert d["bid_10"] == pytest.approx(99.95 * 10) and d["ask_10"] == pytest.approx(100.05 * 5)
    assert d["imb_10"] == pytest.approx((999.5 - 500.25) / (999.5 + 500.25))
    assert d["bid_50"] > d["bid_10"]                                   # wider window sees more depth
