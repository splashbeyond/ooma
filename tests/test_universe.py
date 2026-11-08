from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from ooma.adapters.hyperliquid import HyperliquidAdapter
from ooma.director.data import DataHub, add_cross_section, day_cost
from ooma.scripts.binary import block_p, judge, pool
from ooma.scripts.validate import ScriptInvalid, validate

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
H1 = timedelta(hours=1)


def frame(n, drift, seed, funding=0.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.003, n)))
    return pl.DataFrame({"t": [T0 + H1 * i for i in range(n)], "close": close, "funding": np.full(n, funding)})


def test_cross_section_ranks_and_market_return_are_correct_and_causal():
    fr = {"A": frame(300, 0.002, 1, 0.1), "B": frame(300, 0.0, 2, 0.5), "C": frame(300, -0.002, 3, 0.9)}
    out = add_cross_section(fr)
    row = {m: out[m].row(299, named=True) for m in fr}
    assert row["A"]["xs_rank_ret_24"] == 1.0 and row["C"]["xs_rank_ret_24"] == 0.0 and row["B"]["xs_rank_ret_24"] == 0.5
    assert row["C"]["xs_rank_funding"] == 1.0 and row["A"]["xs_rank_funding"] == 0.0
    mean24 = np.mean([fr[m]["close"][299] / fr[m]["close"][275] - 1 for m in fr])
    assert row["A"]["mkt_ret_24"] == pytest.approx(mean24)
    cut = add_cross_section({m: d.head(200) for m, d in fr.items()})        # appending the future must not change the past
    a, b = out["A"].head(200)["xs_rank_ret_24"].to_numpy(), cut["A"]["xs_rank_ret_24"].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_cost_tiers_charge_thin_coins_more():
    assert day_cost(300e6) < day_cost(20e6) < day_cost(6e6)


def test_validator_accepts_universe_and_named_markets_rejects_garbage():
    base = 'HORIZON_BARS = 1\nTIMEFRAME = "1h"\nMARKETS = {}\ndef decide(bars):\n    return (True, True)\n'
    assert validate(base.format('"ALL"')).markets == ["ALL"]
    assert validate(base.format('["ZEC", "kPEPE"]')).markets == ["ZEC", "kPEPE"]
    with pytest.raises(ScriptInvalid):
        validate(base.format('["BTC; rm -rf /"]'))


def test_block_bootstrap_does_not_overcount_markets_that_move_together():
    """15 markets all right in the same weeks look hugely significant per-trade, but they are one bet per week."""
    rng = np.random.default_rng(0)
    weeks = 40
    week_edge = rng.normal(0, 0.15, weeks)                          # each week is all-right or all-wrong across markets
    t, ex = [], []
    for w in range(weeks):
        for m in range(15):
            for k in range(5):
                t.append((w * 7 + k) * 86_400_000.0)
                ex.append(0.05 + week_edge[w] + rng.normal(0, 0.3))
    t, ex = np.array(t), np.array(ex)
    naive_z = ex.mean() / (ex.std() / np.sqrt(len(ex)))
    assert naive_z > 4                                              # naive per-trade test would scream "edge"
    blocked = block_p(t, ex)
    assert blocked > 0.001 and blocked > 100 * (0.5 * np.exp(-naive_z ** 2 / 2))   # block test is far less impressed


def test_block_bootstrap_still_detects_a_real_persistent_edge():
    rng = np.random.default_rng(1)
    t = np.array([(w * 7 + k) * 86_400_000.0 for w in range(60) for k in range(7) for _ in range(5)])
    ex = 0.08 + rng.normal(0, 0.3, len(t))
    assert block_p(t, ex) < 0.001


def test_pool_uses_block_p_across_markets_and_judge_records_trades():
    n = 1500
    rng = np.random.default_rng(2)
    res = []
    for seed in range(4):
        r = np.random.default_rng(seed).normal(0, 0.004, n)
        close = 100 * np.exp(np.cumsum(r)); open_ = np.r_[close[0], close[:-1]]
        t = np.array([(T0 + H1 * i).timestamp() * 1000 for i in range(n)])
        d = np.ones(n, bool); a = np.ones(n, bool)
        res.append(judge(d, a, open_, close, 4, 100, n, t_ms=t))
    pooled = pool(res)
    assert pooled.p_binom is not None and pooled.n == sum(r.n for r in res)
    assert pooled.p >= 1 / 20000 and "trades" not in pooled.dump()


class FakeAdapter(HyperliquidAdapter):
    def __init__(self, ctxs): self.ctxs = ctxs
    def all_contexts(self): return self.ctxs


def test_universe_ranks_by_volume_and_filters_thin_coins(tmp_path):
    from ooma.adapters.base import MarketContext
    mk = lambda m, v: MarketContext(m, T0, 0.0, 1.0, 1.0, 1.0, 0.0, v)
    a = FakeAdapter([mk("AAA", 9e6), mk("BTC", 5e8), mk("TINY", 1e5), mk("ETH", 2e8), mk("MID", 6e6)])
    hub = DataHub(a, root=str(tmp_path), top_n=3, min_day_volume=5e6)
    assert hub.universe() == ["BTC", "ETH", "AAA"]                  # TINY filtered, MID cut by top_n, ranked by volume
