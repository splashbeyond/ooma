from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from ooma.execution.book import walk_book
from ooma.execution.fees import FeeSchedule
from ooma.execution.model import ExecutionModel, from_config
from ooma.execution.resting import place_limit
from ooma.execution.rules import MarketRules
from ooma.grading.grader import grade
from ooma.history.simulate import simulate
from ooma.types import Setup

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def test_price_and_size_rules_match_hl_docs():
    eth = MarketRules(sz_decimals=4, max_leverage=25)
    assert eth.round_price(2240.4567) == 2240.5 and eth.tick(2240.4) == 0.1      # 5 sig figs
    assert MarketRules(5, 40).round_price(76042.37) == 76042.0                    # integer price always allowed
    # HL docs example: szDecimals=1 -> 0.01234 valid, 0.012345 not (max 5 dp = 6-1)
    assert MarketRules(1, 10).round_price(0.012345) == 0.01234 or MarketRules(1, 10).round_price(0.012345) == 0.01235
    assert MarketRules(1, 10).price_decimals(0.012345) == 5
    assert MarketRules(2, 10).price_decimals(77.434) == 3 and MarketRules(2, 20).price_decimals(150.123) == 2
    assert eth.round_size(0.123456) == 0.1234                                       # floored, never rounded up
    assert not eth.valid_order(2240, 0.004)                                         # $8.96 < $10 minimum


def test_fees_and_book_walk():
    f = FeeSchedule()
    assert f.fee(10_000, "taker") == pytest.approx(4.5) and f.fee(10_000, "maker") == pytest.approx(1.5)
    assert FeeSchedule(staking_discount=0.4).bps("taker") == pytest.approx(2.7)
    asks = [(100.0, 1.0), (100.5, 2.0), (101.0, 5.0)]
    avg, got, ex = walk_book(asks, 2.0)
    assert avg == pytest.approx((100 + 100.5) / 2) and got == 2.0 and not ex
    _, _, ex = walk_book(asks, 50.0)
    assert ex                                                                        # flags unrealistic size


def test_liquidation_matches_hl_formula_and_funding_sign():
    m = ExecutionModel()
    # long, 3x, maxLev 25 (l=0.02): p - (p/3 - 0.02p)/(0.98) = 0.6803p
    assert m.liq_price("ETH", +1, 1000.0) == pytest.approx(680.27, abs=0.1)
    assert m.liq_price("ETH", -1, 1000.0) == pytest.approx(1000 + (1000 / 3 - 20) / 1.02, abs=0.1)
    assert m.funding_cash(+1, 10_000, 1.0, 10) < 0 < m.funding_cash(-1, 10_000, 1.0, 10)   # longs pay positive funding


def test_target_needs_trade_through_and_touch_is_probabilistic_but_deterministic():
    m = ExecutionModel(touch_fill_prob=0.35)
    through = m.target_fill("ETH", +1, 2300.0, 2300.5, 2200, 5e6, 5e3, "k")
    assert through == 1.0
    assert m.target_fill("ETH", +1, 2300.0, 2299.9, 2200, 5e6, 5e3, "k") == 0.0      # never reached
    results = {m.target_fill("ETH", +1, 2300.0, 2300.0, 2200, 5e6, 5e3, f"k{i}") for i in range(200)}
    assert results == {0.0, 1.0}                                                      # touch-only: sometimes
    hits = sum(m.target_fill("ETH", +1, 2300.0, 2300.0, 2200, 5e6, 5e3, f"k{i}") > 0 for i in range(2000)) / 2000
    assert 0.30 < hits < 0.40
    assert m.target_fill("ETH", +1, 2300.0, 2300.0, 2200, 5e6, 5e3, "same") == m.target_fill("ETH", +1, 2300.0, 2300.0, 2200, 5e6, 5e3, "same")
    assert m.target_fill("ETH", +1, 2300.0, 2310, 2200, 1e4, 5e3, "k") < 1.0          # too big for the bar's volume


def bars(closes, volume=1000.0, funding=0.0):
    return pl.DataFrame({"t": [T0 + H * i for i in range(len(closes))],
                         "open": [closes[0]] + list(closes[:-1]), "high": [c + 0.5 for c in closes],
                         "low": [c - 0.5 for c in closes], "close": list(map(float, closes)),
                         "volume": [volume] * len(closes), "funding": [funding] * len(closes)})


def setup(**kw):
    d = dict(id="s", strategy="x", strategy_version=1, market="ETH", timeframe="1h", as_of=T0, direction="long",
             entry=2000.0, stop=1960.0, targets=[2080.0], horizon_hours=12)
    return Setup(**{**d, **kw})


def run(m, closes, **kw):
    t = simulate(setup(**kw), bars([2000.0] * 1 + closes), model=m)
    return t, grade(setup(**kw), t) if t and t.closed_at else None


def test_realistic_costs_exceed_flat_zero_costs_and_target_is_maker():
    path = [2010, 2040, 2085, 2090]
    t, g = run(ExecutionModel(), path)
    assert t.exit_reason == "target"
    flat = simulate(setup(), bars([2000.0] + path), fee_bps=0, slip_bps=0)
    assert t.entry_fill > flat.entry_fill and g.r_multiple < grade(setup(), flat).r_multiple     # real costs bite
    maker_fee = 2080 * t.size * 1.5e-4
    assert t.fees == pytest.approx(t.size * t.entry_fill * 4.5e-4 + maker_fee, rel=1e-3)


def test_stop_costs_extra_and_funding_accrues():
    m = ExecutionModel()
    t, g = run(m, [1990, 1950, 1940])
    assert t.exit_reason == "stop" and t.exit_fill < 1960 and g.stop_slippage_r > 0           # slipped past the stop
    quiet = [2001, 2002, 2003, 2004, 2005, 2006]
    flat_t = simulate(setup(horizon_hours=5), bars([2000.0] + quiet, funding=0.0), model=m)
    paid_t = simulate(setup(horizon_hours=5), bars([2000.0] + quiet, funding=2.0), model=m)
    assert paid_t.funding < flat_t.funding == 0                                                # long pays +funding
    assert grade(setup(horizon_hours=5), paid_t).r_multiple < grade(setup(horizon_hours=5), flat_t).r_multiple


def test_rejects_unrealistic_trades():
    m = ExecutionModel(leverage=20.0)                   # at 20x on a 25x-max coin, liq is ~3% away
    assert simulate(setup(stop=1900.0), bars([2000.0, 2010, 2020]), model=m) is None          # stop beyond liquidation
    assert simulate(setup(), bars([2000.0, 2010]), risk_usd=0.001, model=ExecutionModel()) is None   # under $10 minimum


def test_resting_order_queue_semantics():
    o = place_limit(+1, 100.0, 2.0, best_bid=100.0, best_ask=100.1, size_at_px=5.0)
    assert o.queue_ahead == 5.0
    assert o.on_trade(1, 100.0, 3.0) == 0 and o.queue_ahead == 2.0          # prints at our price eat the queue first
    assert o.on_trade(2, 100.0, 3.0) == pytest.approx(1.0)                   # then partially fill us
    assert o.on_trade(3, 99.9, 0.1) == pytest.approx(1.0) and o.done          # trade THROUGH fills the rest
    assert place_limit(+1, 100.1, 1.0, 100.0, 100.1, 0.0) is None            # post-only that crosses is rejected


def test_config_loader():
    m = from_config({"fees": {"taker_bps": 4.0, "maker_bps": 1.2}, "leverage": 2.0,
                     "rules": {"ETH": {"sz_decimals": 4, "max_leverage": 25}}})
    assert m.fees.taker_bps == 4.0 and m.leverage == 2.0 and m.mrules("ETH").max_leverage == 25
