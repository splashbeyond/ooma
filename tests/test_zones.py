from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from ooma.lab.enrich import enrich
from ooma.modules.zones import pivot_arrays, zone_columns
from ooma.spec import Condition, StrategySpec
from ooma.strategies.spec_strategy import SpecStrategy

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def bouncing_market():
    """Price oscillates between a floor at 100 and a ceiling at 110, touching each ~6 times."""
    rng = np.random.default_rng(0)
    base = np.tile(np.r_[np.linspace(104, 100.2, 12), np.linspace(100.2, 109.8, 18), np.linspace(109.8, 104, 12)], 8)
    close = base + rng.normal(0, 0.05, len(base))
    high, low = close + 0.3, close - 0.3
    return high, low, close


def test_pivots_are_confirmed_late():
    high = np.array([1, 2, 3, 9, 3, 2, 1, 2, 3, 4, 5, 6.0])
    idx, conf, price = pivot_arrays(high, high - 0.5, L=2)
    assert 3 in idx and conf[list(idx).index(3)] == 5          # pivot at bar 3 only known at bar 5


def test_zones_found_and_measured_in_atr():
    h, l, c = bouncing_market()
    atr = np.full(len(c), 1.0)
    z = zone_columns(h, l, c, atr, L=3, k=0.8, window=300)
    last = ~np.isnan(z["sup_dist"])
    assert last[-100:].any()
    i = np.nonzero(~np.isnan(z["sup_dist"]) & ~np.isnan(z["res_dist"]))[0][-1]
    assert abs((c[i] - z["sup_dist"][i]) - 99.7) < 1.0          # support zone near the 100 floor (low wicks)
    assert abs((c[i] + z["res_dist"][i]) - 110.1) < 1.0         # resistance near the 110 ceiling
    assert z["sup_touches"][i] >= 2 and z["res_touches"][i] >= 2


def test_zone_features_are_causal():
    h, l, c = bouncing_market()
    atr = np.full(len(c), 1.0)
    full = zone_columns(h, l, c, atr, L=3, k=0.8)
    part = zone_columns(h[:200], l[:200], c[:200], atr[:200], L=3, k=0.8)
    for k in full:
        assert np.allclose(full[k][:200], part[k], equal_nan=True), k


def test_enriched_columns_and_spec_uses_zones():
    h, l, c = bouncing_market()
    n = len(c)
    candles = pl.DataFrame({"t": [T0 + timedelta(hours=4 * i) for i in range(n)], "open": c, "high": h, "low": l,
                            "close": c, "volume": np.ones(n), "trades": np.ones(n, dtype=int)})
    funding = pl.DataFrame({"t": [T0 + timedelta(hours=i) for i in range(n * 4)],
                            "funding_rate": np.full(n * 4, 0.00001), "premium": np.zeros(n * 4)})
    df = enrich(candles, "4h", funding, candles)
    for col in ("sup_dist", "res_dist", "sup_touches", "funding", "funding_z", "btc_ret_6", "rel_btc_24"):
        assert col in df.columns
    spec = StrategySpec(family="zone_bounce", hypothesis="h", timeframe="4h", markets=["ETH"],
                        long_when=[Condition(lhs="sup_dist", op="<", rhs=1.0), Condition(lhs="sup_touches", op=">", rhs=1)])
    s = SpecStrategy(spec)
    assert s.scan_batch(df, "ETH", "4h"), "should fire near support"
    assert s.scan_batch(candles, "ETH", "4h") == {}             # un-enriched candles: refuses instead of guessing
