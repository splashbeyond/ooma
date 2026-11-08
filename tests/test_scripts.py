import numpy as np
import pytest

from ooma.scripts.binary import Gates, binom_sf, forward_verdict, holdout_ok, judge, pool, train_ok
from ooma.scripts.sandbox import LOOKBACK, SandboxRunner
from ooma.scripts.validate import ScriptInvalid, validate

GOOD = '''
"""momentum: follow the last bar when it is big"""
import numpy as np
from ooma_lib import ema
HORIZON_BARS = 1
TIMEFRAME = "1h"
MARKETS = "ALL"
def decide(bars):
    c = bars["close"]
    r = c[-1] / c[-2] - 1
    return (r > 0, abs(r) > 0.002)
'''


def wrap(body, extra=""):
    return f'HORIZON_BARS = 1\nTIMEFRAME = "1h"\n{extra}\ndef decide(bars):\n{body}\n'


@pytest.mark.parametrize("bad", [
    wrap("    import os\n    return (True, True)"),
    wrap("    return (eval('1'), True)"),
    wrap("    return (bars.__class__, True)"),
    wrap("    open('/etc/passwd')\n    return (True, True)"),
    wrap("    return (getattr(bars, 'x'), True)"),
    wrap("    import numpy as np\n    np.load('x')\n    return (True, True)"),
    wrap("    return (True, True)", extra="print('hi')"),                      # module-level call
    wrap("    return (True, True)", extra="X = [i for i in range(3)]"),         # non-literal constant
    "TIMEFRAME = '1h'\nHORIZON_BARS = 1\ndef decide(bars, extra):\n    return (True, True)\n",
    wrap("    return (True, True)").replace("HORIZON_BARS = 1", "HORIZON_BARS = 999"),
    wrap("    return (True, True)").replace('"1h"', '"5m"'),
    "def decide(bars):\n    return (True, True)\n",                            # no constants
    "HORIZON_BARS = 1\nTIMEFRAME = '1h'\nclass A:\n    pass\ndef decide(bars):\n    return (True, True)\n",
])
def test_validator_rejects_unsafe_or_nonconforming(bad):
    with pytest.raises(ScriptInvalid):
        validate(bad)


def test_validator_accepts_good_script():
    m = validate(GOOD)
    assert m.horizon_bars == 1 and m.timeframe == "1h" and m.markets == ["ALL"]


def arrays(n=900, seed=0, phi=0.3):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.004)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    return dict(open=open_, high=np.maximum(open_, close) * 1.001, low=np.minimum(open_, close) * 0.999,
                close=close, volume=np.ones(n) * 1000)


def test_sandbox_runs_script_and_respects_no_lookahead():
    a = arrays()
    sb = SandboxRunner()
    r = sb.run(GOOD, a)
    assert r.ok and len(r.direction) == len(a["close"]) - LOOKBACK
    # decisions made with a truncated series must equal the same decisions made with the full series
    k = 700
    r2 = sb.run(GOOD, {kk: v[:k] for kk, v in a.items()})
    m = k - LOOKBACK
    assert np.array_equal(r.direction[:m], r2.direction[:m]) and np.array_equal(r.act[:m], r2.act[:m])


@pytest.mark.parametrize("body,why", [
    ("    while True:\n        pass", "timeout or cpu"),
    ("    x = [0] * (10 ** 10)\n    return (True, True)", "memory"),
    ("    return 1 / 0", "errors"),
])
def test_sandbox_contains_runaway_scripts(body, why):
    sb = SandboxRunner(timeout_s=20, cpu_s=3)
    r = sb.run(wrap(body), arrays(700))
    assert not r.ok                                       # killed or rejected, never hangs the host


def test_sandbox_blocks_imports_even_if_static_check_is_bypassed():
    sb = SandboxRunner()
    src = 'HORIZON_BARS = 1\nTIMEFRAME = "1h"\ndef decide(bars):\n    import os\n    return (True, True)\n'
    r = sb.run(src, arrays(700))                          # skip validate() on purpose
    assert not r.ok and "import" in (r.error or "").lower() or not r.ok


def test_sandbox_cannot_write_files(tmp_path):
    target = tmp_path / "pwned.txt"
    src = f'HORIZON_BARS = 1\nTIMEFRAME = "1h"\ndef decide(bars):\n    f = open({str(target)!r}, "w")\n    return (True, True)\n'
    r = SandboxRunner().run(src, arrays(700))
    assert not target.exists()


def test_binom_exact():
    assert binom_sf(5, 10, 0.5) == pytest.approx(0.623046875)
    assert binom_sf(10, 10, 0.5) == pytest.approx(1 / 1024)
    assert binom_sf(0, 10, 0.5) == 1.0


def mk(direction, act, a, H=1, lo=LOOKBACK, hi=None):
    return judge(direction, act, a["open"], a["close"], H, lo, hi or len(a["close"]))


def test_judge_finds_real_edge_and_not_noise():
    a = arrays(4000, seed=1, phi=0.4)
    sb = SandboxRunner()
    res = sb.run(GOOD, a)
    d = np.zeros(len(a["close"]), bool); ac = np.zeros(len(a["close"]), bool)
    d[LOOKBACK:], ac[LOOKBACK:] = res.direction, res.act
    r = mk(d, ac, a)
    assert r.n > 200 and r.hit_rate > r.chance + 0.03 and r.p < 0.01       # momentum works on autocorrelated data
    b = arrays(4000, seed=2, phi=0.0)
    res = sb.run(GOOD, b)
    d[LOOKBACK:], ac[LOOKBACK:] = res.direction, res.act
    rb = mk(d, ac, b)
    assert rb.p > 0.01 or (rb.net_mean or 0) <= 0                           # noise: no credible edge
    assert not holdout_ok(rb, Gates(), looks=4)


def test_independent_trades_counted_once_and_costs_included():
    a = arrays(2000, seed=3, phi=0.0)
    n = len(a["close"])
    always_up = mk(np.ones(n, bool), np.ones(n, bool), a, H=10)
    assert always_up.n_acts > always_up.n * 5 and abs(always_up.n - always_up.n_bars / 10) < 5   # H=10 -> ~1/10 independent
    assert always_up.chance >= 0.5
    # a perfect oracle that peeks at the future still pays costs: net = |ret| - cost
    c, o = a["close"], a["open"]
    oracle = np.zeros(n, bool); oracle[:-1] = c[np.minimum(np.arange(n - 1) + 1, n - 1)] > o[np.minimum(np.arange(n - 1) + 1, n - 1)]
    r = mk(oracle, np.ones(n, bool), a, H=1)
    assert r.hit_rate == 1.0 and r.net_mean < 0.02


def test_gates_and_forward_verdicts():
    from ooma.scripts.binary import BinaryResult
    g = Gates()
    good = BinaryResult(n=60, hits=40, hit_rate=40 / 60, chance=0.5, p=binom_sf(40, 60, 0.5), net_mean=0.004, net_t=3.0)
    assert train_ok(good, g) and holdout_ok(good, g, looks=3)
    assert forward_verdict(good, g, k_incubating=3) == "promote"
    few = BinaryResult(n=10, hits=9, hit_rate=0.9, chance=0.5, p=0.01, net_mean=0.01, net_t=3)
    assert not train_ok(few, g) and forward_verdict(few, g, 1) == "wait"
    bad = BinaryResult(n=60, hits=22, hit_rate=22 / 60, chance=0.5, p=0.99, net_mean=-0.004, net_t=-3.0)
    assert forward_verdict(bad, g, 1) == "retire"
    lucky = BinaryResult(n=60, hits=36, hit_rate=0.6, chance=0.5, p=binom_sf(36, 60, 0.5), net_mean=0.002, net_t=1.0)
    assert not holdout_ok(lucky, g, looks=20)               # many looks: a 60% hit rate is not enough
