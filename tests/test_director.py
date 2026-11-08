import json
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from ooma.director.session import Director, code_block
from ooma.journal.notes import NoteStore
from ooma.journal.store import Journal
from ooma.scripts.binary import Gates
from ooma.scripts.sandbox import SandboxRunner

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
H1 = timedelta(hours=1)

MOMO = '''"""follow the last bar when it is large"""
HORIZON_BARS = 1
TIMEFRAME = "1h"
MARKETS = ["ETH"]
def decide(bars):
    c = bars["close"]
    r = c[-1] / c[-2] - 1
    return (r > 0, abs(r) > 0.002)
'''
EVIL = '''import os
HORIZON_BARS = 1
TIMEFRAME = "1h"
MARKETS = ["ETH"]
def decide(bars):
    os.system("echo pwned")
    return (True, True)
'''


def frame(n, seed, phi):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.004)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    return pl.DataFrame({"t": [T0 + H1 * i for i in range(n)], "open": open_,
                         "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
                         "close": close, "volume": np.full(n, 1000.0)})


class FakeHub:
    def __init__(self, df):
        self.df, self.cursor = df, df.height

    def history(self, m, tf): return self.df
    def live(self, m, tf, bars=700): return self.df.head(self.cursor).tail(bars)


class FakeLLM:
    model = "fake-glm"

    def __init__(self, scripts, hyps=None):
        self.scripts, self.prompts, self.hyps = list(scripts), [], hyps or [
            {"name": "momo", "hypothesis": "big bars continue", "edge_source": "slow reactors", "timeframe": "1h", "horizon_bars": 1, "markets": ["ETH"]}]

    def chat(self, messages, purpose="", **kw):
        text = messages[-1]["content"]
        self.prompts.append((purpose, text))
        if purpose == "propose":
            return json.dumps(self.hyps)
        if purpose in ("write", "repair", "refine"):
            return "```python\n" + (self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]) + "```"
        return json.dumps(["momentum works on trending data", "always check the holdout"])


def director(df, scripts, **kw):
    j = Journal(); notes = NoteStore(j)
    llm = FakeLLM(scripts, kw.pop("hyps", None))
    d = Director(j, notes, llm, FakeHub(df), SandboxRunner(), hypotheses=1, rounds=1, state_dir="/nonexistent", **kw)
    return d, j, llm


def test_code_block_extraction():
    assert code_block("sure\n```python\nx = 1\n```\nthanks").strip() == "x = 1"


def test_real_edge_found_tested_and_incubated():
    d, j, llm = director(frame(5000, 1, 0.4), [MOMO])
    rep = d.run_session()
    assert rep.proposed == 1 and rep.written == 1 and rep.stopped == ""
    recs = j.all("scripts")
    assert any(r.stage == "incubating" for r in recs), [(r.name, r.stage, r.reason) for r in recs]
    inc = next(r for r in recs if r.stage == "incubating")
    assert {e["stage"] for e in inc.evals} == {"train", "holdout"} and inc.model == "fake-glm"
    assert rep.notes == 2 and any("momentum" in n.text for n in NoteStore(j).find())     # the lesson reached the notebook


def test_noise_earns_nothing():
    d, j, _ = director(frame(5000, 2, 0.0), [MOMO])
    rep = d.run_session()
    assert rep.incubating == [] and all(r.stage != "incubating" for r in j.all("scripts"))


def test_llm_never_sees_holdout_numbers():
    d, j, llm = director(frame(5000, 1, 0.4), [MOMO])
    d.run_session()
    ho = next(e for r in j.all("scripts") for e in r.evals if e["stage"] == "holdout")
    refine_prompts = " ".join(t for p, t in llm.prompts if p == "refine")
    assert "holdout" not in refine_prompts.lower()
    for key in ("hit_rate", "net_mean", "p"):                         # holdout values must not leak into any prompt
        assert f"{ho[key]:.6g}" not in " ".join(t for _, t in llm.prompts)   # exact holdout values never reach the LLM


def test_unsafe_script_is_caught_before_it_runs_and_repaired():
    d, j, llm = director(frame(3000, 3, 0.4), [EVIL, MOMO])           # first answer is malicious; repair returns a good script
    rep = d.run_session()
    assert any(p == "repair" for p, _ in llm.prompts)
    assert not any("pwned" in str(r.reason) for r in j.all("scripts"))
    assert rep.written == 1


def test_script_that_never_gets_fixed_is_rejected_static():
    d, j, _ = director(frame(3000, 3, 0.4), [EVIL])                   # LLM keeps returning the same bad script
    rep = d.run_session()
    assert rep.rejected_static == 1 and all(r.stage == "rejected_static" for r in j.all("scripts"))


def test_budget_stop_is_graceful():
    from ooma.llm import BudgetExceeded
    d, j, llm = director(frame(3000, 3, 0.4), [MOMO])
    llm.chat = lambda *a, **k: (_ for _ in ()).throw(BudgetExceeded("daily budget"))
    rep = d.run_session()
    assert "budget" in rep.stopped and j.all("scripts") == []


def test_only_forward_evidence_promotes_to_active():
    """Incubating script stays incubating until 30+ independent LIVE trades beat chance after costs."""
    df = frame(5600, 4, 0.4)
    d, j, _ = director(df, [MOMO], gates=Gates(fwd_min_n=30))
    d.hub.cursor = 5000
    assert any(r.stage == "incubating" for r in (d.run_session() and j.all("scripts")))
    stats = None
    for step in range(5000, 5100):                                      # 100 live hours replayed one at a time
        d.hub.cursor = step + 1
        stats = d.tick()
        if stats["promoted"]:
            break
    rec = next(r for r in j.all("scripts") if r.name.startswith("momo"))
    fwd = next(e for e in rec.evals if e["stage"] == "forward")
    assert fwd["n"] >= 30 and rec.stage == "active" and stats["promoted"], (rec.stage, fwd)


def test_forward_calls_resolve_correctly():
    d, j, _ = director(frame(5200, 4, 0.4), [MOMO])
    d.run_session()
    d.hub.cursor = 5100
    d.tick()
    d.hub.cursor = 5102
    d.tick()
    rows = j.con.execute("SELECT t_ms, resolved, ret FROM script_calls ORDER BY t_ms").fetchall()
    assert rows and any(r[1] for r in rows)
    df = d.hub.df
    t_ms = rows[0][0]
    i = int(np.where(df["t"].dt.epoch("ms").to_numpy() == t_ms)[0][0])
    assert rows[0][2] == pytest_approx(df["close"][i + 1] / df["open"][i + 1] - 1)


def pytest_approx(x):
    import pytest
    return pytest.approx(x)


def test_focus_restricts_markets_and_cumulative_looks_tighten_the_gate():
    d, j, llm = director(frame(5000, 1, 0.4), [MOMO])
    d.focus = ["ETH"]
    rep = d.run_session()
    rec = next(r for r in j.all("scripts"))
    assert rec.markets == ["ETH"]
    n0 = d.recent_looks()
    assert n0 >= 1                                             # the holdout look was counted
    # the same script again is skipped entirely: no sandbox run, no extra look
    d.llm.scripts = [MOMO]
    rep2 = d.run_session()
    assert rep2.duplicates >= 1 and d.recent_looks() == n0


def test_many_looks_make_a_marginal_script_fail_the_holdout():
    from ooma.scripts.binary import BinaryResult, holdout_ok, binom_sf
    r = BinaryResult(n=60, hits=40, hit_rate=40 / 60, chance=0.5, p=binom_sf(40, 60, 0.5), net_mean=0.003, net_t=2.0)
    assert holdout_ok(r, Gates(), looks=2)
    assert not holdout_ok(r, Gates(), looks=140)               # a week of back-to-back sessions: marginal results no longer pass


def test_slots_cap_paper_trading_and_patience_frees_them():
    from ooma.scripts.registry import ScriptRecord, script_id
    d, j, _ = director(frame(5000, 4, 0.4), [MOMO], gates=Gates(max_incubating=1, fwd_max_days=2.0))
    def rec(name, age_days, stage="incubating"):
        src = MOMO + f"# {name}\n"
        r = ScriptRecord(id=script_id(src), name=name, hypothesis="h", source=src, horizon_bars=1, timeframe="1h", markets=["ETH"],
                         stage=stage, created_at=datetime.now(timezone.utc) - timedelta(days=age_days))
        j.put("scripts", r.id, r)
        return r
    rec("occupant", 0.1)
    assert d.n_incubating() == 1
    rep = d.run_session()                                           # a good new candidate arrives, but the only slot is taken
    assert rep.incubating == [] and any("slots are full" in r.reason for r in j.all("scripts"))
    old = rec("stale_one", 5.0)                                     # older than fwd_max_days with no forward evidence
    stats = d.tick()
    assert "stale_one" in stats["retired"]
    assert next(r for r in j.all("scripts") if r.name == "stale_one").stage == "retired"
    assert next(r for r in j.all("scripts") if r.name == "occupant").stage == "incubating"   # young one keeps its slot


def test_live_summary_reports_paper_pnl():
    d, j, _ = director(frame(5600, 4, 0.4), [MOMO])
    d.hub.cursor = 5000
    d.run_session()
    for step in range(5000, 5040):
        d.hub.cursor = step + 1
        d.tick()
    s = d.live_summary()
    assert s and s[0]["paper_trades"] > 0 and s[0]["paper_pnl_pct"] is not None and s[0]["name"].startswith("momo")
