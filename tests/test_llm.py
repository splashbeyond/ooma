import json

import duckdb
import httpx
import pytest

from ooma.lab.ideation import LLMIdeator
from ooma.llm import BudgetExceeded, LLMClient, extract_json


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('Sure!\n```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json('{"x": {"y": 2}} trailing') == {"x": {"y": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def mock(replies):
    it = iter(replies)

    def h(req):
        code, content, cost = next(it)
        if code != 200:
            return httpx.Response(code, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": cost}})
    return httpx.Client(transport=httpx.MockTransport(h))


def test_fallback_and_cost_logging_and_budget(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    con = duckdb.connect()
    c = LLMClient(model="a", fallbacks=["b"], con=con, client=mock([(500, "", 0)] * 3 + [(200, "hi", 0.4)]),
                  key="k", daily_budget_usd=0.5)
    assert c.chat([{"role": "user", "content": "x"}]) == "hi"      # primary failed 3x, fallback answered
    assert abs(c.spent_today() - 0.4) < 1e-9
    c2 = LLMClient(model="a", con=con, client=mock([(200, "hi", 0.2)]), key="k", daily_budget_usd=0.5)
    c2.chat([{"role": "user", "content": "x"}])
    with pytest.raises(BudgetExceeded):
        c2.chat([{"role": "user", "content": "x"}])               # 0.6 spent > 0.5 cap


def test_llm_ideator_keeps_valid_drops_invalid():
    good = {"family": "x", "hypothesis": "h", "timeframe": "4h", "markets": ["ETH"],
            "long_when": [{"lhs": "rsi_14", "op": "<", "rhs": 25}], "stop_atr": 2, "target_r": 2, "horizon_bars": 12}
    bad_feature = {**good, "long_when": [{"lhs": "os.system", "op": ">", "rhs": 1}]}
    no_rules = {**good, "long_when": []}
    c = LLMClient(client=mock([(200, json.dumps([good, bad_feature, no_rules]), 0.0)]), key="k")
    ide = LLMIdeator(c)
    out = ide.propose(3, ["BTC", "ETH"], ["1h", "4h"], [], None)
    assert len(out) == 1 and out[0][1] == "llm" and ide.dropped == 2
