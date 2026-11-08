"""One LLM client for any OpenAI-compatible API (OpenRouter by default). Retries, fallback models,
JSON extraction, per-call cost logging, and a hard daily budget so an unattended loop cannot run away.

Key lookup: env OPENROUTER_API_KEY, else macOS keychain item 'ooma-openrouter'. Never logged.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"


class BudgetExceeded(RuntimeError):
    pass


def get_key(service: str = "ooma-openrouter") -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    try:
        return subprocess.check_output(["security", "find-generic-password", "-s", service, "-w"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:                                           # noqa: BLE001
        raise RuntimeError("no API key: set OPENROUTER_API_KEY or add keychain item 'ooma-openrouter'") from e


def extract_json(text: str):
    """Parse the first JSON object/array in a model reply, tolerating code fences and chatter."""
    t = re.sub(r"```(?:json)?", "", text).strip()
    for start in (i for i, ch in enumerate(t) if ch in "[{"):
        try:
            return json.JSONDecoder().raw_decode(t[start:])[0]
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON found in model reply")


class LLMClient:
    def __init__(self, model="z-ai/glm-5.3-flash", fallbacks=("z-ai/glm-5.3",), con=None,
                 daily_budget_usd=1.0, client: httpx.Client | None = None, key: str | None = None):
        self.model, self.fallbacks, self.budget = model, list(fallbacks), daily_budget_usd
        self.http = client or httpx.Client(timeout=120)
        self._key = key
        self.con = con
        if con is not None:
            con.execute("CREATE TABLE IF NOT EXISTS llm_calls (ts VARCHAR, model VARCHAR, purpose VARCHAR, "
                        "tokens_in INT, tokens_out INT, cost_usd DOUBLE, ok BOOLEAN)")

    def spent_today(self) -> float:
        if self.con is None:
            return 0.0
        day = datetime.now(timezone.utc).date().isoformat()
        return self.con.execute("SELECT COALESCE(SUM(cost_usd),0) FROM llm_calls WHERE ts LIKE ?", [day + "%"]).fetchone()[0]

    def chat(self, messages: list[dict], *, purpose="", max_tokens=6000, temperature=0.7, json_mode=False) -> str:
        if self.spent_today() >= self.budget:
            raise BudgetExceeded(f"daily LLM budget ${self.budget:.2f} reached")
        key = self._key or get_key()
        last = None
        for model in [self.model, *self.fallbacks]:
            for attempt in range(3):
                try:
                    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                            "temperature": temperature, "usage": {"include": True}}
                    if json_mode:
                        body["response_format"] = {"type": "json_object"}
                    r = self.http.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                    r.raise_for_status()
                    d = r.json()
                    text = d["choices"][0]["message"].get("content") or ""
                    if not text.strip():
                        raise ValueError("empty content (reasoning exhausted max_tokens?)")
                    self._log(d, model, purpose, True)
                    return text
                except (httpx.HTTPError, ValueError, KeyError) as e:
                    last = e
                    self._log({}, model, purpose, False)
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"all models failed: {last}")

    def chat_json(self, messages, **kw):
        return extract_json(self.chat(messages, json_mode=False, **kw))

    def _log(self, d: dict, model, purpose, ok):
        if self.con is None:
            return
        u = d.get("usage", {}) or {}
        self.con.execute("INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?)",
                         [datetime.now(timezone.utc).isoformat(), model, purpose,
                          u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost", 0.0) or 0.0, ok])
