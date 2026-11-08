"""Idea generators. Anything that returns StrategySpecs plugs into the Lab.

TemplateIdeator: no-LLM parametric search over known families + mutation of survivors (free, runs now).
LLMIdeator: the open-ended one. Given the notebook, the scorecard and the spec schema, it invents new
rule combinations and families. Its output is validated by StrategySpec like everything else.
"""
from __future__ import annotations

import random
import re
from typing import Protocol

from ..spec import Condition, StrategySpec

MISSION = ("Make more than you lose, after fees and slippage. Use any approach expressible as a "
           "StrategySpec. Judged on data you never trained on, so overfitting is punished.")


class Ideator(Protocol):
    def propose(self, n: int, markets: list[str], timeframes: list[str],
                seeds: list[StrategySpec], rng: random.Random) -> list[tuple[StrategySpec, str]]:
        """Return (spec, origin) pairs; origin is 'template' | 'llm' | 'mutation'."""


def C(l, op, r) -> Condition:
    return Condition(lhs=l, op=op, rhs=r)


def _breakout(rng, tf, mk):
    n = rng.choice([10, 20, 40, 55]); v = rng.choice([1.0, 1.3, 1.6])
    return StrategySpec(family="breakout", timeframe=tf, markets=mk,
        hypothesis=f"Closing above the prior {n}-bar high on {v}x volume starts a continuation move.",
        long_when=[C("close", "cross_above", f"hh_{n}"), C("volx_20", ">", v)],
        short_when=[C("close", "cross_below", f"ll_{n}"), C("volx_20", ">", v)],
        stop_atr=rng.choice([1.5, 2, 3]), target_r=rng.choice([1.5, 2, 3]), horizon_bars=rng.choice([12, 24, 48]))


def _mean_rev(rng, tf, mk):
    lo = rng.choice([20, 25, 30]); z = rng.choice([1.5, 2.0, 2.5])
    return StrategySpec(family="mean_reversion", timeframe=tf, markets=mk,
        hypothesis=f"Oversold (RSI<{lo}) and {z} sigma below the 20-bar mean snaps back.",
        long_when=[C("rsi_14", "<", lo), C("z_20", "<", -z)],
        short_when=[C("rsi_14", ">", 100 - lo), C("z_20", ">", z)],
        stop_atr=rng.choice([1.5, 2, 3]), target_r=rng.choice([0.8, 1.0, 1.5]), horizon_bars=rng.choice([6, 12, 24]))


def _trend(rng, tf, mk):
    f, s = rng.choice([(10, 30), (20, 50), (20, 100)])
    return StrategySpec(family="trend", timeframe=tf, markets=mk,
        hypothesis=f"In an EMA{f}>EMA{s} uptrend, reclaiming EMA{f} after a dip resumes the trend.",
        long_when=[C(f"ema_{f}", ">", f"ema_{s}"), C("close", "cross_above", f"ema_{f}")],
        short_when=[C(f"ema_{f}", "<", f"ema_{s}"), C("close", "cross_below", f"ema_{f}")],
        stop_atr=rng.choice([1.5, 2, 3]), target_r=rng.choice([2, 3, 4]), horizon_bars=rng.choice([24, 48, 96]))


def _momentum(rng, tf, mk):
    n = rng.choice([3, 6, 12]); x = rng.choice([0.02, 0.03, 0.05])
    return StrategySpec(family="momentum", timeframe=tf, markets=mk,
        hypothesis=f"A {x:.0%}+ move over {n} bars on rising volume tends to extend.",
        long_when=[C(f"ret_{n}", ">", x), C("volx_20", ">", 1.3)],
        short_when=[C(f"ret_{n}", "<", -x), C("volx_20", ">", 1.3)],
        stop_atr=rng.choice([1.5, 2, 3]), target_r=rng.choice([1.5, 2, 3]), horizon_bars=rng.choice([12, 24, 48]))


FAMILIES = [_breakout, _mean_rev, _trend, _momentum]


def mutate(spec: StrategySpec, rng: random.Random) -> StrategySpec:
    """Nudge window lengths, thresholds, stop/target/horizon. Each mutant counts as a new trial."""
    scale = rng.choice([0.7, 0.85, 1.2, 1.4])

    def feat(name):
        m = re.match(r"^([a-z]+)_(\d+)$", name)
        if not m or m.group(1) == "atr":
            return name
        return f"{m.group(1)}_{min(400, max(2, round(int(m.group(2)) * scale)))}"

    def conds(cs):
        out = []
        for c in cs:
            rhs = feat(c.rhs) if isinstance(c.rhs, str) else c.rhs * rng.choice([0.8, 1.0, 1.2])
            out.append(Condition(lhs=feat(c.lhs), op=c.op, rhs=rhs))
        return out

    d = spec.model_copy(deep=True)
    return StrategySpec(
        family=d.family, hypothesis=d.hypothesis + " (mutated)", timeframe=d.timeframe, markets=d.markets,
        long_when=conds(d.long_when), short_when=conds(d.short_when),
        stop_atr=min(6, max(0.5, d.stop_atr * rng.choice([0.8, 1.0, 1.25]))),
        target_r=min(6, max(0.5, d.target_r * rng.choice([0.8, 1.0, 1.25]))),
        horizon_bars=min(200, max(1, round(d.horizon_bars * rng.choice([0.5, 1.0, 2.0])))))


class TemplateIdeator:
    def propose(self, n, markets, timeframes, seeds, rng):
        out = []
        for i in range(n):
            if seeds and i % 3 == 0:
                out.append((mutate(rng.choice(seeds), rng), "mutation"))
            else:
                out.append((rng.choice(FAMILIES)(rng, rng.choice(timeframes), markets), "template"))
        return out


PROMPT = """You are the strategy inventor inside ooma, an autonomous trading research system on Hyperliquid perps.
MISSION: {mission}

Invent {n} NEW, DIVERSE strategies as JSON. Output ONLY a JSON array of objects, no prose.
Each object:
{{"family": str, "hypothesis": "one sentence on WHY this makes money (who is on the other side?)",
 "timeframe": "1h"|"4h"|"1d", "markets": {markets},
 "long_when": [{{"lhs": F, "op": OP, "rhs": F_or_number}}, ...],   // ALL conditions must hold
 "short_when": [same shape, may be empty],
 "stop_atr": 0.5-6, "target_r": 0.5-6, "horizon_bars": 1-200}}
OP is one of: ">", "<", "cross_above", "cross_below".
F (feature) is one of: open high low close volume, or NAME_N with N in 2..400 where NAME is
ema sma atr rsi ret z hh ll volx   (ret_N = N-bar return; z_N = z-score of close vs N-bar mean;
hh_N/ll_N = highest high/lowest low of the PRIOR N bars; volx_N = volume / N-bar avg volume).
Example F: ema_20, rsi_14, ret_6, z_50, hh_20, volx_20. Numbers on the right are plain numbers.

Rules: costs are ~13 bps round trip, so strategies need real edge after fees. Prefer regime filters
(trend/volatility conditions), asymmetric setups, and ideas unlike the ones already tried.
Do NOT repeat these already-tested families/ideas blindly: {tried}

What ooma has learned so far (notebook):
{notes}
"""


class LLMIdeator:
    """Open-ended invention by an LLM. Output is validated by StrategySpec; anything invalid is dropped.
    The LLM never runs code and never sees keys; the lab's honest testing decides what survives."""

    def __init__(self, client, notes=None, journal=None, per_call: int = 8):
        self.client, self.notes, self.j, self.per_call = client, notes, journal, per_call
        self.model = client.model
        self.dropped = 0

    def _context(self):
        notes = "(none yet)"
        if self.notes:
            ns = self.notes.context_for(None, None, k=12)
            if ns:
                notes = "\n".join(f"- [{n.status}] {n.text}" for n in ns)
        tried = "(nothing)"
        if self.j:
            recs = self.j.all("strategy_records")
            from collections import Counter
            c = Counter((r.spec.family, r.stage) for r in recs)
            tried = "; ".join(f"{f}/{st} x{k}" for (f, st), k in c.most_common(15)) or tried
        return notes, tried

    def propose(self, n, markets, timeframes, seeds, rng):
        out, calls = [], 0
        notes, tried = self._context()
        while len(out) < n and calls < max(2, n // self.per_call + 2):
            calls += 1
            want = min(self.per_call, n - len(out))
            prompt = PROMPT.format(mission=MISSION, n=want, markets=markets, tried=tried, notes=notes)
            try:
                raw = self.client.chat_json([{"role": "user", "content": prompt}], purpose="ideate", max_tokens=12000)
            except Exception:                                   # noqa: BLE001  budget/outage: return what we have
                break
            for item in raw if isinstance(raw, list) else [raw]:
                try:
                    item["markets"] = [m for m in item.get("markets", markets) if m in markets] or markets
                    if item.get("timeframe") not in timeframes:
                        item["timeframe"] = timeframes[0]
                    out.append((StrategySpec.model_validate(item), "llm"))
                except Exception:                               # noqa: BLE001
                    self.dropped += 1
        return out[:n]
