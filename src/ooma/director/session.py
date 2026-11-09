"""The director: an LLM runs the research loop, code runs every judgment.

LLM decides: which hypotheses to test, writes scripts, refines them from TRAIN metrics, writes lessons.
Code decides: static validation, sandboxed execution, train/holdout/forward verdicts, promotion and retirement.
The LLM never sees holdout numbers and cannot edit gates, risk rules, config, or deploy anything.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from ..llm import BudgetExceeded, LLMClient, extract_json
from ..scripts.binary import BinaryResult, Gates, holdout_ok, holdout_soft_ok, judge, pool, train_ok
from ..scripts.registry import ScriptRecord, script_id
from ..scripts.sandbox import LOOKBACK, SandboxRunner
from ..scripts.validate import ScriptInvalid, validate
from . import prompts
from .data import DataHub, to_arrays
from .forward import ForwardBook, ask_live

log = logging.getLogger("ooma.director")


@dataclass
class SessionReport:
    session: str
    proposed: int = 0
    written: int = 0
    rejected_static: int = 0
    duplicates: int = 0
    rejected_train: int = 0
    holdout_tested: int = 0
    rejected_holdout: int = 0
    incubating: list = field(default_factory=list)
    notes: int = 0
    llm_calls: int = 0
    stopped: str = ""


def code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


class Director:
    def __init__(self, journal, notes, llm: LLMClient, hub: DataHub, sandbox: SandboxRunner, *,
                 gates: Gates | None = None, hypotheses=4, rounds=2, holdout_frac=0.25, top_k=3, state_dir="data",
                 focus: list | None = None, look_window_days: int = 7):
        self.j, self.notes, self.llm, self.hub, self.sb = journal, notes, llm, hub, sandbox
        self.g, self.n_hyp, self.rounds, self.hf, self.top_k, self.state_dir = gates or Gates(), hypotheses, rounds, holdout_frac, top_k, state_dir
        self.focus, self.look_days = focus, look_window_days
        self.hb = None                                          # optional Heartbeat: progress() is called at every step
        self.book = ForwardBook(journal.con)

    # ---------- context ----------
    def recent_looks(self) -> int:
        """Holdout looks in the last N days across ALL sessions. Every look spends statistical power on the same data."""
        cut = datetime.now(timezone.utc) - timedelta(days=self.look_days)
        return sum(1 for r in self.j.all("scripts") if r.created_at >= cut and any(e.get("stage") == "holdout" for e in r.evals))

    def context(self) -> str:
        parts = []
        if self.focus:
            parts.append(f"FOCUS: research ONLY these markets: {self.focus}. Set MARKETS = {self.focus} in every script. "
                         "HYPE is Hyperliquid's own exchange token: relatively thin liquidity, high volatility, flow-driven. "
                         "Do not assume any HYPE-specific mechanism is true; hypothesise it and let the data judge.")
        try:
            parts.append("Live prediction loop state:\n" + open(f"{self.state_dir}/loop_state.json").read()[:2500])
        except OSError:
            parts.append("Live prediction loop state: (not available)")
        if not self.focus and hasattr(self.hub, "universe_info"):
            u = self.hub.universe_info()
            parts.append("UNIVERSE (liquid Hyperliquid perps by 24h volume, USD M): " + ", ".join(f"{m} {v / 1e6:.0f}" for m, v in u.items())
                         + ". A script runs on EVERY coin here (or the subset in MARKETS) and is judged on all of them together; "
                           "round-trip cost is 13 bps for the biggest coins and 20-30 bps for the thinner ones.")
        ns = self.notes.context_for(None, None, k=12)
        parts.append("Notebook lessons:\n" + ("\n".join(f"- [{n.status}] {n.text}" for n in ns) or "(none yet)"))
        recs = sorted(self.j.all("scripts"), key=lambda r: r.created_at, reverse=True)[:14]
        lines = []
        for r in recs:
            tr = next((e for e in r.evals if e.get("stage") == "train"), {})
            lines.append(f"- {r.name} [{r.stage}] {r.timeframe}/H{r.horizon_bars}: {r.hypothesis[:90]} | train n={tr.get('n')} hit={tr.get('hit_rate')} "
                         f"chance={tr.get('chance')} net={tr.get('net_mean')} {('| ' + r.reason) if r.reason else ''}")
        parts.append("Scripts tried so far (most recent first):\n" + ("\n".join(lines) or "(none yet)"))
        return "\n\n".join(parts)

    # ---------- evaluation (code only) ----------
    def _markets(self, meta) -> list:
        """Which markets a script is tested on: the focus list if set, else the whole liquid universe (or the subset it names)."""
        if self.focus:
            return list(self.focus)
        uni = self.hub.history_markets(meta.timeframe) if hasattr(self.hub, "history_markets") else None
        if meta.markets == ["ALL"]:
            return uni or ["BTC", "ETH", "SOL", "HYPE"]
        return [m for m in meta.markets if uni is None or m in uni]

    def _cost(self, market) -> float:
        info = self.hub.universe_info() if hasattr(self.hub, "universe_info") else {}
        from ..scripts.binary import COST
        from .data import day_cost
        return day_cost(info[market]) if market in info else COST

    def _decisions(self, source, meta):
        out = {}
        for m in self._markets(meta):
            self._p()
            df = self.hub.history(m, meta.timeframe)
            arrays = to_arrays(df)
            r = self.sb.run(source, arrays)
            if not r.ok:
                return None, r.error
            n = len(arrays["close"])
            d, a = np.zeros(n, bool), np.zeros(n, bool)
            d[r.start:], a[r.start:] = r.direction, r.act
            out[m] = (arrays, d, a)
        return out, None

    def _judge(self, dec, H, which: str) -> BinaryResult:
        res = []
        for m, (arrays, d, a) in dec.items():
            n = len(arrays["close"])
            s = LOOKBACK + int((n - LOOKBACK) * (1 - self.hf))
            lo, hi = (LOOKBACK, s) if which == "train" else (s, n)
            res.append(judge(d, a, arrays["open"], arrays["close"], H, lo, hi, cost=self._cost(m), t_ms=arrays["t"]))
        return pool(res)

    @staticmethod
    def _fmt(r: BinaryResult) -> str:
        f = lambda x, k=3: "n/a" if x is None else f"{x:.{k}f}"
        return f"independent_trades={r.n} hit_rate={f(r.hit_rate)} chance={f(r.chance)} net_return_per_trade={f(r.net_mean, 4)} p={f(r.p, 3)}"

    # ---------- LLM steps ----------
    def _p(self):
        if self.hb is not None:
            self.hb.progress()

    def _ask(self, rep, user, **kw):
        self._p()
        rep.llm_calls += 1
        return self.llm.chat([{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": user}], **kw)

    def _write(self, rep, h):
        text = self._ask(rep, prompts.WRITE.format(**h), purpose="write", max_tokens=12000)
        for attempt in range(3):
            src = code_block(text)
            try:
                return src, validate(src)
            except ScriptInvalid as e:
                if attempt == 2:
                    return src, str(e)
                text = self._ask(rep, prompts.REPAIR.format(error=e) + "\n\nYour script:\n```python\n" + src + "```", purpose="repair", max_tokens=12000)

    # ---------- the session ----------
    def run_session(self) -> SessionReport:
        sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        rep = SessionReport(sid)
        try:
            ctx = self.context()
            props = extract_json(self._ask(rep, prompts.PROPOSE.format(context=ctx, n=self.n_hyp), purpose="propose", max_tokens=8000))
            props = [p for p in (props if isinstance(props, list) else [props]) if isinstance(p, dict)][: self.n_hyp]
            rep.proposed = len(props)
            finalists = []
            for p in props:
                h = {"name": re.sub(r"\W+", "_", str(p.get("name", "idea")))[:40], "hypothesis": str(p.get("hypothesis", ""))[:400],
                     "edge_source": str(p.get("edge_source", ""))[:400], "timeframe": p.get("timeframe", "1h"),
                     "horizon_bars": p.get("horizon_bars", 4), "markets": p.get("markets", "ALL")}
                best = self._develop(rep, sid, h)
                if best:
                    finalists.append(best)
            finalists.sort(key=lambda x: x[0], reverse=True)
            for score, rec, dec in finalists[: self.top_k]:
                self._holdout(rep, rec, dec, looks=max(min(len(finalists), self.top_k), self.recent_looks() + 1))
            for score, rec, dec in finalists[self.top_k:]:
                rec.stage, rec.reason = "rejected_train", "passed train but outranked; holdout not spent"
                self.j.put("scripts", rec.id, rec)
            self._reflect(rep)
        except BudgetExceeded as e:
            rep.stopped = str(e)
        except Exception as e:                                           # noqa: BLE001  never crash the service
            log.exception("session failed")
            rep.stopped = f"{type(e).__name__}: {e}"
        return rep

    def _record(self, sid, h, src, meta, stage, evals, reason="", parent=None, name=None):
        rec = ScriptRecord(id=script_id(src), name=name or h["name"], hypothesis=h["hypothesis"], source=src,
                           horizon_bars=meta.horizon_bars if meta else 0, timeframe=meta.timeframe if meta else str(h["timeframe"]),
                           markets=(self.focus or (meta.markets if meta.markets != ["ALL"] else ["ALL"])) if meta else [], stage=stage, created_at=datetime.now(timezone.utc), session=sid,
                           model=self.llm.model, parent_id=parent, evals=evals, reason=reason)
        self.j.put("scripts", rec.id, rec)
        return rec

    def _develop(self, rep, sid, h):
        """Write -> validate -> test on train -> up to `rounds` refinements from TRAIN metrics. Returns (score, record, decisions)."""
        src, meta = self._write(rep, h)
        rep.written += 1
        if not isinstance(meta, str) and self.j.get("scripts", script_id(src)) is not None:
            rep.duplicates += 1                              # already tested: do not spend a look or a sandbox run on it
            return None
        if isinstance(meta, str):
            self._record(sid, h, src, None, "rejected_static", [], meta)
            rep.rejected_static += 1
            return None
        best, parent = None, None
        for rnd in range(self.rounds + 1):
            dec, err = self._decisions(src, meta)
            if dec is None:
                self._record(sid, h, src, meta, "rejected_static", [], f"sandbox: {err}", parent, f"{h['name']}_v{rnd}")
                rep.rejected_static += 1
                break
            tr = self._judge(dec, meta.horizon_bars, "train")
            rec = self._record(sid, h, src, meta, "rejected_train", [{"stage": "train", **tr.dump()}], self._fmt(tr), parent, f"{h['name']}_v{rnd}")
            if train_ok(tr, self.g):
                score = (tr.hit_rate - tr.chance) * (tr.n ** 0.5)
                if best is None or score > best[0]:
                    best = (score, rec, dec)
            if rnd == self.rounds:
                break
            parent = rec.id
            try:
                text = self._ask(rep, prompts.REFINE.format(metrics=self._fmt(tr)) + "\n\nCurrent script:\n```python\n" + src + "```", purpose="refine", max_tokens=12000)
                new_src = code_block(text)
                if script_id(new_src) == rec.id or self.j.get("scripts", script_id(new_src)) is not None:
                    break
                new_meta = validate(new_src)
            except ScriptInvalid:
                break
            src, meta = new_src, new_meta
        if best is None:
            rep.rejected_train += 1
        return best

    def _holdout(self, rep, rec, dec, looks):
        ho = self._judge(dec, rec.horizon_bars, "holdout")
        rep.holdout_tested += 1
        rec.evals.append({"stage": "holdout", **ho.dump()})
        strict = holdout_ok(ho, self.g, looks)
        soft = self.g.soft_holdout and holdout_soft_ok(ho, self.g)
        if (strict or soft) and self.n_incubating() >= self.g.max_incubating:
            rec.stage, rec.reason = "rejected_holdout", "passed holdout but all paper-trading slots are full"
            rep.rejected_holdout += 1
        elif strict or soft:
            rec.stage = "incubating"
            rec.reason = "passed holdout; paper trading" if strict else "plausible on holdout (soft gate); paper trading decides"
            rep.incubating.append(rec.name)
        else:
            rec.stage, rec.reason = "rejected_holdout", "failed holdout"
            rep.rejected_holdout += 1
        self.j.put("scripts", rec.id, rec)

    def _reflect(self, rep):
        recs = [r for r in self.j.all("scripts") if r.session == rep.session]
        lines = []
        for r in recs:
            tr = next((e for e in r.evals if e.get("stage") == "train"), {})
            ho = "holdout PASSED" if r.stage in ("incubating", "active") else ("holdout failed" if r.stage == "rejected_holdout" else "")
            lines.append(f"- {r.name} [{r.stage}] {r.hypothesis[:100]} | train n={tr.get('n')} hit={tr.get('hit_rate')} chance={tr.get('chance')} net={tr.get('net_mean')} {ho}")
        if not lines:
            return
        try:
            lessons = extract_json(self._ask(rep, prompts.REFLECT.format(results="\n".join(lines)), purpose="reflect", max_tokens=4000))
        except Exception:                                                # noqa: BLE001
            return
        for t in [x for x in lessons if isinstance(x, str)][:4]:
            self.notes.add(author="llm", kind="lesson", text=t[:400])
            rep.notes += 1

    def n_incubating(self) -> int:
        return sum(1 for r in self.j.all("scripts") if r.stage == "incubating")

    def live_summary(self) -> list:
        """Paper-trading scoreboard for every live script: forward trades, hit rate vs chance, net return after costs."""
        out = []
        for r in self.j.all("scripts"):
            if r.stage not in ("incubating", "active"):
                continue
            res = self.book.result(r.id, r.timeframe, r.horizon_bars)
            open_n = self.book.con.execute("SELECT COUNT(*) FROM script_calls WHERE sid=? AND act AND NOT resolved", [r.id]).fetchall()[0][0]
            out.append({"name": r.name, "open_trades": int(open_n), "stage": r.stage, "age_h": round((datetime.now(timezone.utc) - r.created_at).total_seconds() / 3600, 1),
                        "timeframe": r.timeframe, "horizon": r.horizon_bars, "paper_trades": res.n,
                        "hit": None if res.hit_rate is None else round(res.hit_rate, 3), "chance": None if res.chance is None else round(res.chance, 3),
                        "net_per_trade_pct": None if res.net_mean is None else round(res.net_mean * 100, 3),
                        "paper_pnl_pct": None if res.net_mean is None else round(res.net_mean * res.n * 100, 2), "p": round(res.p, 4)})
        return sorted(out, key=lambda x: -(x["paper_pnl_pct"] or -99))

    def positions(self) -> list:
        """Every open paper trade marked to the current mark price (one API call for all prices)."""
        marks = {c.market: c.mark_price for c in self.hub.a.all_contexts()}
        out = []
        for r in self.j.all("scripts"):
            if r.stage in ("incubating", "active"):
                for p in self.book.open_positions(r.id, r.timeframe, r.horizon_bars, self.hub.a, marks):
                    out.append({"strategy": r.name, "tf": r.timeframe, "horizon": r.horizon_bars, **p})
        return sorted(out, key=lambda x: -x["net_pct"])

    # ---------- forward testing (code only) ----------
    def tick(self) -> dict:
        """Hourly: ask each live script its two questions on the newest closed bar, resolve old calls, apply forward verdicts."""
        live = [r for r in self.j.all("scripts") if r.stage in ("incubating", "active")]
        stats = {"calls": 0, "resolved": 0, "promoted": [], "retired": []}
        frames = {}
        for r in live:
            ms = self._markets(type("M", (), {"markets": r.markets, "timeframe": r.timeframe})())
            for m in ms:
                if (m, r.timeframe) not in frames:
                    self._p()
                    frames[(m, r.timeframe)] = to_arrays(self.hub.live(m, r.timeframe))
                arrays = frames[(m, r.timeframe)]
                t_last = int(arrays["t"][-1])
                if t_last > self.book.last_t(r.id, m):
                    d, a, err = ask_live(self.sb, r.source, arrays)
                    if err is None:
                        self.book.record(r.id, m, r.timeframe, t_last, r.horizon_bars, d, a)
                        stats["calls"] += 1
                stats["resolved"] += self.book.resolve(r.id, m, arrays, r.timeframe)
        k = max(1, len(live))
        for r in live:
            from ..scripts.binary import forward_verdict
            res = self.book.result(r.id, r.timeframe, r.horizon_bars)
            v = forward_verdict(res, self.g, k)
            r.evals = [e for e in r.evals if e.get("stage") != "forward"] + [{"stage": "forward", **res.dump()}]
            if v == "promote" and r.stage == "incubating":
                r.stage, r.reason = "active", "forward paper evidence: " + self._fmt(res)
                stats["promoted"].append(r.name)
                self.notes.add(author="code", kind="lesson", text=f"{r.name} earned 'active' on forward paper data: {self._fmt(res)}", strategy=r.name)
            elif v == "retire":
                r.stage, r.reason = "retired", "forward paper evidence against: " + self._fmt(res)
                stats["retired"].append(r.name)
            elif v == "wait" and r.stage == "incubating" and (
                    (datetime.now(timezone.utc) - r.created_at).total_seconds() / 86400 > self.g.fwd_max_days or res.n >= 3 * self.g.fwd_min_n):
                r.stage, r.reason = "retired", "no forward edge after patience window: " + self._fmt(res)   # frees the paper-trading slot
                stats["retired"].append(r.name)
            self.j.put("scripts", r.id, r)
        return stats
