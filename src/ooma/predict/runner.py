"""Replay (warm start from history) and live ticking. State lives in DuckDB, so restarts lose nothing."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .features import feature_frame
from .loop import RecursiveLoop
from .report import digest

log = logging.getLogger("ooma")
REF = "BTC"


def _frames(adapter, markets, start, end):
    ref = adapter.candles(REF, "1h", start, end)
    out = {}
    for m in markets:
        c = ref if m == REF else adapter.candles(m, "1h", start, end)
        out[m] = feature_frame(c, adapter.funding_history(m, start, end), ref)
    return out


def replay(loop: RecursiveLoop, adapter, markets, days: int, end: datetime | None = None) -> dict:
    """Walk-forward over history: predict, wait for the outcome, learn, repeat. No look-ahead by construction."""
    end = end or datetime.now(timezone.utc)
    frames = _frames(adapter, markets, end - timedelta(days=days), end)
    loop.warm_start(frames, born=frames[markets[0]]["ct"][0].isoformat())
    n = {m: loop.process_market(m, fr, persist=False) for m, fr in frames.items()}
    loop.save()
    return n


def tick(loop: RecursiveLoop, adapter, markets, lookback_h: int = 900) -> dict:
    end = datetime.now(timezone.utc)
    frames = _frames(adapter, markets, end - timedelta(hours=lookback_h), end)
    if not any(len(P) for P in loop.preds.values()):
        loop.warm_start(frames, born=end.isoformat())
    n = {m: loop.process_market(m, fr, persist=True) for m, fr in frames.items()}
    for m in markets:
        try:
            loop.s.log_context(adapter.context(m))          # OI has no history API: record it from now on
        except Exception as e:                               # noqa: BLE001
            log.warning("context log failed for %s: %s", m, e)
    loop.save()
    return n


def run_forever(loop: RecursiveLoop, adapter, markets, report_dir="data/reports", heartbeat="data/heartbeat.json"):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    last_report = None
    while True:
        try:
            n = tick(loop, adapter, markets)
            log.info("tick ok, new bars: %s", n)
            today = datetime.now(timezone.utc).date()
            if last_report != today:
                Path(report_dir, f"{today}.md").write_text(digest(loop))
                last_report = today
            Path(heartbeat).write_text(json.dumps({"ok": True, "at": datetime.now(timezone.utc).isoformat(), "bars": loop.bars}))
        except Exception:                                    # noqa: BLE001  never die: log and retry next hour
            log.exception("tick failed")
            Path(heartbeat).write_text(json.dumps({"ok": False, "at": datetime.now(timezone.utc).isoformat()}))
        now = datetime.now(timezone.utc)
        nxt = (now + timedelta(hours=1)).replace(minute=0, second=20, microsecond=0)
        time.sleep(max(30, (nxt - now).total_seconds()))
