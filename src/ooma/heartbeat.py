"""Heartbeat + watchdog. A background thread writes a status file every minute (so you can always see it is alive and what it
is doing) and pings systemd's watchdog ONLY while the main loop is making progress. If the main loop hangs (stuck API call,
stuck sandbox), the pings stop and systemd restarts the service."""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("ooma.heartbeat")


def sd_notify(msg: str) -> bool:
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect("\0" + addr[1:] if addr.startswith("@") else addr)
            s.sendall(msg.encode())
        return True
    except OSError:
        return False


def rss_mb() -> float | None:
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except OSError:
        return None


class Heartbeat:
    def __init__(self, path: str, name: str, interval: float = 60.0, stale_after: float = 1500.0):
        self.path, self.name, self.interval, self.stale = Path(path), name, interval, stale_after
        self.started = time.time()
        self.fields: dict = {}
        self.activity = "starting"
        self.activity_since = time.time()
        self.last_progress = time.time()
        self.beats = 0
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        sd_notify("READY=1")
        self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def progress(self):
        self.last_progress = time.time()

    def update(self, activity: str | None = None, **fields):
        changed = False
        with self._lock:
            if activity and activity != self.activity:
                self.activity, self.activity_since, changed = activity, time.time(), True
            self.fields.update(fields)
        self.progress()
        if changed:
            try:
                self._write()                                          # a new activity shows up at once, not at the next timer beat
            except Exception:                                          # noqa: BLE001
                log.exception("heartbeat write failed")

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            d = {"service": self.name, "ok": now - self.last_progress < self.stale, "at": datetime.now(timezone.utc).isoformat(),
                 "uptime_s": round(now - self.started), "pid": os.getpid(), "activity": self.activity,
                 "activity_for_s": round(now - self.activity_since), "idle_s": round(now - self.last_progress),
                 "beats": self.beats, "rss_mb": rss_mb(), **self.fields}
        return d

    def _write(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.snapshot(), default=str))
        os.replace(tmp, self.path)                                     # atomic: readers never see a half-written file

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.beats += 1
                self._write()
                if time.time() - self.last_progress < self.stale:
                    sd_notify("WATCHDOG=1")
                else:
                    log.error("no progress for %ds: withholding watchdog so systemd restarts the service", time.time() - self.last_progress)
            except Exception:                                          # noqa: BLE001  the heartbeat itself must never die
                log.exception("heartbeat write failed")
