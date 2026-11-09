"""Read-only dashboard: serves one HTML page plus /api/state, built from the heartbeat/state files the services already write.
Never touches the DuckDB files (single-writer), never accepts input."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA = Path("data")


def _load(name):
    try:
        return json.loads((DATA / name).read_text())
    except Exception:  # noqa: BLE001
        return None


def _age(d):
    try:
        return round((datetime.now(timezone.utc) - datetime.fromisoformat(d["at"])).total_seconds())
    except Exception:  # noqa: BLE001
        return None


def _sessions(n=60):
    p = DATA / "sessions.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def state() -> dict:
    loop, dire = _load("heartbeat.json"), _load("heartbeat_director.json")
    for d in (loop, dire):
        if d:
            d["age_s"] = _age(d)
    return {"now": datetime.now(timezone.utc).isoformat(), "loop": loop, "director": dire, "loop_state": _load("loop_state.json"),
            "scripts": _load("director_scripts.json") or [], "sessions": _sessions()}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/state"):
            body, ctype = json.dumps(state()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = (Path(__file__).parent / "dashboard.html").read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


def serve(host="127.0.0.1", port=8765):
    ThreadingHTTPServer((host, port), Handler).serve_forever()
