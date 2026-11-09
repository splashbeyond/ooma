import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from ooma import dashboard


def test_dashboard_serves_state_and_page(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DATA", tmp_path)
    (tmp_path / "heartbeat.json").write_text(json.dumps({"at": "2026-01-01T00:00:00+00:00", "ok": True}))
    (tmp_path / "sessions.jsonl").write_text('{"n": 1, "proposed": 2}\nnot json\n')
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    st = json.load(urllib.request.urlopen(base + "/api/state"))
    assert st["loop"]["ok"] and st["director"] is None and st["sessions"] == [{"n": 1, "proposed": 2}] and st["scripts"] == []
    assert b"<title>ooma</title>" in urllib.request.urlopen(base + "/").read()
    srv.shutdown()
