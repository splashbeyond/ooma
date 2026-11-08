import json
import time

from ooma.heartbeat import Heartbeat, sd_notify
from ooma.scripts.binary import BinaryResult, Gates, forward_verdict, holdout_soft_ok, pool


def test_heartbeat_writes_file_repeatedly_and_reports_activity(tmp_path):
    hb = Heartbeat(str(tmp_path / "hb.json"), "t", interval=0.1).start()
    hb.update("researching", sessions=3)
    time.sleep(0.45)
    d = json.loads((tmp_path / "hb.json").read_text())
    assert d["ok"] and d["activity"] == "researching" and d["sessions"] == 3 and d["beats"] >= 2
    hb.stop()


def test_heartbeat_goes_unhealthy_when_main_loop_stops_making_progress(tmp_path):
    hb = Heartbeat(str(tmp_path / "hb.json"), "t", interval=0.05, stale_after=0.2).start()
    time.sleep(0.5)                                           # nobody calls progress(): a hung main loop
    d = json.loads((tmp_path / "hb.json").read_text())
    assert d["ok"] is False and d["idle_s"] >= 0
    hb.stop()


def test_sd_notify_is_a_safe_noop_without_systemd(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert sd_notify("WATCHDOG=1") is False


def test_soft_gate_admits_plausible_but_not_bad_candidates():
    g = Gates()
    ok = BinaryResult(n=25, hits=14, hit_rate=0.56, chance=0.52, p=0.3, net_mean=0.001)
    assert holdout_soft_ok(ok, g)
    assert not holdout_soft_ok(BinaryResult(n=25, hits=9, hit_rate=0.36, chance=0.52, p=0.99, net_mean=-0.01), g)
    assert not holdout_soft_ok(BinaryResult(n=5, hits=5, hit_rate=1.0, chance=0.5, p=0.03, net_mean=0.02), g)     # too few trades


def test_forward_verdict_is_still_strict_even_though_entry_is_soft():
    g = Gates()
    marginal = BinaryResult(n=40, hits=22, hit_rate=0.55, chance=0.52, p=0.3, net_mean=0.001, net_t=0.5)
    assert forward_verdict(marginal, g, k_incubating=12) == "wait"            # soft entry never means soft promotion
