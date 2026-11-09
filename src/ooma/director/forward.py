"""Forward (live paper) testing of scripts. Each closed bar the script is asked its two binary questions per market; the
answers are frozen in a table, then resolved when the horizon passes. Only these forward calls can promote a script."""
from __future__ import annotations

import numpy as np

from ..adapters.hyperliquid import INTERVAL_MS
from ..scripts.binary import COST, BinaryResult, binom_sf, pool
from ..scripts.sandbox import SandboxRunner


class ForwardBook:
    def __init__(self, con):
        self.con = con
        con.execute("CREATE TABLE IF NOT EXISTS script_calls (sid VARCHAR, market VARCHAR, tf VARCHAR, t_ms BIGINT, "
                    "horizon INT, direction BOOLEAN, act BOOLEAN, resolved BOOLEAN, ret DOUBLE, PRIMARY KEY (sid, market, t_ms))")
        con.execute("ALTER TABLE script_calls ADD COLUMN IF NOT EXISTS entry DOUBLE")

    def record(self, sid, market, tf, t_ms, horizon, direction, act):
        self.con.execute("INSERT OR IGNORE INTO script_calls (sid, market, tf, t_ms, horizon, direction, act, resolved, ret) VALUES (?,?,?,?,?,?,?,FALSE,NULL)",
                         [sid, market, tf, int(t_ms), horizon, bool(direction), bool(act)])

    def last_t(self, sid, market):
        r = self.con.execute("SELECT MAX(t_ms) FROM script_calls WHERE sid=? AND market=?", [sid, market]).fetchall()[0][0]
        return r or 0

    def resolve(self, sid, market, arrays: dict, tf: str) -> int:
        """Fill in realised returns for calls whose horizon has fully elapsed in `arrays` (live bars)."""
        t = arrays["t"].astype("int64")
        pos = {int(v): i for i, v in enumerate(t)}
        rows = self.con.execute("SELECT t_ms, horizon FROM script_calls WHERE sid=? AND market=? AND NOT resolved",
                                [sid, market]).fetchall()
        n = 0
        for t_ms, H in rows:
            i = pos.get(int(t_ms))
            if i is None or i + H >= len(t):
                continue
            ret = float(arrays["close"][i + H] / arrays["open"][i + 1] - 1.0)
            self.con.execute("UPDATE script_calls SET resolved=TRUE, ret=? WHERE sid=? AND market=? AND t_ms=?", [ret, sid, market, int(t_ms)])
            n += 1
        return n

    def open_positions(self, sid, tf, H, adapter, marks: dict, cost=COST) -> list[dict]:
        """Live mark-to-market of paper trades that have not reached their horizon. Mirrors result(): an 'act' call only counts as a
        trade if it is at least H bars after the previous counted trade in that market. Entry = open of the bar after the signal."""
        from datetime import datetime, timedelta, timezone
        step, out = INTERVAL_MS[tf], []
        for (market,) in self.con.execute("SELECT DISTINCT market FROM script_calls WHERE sid=? AND act", [sid]).fetchall():
            last = None
            for t_ms, d, resolved, entry in self.con.execute(
                    "SELECT t_ms, direction, resolved, entry FROM script_calls WHERE sid=? AND market=? AND act ORDER BY t_ms", [sid, market]).fetchall():
                if last is not None and t_ms < last + H * step:
                    continue                                    # overlapping call, not a counted trade
                last = t_ms
                if resolved:
                    continue
                if entry is None:
                    t0 = datetime.fromtimestamp((t_ms + step) / 1000, timezone.utc)
                    try:
                        c = adapter.candles(market, tf, t0, t0 + timedelta(milliseconds=step))
                        if c.height:
                            entry = float(c["open"][0])
                            self.con.execute("UPDATE script_calls SET entry=? WHERE sid=? AND market=? AND t_ms=?", [entry, sid, market, int(t_ms)])
                    except Exception:                           # noqa: BLE001
                        entry = None
                mark = marks.get(market)
                if not entry or not mark:
                    continue
                gross = (mark / entry - 1.0) * (1 if d else -1)
                out.append({"market": market, "side": "LONG" if d else "SHORT", "entry": entry, "mark": mark,
                            "gross_pct": round(gross * 100, 3), "net_pct": round((gross - cost) * 100, 3),
                            "opened": int(t_ms + step), "closes": int(t_ms + step * (H + 1))})
        return out

    def result(self, sid, tf, H, cost=COST) -> BinaryResult:
        step = INTERVAL_MS[tf]
        per = []
        for (market,) in self.con.execute("SELECT DISTINCT market FROM script_calls WHERE sid=?", [sid]).fetchall():
            rows = self.con.execute("SELECT t_ms, direction, act, ret FROM script_calls WHERE sid=? AND market=? AND resolved ORDER BY t_ms",
                                    [sid, market]).fetchall()
            if not rows:
                continue
            ret = np.array([r[3] for r in rows]); d = np.array([r[1] for r in rows]); a = np.array([r[2] for r in rows])
            up = ret > 0
            res = BinaryResult(n_bars=len(rows), n_acts=int(a.sum()), up_rate=float(up.mean()))
            chosen, last = [], -1
            for j, r in enumerate(rows):
                if a[j] and r[0] >= last + H * step:
                    chosen.append(j); last = r[0]
            if chosen:
                c = np.array(chosen)
                correct = d[c] == up[c]
                net = np.where(d[c], ret[c], -ret[c]) - cost
                res.n, res.hits = len(c), int(correct.sum())
                res.hit_rate, res.chance = res.hits / res.n, float(max(up.mean(), 1 - up.mean()))
                res.p = res.p_binom = binom_sf(res.hits, res.n, res.chance)
                res.trades = (np.array([rows[j][0] for j in chosen], dtype=float), correct.astype(float) - res.chance, net)
                res.net_mean = float(net.mean())
                if res.n >= 3 and net.std(ddof=1) > 0:
                    res.net_t = float(net.mean() / (net.std(ddof=1) / np.sqrt(res.n)))
            per.append(res)
        return pool(per, block_days=1.0)


def ask_live(sandbox: SandboxRunner, source: str, arrays: dict):
    """The script's two answers for the latest closed bar."""
    r = sandbox.run(source, arrays, start=0, mode="last")
    if not r.ok:
        return None, None, r.error
    return bool(r.direction[0]), bool(r.act[0]), None
