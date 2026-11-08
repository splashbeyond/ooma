"""Discover -> fetch -> profile -> classify -> track. State lives in DuckDB (same file as everything else).

Discovery uses the public leaderboard, which lists TODAY'S winners: a survivorship-biased pool. So nothing here
claims these wallets "will" win. Their skill is re-scored walk-forward, and the only proof that matters is whether
their positioning predicts price going FORWARD, which the prediction loop measures out-of-sample from the first snapshot.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import polars as pl

from .api import WalletAPI, spot_hedge_ratio
from .profile import Profile, prepare, profile_wallet
from .scoring import status

FILL_COLS = ["address", "time", "coin", "px", "sz", "side", "dir", "closed_pnl", "fee", "maker", "start_pos", "twap"]


def _pl(con, sql: str, params=()) -> pl.DataFrame:
    """Query -> polars using only fetchall + description (no pl()/arrow() methods, which vary by DuckDB build)."""
    cur = con.execute(sql, list(params))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None) if rows else pl.DataFrame({c: [] for c in cols})


class WalletTracker:
    def __init__(self, con, api: WalletAPI, cfg: dict | None = None):
        self.con, self.api, self.cfg = con, api, cfg or {}
        con.execute("CREATE TABLE IF NOT EXISTS wallets (address VARCHAR PRIMARY KEY, source VARCHAR, added VARCHAR, "
                    "role VARCHAR, status VARCHAR, kind VARCHAR, profile VARCHAR, scored_at VARCHAR, lb_json VARCHAR)")
        con.execute("CREATE TABLE IF NOT EXISTS wallet_fills (address VARCHAR, time BIGINT, coin VARCHAR, px DOUBLE, sz DOUBLE, "
                    "side VARCHAR, dir VARCHAR, closed_pnl DOUBLE, fee DOUBLE, maker BOOLEAN, start_pos DOUBLE, twap BOOLEAN)")
        con.execute("CREATE TABLE IF NOT EXISTS wallet_positions (t VARCHAR, address VARCHAR, grp VARCHAR, coin VARCHAR, "
                    "szi DOUBLE, notional DOUBLE, entry DOUBLE, leverage DOUBLE, liq DOUBLE)")
        con.execute("CREATE TABLE IF NOT EXISTS cohort (t VARCHAR, coin VARCHAR, grp VARCHAR, net DOUBLE, gross DOUBLE, "
                    "n_long INT, n_short INT, n_wallets INT)")

    # ---- discovery ----
    def discover(self, n: int = 400, min_av: float = 50_000, max_turnover: float = 100.0, min_age_pnl: float = 0.0) -> list[str]:
        """Pre-filter the leaderboard BEFORE spending API calls. Turnover (30d volume / account value) above ~100x is
        a market maker or a high-frequency book, not a directional trader. Rank by all-time ROI among profitable accounts."""
        lb = self.api.leaderboard()
        lb = lb.filter((pl.col("account_value") >= min_av) & (pl.col("allTime_pnl") > min_age_pnl)
                       & (pl.col("month_pnl") > 0) & (pl.col("month_vlm") / pl.col("account_value") <= max_turnover)
                       & (pl.col("month_vlm") > 0))
        pick = lb.sort("allTime_roi", descending=True).head(n)
        now = datetime.now(timezone.utc).isoformat()
        for r in pick.to_dicts():
            self.con.execute("INSERT OR IGNORE INTO wallets VALUES (?,?,?,?,?,?,?,?,?)",
                             [r["address"], "leaderboard", now, None, "new", None, None, None, json.dumps(r)])
        return pick["address"].to_list()

    # ---- fetch + profile ----
    def _fills_df(self, address: str) -> pl.DataFrame:
        return _pl(self.con, f"SELECT {','.join(FILL_COLS[1:])} FROM wallet_fills WHERE address=? ORDER BY time", [address])

    def refresh(self, address: str, n_screened: int = 1) -> Profile:
        last = self.con.execute("SELECT COALESCE(MAX(time),0) FROM wallet_fills WHERE address=?", [address]).fetchall()[0][0]
        new = self.api.fills(address, start_ms=last + 1 if last else 0)
        if new:
            df = prepare(new).with_columns(pl.lit(address).alias("address")).select(FILL_COLS)
            self.con.register("_nf", df.to_arrow())
            self.con.execute(f"INSERT INTO wallet_fills SELECT {','.join(FILL_COLS)} FROM _nf")
            self.con.unregister("_nf")
        df = self._fills_df(address)
        role = self.api.role(address)
        hedge, fpnl = 0.0, None
        if role == "user" and df.height >= 60:
            hedge = spot_hedge_ratio(self.api.state(address), self.api.spot(address))
            fpnl = self.api.funding_pnl(address, int(df["time"].min()))
        p = profile_wallet(address, df, role=role, spot_hedge=hedge, funding_pnl=fpnl, cfg=self.cfg.get("profile"))
        self.con.execute("UPDATE wallets SET role=?, kind=?, profile=?, scored_at=? WHERE address=?",
                         [role, p.kind, json.dumps(p.to_dict()), datetime.now(timezone.utc).isoformat(), address])
        return p

    def profile_all(self, addresses: list[str] | None = None, log=lambda *_: None) -> dict:
        addrs = addresses or [r[0] for r in self.con.execute("SELECT address FROM wallets").fetchall()]
        profs = {}
        for i, a in enumerate(addrs):
            try:
                profs[a] = self.refresh(a)
            except Exception as e:                                   # noqa: BLE001  one bad wallet must not stop the batch
                log(f"skip {a[:10]}: {e}")
            if i % 25 == 0:
                log(f"profiled {i + 1}/{len(addrs)}")
        k = sum(p.kind == "directional" for p in profs.values())     # multiple-testing denominator
        for a, p in profs.items():
            self.con.execute("UPDATE wallets SET status=? WHERE address=?", [status(p, k, self.cfg.get("profile")), a])
        return profs

    def restatus(self) -> dict:
        """Re-apply the status rules to stored profiles. No API calls (use after changing rules)."""
        rows = self.con.execute("SELECT address, profile FROM wallets WHERE profile IS NOT NULL").fetchall()
        profs = {a: Profile(**json.loads(j)) for a, j in rows}
        k = sum(p.kind == "directional" for p in profs.values())
        out = {}
        for a, p in profs.items():
            out[a] = status(p, k, self.cfg.get("profile"))
            self.con.execute("UPDATE wallets SET status=? WHERE address=?", [out[a], a])
        return out

    def by_status(self, st: str) -> list[dict]:
        rows = self.con.execute("SELECT address, profile FROM wallets WHERE status=?", [st]).fetchall()
        return [json.loads(p) for _, p in rows]

    # ---- live tracking ----
    def snapshot(self, now: datetime | None = None) -> int:
        """Positions of skilled and weak wallets right now -> per-coin cohort aggregates."""
        now = now or datetime.now(timezone.utc)
        t = now.isoformat()
        rows = self.con.execute("SELECT address, status FROM wallets WHERE status IN ('skilled','weak')").fetchall()
        agg: dict = {}
        for addr, st in rows:
            try:
                state = self.api.state(addr)
            except Exception:                                        # noqa: BLE001
                continue
            for ap in state.get("assetPositions", []):
                p = ap["position"]
                szi, val = float(p["szi"]), float(p.get("positionValue", 0) or 0)
                if szi == 0:
                    continue
                signed = val if szi > 0 else -val
                self.con.execute("INSERT INTO wallet_positions VALUES (?,?,?,?,?,?,?,?,?)",
                                 [t, addr, st, p["coin"], szi, signed, float(p.get("entryPx") or 0),
                                  float(p["leverage"]["value"]), float(p["liquidationPx"]) if p.get("liquidationPx") else None])
                a = agg.setdefault((p["coin"], st), [0.0, 0.0, 0, 0, set()])
                a[0] += signed; a[1] += abs(signed); a[2] += szi > 0; a[3] += szi < 0; a[4].add(addr)
        for (coin, grp), (net, gross, nl, ns, ws) in agg.items():
            self.con.execute("INSERT INTO cohort VALUES (?,?,?,?,?,?,?,?)", [t, coin, grp, net, gross, nl, ns, len(ws)])
        return len(rows)


def cohort_frame(con, coin: str) -> pl.DataFrame:
    """Per-snapshot features for one coin: skill_net / weak_net in [-1, 1] and 4h change. Empty until snapshots exist."""
    d = _pl(con, "SELECT t, grp, net, gross FROM cohort WHERE coin=? ORDER BY t", [coin])
    if d.height == 0:
        return pl.DataFrame()
    d = d.with_columns(pl.col("t").str.to_datetime(time_zone="UTC", time_unit="ms"),
                       (pl.col("net") / pl.col("gross")).alias("r"))
    s = d.filter(pl.col("grp") == "skilled").select("t", pl.col("r").alias("skill_net")).sort("t")
    w = d.filter(pl.col("grp") == "weak").select("t", pl.col("r").alias("weak_net")).sort("t")
    out = s.join(w, on="t", how="full", coalesce=True).sort("t") if s.height and w.height else (s if s.height else w)
    for col in ("skill_net", "weak_net"):
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return out.with_columns((pl.col("skill_net") - pl.col("skill_net").shift(4)).alias("skill_chg_4h"))
