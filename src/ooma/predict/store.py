"""Persistence for the loop: predictors (with learned state), snapshots (the prediction ledger), meta."""
from __future__ import annotations

import json
from datetime import datetime

COLS = ["id", "horizon", "kind", "params", "state", "stats", "status", "note_id", "born", "parent", "w"]


class PredictStore:
    def __init__(self, con):
        self.con = con
        con.execute("CREATE TABLE IF NOT EXISTS predictors (id VARCHAR PRIMARY KEY, horizon INT, kind VARCHAR, "
                    "params VARCHAR, state VARCHAR, stats VARCHAR, status VARCHAR, note_id VARCHAR, born VARCHAR, "
                    "parent VARCHAR, w DOUBLE)")
        con.execute("CREATE TABLE IF NOT EXISTS snapshots (id VARCHAR PRIMARY KEY, market VARCHAR, ct VARCHAR, "
                    "horizon INT, price0 DOUBLE, features VARCHAR, preds VARCHAR, base_p DOUBLE, ens_p DOUBLE, "
                    "label BOOLEAN, ret DOUBLE, resolved_at VARCHAR)")
        con.execute("CREATE TABLE IF NOT EXISTS meta (k VARCHAR PRIMARY KEY, v VARCHAR)")
        con.execute("CREATE TABLE IF NOT EXISTS context_log (market VARCHAR, t VARCHAR, funding DOUBLE, "
                    "open_interest DOUBLE, premium DOUBLE, mark DOUBLE, day_volume DOUBLE)")

    def save_predictors(self, rows: list[dict]):
        import polars as pl
        df = pl.DataFrame({c: [r[c] for r in rows] for c in COLS},
                          schema_overrides={"note_id": pl.Utf8, "parent": pl.Utf8})
        self.con.register("_tmp_pred", df.to_arrow())
        self.con.execute(f"INSERT OR REPLACE INTO predictors SELECT {','.join(COLS)} FROM _tmp_pred")
        self.con.unregister("_tmp_pred")

    def load_predictors(self) -> list[dict]:
        return [dict(zip(COLS, r)) for r in self.con.execute(f"SELECT {','.join(COLS)} FROM predictors").fetchall()]

    def add_snapshot(self, sid, market, ct: datetime, H, price0, f, preds, base_p, ens_p):
        self.con.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,NULL,NULL,NULL)",
                         [sid, market, ct.isoformat(), H, price0, json.dumps(f),
                          json.dumps(preds), base_p, ens_p])

    def resolve_snapshot(self, sid, label: bool, ret: float, at: datetime):
        self.con.execute("UPDATE snapshots SET label=?, ret=?, resolved_at=? WHERE id=?", [label, ret, at.isoformat(), sid])

    def pending(self) -> list[dict]:
        cols = ["id", "market", "ct", "horizon", "price0", "features", "preds", "base_p", "ens_p"]
        rows = self.con.execute(f"SELECT {','.join(cols)} FROM snapshots WHERE label IS NULL ORDER BY ct").fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def resolved(self, horizon: int | None = None, limit: int = 100000) -> list[tuple]:
        q = "SELECT ct, market, horizon, base_p, ens_p, label, ret FROM snapshots WHERE label IS NOT NULL"
        if horizon:
            q += f" AND horizon={int(horizon)}"
        return self.con.execute(q + f" ORDER BY ct DESC LIMIT {int(limit)}").fetchall()

    def set_meta(self, k, v):
        self.con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", [k, json.dumps(v)])

    def get_meta(self, k, default=None):
        r = self.con.execute("SELECT v FROM meta WHERE k=?", [k]).fetchall()
        return json.loads(r[0][0]) if r else default

    def log_context(self, ctx):
        self.con.execute("INSERT INTO context_log VALUES (?,?,?,?,?,?,?)",
                         [ctx.market, ctx.as_of.isoformat(), ctx.funding_rate, ctx.open_interest,
                          ctx.premium, ctx.mark_price, ctx.day_notional_volume])
