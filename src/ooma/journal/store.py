"""DuckDB journal. One table per model, JSON payload, so schema changes cost nothing."""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import duckdb
from pydantic import BaseModel

from ..types import Grade, Note, Setup, Trade

M = TypeVar("M", bound=BaseModel)
TABLES = {"setups": Setup, "trades": Trade, "grades": Grade, "notes": Note}


class Journal:
    def __init__(self, path: str | Path = ":memory:"):
        self.con = duckdb.connect(str(path))
        for t in TABLES:
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {t} (id VARCHAR PRIMARY KEY, payload VARCHAR)")

    def put(self, table: str, id: str, obj: BaseModel) -> None:
        self.con.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)", [id, obj.model_dump_json()])

    def get(self, table: str, id: str) -> BaseModel | None:
        r = self.con.execute(f"SELECT payload FROM {table} WHERE id=?", [id]).fetchone()
        return TABLES[table].model_validate_json(r[0]) if r else None

    def all(self, table: str) -> list:
        rows = self.con.execute(f"SELECT payload FROM {table}").fetchall()
        return [TABLES[table].model_validate_json(r[0]) for r in rows]

    # convenience
    def save_setup(self, s: Setup): self.put("setups", s.id, s)
    def save_trade(self, t: Trade): self.put("trades", t.id, t)
    def save_grade(self, g: Grade): self.put("grades", g.trade_id, g)

    def ungraded_closed_trades(self) -> list[Trade]:
        graded = {g.trade_id for g in self.all("grades")}
        return [t for t in self.all("trades") if t.closed_at and t.id not in graded]
