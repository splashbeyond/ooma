"""ooma's notebook. Notes are written by code, the LLM, or the user; each can gather evidence."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..types import Note
from .store import Journal

MIN_EVIDENCE = 10   # evidence needed before a hypothesis may leave 'open'


class NoteStore:
    def __init__(self, journal: Journal):
        self.j = journal

    def add(self, *, author, kind, text, strategy=None, market=None, trade_ids=()) -> Note:
        n = Note(id=uuid.uuid4().hex[:12], created_at=datetime.now(timezone.utc), author=author,
                 kind=kind, text=text, strategy=strategy, market=market, trade_ids=list(trade_ids))
        self.j.put("notes", n.id, n)
        return n

    def record_evidence(self, note_id: str, supports: bool, trade_id: str) -> Note:
        n: Note = self.j.get("notes", note_id)
        if trade_id in n.trade_ids:
            return n                      # a trade counts once
        n.trade_ids.append(trade_id)
        if supports:
            n.n_support += 1
        else:
            n.n_against += 1
        self.j.put("notes", n.id, n)
        return n

    def set_status(self, note_id: str, status: str) -> Note:
        n: Note = self.j.get("notes", note_id)
        n.status = status
        self.j.put("notes", n.id, n)
        return n

    def find(self, *, strategy=None, market=None, status=None, kind=None) -> list[Note]:
        out = self.j.all("notes")
        for k, v in (("strategy", strategy), ("market", market), ("status", status), ("kind", kind)):
            if v is not None:
                out = [n for n in out if getattr(n, k) == v]
        return sorted(out, key=lambda n: n.created_at, reverse=True)

    def context_for(self, strategy: str | None, market: str | None, k: int = 10) -> list[Note]:
        """Notes to show the analyst: supported/promoted first, never refuted, scoped to this strategy/market or global."""
        rank = {"promoted": 0, "supported": 1, "open": 2}
        pool = [n for n in self.j.all("notes") if n.status != "refuted"
                and n.strategy in (None, strategy) and n.market in (None, market)]
        pool.sort(key=lambda n: (rank[n.status], -n.created_at.timestamp()))
        return pool[:k]
