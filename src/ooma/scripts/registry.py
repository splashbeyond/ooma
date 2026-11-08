from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Stage = Literal["rejected_static", "rejected_train", "rejected_holdout", "incubating", "active", "retired"]


class ScriptRecord(BaseModel):
    id: str
    name: str
    hypothesis: str
    source: str
    horizon_bars: int
    timeframe: str
    markets: list[str]
    stage: Stage
    created_at: datetime
    session: str = ""
    model: str | None = None
    parent_id: str | None = None
    evals: list[dict] = []
    reason: str = ""


def script_id(source: str) -> str:
    norm = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return hashlib.sha1(norm.encode()).hexdigest()[:12]
