from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..types import Bias


class Thesis(BaseModel):
    """LLM output (PRD s5). Numbers must come from module/strategy inputs, never invented."""
    market: str
    timeframe: str
    as_of: datetime
    decision: Literal["long", "short", "no_trade"]
    setup_tags: list[str] = []
    thesis: str
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    entry: float | None = None
    stop: float | None = None
    targets: list[float] = []
    invalidation: str = ""
    horizon_hours: int = 24
    confidence: float = Field(0.0, ge=0, le=1)
    module_agreement: dict[str, Bias] = {}
    model: str = ""
    prompt_version: str = ""
