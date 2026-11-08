"""News/events/catalysts via LLM web search. Every item needs URL + timestamp; cache 1-4h (milestone 5)."""
from __future__ import annotations

from datetime import datetime

from ..types import Findings


def run(market: str, timeframe: str, as_of: datetime, **data) -> Findings:
    raise NotImplementedError("research")
