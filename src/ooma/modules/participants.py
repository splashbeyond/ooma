"""Skilled vs weak wallet cohort positioning from wallets/ tables, using only scores known before as_of."""
from __future__ import annotations

from datetime import datetime

from ..types import Findings


def run(market: str, timeframe: str, as_of: datetime, **data) -> Findings:
    raise NotImplementedError("participants")
