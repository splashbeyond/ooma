from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..types import Findings


class Module(Protocol):
    """Pure function of data at or before `as_of`. Never reads the future."""
    name: str

    def run(self, market: str, timeframe: str, as_of: datetime, **data) -> Findings: ...
