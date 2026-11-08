"""Market-agnostic adapter interface. Modules depend only on this."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol

import polars as pl


class Capability(str, Enum):
    CANDLES = "candles"
    CONTEXT = "context"      # funding, OI, premium
    BOOK = "book"
    TRADES = "trades"
    PARTICIPANTS = "participants"


@dataclass(frozen=True)
class MarketContext:
    market: str
    as_of: datetime
    funding_rate: float      # per funding interval
    open_interest: float     # base units
    mark_price: float
    oracle_price: float
    premium: float
    day_notional_volume: float


@dataclass(frozen=True)
class BookSnapshot:
    market: str
    time: datetime
    bids: list[tuple[float, float]] = field(default_factory=list)  # (px, size), best first
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.bids[0][0] + self.asks[0][0]) / 2


class MarketAdapter(Protocol):
    name: str
    capabilities: frozenset[Capability]

    def markets(self) -> list[str]: ...

    def candles(self, market: str, interval: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Columns: t (UTC datetime, bar open), open, high, low, close, volume, trades.
        Only bars fully closed at `end` are returned (no look-ahead)."""
        ...

    def context(self, market: str) -> MarketContext: ...
    def funding_history(self, market: str, start: datetime, end: datetime) -> pl.DataFrame: ...
    def book(self, market: str) -> BookSnapshot: ...
