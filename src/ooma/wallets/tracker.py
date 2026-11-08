"""Wallet tracking from the Hyperliquid API.

Pipeline: discover -> pull history -> score -> cohort positioning (feeds modules/participants.py).
VERIFY against live responses before relying on any of these (PRD s13):
  - discovery: `trades` websocket carries `users` [buyer, seller]; also leaderboard stats endpoint (unofficial)
  - positions: info `clearinghouseState` {user}
  - history:   info `userFills` / `userFillsByTime` {user, startTime}
"""
from __future__ import annotations

from datetime import datetime

import polars as pl


class WalletTracker:
    def __init__(self, store_path: str = "data/wallets.duckdb"):
        self.store_path = store_path

    def discover(self, market: str, since: datetime) -> list[str]:
        """Addresses seen trading `market` (large size first). Needs the trades recorder."""
        raise NotImplementedError

    def fills(self, address: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Historical fills for one wallet. Paginate; respect rate limits."""
        raise NotImplementedError

    def positions(self, address: str) -> pl.DataFrame:
        """Current open positions: coin, size, entry, leverage, liq price."""
        raise NotImplementedError

    def snapshot_cohort(self, market: str, as_of: datetime) -> pl.DataFrame:
        """Positions of all tracked wallets at as_of, joined with their score. Stored for replay."""
        raise NotImplementedError
