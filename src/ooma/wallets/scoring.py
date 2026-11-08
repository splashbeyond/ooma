"""Wallet skill scoring. Walk-forward only: a wallet's score at T uses fills before T."""
from __future__ import annotations

from datetime import datetime

import polars as pl
from pydantic import BaseModel


class WalletScore(BaseModel):
    address: str
    as_of: datetime
    n_trades: int
    win_rate: float
    net_pnl: float
    pnl_per_risk: float | None = None     # needs stop/size proxy; consistency beats one lucky trade
    max_drawdown: float
    skill: float                           # shrunk toward 0 for small n; rank on this
    decile: int                            # 1 = weak ... 10 = skilled


def score_wallet(address: str, fills: pl.DataFrame, as_of: datetime) -> WalletScore:
    raise NotImplementedError


def assign_deciles(scores: list[WalletScore]) -> list[WalletScore]:
    raise NotImplementedError
