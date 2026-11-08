"""Paper broker: book-walked fills, taker fees, funding accrual, 2-5s delay, limits fill only on trade-through.
Live prices from the adapter. Milestone 6. Backtests use history.simulate instead."""
from __future__ import annotations

from ..types import Setup, Trade


class PaperBroker:
    def __init__(self, adapter, cfg: dict): self.adapter, self.cfg = adapter, cfg
    def open(self, setup: Setup, size: float) -> Trade: raise NotImplementedError
    def tick(self) -> list[Trade]:
        """Check stops/targets/horizon, accrue funding; return trades closed."""
        raise NotImplementedError

    def equity(self) -> float: raise NotImplementedError
