"""Hyperliquid perp fees. Base tier (verified in secondary sources citing the docs): taker 0.045%, maker 0.015%.
Higher tiers use 14-day volume; staking discounts up to 40% on taker. Rebate tiers unverified: not modelled."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeSchedule:
    taker_bps: float = 4.5
    maker_bps: float = 1.5
    staking_discount: float = 0.0       # 0..0.4

    def bps(self, role: str) -> float:
        base = self.taker_bps if role == "taker" else self.maker_bps
        return base * (1 - self.staking_discount)

    def fee(self, notional: float, role: str) -> float:
        return abs(notional) * self.bps(role) / 1e4
