"""Hyperliquid order precision and limits. Source: HL docs "Tick and lot size" (verified):
price <= 5 significant figures AND <= (6 - szDecimals) decimal places for perps; integer prices always allowed;
size is floored to szDecimals. Min order value ($10) is from memory, NOT verified: keep it configurable."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

MAX_DECIMALS_PERP = 6


@dataclass(frozen=True)
class MarketRules:
    sz_decimals: int
    max_leverage: int
    min_notional: float = 10.0

    def price_decimals(self, px: float) -> int:
        sig = 5 - 1 - math.floor(math.log10(abs(px))) if px > 0 else 0
        return max(0, min(sig, MAX_DECIMALS_PERP - self.sz_decimals))

    def tick(self, px: float) -> float:
        return 10.0 ** -self.price_decimals(px)

    def round_price(self, px: float, mode: str = "nearest") -> float:
        d = self.price_decimals(px)
        q = Decimal(1).scaleb(-d)
        rnd = {"nearest": ROUND_HALF_EVEN, "down": ROUND_FLOOR, "up": ROUND_CEILING}[mode]
        return float(Decimal(repr(px)).quantize(q, rounding=rnd))

    def round_size(self, sz: float) -> float:
        return float(Decimal(repr(sz)).quantize(Decimal(1).scaleb(-self.sz_decimals), rounding=ROUND_FLOOR))

    def valid_order(self, px: float, sz: float) -> bool:
        return sz > 0 and sz * px >= self.min_notional
