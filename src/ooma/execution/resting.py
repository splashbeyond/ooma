"""Queue-position model for resting limit orders, for live paper trading with a trade stream.

At placement we join the BACK of the visible queue at our price. Only trades that print at our price
reduce the queue ahead of us; a print THROUGH our price fills us in full. Cancels ahead of us are NOT
credited (conservative: real queues move faster, so real fills are somewhat better than this).
Post-only orders that would cross the book are rejected, as on the exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RestingOrder:
    side: int                    # +1 resting buy, -1 resting sell
    px: float
    qty: float
    queue_ahead: float           # size already resting at px when we joined
    post_only: bool = True
    ttl_s: float | None = None
    placed_at: float = 0.0
    filled: float = 0.0
    fills: list = field(default_factory=list)   # (t, px, qty)

    @property
    def done(self) -> bool:
        return self.filled >= self.qty - 1e-12

    def on_trade(self, t: float, trade_px: float, trade_sz: float) -> float:
        """Feed every public trade. Returns qty filled by this print."""
        if self.done:
            return 0.0
        if self.ttl_s is not None and t - self.placed_at > self.ttl_s:
            return 0.0
        left = self.qty - self.filled
        if (self.side > 0 and trade_px < self.px) or (self.side < 0 and trade_px > self.px):
            take = left                                   # printed through our price: we are filled
        elif trade_px == self.px:
            eat = min(self.queue_ahead, trade_sz)
            self.queue_ahead -= eat
            take = min(left, trade_sz - eat)
        else:
            return 0.0
        if take > 0:
            self.filled += take
            self.fills.append((t, self.px, take))
        return take


def place_limit(side: int, px: float, qty: float, best_bid: float, best_ask: float, size_at_px: float,
                post_only: bool = True, ttl_s=None, t: float = 0.0) -> RestingOrder | None:
    """None if a post-only order would cross (exchange rejects it)."""
    if post_only and ((side > 0 and px >= best_ask) or (side < 0 and px <= best_bid)):
        return None
    return RestingOrder(side, px, qty, queue_ahead=size_at_px, post_only=post_only, ttl_s=ttl_s, placed_at=t)
