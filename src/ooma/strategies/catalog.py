"""Strategy catalog: the map of ways to make money that ooma will test.

Add a class, set status='active' once it has tests, and the scanner/backtester/scorecard pick it up.
"""
from .base import ScanContext, Strategy, register


def _scan_stub(self, ctx: ScanContext):
    raise NotImplementedError(self.name)


def _stub(name, description, requires_modules=()):
    return register(type(name, (Strategy,), {
        "name": name, "description": description, "status": "stub",
        "requires_modules": frozenset(requires_modules), "scan": _scan_stub}))


# price action
_stub("range_breakout", "Close beyond an accumulation range edge with volume; enter on break or retest.", ["price_action"])
_stub("trend_pullback", "Trend intact (MA stack + HH/HL), buy/sell pullback into MA or support/resistance.", ["price_action"])
_stub("level_reversal", "Rejection/sweep at a tested support or resistance zone.", ["price_action"])
# momentum
_stub("momentum_continuation", "Strong range expansion with volume; ride continuation with ATR stop.", ["price_action"])
_stub("mean_reversion", "Overextended from MA in low-trend regime; fade back toward mean.", ["price_action"])
# positioning / flow
_stub("funding_squeeze", "Extreme funding + rising OI = crowded side; trade the unwind.", ["research"])
_stub("liquidation_magnet", "Price drawn toward dense liquidation clusters.", ["order_flow"])
_stub("order_flow_imbalance", "Persistent aggressive buying/selling with book imbalance.", ["order_flow"])
# participants
_stub("smart_wallet_follow", "Follow when historically skilled wallets add in one direction.", ["participants"])
_stub("weak_wallet_fade", "Fade when historically weak wallets pile in one direction.", ["participants"])
# model / news
_stub("kronos_forecast", "Trade when Kronos P(up) and quantiles show asymmetric payoff.", ["kronos"])
_stub("news_catalyst", "Sourced catalyst with a dated event; position around it.", ["research"])
_stub("confluence", "Multiple independent modules agree; meta-strategy graded by agreement level.",
      ["price_action", "order_flow", "participants", "kronos", "research"])
