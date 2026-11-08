"""Hard risk rules in code. The LLM and the learning loop cannot change these (config edits are manual).

Checks: 1% risk/trade, 3x leverage, 3 positions, 1 per market, 3% daily loss cap, 10% drawdown kill,
data_quality == ok, stop present, confidence >= threshold. Every rejection logged with reason. Milestone 6.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import Setup


@dataclass
class Decision:
    approved: bool
    size: float = 0.0
    reasons: list[str] | None = None


class Firewall:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def check(self, setup: Setup, account) -> Decision:
        raise NotImplementedError
