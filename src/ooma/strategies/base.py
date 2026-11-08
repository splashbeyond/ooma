"""A Strategy is a scanner: look at one market/timeframe at `as_of`, emit zero or more Setups.

Everything tradeable becomes a Strategy, so the scorecard can grade each one separately.
Stubs (status='stub') are skipped by the scanner until implemented.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from ..adapters.base import Capability, MarketContext
from ..types import Findings, Setup


@dataclass
class ScanContext:
    market: str
    timeframe: str
    as_of: datetime
    candles: pl.DataFrame                      # closed bars only, up to as_of
    findings: dict[str, Findings] = field(default_factory=dict)   # module name -> findings
    market_ctx: MarketContext | None = None
    params: dict = field(default_factory=dict)  # this strategy's config (versioned)


class Strategy(ABC):
    name: str = ""
    version: int = 1
    status: str = "stub"                       # 'stub' | 'active' | 'disabled'
    requires: frozenset[Capability] = frozenset({Capability.CANDLES})
    requires_modules: frozenset[str] = frozenset()
    description: str = ""

    @abstractmethod
    def scan(self, ctx: ScanContext) -> list[Setup]: ...


_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    assert cls.name, "strategy needs a name"
    _REGISTRY[cls.name] = cls
    return cls


def all_strategies(include_stubs: bool = False) -> dict[str, type[Strategy]]:
    from . import catalog  # noqa: F401  (import registers the built-in strategies)
    return {k: v for k, v in _REGISTRY.items() if include_stubs or v.status == "active"}
