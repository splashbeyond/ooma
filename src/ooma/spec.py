"""Declarative strategy specs: what ooma's lab invents. Pure data, no code execution.

A spec can only reference whitelisted features and fixed operators, so an invented strategy
cannot run arbitrary code, touch the network, or change risk rules.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PLAIN = {"open", "high", "low", "close", "volume"}
PARAM = re.compile(r"^(ema|sma|atr|rsi|ret|z|hh|ll|volx)_(\d{1,3})$")
# ema/sma: moving averages | atr: avg true range | rsi: Wilder RSI | ret: N-bar return
# z: z-score of close vs N-bar mean | hh/ll: highest high / lowest low of the PRIOR N bars
# volx: volume / N-bar average volume


def valid_feature(name: str) -> bool:
    if name in PLAIN:
        return True
    m = PARAM.match(name)
    return bool(m) and 2 <= int(m.group(2)) <= 400


class Condition(BaseModel):
    lhs: str
    op: Literal[">", "<", "cross_above", "cross_below"]
    rhs: str | float

    @model_validator(mode="after")
    def _check(self):
        if not valid_feature(self.lhs):
            raise ValueError(f"unknown feature {self.lhs!r}")
        if isinstance(self.rhs, str) and not valid_feature(self.rhs):
            raise ValueError(f"unknown feature {self.rhs!r}")
        return self


class StrategySpec(BaseModel):
    name: str = ""
    family: str                                   # breakout | mean_reversion | trend | momentum | ...
    hypothesis: str                               # why this should make money, in one sentence
    timeframe: Literal["1h", "4h", "1d"]
    markets: list[str]
    long_when: list[Condition] = []               # ALL must hold
    short_when: list[Condition] = []
    stop_atr: float = Field(2.0, ge=0.5, le=6.0)  # a stop is mandatory
    target_r: float = Field(2.0, ge=0.5, le=6.0)
    horizon_bars: int = Field(24, ge=1, le=200)

    @field_validator("markets")
    @classmethod
    def _markets(cls, v):
        if not v:
            raise ValueError("need at least one market")
        return v

    @model_validator(mode="after")
    def _has_rule(self):
        if not (self.long_when or self.short_when):
            raise ValueError("spec needs long_when or short_when")
        if not self.name:
            self.name = f"{self.family}_{self.digest()[:6]}"
        return self

    def digest(self) -> str:
        d = self.model_dump(exclude={"name", "hypothesis"})
        return hashlib.sha1(repr(sorted(d.items())).encode()).hexdigest()


Stage = Literal["rejected_train", "rejected_holdout", "incubating", "active", "retired"]


class StrategyRecord(BaseModel):
    """Lifecycle of an invented strategy. incubating = forward paper trading to earn trust."""
    id: str                                        # spec digest
    spec: StrategySpec
    stage: Stage
    created_at: datetime
    origin: Literal["template", "llm", "mutation"] = "template"
    parent_id: str | None = None
    model: str | None = None                       # which LLM invented it (origin == "llm")
    evaluations: list[dict] = []
    reason: str = ""
