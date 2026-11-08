"""Shared data contracts. Every module, strategy and store speaks these."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Bias = Literal["bullish", "bearish", "neutral"]
Direction = Literal["long", "short"]
Quality = Literal["ok", "stale", "partial"]
ExitReason = Literal["stop", "target", "horizon", "manual", "invalidated"]


class Signal(BaseModel):
    name: str
    value: bool | float | str
    detail: str = ""


class Level(BaseModel):
    price: float
    type: Literal["support", "resistance", "liquidation", "range_edge"]
    touches: int = 0


class Findings(BaseModel):
    """Output of every analysis module (PRD s4)."""
    module: str
    market: str
    timeframe: str
    as_of: datetime
    bias: Bias = "neutral"
    strength: float = Field(0.0, ge=0, le=1)
    signals: list[Signal] = []
    levels: list[Level] = []
    data_quality: Quality = "ok"
    notes: str = ""


class Setup(BaseModel):
    """A candidate trade emitted by a strategy. Gradable and replayable."""
    id: str
    strategy: str
    strategy_version: int
    market: str
    timeframe: str
    as_of: datetime
    direction: Direction
    entry: float
    stop: float
    targets: list[float]
    horizon_hours: int
    confidence: float = Field(0.5, ge=0, le=1)
    rationale: str = ""
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    features: dict[str, float | str | bool] = {}   # snapshot of inputs, for later learning


class Trade(BaseModel):
    id: str
    setup_id: str
    strategy: str
    strategy_version: int
    market: str
    direction: Direction
    opened_at: datetime
    entry_fill: float
    stop: float
    targets: list[float]
    size: float                      # base units
    fees: float = 0.0                # quote currency, positive = cost
    funding: float = 0.0             # quote currency, positive = received
    best_price: float                # most favorable price seen while open
    worst_price: float               # most adverse price seen while open
    closed_at: datetime | None = None
    exit_fill: float | None = None
    exit_reason: ExitReason | None = None
    source: Literal["paper", "backtest"] = "paper"


class Grade(BaseModel):
    """Grades the decision, not just the result (see grading/grader.py)."""
    trade_id: str
    strategy: str
    strategy_version: int
    r_multiple: float
    outcome_grade: str               # A-F from R
    process_score: float             # 0-1
    process_grade: str               # A-F
    mfe_r: float
    mae_r: float
    entry_slippage_r: float
    stop_slippage_r: float
    direction_correct: bool | None


class Note(BaseModel):
    """A line in ooma's own notebook. Hypotheses must earn their way to 'supported'."""
    id: str
    created_at: datetime
    author: Literal["code", "llm", "user"]
    kind: Literal["observation", "lesson", "hypothesis", "rule_proposal"]
    strategy: str | None = None
    market: str | None = None
    text: str
    status: Literal["open", "supported", "refuted", "promoted"] = "open"
    n_support: int = 0
    n_against: int = 0
    trade_ids: list[str] = []
