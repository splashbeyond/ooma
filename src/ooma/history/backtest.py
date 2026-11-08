"""Historical analysis: replay strategies bar by bar with no look-ahead.

At each bar close the strategy sees only candles up to and including that bar. Setups are simulated
on the bars after. Results feed the same grader/stats as paper trades, tagged source='backtest'.
Holdout: the most recent `holdout_frac` of history is reserved; check it rarely (PRD s11).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import polars as pl

from ..adapters.hyperliquid import INTERVAL_MS
from ..grading.grader import grade
from ..journal.stats import Stats, summarize
from ..strategies.base import ScanContext, Strategy
from ..types import Findings, Grade, Setup, Trade
from .simulate import simulate

Loader = Callable[[str, str, datetime, datetime], pl.DataFrame]
FindingsFn = Callable[[str, str, datetime, pl.DataFrame], dict[str, Findings]]


@dataclass
class BacktestResult:
    setups: list[Setup] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    grades: list[Grade] = field(default_factory=list)

    def stats_by_strategy(self) -> dict[str, Stats]:
        out: dict[str, list[float]] = {}
        for g in self.grades:
            out.setdefault(f"{g.strategy}@v{g.strategy_version}", []).append(g.r_multiple)
        return {k: summarize(v) for k, v in out.items()}


class Backtester:
    def __init__(self, loader: Loader, findings_fn: FindingsFn | None = None,
                 warmup_bars: int = 250, risk_usd: float = 100.0):
        self.loader, self.findings_fn = loader, findings_fn
        self.warmup, self.risk_usd = warmup_bars, risk_usd

    def run(self, strategies: list[Strategy], markets: list[str], timeframe: str,
            start: datetime, end: datetime, params: dict[str, dict] | None = None) -> BacktestResult:
        step = timedelta(milliseconds=INTERVAL_MS[timeframe])
        res = BacktestResult()
        for market in markets:
            candles = self.loader(market, timeframe, start - step * self.warmup, end)
            busy_until: dict[str, datetime] = {}          # one open trade per strategy per market
            for i in range(self.warmup, candles.height - 1):
                as_of = candles["t"][i] + step             # close time of bar i
                hist = candles.head(i + 1)
                findings = self.findings_fn(market, timeframe, as_of, hist) if self.findings_fn else {}
                for strat in strategies:
                    if busy_until.get(strat.name, as_of) > as_of:
                        continue
                    ctx = ScanContext(market, timeframe, as_of, hist, findings,
                                      params=(params or {}).get(strat.name, {}))
                    for setup in strat.scan(ctx):
                        trade = simulate(setup, candles.slice(i + 1), risk_usd=self.risk_usd)
                        if trade is None:
                            continue
                        res.setups.append(setup)
                        res.trades.append(trade)
                        if trade.closed_at:
                            busy_until[strat.name] = trade.closed_at + step
                            res.grades.append(grade(setup, trade))
                        else:
                            busy_until[strat.name] = end
        return res
