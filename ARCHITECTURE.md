# ooma architecture (outline v0)

```
adapters (Hyperliquid)  ->  modules (price_action, order_flow, participants, kronos, research)
        |                              |
        |                              v
        |                     strategies (scanners: each emits Setups)  <- catalog.py
        |                              |
        v                              v
 history/backtest + simulate   scanner (live)  ->  analyst (LLM thesis) -> risk firewall -> paper broker
        \______________________________|______________________________________________/
                                       v
                       journal (setups, trades, grades, notes)
                                       v
              grading -> stats (n, CI) -> LearningCycle -> notes -> proposals -> YOU approve -> version bump
```

Status legend: REAL = implemented + tested, STUB = signature only.

| Area | File | State |
|---|---|---|
| Types / contracts | types.py | REAL |
| Hyperliquid adapter (candles, context, book, funding) | adapters/ | REAL (trades WS stub) |
| Strategy base, registry, 13-strategy catalog | strategies/ | REAL registry, all strategies STUB |
| Historical replay, no look-ahead | history/simulate.py, backtest.py | REAL |
| Candle cache (parquet) | history/ | not started |
| Grading (outcome + process) | grading/grader.py | REAL |
| Stats with n and bootstrap CI | journal/stats.py | REAL |
| Journal + notebook | journal/store.py, notes.py | REAL |
| Learning cycle (grade, measure, reflect) | learning/cycle.py | REAL; test + propose STUB |
| Price action | modules/price_action.py | STUB (milestone 2) |
| Wallet tracking + scoring | wallets/, modules/participants.py | STUB |
| Order flow, Kronos, Research | modules/ | STUB |
| Analyst, firewall, paper broker | analyst/, risk/, paper/ | STUB (milestone 6) |

Learning rules: grade the decision as well as the result; nothing acts on n < 30; hypotheses need
evidence before leaving `open`; proposals are validated on a holdout; a human approves every change.

## Strategy lab (ooma invents its own strategies)

Mission: make more than you lose, net of fees and slippage, found honestly.

```
Ideator (templates now, LLM next) -> StrategySpec (validated JSON, whitelisted features, mandatory stop)
  -> train backtest -> rank top-K -> untouched holdout (p-value, Bonferroni by looks)
  -> incubating (forward paper trading) -> active (live evidence CI > 0) -> retired (CI < 0)
```
- Specs are data, not code: an invented strategy cannot execute code, hit the network, or change risk rules.
- Autonomy: the lab creates, tests, runs and retires strategies on paper by itself. Risk firewall, config
  and real money stay manual. (This supersedes "human approves every change" for strategies only.)
- Overfitting defenses: holdout never used for selection beyond top-K; every trial is a stored record;
  n >= 30 and p < 0.05 (adjusted); only forward paper results can promote to active.
- Tested: 12 pure-noise datasets (~470 ideas) -> 0 false positives; planted trend edge -> found.
- First real run (HL, 540d, 1h+4h, 39 ideas): 0 survivors. 34 failed train, 3 passed train and failed holdout.

Gaps before the lab can use "any strategy":
- Features are candle-only. Needed: funding/OI (context), cross-market ratios (pairs / stat arb),
  wallet cohort signals (participants), order flow, time-of-day.
- Backtests ignore funding costs. Add before trusting perp carry or long holds.
- True arbitrage: Hyperliquid alone has little risk-free arb. Funding carry needs spot + perp (HL spot adapter);
  cross-venue arb needs another exchange adapter. Both are adapters, then new feature families.
- LLMIdeator is a stub (open-ended invention). Sandboxed code strategies are deliberately not built.
