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
