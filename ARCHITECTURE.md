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

## Recursive prediction loop (predict/)

Binary framing: every closed 1h bar, for each market and horizon H in {1,4,24}h, the question is
"will the close H hours from now be above the close now?". Measurables (23 causal features: returns, RSI,
z-scores, volume, ATR, MA distance, range position, candle shape, hour/day, funding, funding z, BTC lead) feed
a population of predictors (rule stumps, conjunctions, an online logistic model, a base-rate baseline).

```
features -> predictions (frozen, stored) -> wait H hours -> outcome
   -> score vs baseline (log loss, hit rate, net edge after costs; n_eff = n/H)
   -> every predictor learns online; ensemble weights shift (Hedge)
   -> trusted if Bonferroni-adjusted p <= 0.05 and n_eff >= 30  -> notebook (supported lesson)
   -> retired if clearly worse than baseline
   -> new hypotheses spawned from what works (jitter / conjunction / fresh), judged only on outcomes after birth
```
- Out-of-sample by construction: a prediction exists before its outcome. Tests prove features are causal,
  that restart == one straight run, that noise earns no trust, and that a planted edge does.
- Run: `ooma predict-replay` (warm start from history), `ooma run` (hourly forever), `ooma predict-status`.
- Server: deploy/setup.sh + deploy/ooma.service (systemd user service). State is in data/ooma.duckdb.
- First real-data replay (120d, 4 markets): statistically detectable but economically tiny skill at 1h
  (+0.0003 nats/pred, z=2.3); trusted predictors had NEGATIVE net edge after costs. Read: no tradable edge yet.
- Not yet: live OI/book features (OI is logged to context_log from now, no history API), turning confident
  predictions into graded paper trades, feeding trusted notes to the LLM analyst and the strategy lab.

## Execution model (execution/)

Realistic, conservative costs for candle backtests; reusable for live paper trading.
- Fees: HL base tier, taker 4.5 bps / maker 1.5 bps (staking discount configurable). Verified in secondary sources.
- Entry/horizon exits: taker, reference price + half-spread (calibrated to the live book via `ooma exec-calibrate`)
  + square-root market impact scaled by bar volatility and bar volume.
- Stops: taker stop-market with extra slippage; gaps fill at the open; liquidation per HL formula (isolated margin),
  trades whose stop sits beyond liquidation are rejected.
- Targets: resting maker limits. Fill only if the bar trades THROUGH the target by a tick; a touch fills with
  probability 0.35 (deterministic per trade); needs the bar's volume to absorb the order (10% participation cap).
  A target that merely touches does not fill: this is the main source of "backtest says win, real life says no".
- Funding: hourly rate x notional, longs pay positive funding, charged every bar held.
- Exchange rules: 5 significant figure prices, (6 - szDecimals) decimals, sizes floored to szDecimals, min order value.
- Live paper trading: execution/book.py walks the real l2Book; execution/resting.py is a queue-position model
  (join the back of the queue, only prints at/through our price fill us, cancels ahead not credited).
Unverified (configurable): $10 minimum order, liquidation penalty, maker rebate tiers, large-position margin tiers.
Not modelled yet: partial target fills (treated as not filled), adverse selection of maker ENTRY orders
(entries are taker), latency inside a bar (entry uses the next open), spread widening in crashes beyond stop_extra_bps.
