# OOMA — Open Order-flow Market Analyst
## Basic PRD (v0)

## 1. What it is

OOMA is a personal AI market analyst. For any market it covers, it combines research, price action, order flow, participant behavior and a forecasting model into a clear trade thesis (or "no trade"), paper-trades it with strict risk rules, and keeps an honest scorecard of what works.

- **v0 market:** Hyperliquid perps (BTC, ETH, SOL, HYPE).
- **Design goal:** market-agnostic. Stocks, futures, FX and prediction markets plug in later through data adapters.
- **v0 user:** me. No public features, no real money.

## 2. Goals

1. Produce evidence-backed theses on demand and on a schedule.
2. Paper-trade them realistically (fees, slippage, delay).
3. Measure which evidence and setups have edge, with sample sizes, against simple baselines.

## 3. Non-goals (v0)

- Live trading or holding keys
- Public posting, multi-user, or managing anyone else's money
- The LLM doing any math (code computes every number)

## 4. Core design: adapters + modules

**Market adapters** turn any market's data into one standard format:

| Data | Standard output | Hyperliquid source (v0) | Later examples |
|---|---|---|---|
| Candles | OHLCV bars | `candleSnapshot` / trades | Equities/futures data vendors |
| Context | funding, OI, basis, events | `metaAndAssetCtxs`, funding history | Options OI, earnings calendar, macro calendar |
| Order flow | book snapshots, trades with side | `l2Book`, `trades` WS | Level-2 feeds where licensed |
| Participants | identified wallets/accounts | `trades` users + wallet scorer | Usually unavailable; module disabled |

Each adapter declares which capabilities it supports. Modules that need missing data are disabled for that market rather than faked.

**Analysis modules** (all return the same JSON schema: bias, strength, signals, levels, data quality, as_of):

1. **Research:** news, scheduled events, fundamentals (funding/OI for crypto; earnings/macro later). Every item must carry a source URL and timestamp.
2. **Price action (code only):** moving averages (20/50/200), ATR, swing structure, support/resistance zones, accumulation ranges, breakouts/breakdowns and retests.
3. **Order flow:** book imbalance, aggressive buy vs sell volume, large trades, crowding; liquidation clusters where available.
4. **Participants:** what historically skilled vs. weak accounts are doing (from the wallet scorer). Hyperliquid-only in v0.
5. **Kronos:** Kronos-small (Hugging Face, MIT) forecasts on recent candles; multiple sampled paths → probability of up move, quantiles, expected volatility. CPU inference.

Every module takes `as_of` so it can be replayed historically without look-ahead.

## 5. Analyst (LLM)

- Input: module outputs + open paper positions + recent journal lessons.
- Output (schema-validated JSON): decision (long/short/no_trade), setup tags, short thesis, evidence for and against, entry, stop, targets, invalidation, horizon, confidence, and which modules agree.
- Rules: no numbers that aren't in the inputs; evidence against is required; "no_trade" is normal; stop required for any trade.

## 6. Risk firewall (code only, not changeable by the LLM)

- Paper equity $10,000; max 1% risk per trade; max 3x leverage; max 3 open positions; 1 per market.
- Daily loss cap 3%; pause at 10% drawdown from peak.
- Reject trades with stale/partial data, missing stops, or low confidence. Log every rejection.

## 7. Paper broker

- Live prices and book; market orders walk the book plus extra slippage; taker fees; funding for perps.
- Limits fill only when price trades through. Assume 2–5s execution delay.

## 8. Journal and scorecard

- Log every thesis (including no_trade), decision, fill, exit and net PnL.
- Nightly scorecard: hit rate, average R, expectancy, drawdown, by setup, market, timeframe and module-agreement level; confidence calibration.
- Baselines: buy-and-hold, MA-crossover only, Kronos only, random entries with same stops.
- Show sample size on every metric; flag n < 30.
- Weekly auto-review with suggested changes; I approve changes manually. Every config change bumps a version.

## 9. Interfaces

- CLI: `ooma analyze ETH 4h`, `ooma positions`, `ooma scorecard`, `ooma review`.
- Schedule: hourly (1h) and every 4h (4h) per market; daily brief in markdown.
- Optional: private Telegram alerts for theses and kill-switch events.

## 10. Stack

Python 3.11+, polars, duckdb, pydantic, httpx, websockets, torch (CPU) for Kronos, pytest. Pluggable LLM provider with web search for Research. Runs on one small server (~$20/month); LLM ~$10–30/month.

```
ooma/
  config.yaml
  src/ooma/
    adapters/{base.py, hyperliquid.py}
    modules/{research.py, price_action.py, order_flow.py, participants.py, kronos_mod.py}
    analyst/{prompt.md, analyst.py, schema.py}
    risk/firewall.py
    paper/broker.py
    journal/{store.py, scorecard.py, review.py}
    cli.py, scheduler.py
  tests/
```

## 11. Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Adapter interface + Hyperliquid adapter | Candles, context, book, trades load for 4 markets; capabilities declared |
| 2 | Price action module | All signals unit-tested on synthetic data |
| 3 | Kronos module | Forecast + quantiles per market in < 60s on CPU |
| 4 | Order flow + participants modules | Use recorder/scorer data; test proves no look-ahead |
| 5 | Research module | Sourced catalysts; cached; cost logged |
| 6 | Analyst + firewall + paper broker | Valid thesis every run; rejections logged; realistic fills |
| 7 | Journal, scorecard, daily brief, weekly review | Nightly metrics with baselines and n |

Target: milestones 1–4 in weeks 1–2, 5–7 in weeks 3–5, then 2–3 months of paper trading.

## 12. Decision points

- After ~8 weeks / 50+ theses: if no setup beats baselines net of costs, cut the weakest modules before adding anything.
- Real money only after 3+ months of positive expectancy net of costs, acceptable drawdown, and results holding on held-out data. Start ≤ $500.
- Second market (e.g., US equities via a data vendor) only after Hyperliquid results are clear.

## 13. Notes for Claude Code

- Build one milestone at a time and stop for review.
- Tests first for price-action math, firewall, paper fills and look-ahead protection.
- Verify Hyperliquid endpoints/fields and the Kronos predictor API against live responses or the repo before relying on them.
- Keep adapters thin and modules market-agnostic; no market-specific logic inside modules.
- No live order code.
