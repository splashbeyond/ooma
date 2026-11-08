# PRD: Personal Trading Analyst (v0)

## 1. Purpose

A personal AI trading analyst for Hyperliquid perps. It combines five kinds of evidence (fundamentals/research, price action, order flow, wallet behavior, Kronos forecasts) into written trade theses, paper-trades them through a risk firewall, and keeps a scorecard of which evidence and setups actually work.

v0 is for one user (me). No public posting, no real money. Later versions may power public products (X bot, app), so keep outputs structured and logged from day one.

Depends on: `PRD.md` (Wallet Scorer). The analyst reads the scorer's tables; build the scorer's M1–M3 first or in parallel.

## 2. Goals

1. On demand (`analyze ETH 4h`) and on a schedule, produce a clear, evidence-backed thesis or an explicit "no trade."
2. Paper-trade approved theses with realistic fills, fees and slippage.
3. Learn which modules, combinations and setups have edge, using a scorecard, without fooling itself.

## 3. Non-goals (v0)

- Live trading or holding private keys
- Public posting or multi-user features
- The LLM calculating numbers or levels (code does all math)
- Self-modifying risk rules

## 4. Constraints

- Budget: ~$30–60/month total (server ~$20, LLM API ~$10–30, occasional GPU ~$5).
- Markets: BTC, ETH, SOL, HYPE perps (configurable). Timeframes: 1h, 4h, 1d.
- Data: Hyperliquid public API (+ wallet scorer tables). Web research via an LLM with web search.
- Respect Hyperliquid rate limits via the scorer's shared rate limiter.

## 5. Architecture

```
          ┌──────────── data layer ────────────┐
HL API ──►│ candles, funding/OI, book, trades  │◄── wallet scorer tables
          └────────────────┬───────────────────┘
                           ▼
   ┌───────────┬───────────┬───────────┬───────────┬───────────┐
   │ Research  │ PriceAct  │ OrderFlow │ Wallets   │ Kronos    │  (modules → JSON findings)
   └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘
         └───────────┴─────┬─────┴───────────┴───────────┘
                           ▼
                   Analyst (LLM) → Thesis JSON
                           ▼
                   Risk firewall → Paper broker
                           ▼
                Journal → Scorecard → Weekly review
```

Each module is a pure function: `(coin, timeframe, as_of) -> Findings`. `as_of` is mandatory so every module can be replayed historically without look-ahead.

## 6. Modules

All modules return a common schema:

```json
{
  "module": "price_action",
  "coin": "ETH", "timeframe": "4h", "as_of": "2026-11-08T12:00:00Z",
  "bias": "bullish | bearish | neutral",
  "strength": 0.0,
  "signals": [{"name": "ma_reclaim_50", "value": true, "detail": "..."}],
  "levels": [{"price": 3050.0, "type": "support", "touches": 4}],
  "data_quality": "ok | stale | partial",
  "notes": "short factual summary"
}
```

### 6.1 Research (fundamentals)
- Inputs: funding rate and history, open interest and change, premium, upcoming token unlocks, major news and scheduled events (via LLM web search), protocol revenue where relevant.
- Output: bias + list of catalysts with dates and sources (URLs stored).
- Rule: every catalyst must carry a source URL and timestamp; unsourced items are dropped.
- Cache research per coin for 1–4 hours to control LLM cost.

### 6.2 Price action (pure code)
- Moving averages: 20/50/200 (EMA and SMA configurable); slope; price vs MA; reclaims/losses.
- Volatility: ATR(14), range compression (ATR percentile).
- Swing structure: pivot highs/lows (configurable lookback), trend = higher highs/higher lows vs lower highs/lower lows.
- Levels: cluster pivots within k×ATR into support/resistance zones; count touches; recency weight.
- Accumulation/distribution range: N+ bars inside a band ≤ x×ATR with ≥2 touches of each edge; flag volume behavior inside the range.
- Breakout/breakdown: close beyond a level by ≥ y×ATR with volume ≥ z× average; track retest success within M bars.
- All thresholds in `config.yaml`; every signal unit-tested on synthetic candles.

### 6.3 Order flow
- From recorder/scorer data: order-book imbalance near mid, aggressive buy vs sell notional (5m/15m/1h), large trades, funding/OI crowding.
- Liquidation map: notional within 1%/3%/5% above and below price, from position snapshots.
- Output: bias + key liquidation clusters as levels.

### 6.4 Wallets
- From scorer tables: net positioning and recent changes of top-decile (skilled) and bottom-decile (weak) cohorts per coin; count of skilled wallets adding/reducing in last 1h/4h/24h; their average leverage.
- Uses only scores computed before `as_of`.

### 6.5 Kronos
- Model: Kronos-small from Hugging Face (MIT license). CPU inference.
- Input: last up to 512 candles of the timeframe.
- Sample multiple forecast paths (e.g., 20–50) for the next H bars → outputs: P(up over H), median path, 10/90% quantiles, forecast realized volatility.
- Verify predictor API arguments against the Kronos repo before implementing.
- Weekly optional fine-tune on a rented GPU (separate script); new weights only used if they beat current on the last 2 weeks out-of-sample (promotion gate).

## 7. Analyst (LLM)

- Input: the five Findings JSONs + current paper positions + recent journal lessons (last 10).
- Output: Thesis JSON (validated by schema; invalid output is rejected and retried once):

```json
{
  "coin": "ETH", "timeframe": "4h", "as_of": "...",
  "decision": "long | short | no_trade",
  "setup_tags": ["range_breakout", "ma_reclaim", "skilled_wallets_agree"],
  "thesis": "plain-English, 3–6 sentences, each claim referencing a module",
  "evidence_for": ["module:signal", "..."],
  "evidence_against": ["module:signal", "..."],
  "entry": {"type": "market | limit", "price": 0.0},
  "stop": 0.0, "targets": [0.0], "invalidation": "text",
  "horizon_hours": 24,
  "confidence": 0.0,
  "module_agreement": {"research": "bull", "price_action": "bull", "order_flow": "neutral", "wallets": "bull", "kronos": "bear"}
}
```

- Rules in the system prompt: no numbers that are not in the inputs; must list evidence against; "no_trade" is a valid and common answer; stop and invalidation required for any trade.
- Default model: a capable API model (e.g., Claude Sonnet) for quality; optional local model via llama.cpp for cheap runs. Log model name and prompt version with every thesis.

## 8. Risk firewall (hard rules, code only, not editable by the LLM)

- Paper equity starts at $10,000 (configurable).
- Max risk per trade: 1% of equity (size derived from entry–stop distance).
- Max leverage: 3x. Max 3 concurrent positions. Max 1 position per coin.
- Daily loss cap: 3% → no new trades until next UTC day.
- Drawdown kill switch: 10% from peak → pause all new trades, notify me.
- Reject if any module `data_quality != ok`, if stop is missing, or if confidence < threshold.
- Every rejection is logged with the reason.

## 9. Paper broker

- Uses live mainnet mid and book from the recorder.
- Market orders fill at book-walked price for the order size + configurable extra slippage; taker fee applied.
- Limit orders fill only if price trades through the limit (not just touches).
- Stops and targets checked on every new trade/mid tick; funding accrued hourly.
- Assume 2–5 seconds of execution delay.

## 10. Journal and scorecard

- Journal (DuckDB/Parquet): every thesis (including no_trade), every firewall decision, every paper fill, exit reason, PnL net of fees and funding, and a post-mortem written by the LLM after exit (stored separately from facts).
- Scorecard (nightly):
  - Hit rate, average R-multiple, expectancy, max drawdown, by setup tag, by coin, by timeframe.
  - Per-module: accuracy of each module's bias over the thesis horizon, whether or not traded.
  - Agreement analysis: performance when N of 5 modules agree.
  - Confidence calibration: realized win rate per confidence bucket.
  - Baselines: buy-and-hold, MA-crossover-only, Kronos-only, random entries with same stops.
- Weekly review (auto-generated markdown): what worked, what didn't, sample sizes, and suggested changes. Suggestions are proposals only; I approve changes manually.

## 11. Overfitting guardrails

- Every configuration change bumps a version; results are reported per version.
- Hold out the most recent 25% of history for backtests; check it at most weekly.
- Report sample size (n) and confidence intervals next to every metric; flag n < 30 as "insufficient."
- No metric definitions change after launch without a version bump and a note.

## 12. Backtest mode

- Same modules and analyst, driven by `as_of` over historical data (candles from API/S3; order flow and wallets only where recorded).
- LLM analyst is optional in backtests (cost and look-ahead concerns); a rules-only "proxy analyst" (weighted module vote) can be used to backtest combinations cheaply. Clearly label which mode produced each result.

## 13. Interfaces

- CLI: `analyst analyze ETH 4h`, `analyst positions`, `analyst scorecard`, `analyst review`.
- Scheduled: every 1h (1h timeframe) and every 4h (4h timeframe) per coin; daily brief at a fixed time.
- Daily brief (markdown): market summary per coin, open paper positions, new theses, scorecard snapshot.
- Optional: private Telegram bot to send theses and kill-switch alerts to me only.

## 14. Tech stack and layout

- Python 3.11+, polars/pandas, duckdb, pydantic, httpx, websockets, torch (CPU) for Kronos, lightgbm (later), pytest.
- LLM client with web search capability for Research; pluggable provider.

```
analyst/
  config.yaml
  src/analyst/
    data/            # loaders over recorder + scorer tables + HL API
    modules/{research.py, price_action.py, order_flow.py, wallets.py, kronos_mod.py}
    brain/{prompt.md, analyst.py, schema.py}
    risk/firewall.py
    paper/broker.py
    journal/{store.py, scorecard.py, review.py}
    cli.py, scheduler.py
  tests/
```

## 15. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| A1 | Price action module | All signals unit-tested on synthetic candles; `analyze` prints levels/MAs/ranges for 4 coins |
| A2 | Kronos module | CPU forecasts with quantiles for 1h/4h in < 60s per coin; outputs logged |
| A3 | Order flow + wallets modules | Read from recorder/scorer tables; `as_of` respected (test proves no look-ahead) |
| A4 | Research module | Catalysts with sources; cached; cost per run logged |
| A5 | Analyst + firewall + paper broker | Valid thesis JSON every run; firewall rejections logged; paper fills include fees/slippage/funding |
| A6 | Journal + scorecard + daily brief | Nightly scorecard with baselines and n; daily brief generated |
| A7 | Weekly review + backtest proxy mode | Review produced; proxy backtest runs over available history |

Target: A1–A3 week 1–2; A4–A6 week 3–4; A7 week 5–6. Then 2–3 months of paper trading before any decision about real money.

## 16. Decision points

- After ~8 weeks / ≥50 theses: does any setup or module-agreement level beat the baselines net of costs with a reasonable sample? If none, simplify (drop weakest modules) before adding anything.
- Real money only if: positive expectancy net of costs over ≥3 months, max drawdown within limits, and results hold on the held-out period. Start ≤ $500.

## 17. Instructions for Claude Code

- Build milestone by milestone; stop after each for review.
- Tests first for price-action math, firewall rules, paper fills, and `as_of` look-ahead protection.
- Verify every Hyperliquid endpoint/field and the Kronos predictor API against live responses or the repo before relying on them; log differences from this PRD.
- Keep the LLM away from arithmetic; all numbers come from modules.
- No live order placement code in v0.
