# ooma STATUS

Market-agnostic AI analyst, v0 on Hyperliquid perps, paper only. Spec: PRD.md (OOMA_PRD).

## Now
- Recursive prediction loop built (2026-11-08), 27 tests pass, replayed on live HL data, live tick + restart verified. Not deployed to any server yet (waiting on Davis). Deploy: deploy/setup.sh.
- Local data/ooma.duckdb holds a replay state (gitignored); a server should build its own.
- Strategy lab built (2026-11-08): ooma invents/tests/incubates strategies. `ooma lab`, `ooma lab-status`. First real run: 0 survivors (honest). 21 tests.
- Next: milestone 2 price action is still optional; higher value now is richer features (funding/OI, cross-market, wallets) + LLMIdeator, + paper loop to run incubating strategies forward.
- Outline in code done (2026-11-08): see ARCHITECTURE.md. 15 tests pass. All strategies are stubs; replay, grading, notes, learning cycle are real.
- Next: milestone 2 price action, then first real strategy (range_breakout or trend_pullback) so backtest + learn run on real data.
- Milestone 1 done (2026-11-08): adapter interface + Hyperliquid adapter, 5 tests pass incl. live check on BTC/ETH/SOL/HYPE.
- Next: milestone 2, price action module.

## Decisions
- Location ~/Desktop/ooma, package `ooma` (src/ooma), Python 3.11 via uv.
- Wallet Scorer PRD dropped for now. Hyperliquid API only; participants module deferred (needs trades WS recorder).
- Candles adapter returns only fully closed bars at `end` (look-ahead guard).

## Open / to discuss
- Scope vs the vision (scan many strategies, find edge over time): PRD is one thesis per market. Proposed addition: a strategy/scanner layer where each strategy emits candidate setups tagged by name, so the scorecard measures every strategy separately.
- Verified HL fields: candleSnapshot (max 5000/call), metaAndAssetCtxs, l2Book (20 levels/side), fundingHistory (hourly).
