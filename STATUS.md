# ooma STATUS

Market-agnostic AI analyst, v0 on Hyperliquid perps, paper only. Spec: PRD.md (OOMA_PRD).

## Now
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
