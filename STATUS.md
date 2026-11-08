# ooma STATUS

Market-agnostic AI analyst, v0 on Hyperliquid perps, paper only. Spec: PRD.md (OOMA_PRD).

## Now
- Milestone 1 done (2026-11-08): adapter interface + Hyperliquid adapter, 5 tests pass incl. live check on BTC/ETH/SOL/HYPE.
- Next: milestone 2, price action module.

## Decisions
- Location ~/Desktop/ooma, package `ooma` (src/ooma), Python 3.11 via uv.
- Wallet Scorer PRD dropped for now. Hyperliquid API only; participants module deferred (needs trades WS recorder).
- Candles adapter returns only fully closed bars at `end` (look-ahead guard).

## Open / to discuss
- Scope vs the vision (scan many strategies, find edge over time): PRD is one thesis per market. Proposed addition: a strategy/scanner layer where each strategy emits candidate setups tagged by name, so the scorecard measures every strategy separately.
- Verified HL fields: candleSnapshot (max 5000/call), metaAndAssetCtxs, l2Book (20 levels/side), fundingHistory (hourly).
