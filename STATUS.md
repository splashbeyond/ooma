# ooma STATUS

Personal AI trading analyst for Hyperliquid perps (paper only, v0). Spec: PRD_analyst.md.

## Now
- Scaffold done (2026-11-08). Next: A1 price action module (tests first, synthetic candles).

## Decisions
- Project name: ooma. Location: ~/Desktop/ooma. Package: analyst (src/analyst).
- Python 3.11 via uv (system python is 3.9).
- Build one milestone at a time, stop for review after each (PRD s17).

## Open
- PRD depends on Wallet Scorer (PRD.md). Not found on this machine. Needed for A3 (order flow, wallets); A1, A2, A4 do not need it.
- Verify HL endpoints and Kronos predictor API against live/repo before use; log diffs from PRD.
