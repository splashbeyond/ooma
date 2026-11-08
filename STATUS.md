# ooma STATUS

Market-agnostic AI analyst, v0 on Hyperliquid perps, paper only. Spec: PRD.md (OOMA_PRD).

## Now
- HEARTBEAT + WATCHDOG live (2026-11-08): both services write data/heartbeat*.json every 60s (and immediately on activity change); systemd Type=notify + WatchdogSec=600 restarts a hung service (pings withheld if no progress for 25 min). `deploy/status.sh` (from Mac) or `ooma status` (on server) shows alive/stale, current activity, sessions, scripts by stage, LLM spend, and the PAPER TRADING scoreboard (trades, hit vs chance, net/trade, paper PnL).
- Paper-trading funnel: soft holdout gate admits plausible candidates (n>=20, hit>chance, net>0 after costs) into up to 12 free paper-trading slots; promotion to `active` still needs strict forward evidence (n>=30 independent, 1-day-block bootstrap p adj. by slots <= 0.05, net>0); retire after 10 days or 90 trades without promotion (frees the slot). Pooled forward p uses block bootstrap.
- CORRECTION (2026-11-08): "hype api" = the whole Hyperliquid API. Director now researches the WHOLE liquid universe (top 17 perps by 24h volume >= $5M, refreshed daily), not just HYPE. Scripts get cross-sectional columns (mkt_ret_6/24, xs_rank_ret_24/funding/vol), trades/activity, BTC-relative, zones, funding. Cost by liquidity tier (13/20/30 bps). Pooled results use a WEEKLY BLOCK BOOTSTRAP (coins move together, so per-trade binomial would overstate significance). Holdout looks counted across all sessions (7 days).
- Loop now logs ALL ~179 coins' funding/OI/premium/volume hourly in one call (context_log) and order-book depth/imbalance for the top 20 (book_log). Not yet exposed to scripts (director can't read the loop DB); next step is exporting those tables.
- Director now runs CONTINUOUSLY on HYPE only (config director: focus [HYPE], 2 ideas/session, 1 refinement, 120s pause): ~3-4 min per cycle. ~$0.007 per session; $2/day LLM cap (sessions pause 15 min at a time once hit, forward testing continues). Holdout correction counts ALL holdout looks in the last 7 days (recent_looks), duplicates skipped. First 2 HYPE sessions: 8 scripts, 1 passed train (donchian breakout+volume, n=71 hit .549 vs .52), failed holdout (n=12).
- DIRECTOR LIVE on the droplet (service `ooma-director`, env in ~/.ooma.env incl. OpenRouter key, mode 600). GLM-5.3-flash proposes hypotheses, writes Python scripts (contract: decide(bars)->(direction, act), both booleans), refines from TRAIN metrics only, notes lessons. Code judges: sandbox -> binary judge (non-overlapping trades, exact binomial vs best constant guess, net of costs) -> holdout (LLM never sees it) -> forward paper (only path to `active`). Session every 6h; hourly forward tick; $1/day LLM cap; ~$0.025 per session.
- Sandbox: scripts run as OS user ooma-sbx via sudo (only /opt/ooma-sandbox/sandbox_main.py), no network (iptables owner rule), /home/ooma unreadable, rlimits in header + self-kill alarm, numpy file I/O disabled, AST allowlist. Setup: deploy/sandbox-setup.sh (as root). Verified with hostile scripts.
- First server session (2026-11-08): 4 hypotheses, 0 passed train (funding reversion, zone rejection, US-hours drift, BTC-lead alt catch-up). All hit rates ~chance. Notes written. Cost wall (~13 bps) dominates.
- Director DB is data/director.duckdb (separate from ooma.duckdb because DuckDB is single-writer). It reads data/loop_state.json written by the loop each tick.
- Only run things on the server; Mac is for dev/tests. Deploy: deploy/push.sh.
- LIVE on droplet 104.236.76.249 (DO, 1 vCPU / 2 GB / 48 GB, Ubuntu 24.04), started 2026-11-08 ~18:00 UTC. Service `ooma` (systemd --user, linger, Restart=always) runs `ooma run`: hourly predictions, scoring, learning, wallet snapshots, OI logging. Paper only, no keys on server.
- Access: `ssh -i ~/.ssh/ooma_do ooma@104.236.76.249` (non-root user; root login only for setup). Deploy code: `deploy/push.sh` (never copies data/). State: ~/ooma/data/ooma.duckdb on the droplet. Logs: `journalctl --user -u ooma`. Digests: ~/ooma/data/reports/.
- Wallets on server: 150 profiled, 8 skilled+followable, imported from local run. Persistence check was NEGATIVE (spearman -0.17): past skill did not predict next-period profit. Treat wallet signal as unproven until forward data says otherwise.
- First real-data replay with the recency ensemble: no trusted predictors, ensemble skill ~0, confident calls lose ~0.15% each net of costs. No tradable edge yet. Watch for winner's-curse overconfidence in the ensemble (consider a calibration layer).
- TODO: nightly DB backup off the droplet; LLM key is NOT on the server (lab/LLM runs happen locally); trim old ledger rows eventually.
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
