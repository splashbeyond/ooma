# ooma

Read STATUS.md and PRD_analyst.md first. Paper trading only: no live order code, no private keys.
Rules: tests first (price-action math, firewall, paper fills, as_of look-ahead); LLM never does arithmetic; all thresholds in config.yaml; modules are pure `(coin, timeframe, as_of) -> Findings`; one milestone at a time, stop after each.
Run: `uv run pytest`, `uv run analyst ...`
