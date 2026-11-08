#!/usr/bin/env bash
# Run ON the server as the service user, from the repo root (~/ooma). Paper only: no keys, no orders.
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 -q
uv pip install -q -e .
[ -f data/ooma.duckdb ] || uv run ooma predict-replay --days 120      # warm start (~30s)
mkdir -p ~/.config/systemd/user
cp deploy/ooma.service ~/.config/systemd/user/ooma.service
systemctl --user daemon-reload
systemctl --user enable --now ooma
loginctl enable-linger "$USER"                                          # keep running after logout
echo "started. logs: journalctl --user -u ooma -f   status: uv run ooma predict-status"
