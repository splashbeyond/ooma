#!/usr/bin/env bash
# Copy CODE ONLY to the droplet (never data/, never the venv, never keys) and restart the service.
# usage: deploy/push.sh [host]   (default: the ooma droplet)
set -euo pipefail
HOST="${1:-104.236.76.249}"; KEY="$HOME/.ssh/ooma_do"
cd "$(dirname "$0")/.."
rsync -az --delete -e "ssh -i $KEY -o IdentitiesOnly=yes" \
  --exclude '.venv' --exclude 'data' --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.pytest_cache' --exclude '.env' --exclude '*.duckdb*' \
  ./ "ooma@$HOST:ooma/"
ssh -i "$KEY" -o IdentitiesOnly=yes "ooma@$HOST" 'cd ooma && .venv/bin/python -m pip install -q -e . 2>/dev/null || ~/.local/bin/uv pip install -q -e . 2>&1 | tail -1; systemctl --user restart ooma 2>/dev/null && echo restarted || echo "service not installed yet"'
