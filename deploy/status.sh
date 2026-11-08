#!/usr/bin/env bash
# One-command health + paper-trading scoreboard from your Mac. usage: deploy/status.sh [host]
HOST="${1:-104.236.76.249}"
ssh -i "$HOME/.ssh/ooma_do" -o IdentitiesOnly=yes "ooma@$HOST" 'cd ooma && .venv/bin/ooma status; echo; systemctl --user is-active ooma ooma-director | paste -sd" " | sed "s/^/services: /"; echo "server time: $(date -u +%H:%M:%S) UTC   ram free: $(free -m | awk "/Mem:/{print \$7}") MB   disk free: $(df -h / | awk "NR==2{print \$4}")"'
