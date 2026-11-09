#!/usr/bin/env bash
# Open the ooma dashboard from your Mac via an SSH tunnel (the server never exposes it publicly). usage: deploy/dashboard.sh [host]
HOST="${1:-104.236.76.249}"
echo "dashboard: http://localhost:8765   (Ctrl-C to close the tunnel)"
(sleep 2; open http://localhost:8765) &
ssh -i "$HOME/.ssh/ooma_do" -o IdentitiesOnly=yes -N -L 8765:127.0.0.1:8765 "ooma@$HOST"
