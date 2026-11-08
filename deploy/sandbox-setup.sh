#!/usr/bin/env bash
# Run as ROOT on the droplet. Creates the locked-down user that LLM-written scripts execute as.
#  - user ooma-sbx: no shell, no home access to /home/ooma (750), cannot read keys or the database
#  - network egress blocked for that uid (iptables owner match, v4 + v6), re-applied at boot
#  - runner + helper copied to /opt/ooma-sandbox (root-owned, read-only to the sandbox)
#  - ooma may run ONLY the runner as ooma-sbx via sudo (no password, no other commands)
set -euo pipefail
id ooma-sbx >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin ooma-sbx
apt-get install -y -qq python3-numpy iptables >/dev/null
install -d -m 755 -o root -g root /opt/ooma-sandbox
install -m 644 -o root -g root /home/ooma/ooma/src/ooma/scripts/sandbox_main.py /opt/ooma-sandbox/sandbox_main.py
install -m 644 -o root -g root /home/ooma/ooma/src/ooma/scripts/ooma_lib.py /opt/ooma-sandbox/ooma_lib.py
chmod 750 /home/ooma
cat > /etc/sudoers.d/ooma-sandbox <<'SUDO'
ooma ALL=(ooma-sbx) NOPASSWD: /usr/bin/python3 -I /opt/ooma-sandbox/sandbox_main.py
SUDO
chmod 440 /etc/sudoers.d/ooma-sandbox && visudo -cf /etc/sudoers.d/ooma-sandbox >/dev/null
cat > /usr/local/sbin/ooma-sbx-firewall <<'FW'
#!/bin/sh
for ipt in iptables ip6tables; do
  $ipt -C OUTPUT -m owner --uid-owner ooma-sbx -j REJECT 2>/dev/null || $ipt -I OUTPUT 1 -m owner --uid-owner ooma-sbx -j REJECT
done
FW
chmod 755 /usr/local/sbin/ooma-sbx-firewall
cat > /etc/systemd/system/ooma-sbx-firewall.service <<'UNIT'
[Unit]
Description=Block network egress for the ooma sandbox user
After=network-pre.target ufw.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/ooma-sbx-firewall
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable --now ooma-sbx-firewall >/dev/null
cat > /etc/cron.hourly/ooma-sbx-clean <<'CRON'
#!/bin/sh
find /tmp -maxdepth 2 -user ooma-sbx -delete 2>/dev/null
pkill -u ooma-sbx -f sandbox_main --older 600 2>/dev/null || true
CRON
chmod 755 /etc/cron.hourly/ooma-sbx-clean
echo "sandbox ready"
