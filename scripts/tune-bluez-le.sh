#!/usr/bin/env bash
# Raise the Linux BLE supervision timeout used by BlueZ.
#
# The kernel default is only 0x002a = 42 x 10 ms = 420 ms.  If the sensor's
# radio pauses for half a second (firmware housekeeping, weak signal, a power
# glitch), the link is declared dead.  Android's Bluetooth stack typically uses
# 20 s for GATT connections, which is why the phone app survived small stalls
# that kill a Linux link.
#
# This script is OPT-IN and changes /etc/bluetooth/main.conf.  Without --apply
# it only prints what it would do.  With --apply it backs up the file, sets the
# [LE] defaults below, and restarts bluetooth.  The values apply to every new
# LE connection, not just the Beddit.
#
# Usage:
#   ./scripts/tune-bluez-le.sh            # dry run
#   sudo ./scripts/tune-bluez-le.sh --apply
#
# A one-off alternative for the current connection (find the handle with
# `hcitool con`):
#   sudo hcitool lecup --handle 0x80 --min 24 --max 40 --latency 0 --timeout 2000
#
# Units: connection interval is 1.25 ms, supervision timeout is 10 ms.
# 24 -> 30 ms, 40 -> 50 ms, 2000 -> 20 s.

set -euo pipefail

CONF="${CONF:-/etc/bluetooth/main.conf}"
APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
    APPLY=1
fi

settings=(
    "MinConnectionInterval=24"
    "MaxConnectionInterval=40"
    "ConnectionLatency=0"
    "ConnectionSupervisionTimeout=2000"
)

echo "BlueZ config: $CONF"
for line in "${settings[@]}"; do
    key="${line%%=*}"
    printf '  %-32s -> %s\n' "$key" "${line#*=}"
done

if [[ ! -f "$CONF" ]]; then
    echo "error: $CONF does not exist" >&2
    exit 1
fi

if [[ "$APPLY" -ne 1 ]]; then
    echo
    echo "Dry run. Re-run with: sudo $0 --apply"
    exit 0
fi

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "error: --apply needs root (use sudo)" >&2
    exit 1
fi

backup="${CONF}.bak.$(date +%Y%m%d%H%M%S)"
cp -a "$CONF" "$backup"
echo "Backed up to $backup"

for line in "${settings[@]}"; do
    key="${line%%=*}"
    value="${line#*=}"
    # The Debian main.conf ships these as commented-out keys in [LE].
    # Match both the commented and an already-set form.
    if grep -qE "^[#[:space:]]*${key}[[:space:]]*=" "$CONF"; then
        sed -i -E "s|^[#[:space:]]*${key}[[:space:]]*=.*|${key}=${value}|" "$CONF"
    else
        # Insert into the [LE] section if the key is missing entirely.
        sed -i "/^\[LE\]/a ${key}=${value}" "$CONF"
    fi
done

echo "Restarting bluetooth..."
systemctl restart bluetooth
echo "Done. New LE connections use the values above."
echo "Verify the current connection with: hcitool con"
