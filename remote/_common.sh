#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
[ -f "$HERE/pi.env" ] || { echo "copy remote/pi.env.example to remote/pi.env and edit it"; exit 1; }
source "$HERE/pi.env"
PI="$PI_USER@$PI_HOST"
pssh() { ssh $SSH_OPTS "$PI" "$@"; }
