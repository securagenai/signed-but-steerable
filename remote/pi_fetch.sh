#!/usr/bin/env bash
# Pulls the device results into results/ (merge, no deletion), then the analysis can run on the desktop.
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
rsync -az --info=stats1 -e "ssh $SSH_OPTS" "$PI:$PI_DIR/results/" ./results/
echo "fetched into $ROOT/results; run: bash analysis/reproduce_all.sh"
