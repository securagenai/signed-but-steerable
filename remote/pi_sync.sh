#!/usr/bin/env bash
# Pushes code, data and artefacts to the Pi, plus the desktop results the device needs
# (E1 sentence scores, E2 contexts, E8 cases). Never deletes anything on the Pi.
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
pssh "mkdir -p '$PI_DIR/results'"
rsync -az --info=stats1 -e "ssh $SSH_OPTS" \
  --exclude 'venv*/' --exclude 'results/' --exclude 'third_party/' --exclude '.git/' --exclude '__pycache__/' \
  --exclude 'remote/pi.env' --exclude 'artifacts/models/minilm-squad2/' \
  ./ "$PI:$PI_DIR/"
for p in results/e1/sentences_top10.jsonl results/e1/features.jsonl results/e2/contexts results/e8/packs \
         results/e8/keys results/e8/cases.json; do
  [ -e "$p" ] && rsync -az --relative -e "ssh $SSH_OPTS" "./$p" "$PI:$PI_DIR/"
done
echo "synced to $PI:$PI_DIR"
