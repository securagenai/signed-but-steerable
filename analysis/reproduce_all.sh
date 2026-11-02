#!/usr/bin/env bash
# Regenerates the E1, E2, E7 and E8 summaries from the per-query/per-call outputs in results/ (no hardware needed).
# Each step runs only if its inputs exist, so it can be used at any point of the study.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
run() { echo "== $1"; $PY "analysis/$1" "${@:2}" || echo "   ($1 failed)"; }
any() { for p in "$@"; do [ -e "$p" ] && return 0; done; return 1; }
[ -f results/e1/features.jsonl ] && run analyze_e1.py
[ -f data/prior/results_v2/gen_q3_k3.jsonl ] && run analyze_e2.py ce_max
[ -f results/e7/summary.json ] && run analyze_e7.py
any results/e8/device_results_*.json && run analyze_e8.py
echo "done"
