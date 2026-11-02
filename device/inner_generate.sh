#!/usr/bin/env bash
# Runs inside the network namespace (or directly when OFFLINE_NETNS=0). One llama-server for all
# configurations: same model, same build, same request format as the prior study.
set -uo pipefail
cd "$(dirname "$0")/.."; ROOT="$(pwd)"
source device/config.env
source venv_device/bin/activate
python -c "import requests; requests.get('https://example.com', timeout=3)" 2>/dev/null \
  && echo "WARNING: network reachable" || echo "network: offline (isolated)"
"$LLAMA_SERVER" -m "$MODEL_3B" -t "$LLM_THREADS" -c 4096 -np 1 --host 127.0.0.1 --port 8080 \
  > results/server_e2.log 2>&1 &
SPID=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
run() { if [ -f "results/e2/status_$1.json" ]; then echo "skip $1 (done)"; else python device/e2_generate.py "${@:2}" --config "$1" || echo "config $1 failed"; fi; }
# reproducibility check of the prior main configuration (60 questions, ~25 min)
python device/e2_generate.py --config k3_full --limit 60
# pruned contexts selected on the desktop (FP32): the algorithmic effect
for c in k5_ce3 k10_ce3 k10_ce5 k5_bi3; do run "$c"; done
# end-to-end selection on the device (cost + agreement with FP32 selection)
python device/e2_generate.py --config k5_ce3 --online-selector ort-int8 --selector-model ms-marco-minilm-l6
# E7 end-to-end: payload chunks with desynchronised vectors (answerable questions only)
for c in e7_payload_unconstrained e7_payload_tau0.999 e7_payload_tau_cpu_int8; do
  [ -f "results/e2/contexts/$c.json" ] && run "$c"
done
kill $SPID; wait $SPID 2>/dev/null
echo "INNER GENERATE DONE $(date)"
