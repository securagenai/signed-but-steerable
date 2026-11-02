#!/usr/bin/env bash
# E7 end-to-end with the CPU-INT8 tolerance: one llama-server, same model, build, settings and
# request format as the other E7 configurations; runs only e7_payload_tau_cpu_int8.
set -uo pipefail
cd "$(dirname "$0")/.."
source device/config.env
source venv_device/bin/activate
[ -f results/e2/contexts/e7_payload_tau_cpu_int8.json ] || { echo "missing contexts: run desktop/e7_desync.py and sync"; exit 1; }
"$LLAMA_SERVER" -m "$MODEL_3B" -t "$LLM_THREADS" -c 4096 -np 1 --host 127.0.0.1 --port 8080 > results/server_e7_int8.log 2>&1 &
SPID=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
echo "== E7 tau_cpu_int8 $(date)"
python device/e2_generate.py --config e7_payload_tau_cpu_int8 || echo "config failed"
kill $SPID; wait $SPID 2>/dev/null
echo "E7 CPU-INT8 DONE $(date)"
