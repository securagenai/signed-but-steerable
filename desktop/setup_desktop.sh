#!/usr/bin/env bash
# Desktop environment for the desktop experiments, model export and analysis (Linux x86-64 or WSL2).
# Usage:  bash desktop/setup_desktop.sh
set -euo pipefail
cd "$(dirname "$0")/.."; ROOT="$(pwd)"
python3 -m venv venv_desktop
source venv_desktop/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
pip install -r requirements_desktop.txt
pip install onnxsim || echo "onnxsim not installed (optional)"

# prior study: question set, released per-query outputs (results_v2/v3) and build statistics
PRIOR_REPO=${PRIOR_REPO:-https://github.com/securagenai/edge-rag-trust-mechanisms}
mkdir -p third_party data/prior results/env
[ -d third_party/edge-rag-trust-mechanisms ] || git clone --depth 1 "$PRIOR_REPO" third_party/edge-rag-trust-mechanisms
cp third_party/edge-rag-trust-mechanisms/data/questions_v2.json third_party/edge-rag-trust-mechanisms/data/pack_v2_build_stats.json data/prior/
cp -r third_party/edge-rag-trust-mechanisms/data/results_v2 third_party/edge-rag-trust-mechanisms/data/results_v3 data/prior/

# SQuAD 2.0 dev (CC BY-SA 4.0)
[ -f data/dev-v2.0.json ] || wget -O data/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
pip freeze > results/env/pip_freeze_desktop.txt
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "Desktop ready. Next: python desktop/freeze_pins.py  (once, needs internet), then python desktop/rebuild_corpus.py"
