"""Resolves and records the current Hugging Face revision (commit hash) of every model used by the
study into pins.json, keeping revisions that are already pinned (the prior study's BGE-M3 and NLI
checkpoints). Run once on the desktop with internet access, commit pins.json, then run everything
with REPRO=1 (the default) so that desktop and Pi load exactly the same weights.

Usage: python desktop/freeze_pins.py [--force MODEL ...]"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import ROOT  # noqa: E402

MODELS = [
    "BAAI/bge-m3",                               # retrieval embedder of the prior study (pinned there)
    "cross-encoder/nli-deberta-v3-base",         # NLI verifier of the prior study (pinned there)
    "sentence-transformers/all-MiniLM-L6-v2",    # bi-encoder
    "cross-encoder/ms-marco-MiniLM-L-6-v2",      # cross-encoder relevance scorer (same 6x384 trunk)
    "deepset/minilm-uncased-squad2",             # extractive QA with no-answer score (12 layers; CPU)
    "cross-encoder/nli-MiniLM2-L6-H768",         # small NLI cross-encoder (RoBERTa-style, 768 hidden)
    "Qwen/Qwen2.5-3B-Instruct",                  # tokenizer only (prompt-length bookkeeping)
]
ap = argparse.ArgumentParser()
ap.add_argument("--force", nargs="*", default=[])
a = ap.parse_args()
from huggingface_hub import HfApi  # noqa: E402

pins = json.load(open(ROOT / "pins.json"))
revs = pins.setdefault("hf_revisions", {})
api = HfApi()
for m in MODELS:
    if revs.get(m) and m not in a.force:
        print(f"keep    {m} @ {revs[m]}")
        continue
    sha = api.model_info(m).sha
    revs[m] = sha
    print(f"pinned  {m} @ {sha}")
json.dump(pins, open(ROOT / "pins.json", "w"), indent=1)
print("pins.json updated; commit it before running the experiments.")
