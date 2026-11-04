"""Repository paths, pinned artefacts and small JSON helpers shared by desktop and device scripts.

Layout (relative to the repository root; override with RAG_DATA, RAG_ARTIFACTS, RAG_RESULTS):
  data/        inputs: SQuAD 2.0 dev, files copied from the prior study (data/prior), rebuilt corpus,
               desktop retrieval and embedding matrices
  artifacts/   models exported on the desktop (ONNX graphs, tokenizer.json, weights)
  results/     outputs, one sub-folder per experiment (e1, e2, e4, e5, e7, e8)
PRIOR_DIR is the folder of the previous study on the Pi (llama.cpp build, GGUF models); default
~/edge_rag_experiment. REPRO=1 (default) enforces pinned model revisions; REPRO=0 allows exploration.
"""
import hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("RAG_DATA", ROOT / "data"))
ARTIFACTS = Path(os.environ.get("RAG_ARTIFACTS", ROOT / "artifacts"))
RESULTS = Path(os.environ.get("RAG_RESULTS", ROOT / "results"))
PRIOR_DATA = DATA / "prior"
PRIOR_DIR = Path(os.environ.get("PRIOR_DIR", Path.home() / "edge_rag_experiment"))
STRICT = os.environ.get("REPRO", "1") == "1"


def out_dir(name):
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(p):
    with open(p) as f:
        return json.load(f)


def _json_safe(o):
    """Replace NaN and infinities with None, so that every written file is strict JSON."""
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def write_json(p, obj, indent=1):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(_json_safe(obj), f, indent=indent, allow_nan=False)
    return p


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(fh, row):
    fh.write(json.dumps(row) + "\n")
    fh.flush()


def done_keys(p, key="qid"):
    p = Path(p)
    return {json.loads(line)[key] for line in open(p) if line.strip()} if p.exists() else set()


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_PINS = None


def pins():
    global _PINS
    if _PINS is None:
        _PINS = read_json(ROOT / "pins.json")
    return _PINS


def hf_revision(name):
    """Pinned Hugging Face revision. In strict mode an unpinned model aborts the run."""
    rev = pins().get("hf_revisions", {}).get(name)
    if STRICT and not rev:
        raise SystemExit(f"[REPRO=1] no pinned revision for {name}: run desktop/freeze_pins.py "
                         f"once (needs internet), or set REPRO=0 to explore")
    return rev


def questions():
    return read_json(PRIOR_DATA / "questions_v2.json")


def corpus():
    p = DATA / "corpus.json"
    if not p.exists():
        raise SystemExit("data/corpus.json missing: run desktop/rebuild_corpus.py first")
    return read_json(p)


def model_dir(name):
    d = ARTIFACTS / "models" / name
    if not (d / "model_card.json").exists():
        raise SystemExit(f"{d}/model_card.json missing: run desktop/export_onnx.py --models {name}")
    return d
