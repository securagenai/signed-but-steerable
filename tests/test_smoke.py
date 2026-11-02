"""Smoke tests: runs desktop/e7_desync.py, analysis/analyze_e1.py and analysis/analyze_e2.py end to end on
synthetic inputs (and on the prior study's released outputs when available) in a temporary data/results
tree. Checks that the scripts run and that key quantities behave as they must."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "third_party" / "edge-rag-trust-mechanisms" / "data"
FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def run(script, *args, env=None):
    p = subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True, env=env, cwd=ROOT)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:])
    return p.returncode == 0


rng = np.random.default_rng(5)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    data, res = td / "data", td / "results"
    (data / "prior").mkdir(parents=True)
    env = dict(os.environ, RAG_DATA=str(data), RAG_RESULTS=str(res), REPRO="0", PYTHONPATH=str(ROOT))
    # ---------------- synthetic corpus, questions, vectors (gold passage is the nearest one)
    n_p, n_q, d = 60, 30, 32
    P = rng.normal(size=(n_p, d)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    Q, qs = [], []
    for i in range(n_q):
        cond = ["in_answerable", "in_unanswerable", "out_of_corpus"][i % 3]
        g = i % n_p
        v = P[g] * 3 + rng.normal(size=d) * (0.5 if cond == "in_answerable" else 2.0)
        Q.append(v / np.linalg.norm(v))
        qs.append({"qid": f"q{i}", "question": f"question {i}?", "golds": ["x"], "gold_pid": g if cond != "out_of_corpus" else None,
                   "title": "t", "condition": cond})
    Q = np.array(Q)
    json.dump(qs, open(data / "prior" / "questions_v2.json", "w"))
    json.dump([{"id": i, "title": "t", "text": f"passage {i}"} for i in range(n_p)], open(data / "corpus.json", "w"))
    np.savez(data / "vec_bge_m3.npz", P=P.astype(np.float32), Q=Q.astype(np.float32), qids=np.array([q["qid"] for q in qs]))
    W = np.repeat(P, 2, axis=0) + rng.normal(size=(2 * n_p, d)) * 0.05
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    np.savez(data / "vec_minilm_L128.npz", W=W.astype(np.float32), Q=Q.astype(np.float32), qids=np.array([q["qid"] for q in qs]),
             pid=np.repeat(np.arange(n_p), 2))
    ok = run("desktop/e7_desync.py", "--no-embed", env=env)
    check("e7_desync.py runs on synthetic vectors", ok)
    if ok:
        s = json.load(open(res / "e7" / "summary.json"))
        for sp in ("bge_m3", "minilm"):
            r = s["spaces"][sp]
            h = {t: r[f"tau={t}"]["A1_hijack_top1"][0] for t in (1.0, 0.999, 0.95, 0.9, -1.0)}
            check(f"E7 {sp}: no hijack without deviation, full hijack without a check, monotone in tolerance",
                  h[1.0] == 0 and h[-1.0] == 1.0 and h[0.999] <= h[0.95] <= h[0.9] <= h[-1.0], str(h))
    # ---------------- synthetic E1 features -> analyze_e1
    (res / "e1").mkdir(parents=True)
    with open(res / "e1" / "features.jsonl", "w") as f:
        for q in qs:
            a = q["condition"] == "in_answerable"
            feats = {}
            for k in ("k3", "k5", "k10"):
                feats[k] = {"top1": float(rng.normal(0.6 if a else 0.55, 0.07)), "bi_max": float(rng.normal(0.7 if a else 0.5, 0.1)),
                            "ce_max": float(rng.normal(5 if a else 0, 2)), "cep_max": float(rng.normal(4 if a else 1, 2)),
                            "qa_margin": float(rng.normal(3 if a else -2, 2)), "n_sentences": 25}
            f.write(json.dumps({"qid": q["qid"], "condition": q["condition"], "gold_pid": q["gold_pid"], "gold_rank": 0, "f": feats}) + "\n")
    ok = run("analysis/analyze_e1.py", env=env)
    check("analyze_e1.py runs and applies the pre-registered H1 rule", ok and "H1_decision" in json.load(open(res / "e1" / "summary.json")))
    # ---------------- analyze_e2 on the prior study's released outputs (no new runs needed)
    if PRIOR.exists():
        shutil.rmtree(data / "prior")
        shutil.copytree(PRIOR, data / "prior")
        shutil.rmtree(res / "e1")
        ok = run("analysis/analyze_e2.py", env=env)
        check("analyze_e2.py runs on the prior outputs", ok)
        if ok:
            s = json.load(open(res / "e2" / "summary.json"))["configs"]
            check("analyze_e2 reproduces prior lenient accuracy (k3 0.75, k5 0.52)",
                  abs(s["prior_q3_k3"]["acc_answerable_lenient"] - 0.75) < 1e-9 and abs(s["prior_q3_k5"]["acc_answerable_lenient"] - 0.52) < 1e-9)
            check("analyze_e2 reproduces the prior model-only abstention on unanswerables (k3 0.66)",
                  abs(s["prior_q3_k3"]["M"]["unans_abstained"] - 0.66) < 1e-9, str(s["prior_q3_k3"]["M"]))
print(f"\n{len(FAILED)} failed" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)
