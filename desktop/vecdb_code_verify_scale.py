"""Exact code-level verification at scale (exploratory; PROTOCOL.md deviations log).

Same question as analysis/vecdb_code_verify.py (do PQ / binary / SQ codes computed on two platforms
coincide exactly?), on a large corpus and two models: FiQA (57,638 documents, BEIR) embedded with
all-MiniLM-L6-v2 (128 tokens, all documents) and BGE-M3 (512 tokens, a seeded subset), each on the desktop
GPU (publisher; codebooks trained here) and on the desktop CPU (verifier), both FP32 with the pinned
revisions. Reports, with Wilson 95% intervals: share of vectors with identical codes; share of code
entries that differ; for SQ a guard band at bin edges and for PQ a guard band on near-ties between the
two closest centroids (both: accept either candidate only when the publisher-side value is within eps
of the decision boundary), and the share of vectors verified exactly after the guard band.
Writes results/e7/vecdb_code_verify_scale.json; embeddings are cached in results/e7/scale_*.npy.
Usage: python desktop/vecdb_code_verify_scale.py [--n-bge 5000]"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, hf_revision, out_dir, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-bge", type=int, default=5000)
ap.add_argument("--seed", type=int, default=20261101)
a = ap.parse_args()
docs = []
for line in open(DATA / "beir" / "fiqa" / "corpus.jsonl"):
    d = json.loads(line)
    docs.append((d.get("title", "") + " " + d.get("text", "")).strip())
idx_bge = sorted(random.Random(a.seed).sample(range(len(docs)), min(a.n_bge, len(docs))))
MODELS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", 128, list(range(len(docs))), 256),
          "bge_m3": ("BAAI/bge-m3", 512, idx_bge, 32)}
E = out_dir("e7")


def embed(key, device):
    f = E / f"scale_{key}_{device}.npy"
    if f.exists():
        return np.load(f)
    from sentence_transformers import SentenceTransformer
    hf, msl, idx, bs = MODELS[key]
    m = SentenceTransformer(hf, revision=hf_revision(hf), device=device)
    m.max_seq_length = msl
    V = m.encode([docs[i] for i in idx], batch_size=bs, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(f, V)
    return V


def sq(V, lo, hi, bits):
    n = 2 ** bits - 1
    t = (V - lo) / np.maximum(hi - lo, 1e-12) * n
    return np.clip(np.floor(t + 0.5), 0, n).astype(np.int32), np.abs((t - np.floor(t)) - 0.5) * (hi - lo) / n


def pq_train(V, sub=8, k=256, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    tr = V[np.random.default_rng(seed).choice(len(V), min(len(V), 20000), replace=False)]
    return [MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=100, batch_size=4096, random_state=seed).fit(tr[:, j:j + sub]).cluster_centers_
            for j in range(0, V.shape[1], sub)]


def pq(V, C, sub=8):
    codes, gaps = [], []
    for c, j in zip(C, range(0, V.shape[1], sub)):
        d = np.empty((len(V), len(c)), np.float32)
        for s in range(0, len(V), 8192):
            x = V[s:s + 8192, j:j + sub]
            d[s:s + 8192] = (x * x).sum(1, keepdims=True) - 2 * x @ c.T + (c * c).sum(1)[None]
        two = np.partition(d, 1, axis=1)[:, :2]
        codes.append(np.argmin(d, 1))
        gaps.append(np.abs(two[:, 1] - two[:, 0]))
    return np.stack(codes, 1), np.stack(gaps, 1)


res = {"corpus": "BEIR FiQA", "n_docs": len(docs), "models": {}}
for key in MODELS:
    G, C = embed(key, "cuda"), embed(key, "cpu")
    n = len(G)
    lo, hi = G.min(0), G.max(0)
    out = {"n_vectors": n, "float_max_abs_diff": float(np.abs(G - C).max()), "schemes": {}}
    for name, bits in (("SQ8", 8), ("SQ4", 4)):
        cg, dist = sq(G, lo, hi, bits)
        cc, _ = sq(C, lo, hi, bits)
        diff = cg != cc
        r = {"identical": wilson(int((~diff.any(1)).sum()), n), "entries_differing": float(diff.mean())}
        for eps in (1e-6, 1e-5):
            near = dist < eps
            r[f"guard_eps{eps:g}"] = {"entries_flagged": float(near.mean()),
                                      "identical_after": wilson(int((~(diff & ~near).any(1)).sum()), n)}
        out["schemes"][name] = r
    CB = pq_train(G)
    cg, gap = pq(G, CB)
    cc, _ = pq(C, CB)
    diff = cg != cc
    r = {"identical": wilson(int((~diff.any(1)).sum()), n), "entries_differing": float(diff.mean())}
    for eps in (1e-6, 1e-5):
        near = gap < eps
        r[f"guard_eps{eps:g}"] = {"entries_flagged": float(near.mean()), "identical_after": wilson(int((~(diff & ~near).any(1)).sum()), n)}
    out["schemes"]["PQ"] = r
    diff = (G > 0) != (C > 0)
    near = np.abs(G) < 1e-6
    out["schemes"]["binary"] = {"identical": wilson(int((~diff.any(1)).sum()), n), "entries_differing": float(diff.mean()),
                                "guard_eps1e-06": {"entries_flagged": float(near.mean()),
                                                   "identical_after": wilson(int((~(diff & ~near).any(1)).sum()), n)}}
    res["models"][key] = out
    print(f"{key}: {n} vectors, max |GPU - CPU| = {out['float_max_abs_diff']:.2e}")
    for s, r in out["schemes"].items():
        g = r.get("guard_eps1e-06", {})
        print(f"   {s:7s} identical codes {r['identical'][0]:.5f} [{r['identical'][1]:.5f}, {r['identical'][2]:.5f}]   "
              f"entries differing {r['entries_differing']:.1e}   with guard 1e-6: {g.get('identical_after', [float('nan')])[0]:.5f} "
              f"(flags {g.get('entries_flagged', float('nan')):.1e} of entries)")
write_json(E / "vecdb_code_verify_scale.json", res)
