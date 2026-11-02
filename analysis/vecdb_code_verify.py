"""Exact verification at the level of quantisation CODES (exploratory; PROTOCOL.md deviations log).

Proposed defence for vector databases: instead of comparing a stored vector with a recomputed one under
a tolerance (which leaves an attack budget, see analysis/vecdb_quant.py), the verifier recomputes the
embedding from the text, quantises it with the SAME codebook, and requires the codes to be IDENTICAL.
This only works if float noise between platforms rarely moves a value across a quantisation boundary.

Measured here with the same 1,745 MiniLM windows embedded on three platforms:
  x86   ONNX Runtime FP32 on the desktop CPU (publisher; computed now)
  ARM   ONNX Runtime FP32 on the Raspberry Pi 5 (device; results/e5/vectors_ort-fp32.npz)
  GPU   PyTorch FP32 on the desktop GPU (sentence-transformers; data/vec_minilm_L128.npz)
Codebooks are trained on the publisher's (x86) vectors: SQ8 and SQ4 (per-dimension min/max), PQ
(d/8 sub-vectors x 256 centroids) and binary (sign). For each platform pair the script reports the share
of vectors whose codes are identical, the share of code entries that differ, and, for SQ, the share of
entries that a guard band (accept either neighbouring code when the value lies within eps of a bin edge)
would have to tolerate. Writes results/e7/vecdb_code_verify.json. Usage: python analysis/vecdb_code_verify.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, out_dir, read_json, write_json  # noqa: E402

texts = [w["text"] for w in read_json(DATA / "windows_minilm_L128.json")]
arm = np.load(RESULTS / "e5" / "vectors_ort-fp32.npz")["W"].astype(np.float32)
gpu = np.load(DATA / "vec_minilm_L128.npz")["W"].astype(np.float32)
cache = RESULTS / "e7" / "vectors_x86_ort-fp32.npy"
if cache.exists():
    x86 = np.load(cache)
else:
    from ragtrust.encoders import Encoder
    x86 = Encoder("all-minilm-l6-v2", "ort-fp32", threads=4, batch=32, seq_len=128, cpu_padding="longest").embed(texts).astype(np.float32)
    np.save(cache, x86)
P = {"x86": x86, "ARM": arm, "GPU": gpu}
lo, hi = x86.min(0), x86.max(0)


def sq_codes(V, bits):
    n = 2 ** bits - 1
    t = (V - lo) / np.maximum(hi - lo, 1e-12) * n
    return np.clip(np.floor(t + 0.5), 0, n).astype(np.int32), t


def train_pq(V, sub=8, k=256, seed=0):
    from sklearn.cluster import KMeans
    return [KMeans(n_clusters=k, n_init=1, random_state=seed).fit(V[:, j:j + sub]).cluster_centers_
            for j in range(0, V.shape[1], sub)]


PQ = train_pq(x86)


def pq_codes(V, sub=8):
    out = []
    for c, j in zip(PQ, range(0, V.shape[1], sub)):
        d = ((V[:, None, j:j + sub] - c[None]) ** 2).sum(-1)
        out.append(np.argmin(d, 1))
    return np.stack(out, 1)


schemes = {"SQ8": lambda V: sq_codes(V, 8)[0], "SQ4": lambda V: sq_codes(V, 4)[0], "PQ": pq_codes,
           "binary": lambda V: (V > 0).astype(np.int8)}
res = {"n_vectors": int(len(x86)), "float_max_abs_diff": {}, "schemes": {}}
pairs = [("x86", "ARM"), ("x86", "GPU"), ("ARM", "GPU")]
for a, b in pairs:
    res["float_max_abs_diff"][f"{a}-{b}"] = float(np.abs(P[a] - P[b]).max())
print(f"{'scheme':7s} {'pair':9s} {'vectors with identical codes':>29s} {'differing code entries':>23s} {'max entries differing in one vector':>36s}")
for name, fn in schemes.items():
    C = {k: fn(v) for k, v in P.items()}
    res["schemes"][name] = {}
    for a, b in pairs:
        diff = C[a] != C[b]
        r = {"vectors_identical": float((~diff.any(1)).mean()), "entries_differing": float(diff.mean()),
             "max_entries_differing_per_vector": int(diff.sum(1).max())}
        if name.startswith("SQ"):
            bits = int(name[2:])
            for eps in (1e-6, 1e-5, 1e-4):
                t = sq_codes(P[a], bits)[1]
                near = np.abs((t - np.floor(t)) - 0.5) * (hi - lo) / (2 ** bits - 1) < eps
                r[f"guard_band_eps{eps:g}_entries_flagged"] = float(near.mean())
                r[f"guard_band_eps{eps:g}_vectors_identical_after"] = float((~(diff & ~near).any(1)).mean())
        res["schemes"][name][f"{a}-{b}"] = r
        print(f"{name:7s} {a + '-' + b:9s} {r['vectors_identical']:29.4f} {r['entries_differing']:23.2e} {r['max_entries_differing_per_vector']:36d}")
for name in ("SQ8", "SQ4"):
    r = res["schemes"][name]["x86-ARM"]
    print(f"{name} x86-ARM with guard band: eps=1e-6 flags {r['guard_band_eps1e-06_entries_flagged']:.2e} of entries -> "
          f"{r['guard_band_eps1e-06_vectors_identical_after']:.4f} of vectors verified exactly")
write_json(out_dir("e7") / "vecdb_code_verify.json", res)
