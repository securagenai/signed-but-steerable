"""How much attack budget do the storage formats of vector databases leave?

An integrity audit that recomputes an embedding from its text and compares it with the STORED vector
must tolerate the storage format's own error, otherwise honest vectors fail. That tolerance is the
attacker's angular budget, and the margin law turns it into the fraction of queries an insider can
hijack. For every dataset/model of desktop/beir_margins.py this script quantises the corpus vectors
with common schemes, measures the honest angle between each vector and its reconstruction, sets the
tolerance at the largest honest angle (no false alarms) and at its 99.9th percentile, and reports the
predicted hijack rate with Wilson 95% intervals:
  fp16    half-precision storage                          16 bits/dim
  sq8     per-dimension 8-bit scalar quantisation         8 bits/dim
  sq4     per-dimension 4-bit scalar quantisation         4 bits/dim
  pq      product quantisation, d/8 sub-vectors x 256     1 bit/dim
  binary  sign bits (rescaled)                            1 bit/dim
Writes results/e7/vecdb_quant.json. Usage: python analysis/vecdb_quant.py"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import RESULTS, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402


def sq(D, bits):
    lo, hi = D.min(0), D.max(0)
    n = 2 ** bits - 1
    return np.round((D - lo) / np.maximum(hi - lo, 1e-12) * n) / n * (hi - lo) + lo


def pq(D, sub=8, k=256, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    d = D.shape[1]
    R = np.empty_like(D)
    rng = np.random.default_rng(seed)
    train = D[rng.choice(len(D), min(len(D), 20000), replace=False)]
    for j in range(0, d, sub):
        km = MiniBatchKMeans(n_clusters=min(k, len(train)), n_init=1, max_iter=50, batch_size=4096, random_state=seed)
        km.fit(train[:, j:j + sub])
        R[:, j:j + sub] = km.cluster_centers_[km.predict(D[:, j:j + sub])]
    return R


SCHEMES = {"fp16": (16, lambda D: D.astype(np.float16).astype(np.float32)), "sq8": (8, lambda D: sq(D, 8)),
           "sq4": (4, lambda D: sq(D, 4)), "pq": (1, pq), "binary": (1, lambda D: np.sign(D) + (D == 0))}
out = {}
print(f"{'dataset/model':18s} {'scheme':7s} {'bits/d':>6s} {'honest angle median / max':>27s} "
      f"{'hijack @max-tol':>20s} {'hijack @p99.9-tol':>20s}")
for f in sorted((RESULTS / "e7" / "vecs").glob("*.npz")):
    key = f.stem.replace("_", "/", 1)
    D = np.load(f)["D"].astype(np.float32)
    mg = np.load(RESULTS / "e7" / "margins" / f.name)["margins"]
    out[key] = {}
    for name, (bits, fn) in SCHEMES.items():
        R = fn(D)
        R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        ang = np.arccos(np.clip((D * R).sum(1), -1, 1))
        a_max, a_999 = float(ang.max()), float(np.quantile(ang, 0.999))
        h_max = wilson(int((mg < a_max).sum()), len(mg))
        h_999 = wilson(int((mg < a_999).sum()), len(mg))
        out[key][name] = {"bits_per_dim": bits, "honest_angle_deg_median": float(np.degrees(np.median(ang))),
                          "honest_angle_deg_max": float(np.degrees(a_max)), "honest_angle_deg_p999": float(np.degrees(a_999)),
                          "hijack_at_max_tolerance": h_max, "hijack_at_p999_tolerance": h_999}
        print(f"{key:18s} {name:7s} {bits:6d} {np.degrees(np.median(ang)):11.3f}° / {np.degrees(a_max):8.3f}°   "
              f"{h_max[0]:7.4f} [{h_max[1]:.3f},{h_max[2]:.3f}]  {h_999[0]:7.4f} [{h_999[1]:.3f},{h_999[2]:.3f}]", flush=True)
write_json(RESULTS / "e7" / "vecdb_quant.json", out)
