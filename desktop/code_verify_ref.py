"""Code-level verification with the reference verifier (ragtrust/codeverify.py) on the desktop data
(exploratory; PROTOCOL.md deviations log). Same rule as on the Apple M2: ambiguity is decided ONLY from the
verifier's recomputation, a neighbouring code is accepted only at ambiguous entries, accepted vectors are
canonicalised. Codebooks are trained on the publisher's vectors. Both directions are measured.
  scale:  BEIR FiQA, MiniLM on all 57,638 documents and BGE-M3 on the seeded 5,000; NVIDIA GPU <-> x86 CPU
          (results/e7/scale_*_{cuda,cpu}.npy from desktop/vecdb_code_verify_scale.py).
  arch:   1,745 MiniLM windows embedded with ONNX Runtime FP32 on x86 (results/e7/vectors_x86_ort-fp32.npy)
          and on the Raspberry Pi ARM CPU (results/e5/vectors_ort-fp32.npz), plus PyTorch on the GPU
          (data/vec_minilm_L128.npz).
Guard widths are scanned (SQ and binary in value units, PQ in squared-distance units). For each width:
accepted share (Wilson 95%), share of vectors that needed an exception, largest number of exceptions in a
vector, share of entries marked ambiguous. Writes results/e7/code_verify_ref.json.
Usage: python desktop/code_verify_ref.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.codeverify import PQ, SQ, Binary, verify  # noqa: E402
from ragtrust.paths import DATA, RESULTS, out_dir, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

EPS_VALUE = [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5]
EPS_PQ = [0.0, 1e-7, 1e-6, 1e-5, 1e-4]


def measure(P, R, label):
    P, R = P.astype(np.float64), R.astype(np.float64)
    n = len(P)
    out = {"n": n, "float_max_abs_diff": float(np.abs(P - R).max()), "schemes": {}}
    for name, q, grid in (("SQ8", SQ(8).fit(P), EPS_VALUE), ("SQ4", SQ(4).fit(P), EPS_VALUE),
                          ("PQ", PQ().fit(P), EPS_PQ), ("binary", Binary(), EPS_VALUE)):
        stored = q.encode(P)
        rows = {}
        for eps in grid:
            r = verify(stored, R, q, eps)
            rows[f"{eps:g}"] = {"accepted": wilson(int(r["accept"].sum()), n),
                                "vectors_with_exceptions": float((r["exceptions_per_vector"] > 0).mean()),
                                "max_exceptions_in_a_vector": int(r["exceptions_per_vector"].max()),
                                "ambiguous_entry_share": r["ambiguous_entry_share"]}
        first = next((e for e in grid if rows[f"{e:g}"]["accepted"][0] == 1.0), None)
        out["schemes"][name] = {"by_eps": rows, "smallest_tested_eps_with_all_accepted": first}
        line = "  ".join(f"{e:g}:{rows[f'{e:g}']['accepted'][0]:.5f}" for e in grid)
        print(f"{label:28s} {name:6s} {line}   (smallest eps with 100%: {first})", flush=True)
    return out


res = {"verifier": "ragtrust/codeverify.py", "scale": {}, "arch": {}}
E7 = RESULTS / "e7"
for key in ("minilm", "bge_m3"):
    G, C = np.load(E7 / f"scale_{key}_cuda.npy"), np.load(E7 / f"scale_{key}_cpu.npy")
    print(f"--- scale {key}: n={len(G)}, max |GPU - CPU| = {np.abs(G.astype(np.float64) - C).max():.2e}")
    res["scale"][key] = {"gpu_to_cpu": measure(G, C, f"{key} GPU->CPU"), "cpu_to_gpu": measure(C, G, f"{key} CPU->GPU")}
x86 = np.load(E7 / "vectors_x86_ort-fp32.npy")
arm = np.load(RESULTS / "e5" / "vectors_ort-fp32.npz")["W"]
gpu = np.load(DATA / "vec_minilm_L128.npz")["W"]
print(f"--- arch MiniLM windows: n={len(x86)}, max |x86 - ARM| = {np.abs(x86.astype(np.float64) - arm).max():.2e}, "
      f"max |GPU - ARM| = {np.abs(gpu.astype(np.float64) - arm).max():.2e}")
for pub, ver, A, B in (("x86", "ARM", x86, arm), ("ARM", "x86", arm, x86), ("GPU", "ARM", gpu, arm), ("ARM", "GPU", arm, gpu)):
    res["arch"][f"{pub}_to_{ver}"] = measure(A, B, f"windows {pub}->{ver}")
write_json(out_dir("e7") / "code_verify_ref.json", res)
