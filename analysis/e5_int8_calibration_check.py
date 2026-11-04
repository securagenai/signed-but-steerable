"""Audit of the CPU-INT8 reference threshold (E5), for the paper's description of its calibration.

On the desktop (default): from the saved E5 vectors, recompute the per-window cosine between dynamic INT8 and FP32
(ONNX Runtime, both on the Raspberry Pi CPU), the exact 0.1% quantile (numpy default, linear interpolation), the
minimum, and how many honest windows fall below the threshold.
On the Raspberry Pi (--rerun): re-encode the windows in dynamic INT8 with batch 64 and with batch 32, exactly as E5
does otherwise (4 threads, sequence length 128, padding to the longest text in each batch), and compare with the
saved vectors, to establish which batch size produced them. Writes results/e5/int8_calibration_check.json.
Usage: python analysis/e5_int8_calibration_check.py [--rerun]"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rerun", action="store_true")
a = ap.parse_args()
E5 = RESULTS / "e5"
V8 = np.load(E5 / "vectors_ort-int8.npz")["W"].astype(np.float64)
V32 = np.load(E5 / "vectors_ort-fp32.npz")["W"].astype(np.float64)
out = {"n_windows": int(len(V8)),
       "norms_int8_min_max": [float(np.linalg.norm(V8, axis=1).min()), float(np.linalg.norm(V8, axis=1).max())],
       "norms_fp32_min_max": [float(np.linalg.norm(V32, axis=1).min()), float(np.linalg.norm(V32, axis=1).max())]}
cw = (V8 * V32).sum(1)
tau = float(np.quantile(cw, 0.001))
out.update({"tau_p001_linear": tau, "tau_in_summary": read_json(E5 / "summary.json")["fidelity"]["ort-int8"]["cos_p001"],
            "min_cosine": float(cw.min()), "n_below_tau": int((cw < tau).sum()), "n_below_0.904": int((cw < 0.904).sum()),
            "lowest_five": [float(x) for x in np.sort(cw)[:5]], "angle_deg_at_tau": float(np.degrees(np.arccos(tau)))})
if a.rerun:
    from ragtrust.encoders import Encoder
    W = read_json(DATA / "windows_minilm_L128.json")
    texts = [w["text"] for w in W]
    for b in (64, 32):
        enc = Encoder("all-minilm-l6-v2", "ort-int8", threads=4, batch=b, seq_len=128, cpu_padding="longest")
        R = enc.embed(texts).astype(np.float64)
        enc.close()
        d = np.abs(R - V8)
        out[f"rerun_batch{b}"] = {"max_abs_coordinate_difference": float(d.max()), "identical": bool(d.max() == 0.0),
                                  "min_cosine_to_saved": float((R * V8).sum(1).min())}
write_json(E5 / "int8_calibration_check.json", out)
print(json.dumps(out, indent=1))
