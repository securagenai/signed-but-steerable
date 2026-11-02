"""E4 across repeated full runs (results/e4/contention_rep*.json): per condition and mode, decoding speed
and p99 inter-token latency relative to the no-encoder reference of the same run; mean and range over runs.
Writes results/e4/contention_summary.json. Usage: python analysis/e4_ci.py"""
import glob, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import RESULTS, read_json, write_json  # noqa: E402

runs = [read_json(f) for f in sorted(glob.glob(str(RESULTS / "e4" / "contention_rep*.json")))]
print(f"{len(runs)} runs")
conds = [("ort-int8:1", "CPU INT8, 1 thread"), ("ort-int8:2", "CPU INT8, 2 threads"), ("ort-int8:4", "CPU INT8, 4 threads")]
out = {"n_runs": len(runs), "reference": {}, "conditions": {}}
base = [next(x for x in r if x["condition"] == "none") for r in runs]
ref_dec = [b["decode_tok_s_server"]["median"] for b in base]
ref_p99 = [b["itl_ms"]["p99"] for b in base]
out["reference"] = {"decode_tok_s": ref_dec, "p99_ms": ref_p99}
print(f"reference: decode {np.mean(ref_dec):.2f} tok/s (runs {', '.join(f'{v:.2f}' for v in ref_dec)}), p99 {np.mean(ref_p99):.0f} ms")
for cond, label in conds:
    for mode in ("saturate", "periodic"):
        loss, p99inc = [], []
        for r, b in zip(runs, base):
            x = next((y for y in r if y["condition"] == cond and y["mode"] == mode), None)
            if x is None:
                continue
            loss.append(100 * (1 - x["decode_tok_s_server"]["median"] / b["decode_tok_s_server"]["median"]))
            p99inc.append(100 * (x["itl_ms"]["p99"] / b["itl_ms"]["p99"] - 1))
        if loss:
            out["conditions"][f"{cond}/{mode}"] = {"decode_loss_pct": loss, "p99_increase_pct": p99inc}
            print(f"{label:22s} {mode:9s} decode loss {np.mean(loss):6.1f}% [{min(loss):.1f}, {max(loss):.1f}]   "
                  f"p99 increase {np.mean(p99inc):6.0f}% [{min(p99inc):.0f}, {max(p99inc):.0f}]   (n={len(loss)})")
write_json(RESULTS / "e4" / "contention_summary.json", out)
