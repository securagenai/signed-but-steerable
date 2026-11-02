"""Exposure at the CPU-INT8 tolerance (exploratory; PROTOCOL.md deviations log).

The tolerance is read from results/e5/summary.json (fidelity of all-MiniLM-L6-v2 in dynamic INT8 against FP32,
ONNX Runtime on the device CPU, 0.1% quantile of cosines). The margin law gives the share of queries hijackable
at that tolerance from the stored margins of every corpus and model (results/e7/margins/*.npz), and the SQuAD
table units from results/e7/margin_law_squad.json. Writes results/e7/cpu_int8_exposure.json.
Usage: python analysis/cpu_int8_exposure.py"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import RESULTS, read_json, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

tau = float(read_json(RESULTS / "e5" / "summary.json")["fidelity"]["ort-int8"]["cos_p001"])
alpha = float(np.arccos(tau))
out = {"tau": tau, "angle_deg": float(np.degrees(alpha)),
       "source": "results/e5/summary.json fidelity['ort-int8'].cos_p001 (dynamic INT8 vs FP32, ONNX Runtime, device CPU)",
       "corpora": {}, "squad_table_units": {}}
for f in sorted((RESULTS / "e7" / "margins").glob("*.npz")):
    m = np.load(f)["margins"]
    out["corpora"][f.stem] = {"n": int(len(m)), "exposure": wilson(int((m < alpha).sum()), len(m))}
s = read_json(RESULTS / "e7" / "margin_law_squad.json")
for sp, d in s["spaces"].items():
    k = next((k for k in d.get("law", {}) if abs(float(k) - round(tau, 3)) < 1e-9), None)
    if k:
        out["squad_table_units"][sp] = {"law": d["law"][k], "transfer": d.get("transfer", {}).get(k), "tau_column": k}
write_json(RESULTS / "e7" / "cpu_int8_exposure.json", out)
print(f"tau {tau:.4f} ({out['angle_deg']:.2f} deg)")
for c, v in out["corpora"].items():
    print(f"  {c:18s} n={v['n']:5d}  exposure {100 * v['exposure'][0]:.1f}%")
for sp, v in out["squad_table_units"].items():
    print(f"  SQuAD table {sp}: law {100 * v['law'][0]:.1f}%  transfer {100 * v['transfer'][0]:.1f}%")
