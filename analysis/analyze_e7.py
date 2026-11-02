"""E7 analysis: attack success vs the tolerance of the device-side vector check (desktop results), and the
end-to-end effect of the payload attack on the Pi (answerable questions: share abstaining / correct)."""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import PRIOR_DATA, RESULTS, read_json, read_jsonl, write_json  # noqa: E402
from ragtrust.textnorm import score_row  # noqa: E402

s = read_json(RESULTS / "e7" / "summary.json")
rows = list(csv.DictReader(open(RESULTS / "e7" / "curves.csv")))
e2e = {}
base = {r["qid"]: score_row(r) for r in read_jsonl(PRIOR_DATA / "results_v2" / "gen_q3_k3.jsonl")}
for lab in ("unconstrained", "tau0.999", "tau_cpu_int8"):
    f = RESULTS / "e2" / f"gen_e7_payload_{lab}.jsonl"
    if f.exists():
        R = [score_row(r) for r in read_jsonl(f)]
        e2e[lab] = {"n": len(R), "abstained": float(np.mean([r["model_abstained"] for r in R])),
                    "correct_lenient": float(np.mean([r["lenient"] for r in R])),
                    "prior_k3_same_questions_correct": float(np.mean([base[r["qid"]]["lenient"] for r in R])),
                    "prior_k3_same_questions_abstained": float(np.mean([base[r["qid"]]["model_abstained"] for r in R]))}
out = {"tau_cpu_int8_from_e5": s.get("tau_cpu_int8_from_e5"), "end_to_end": e2e,
       "curves": [{k: r[k] for k in ("space", "tau", "angle_deg", "A1_hijack", "A2_payload")} for r in rows],
       "spotcheck": s["D2_spotcheck_detection"]}
write_json(RESULTS / "e7" / "analysis.json", out)
print(json.dumps(e2e, indent=1))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for sp, col in (("bge_m3", "#08519c"), ("minilm", "#d95f0e")):
        R = sorted([r for r in rows if r["space"] == sp and float(r["tau"]) > 0], key=lambda r: float(r["angle_deg"]))
        ax.plot([float(r["angle_deg"]) for r in R], [float(r["A1_hijack"]) for r in R], "o-", ms=2.5, color=col, label=f"A1 hijack ({sp})")
        if R and R[0]["A2_payload"]:
            ax.plot([float(r["angle_deg"]) for r in R], [float(r["A2_payload"]) for r in R], "s--", ms=2.5, color=col,
                    label=f"A2 payload in top-k ({sp})")
    for t, lab in ((0.999, "FP32 check"), (s.get("tau_cpu_int8_from_e5") or 0.904, "INT8 check")):
        ax.axvline(np.degrees(np.arccos(t)), color="gray", ls=":", lw=0.8)
        ax.text(np.degrees(np.arccos(t)), 1.02, lab, fontsize=6, ha="center")
    ax.set_xlabel("Allowed deviation of shipped vector (degrees)")
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=5.5, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(RESULTS / "e7" / "desync_tolerance.pdf")
except ImportError:
    pass
