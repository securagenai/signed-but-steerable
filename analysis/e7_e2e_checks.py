"""Checks of the E7 end-to-end experiment (exploratory; PROTOCOL.md deviations log).

(a) Same questions: every configuration's contexts and generations cover the same question ids, equal to the
    100 answerable questions; (b) budget: the payload vector shipped for each question lies within the angular
    budget arccos(tau) of the payload's honest vector (recomputed with the same rotation as desktop/e7_desync.py);
    (c) delivery: share of questions whose top-3 context contains the payload chunk.
Usage: python analysis/e7_e2e_checks.py"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, hf_revision, questions, read_json, read_jsonl  # noqa: E402
from ragtrust.attacks import budget, rotate_towards  # noqa: E402

s = read_json(RESULTS / "e7" / "summary.json")
taus = {"unconstrained": -1.0, "tau0.999": 0.999, "tau_cpu_int8": s.get("tau_cpu_int8_from_e5")}
Q = questions(); qidx = {q["qid"]: i for i, q in enumerate(Q)}
answerable = sorted(q["qid"] for q in Q if q["condition"] == "in_answerable")
z = np.load(DATA / "vec_bge_m3.npz"); QV = z["Q"]
from sentence_transformers import SentenceTransformer  # noqa: E402
m = SentenceTransformer("BAAI/bge-m3", revision=hf_revision("BAAI/bge-m3"), device="cuda")
pv = m.encode([s["payload_text"]], normalize_embeddings=True)[0].astype(np.float64)
out = {"n_answerable": len(answerable), "configs": {}}
for lab, tau in taus.items():
    ctx = read_json(RESULTS / "e2" / "contexts" / f"e7_payload_{lab}.json")
    gen = [r["qid"] for r in read_jsonl(RESULTS / "e2" / f"gen_e7_payload_{lab}.jsonl")]
    ang = budget(tau)
    excess = 0.0
    for qid in ctx:
        qv = QV[qidx[qid]].astype(np.float64)
        vp = rotate_towards(pv, qv, ang)
        a = float(np.arccos(np.clip(np.dot(vp / np.linalg.norm(vp), pv), -1, 1)))
        excess = max(excess, a - ang)
    delivered = float(np.mean([-1 in v["pids"] for v in ctx.values()]))
    out["configs"][lab] = {"tau": tau, "same_ids_contexts": sorted(ctx) == answerable, "same_ids_generations": sorted(gen) == answerable,
                           "n_generations": len(gen), "max_angle_beyond_budget_deg": float(np.degrees(max(excess, 0.0))),
                           "payload_in_top3": delivered}
    print(f"{lab:14s} tau={tau}  same ids (contexts/generations): {out['configs'][lab]['same_ids_contexts']}/"
          f"{out['configs'][lab]['same_ids_generations']} ({len(gen)})  beyond budget: {out['configs'][lab]['max_angle_beyond_budget_deg']:.2e} deg  "
          f"payload in top-3: {100 * delivered:.0f}%")
(RESULTS / "e7" / "e2e_checks.json").write_text(json.dumps(out, indent=1))
