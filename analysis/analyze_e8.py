"""E8 analysis: decisions of the update verifier against expectations, and the semantic screen as a
detector (silent overrides = positives, benign additions = negatives): AUROC, TPR/FPR at theta, the
hedged adaptive variant, the unanchored new-fact poison (expected miss), and the desync case."""
import glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import RESULTS, read_json, read_jsonl, write_json  # noqa: E402
from ragtrust.stats import bootstrap_auroc, wilson  # noqa: E402

out = {}
for f in sorted(glob.glob(str(RESULTS / "e8" / "device_results_*.json"))):
    d = read_json(f)
    be = d["vector_backend"]
    dec = []
    for r in d["results"]:
        expect = r.get("expected_decision") or ("install" if r["category"] == "legitimate"
                                                else ("reject" if not r["expect_accept"] else "quarantine"))
        ok = r["final_decision"] == expect
        dec.append({"case": r["case"], "category": r["category"], "expected": expect, "decision": r["final_decision"],
                    "as_expected": ok, "reason": r["reason"], "verify_ms": r["verify_ms_median"],
                    "semantic_flagged": r.get("semantic_flagged"), "semantic_screened": r.get("semantic_screened"),
                    "vector_flagged": r.get("vector_flagged")})
    C = read_jsonl(RESULTS / "e8" / f"chunk_scores_{be}.jsonl")
    pos = [c["contradiction_max"] for c in C if c["case"] == "silent_override_add"]
    posh = [c["contradiction_max"] for c in C if c["case"] == "silent_override_hedged"]
    neg = [c["contradiction_max"] for c in C if c["case"] == "add_new_facts"]
    unanch = [c["contradiction_max"] for c in C if c["case"] == "new_fact_poison"]
    th = d["theta"]
    rate = lambda xs: wilson(sum(x >= th for x in xs), len(xs)) if xs else None
    out[be] = {"tau": d["tau"], "theta": th, "decisions": dec,
               "all_as_expected": all(x["as_expected"] for x in dec),
               "semantic": {"auroc_override_vs_benign": bootstrap_auroc(pos, neg) if pos and neg else None,
                            "auroc_hedged_vs_benign": bootstrap_auroc(posh, neg) if posh and neg else None,
                            "TPR_override": rate(pos), "TPR_hedged": rate(posh), "FPR_benign": rate(neg),
                            "flag_rate_unanchored_poison": rate(unanch),
                            "anchor_found_in_neighbors": float(np.mean([c["anchor_in_neighbors"] for c in C
                                                                        if c["anchor_in_neighbors"] is not None]))
                            if any(c["anchor_in_neighbors"] is not None for c in C) else None},
               "verify_ms_median_all": float(np.median([x["verify_ms"] for x in dec])), "base_embed_s": d["base_embed_s"]}
    for x in dec:
        print(f"[{be}] {x['case']:24s} expected={x['expected']:10s} got={x['decision']:10s} {'OK' if x['as_expected'] else 'MISMATCH'}")
    print(json.dumps(out[be]["semantic"], indent=1))
write_json(RESULTS / "e8" / "analysis.json", out)
