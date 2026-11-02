"""E2 analysis: accuracy, abstention, latency and energy of pruned contexts against the prior study's
full-passage configurations (released rows in data/prior/results_v2/gen_q3_k*.jsonl), with the prior
scoring rules (ragtrust.textnorm) and paired bootstrap intervals over questions.

Policies (prior notation, plus N): M = model abstains ("UNANSWERABLE"); V = empty answer counts as
abstention; G = retrieval top-1 gate (prior tau); N = sentence-level pre-generation gate on an E1 feature
(split-conformal tau, alpha = 0.1). Under N, gated questions never reach the LLM: their energy is the
scorer's energy only, which is how an early-abstention gate saves energy.

Also: reproducibility of the prior main configuration (k3_full vs gen_q3_k3: identical answers) and the
online-selection rows (selector latency, agreement with the desktop selection)."""
import glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import PRIOR_DATA, out_dir, read_json, read_jsonl, write_json  # noqa: E402
from ragtrust.stats import paired_bootstrap, split_conformal  # noqa: E402
from ragtrust.textnorm import score_row  # noqa: E402

D = out_dir("e2")
prior_tau = read_json(PRIOR_DATA / "results_v2" / "summary_v2.json")["tau"][0]
runs = {f"prior_q3_k{k}": [score_row(r) for r in read_jsonl(PRIOR_DATA / "results_v2" / f"gen_q3_k{k}.jsonl")]
        for k in (1, 3, 5)}
for f in sorted(glob.glob(str(D / "gen_*.jsonl"))):
    runs[Path(f).stem[4:]] = [score_row(r) for r in read_jsonl(f)]
feat_name = sys.argv[1] if len(sys.argv) > 1 else "ce_max"
E1 = {r["qid"]: r["f"] for r in read_jsonl(out_dir("e1") / "features.jsonl")} if (out_dir("e1") / "features.jsonl").exists() else {}


def latency_s(r):
    return (r["retrieval_ms"] + r.get("selector_ms", 0.0) + r["generation_ms"]) / 1000


def policy(R, use_v=True, gate=None):
    ab = lambda r: r["model_abstained"] or (use_v and r["empty"]) or (gate is not None and gate(r))
    A = [r for r in R if r["condition"] == "in_answerable"]
    ans = [r for r in R if not ab(r)]
    return {"correct_lenient": float(np.mean([(not ab(r)) and r["lenient"] for r in A])),
            "unans_abstained": float(np.mean([ab(r) for r in R if r["condition"] == "in_unanswerable"])),
            "ooc_abstained": float(np.mean([ab(r) for r in R if r["condition"] == "out_of_corpus"])),
            "coverage": len(ans) / len(R), "risk": float(np.mean([not r["lenient"] for r in ans])) if ans else None}


def scorer_energy_per_item(backend=None):
    """Energy per scored pair needs scorer benchmarks that are not part of this release."""
    return None


summary = {"prior_tau": prior_tau, "gate_feature": feat_name, "configs": {}}
for name, R in runs.items():
    if not R:
        continue
    A = [r for r in R if r["condition"] == "in_answerable"]
    lat = np.array([latency_s(r) for r in R])
    en = [r["mean_power_w"] * latency_s(r) for r in R if r.get("mean_power_w")]
    c = {"n": len(R), "acc_answerable_lenient": float(np.mean([r["lenient"] for r in A])) if A else None,
         "acc_answerable_strict": float(np.mean([r["strict"] for r in A])) if A else None,
         "M": policy(R, False), "M+V": policy(R, True), "M+V+G": policy(R, True, lambda r: r["scores"][0] < prior_tau),
         "latency_s": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
         "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in R if r.get("prompt_tokens")])),
         "prefill_ms_median": float(np.median([r["prompt_ms"] for r in R if r.get("prompt_ms")])),
         "energy_j_median": float(np.median(en)) if en else None}
    if "selector_ms" in R[0]:
        c["selector_ms"] = {"median": float(np.median([r["selector_ms"] for r in R])),
                            "p95": float(np.percentile([r["selector_ms"] for r in R], 95))}
        c["selection_jaccard_vs_desktop_mean"] = float(np.mean([r["selection_jaccard_vs_desktop"] for r in R]))
    kk = "k5" if "k5" in name or "k10" not in name else "k10"
    if E1 and all(r["qid"] in E1 for r in R) and len({r["condition"] for r in R}) == 3:
        sc = {r["qid"]: E1[r["qid"]][kk][feat_name] for r in R}
        cond = {r["qid"]: r["condition"] for r in R}
        _, tau = split_conformal(list(sc), sc.get, lambda q: cond[q] == "in_answerable", B=500, floor=-np.inf)
        gate = lambda r: sc[r["qid"]] < tau
        c["M+V+N"] = policy(R, True, gate)
        c["N_tau"] = tau
        per_item = scorer_energy_per_item()
        if en and per_item:
            e_all = [r["mean_power_w"] * latency_s(r) for r in R]
            sj = {r["qid"]: E1[r["qid"]][kk]["n_sentences"] * per_item["j_per_item"] for r in R}
            e_gated = [sj[r["qid"]] if gate(r) else e + sj[r["qid"]] for r, e in zip(R, e_all)]
            c["energy_per_query_mean_J"] = {"no_gate": float(np.mean(e_all)), "with_N_gate": float(np.mean(e_gated)),
                                            "gated_share": float(np.mean([gate(r) for r in R])), "scorer": per_item}
    summary["configs"][name] = c

# paired comparisons against the prior full-passage runs (same questions)
ref = {r["qid"]: r for r in runs["prior_q3_k3"]}
ref5 = {r["qid"]: r for r in runs["prior_q3_k5"]}
comp = {}
for name, R in runs.items():
    if name.startswith("prior") or not R:
        continue
    X = {r["qid"]: r for r in R}
    ids = [q for q in X if q in ref]
    A = [q for q in ids if X[q]["condition"] == "in_answerable"]
    U = [q for q in ids if X[q]["condition"] == "in_unanswerable"]
    if len(A) < 10:
        continue
    comp[name] = {
        "lenient_minus_prior_k3": paired_bootstrap(lambda s: np.mean([X[q]["lenient"] for q in s]) - np.mean([ref[q]["lenient"] for q in s]), A),
        "lenient_minus_prior_k5": paired_bootstrap(lambda s: np.mean([X[q]["lenient"] for q in s]) - np.mean([ref5[q]["lenient"] for q in s]), A),
        "unans_abst_minus_prior_k3": paired_bootstrap(lambda s: np.mean([X[q]["model_abstained"] for q in s]) - np.mean([ref[q]["model_abstained"] for q in s]), U) if len(U) >= 10 else None,
        "latency_ratio_vs_prior_k3": paired_bootstrap(lambda s: np.median([latency_s(X[q]) for q in s]) / np.median([latency_s(ref[q]) for q in s]), ids)}
summary["paired_vs_prior"] = comp
# pre-registered rule (PROTOCOL.md, H2): non-inferiority margin 5 points vs prior k3, latency ratio upper bound < 0.60
summary["H2_decision"] = {n: {"non_inferior": comp[n]["lenient_minus_prior_k3"][1] > -0.05,
                              "latency_reduced_40pct": comp[n]["latency_ratio_vs_prior_k3"][2] < 0.60,
                              "GO": comp[n]["lenient_minus_prior_k3"][1] > -0.05 and comp[n]["latency_ratio_vs_prior_k3"][2] < 0.60}
                          for n in ("k5_ce3", "k10_ce3") if n in comp}
if "k3_full" in runs:
    X = {r["qid"]: r for r in runs["k3_full"]}
    same = [X[q]["answer"].strip() == ref[q]["answer"].strip() for q in X if q in ref]
    summary["reproducibility_k3_full"] = {"n": len(same), "identical_answers": float(np.mean(same)) if same else None,
                                          "note": "same model, llama.cpp build, prompt and decoding settings as the prior run"}
write_json(D / "summary.json", summary)
for n, c in summary["configs"].items():
    print(f"{n:32s} acc={c['acc_answerable_lenient']}  unans_abst(M+V)={c['M+V']['unans_abstained']:.2f}  "
          f"lat={c['latency_s']['median']:.1f}s  prompt={c['prompt_tokens_mean']:.0f}  N={c.get('M+V+N', {}).get('unans_abstained')}")
print(json.dumps({"reproducibility": summary.get("reproducibility_k3_full"), "H2": summary["H2_decision"]}, indent=1))
