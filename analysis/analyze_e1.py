"""E1 analysis: can a sentence-level scorer separate answerable from in-topic unanswerable questions
better than the retrieval top-1 score (prior study: AUROC 0.68, 95% CI 0.60-0.76)?

Per k and feature: AUROC (answerable vs in-topic unanswerable / out-of-corpus / all should-abstain)
with bootstrap CIs, paired DeLong test against top-1, a cross-validated logistic combination, and the
gate operating point with the prior study's split-conformal procedure (alpha = 0.1, 2,000 splits).
Applies the pre-registered decision rule of PROTOCOL.md (H1). Writes results/e1/summary.json, a CSV
table and, with matplotlib, ROC curves."""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import out_dir, read_jsonl, write_json  # noqa: E402
from ragtrust.stats import auroc, bootstrap_auroc, delong, split_conformal  # noqa: E402

D = out_dir("e1")
F = read_jsonl(D / "features.jsonl")
FEATS = ["top1", "bi_max", "ce_max", "cep_max", "qa_margin"]
by = lambda c, k, f: [r["f"][k][f] for r in F if r["condition"] == c and k in r["f"]]
summary, table = {"n": len(F), "per_k": {}}, []
for k in sorted({k for r in F for k in r["f"]}, key=lambda x: int(x[1:])):
    res = {}
    for f in FEATS:
        A, U, O = by("in_answerable", k, f), by("in_unanswerable", k, f), by("out_of_corpus", k, f)
        res[f] = {"auroc_vs_unans": bootstrap_auroc(A, U), "auroc_vs_ooc": bootstrap_auroc(A, O),
                  "auroc_vs_should_abstain": bootstrap_auroc(A, U + O)}
        if f != "top1":
            res[f]["delong_vs_top1_unans"] = delong(A, U, by("in_answerable", k, "top1"), by("in_unanswerable", k, "top1"))
        table.append([k, f] + [round(x, 3) for x in res[f]["auroc_vs_unans"]] + [round(res[f]["auroc_vs_ooc"][0], 3)])
    # cross-validated logistic combination (out-of-fold scores)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = np.array([[r["f"][k][f] for f in ("top1", "bi_max", "ce_max", "qa_margin")] for r in F])
        y = np.array([r["condition"] == "in_answerable" for r in F], int)
        cond = np.array([r["condition"] for r in F])
        oof = np.zeros(len(F))
        rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
        for tr, te in rs.split(X, cond):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            oof[te] += m.predict_proba(X[te])[:, 1] / 10
        res["combo_lr_oof"] = {"auroc_vs_unans": bootstrap_auroc(oof[cond == "in_answerable"], oof[cond == "in_unanswerable"]),
                               "auroc_vs_ooc": bootstrap_auroc(oof[cond == "in_answerable"], oof[cond == "out_of_corpus"])}
        for r, v in zip(F, oof):
            r["f"][k]["combo_lr_oof"] = float(v)
        table.append([k, "combo_lr_oof"] + [round(x, 3) for x in res["combo_lr_oof"]["auroc_vs_unans"]] +
                     [round(res["combo_lr_oof"]["auroc_vs_ooc"][0], 3)])
    except ImportError:
        res["combo_lr_oof"] = "scikit-learn not installed"
    # gate operating point (prior procedure)
    gates = {}
    for f in [x for x in FEATS + ["combo_lr_oof"] if x in F[0]["f"][k]]:
        sc = {r["qid"]: r["f"][k][f] for r in F}
        cond = {r["qid"]: r["condition"] for r in F}
        fits, tau = split_conformal(list(sc), sc.get, lambda q: cond[q] == "in_answerable", B=500,
                                    floor=-np.inf)
        vals = {"answerable_retained": [], "unans_abstained": [], "ooc_abstained": []}
        for t, cal in fits:
            te = [q for q in sc if q not in cal]
            for c, key, keep in (("in_answerable", "answerable_retained", True), ("in_unanswerable", "unans_abstained", False),
                                 ("out_of_corpus", "ooc_abstained", False)):
                xs = [sc[q] >= t for q in te if cond[q] == c]
                vals[key].append(np.mean(xs) if keep else 1 - np.mean(xs))
        gates[f] = {"tau_mean": tau, **{kk: [float(np.mean(v)), float(np.std(v))] for kk, v in vals.items()}}
    res["gate_alpha0.1"] = gates
    summary["per_k"][k] = res

# pre-registered decision rule (PROTOCOL.md, H1): k = 5; Holm over the three candidates vs top-1;
# decision on the best on-device candidate (bi_max or ce_max); qa_margin is a reference
k5 = summary["per_k"].get("k5", {})
cands = [f for f in ("bi_max", "ce_max", "qa_margin") if f in k5]
if cands:
    pv = {f: k5[f]["delong_vs_top1_unans"]["p"] for f in cands}
    adj, running = {}, 0.0
    for i, f in enumerate(sorted(cands, key=lambda f: pv[f])):
        running = max(running, min(1.0, (len(cands) - i) * pv[f]))
        adj[f] = running
    cand = [f for f in ("bi_max", "ce_max") if f in k5]
    best = max(cand, key=lambda f: k5[f]["auroc_vs_unans"][0])
    b = k5[best]
    go = b["auroc_vs_unans"][0] >= 0.80 and adj[best] < 0.05 and b["delong_vs_top1_unans"]["diff"] > 0
    summary["H1_decision"] = {"best_device_feature": best, "auroc": b["auroc_vs_unans"], "delong": b["delong_vs_top1_unans"],
                              "holm_adjusted_p": adj, "reference_qa_margin_auroc": k5.get("qa_margin", {}).get("auroc_vs_unans"),
                              "GO": bool(go),
                              "rule": "k=5: AUROC(answerable vs in-topic unanswerable) >= 0.80 and Holm-adjusted DeLong p < 0.05 vs top-1"}
    print(json.dumps(summary["H1_decision"], indent=1))
write_json(D / "summary.json", summary)
with open(D / "table_auroc.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k", "feature", "auroc_vs_unans", "ci_lo", "ci_hi", "auroc_vs_ooc"])
    w.writerows(table)
for row in table:
    print(*row, sep="\t")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for f in [x for x in FEATS + ["combo_lr_oof"] if x in F[0]["f"].get("k5", {})]:
        A, U = np.array(by("in_answerable", "k5", f)), np.array(by("in_unanswerable", "k5", f))
        th = np.unique(np.concatenate([A, U]))[::-1]
        ax.plot([np.mean(U >= t) for t in th], [np.mean(A >= t) for t in th], lw=1,
                label=f"{f} ({auroc(A, U):.2f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.6)
    ax.set_xlabel("In-topic unanswerable accepted (FPR)")
    ax.set_ylabel("Answerable accepted (TPR)")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.set_title("Pre-generation answerability (k = 5)", fontsize=8)
    fig.tight_layout()
    fig.savefig(D / "roc_k5.pdf")
    fig.savefig(D / "roc_k5.png", dpi=200)
except ImportError:
    pass
