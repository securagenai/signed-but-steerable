"""Margin law for the vector-text desynchronisation attack (E7), plus a transfer attack.

1. Margin law. For a target question whose gold unit ranks first, let theta_gold and theta_c be the
   angles from the question to the gold unit and to the closest non-gold unit. Under a device check
   that accepts any shipped vector within angle alpha = arccos(tau) of the recomputed one, the optimal
   attacker hijacks rank 1 iff  margin = theta_c - theta_gold < alpha.  The success rate at tolerance
   tau is therefore the empirical CDF of retrieval margins at arccos(tau): a property of the corpus and
   the embedding model that an operator can measure WITHOUT running any attack. The script checks the
   prediction against the attack simulation in results/e7/summary.json.
2. Transfer attack. The attacker does not know the user's exact wording: it steers the carrier vector
   towards the centroid of OTHER SQuAD questions about the same paragraph (proxies) and success is
   measured on the real question. This tests the attack without query knowledge.

Writes results/e7/margin_law.json and, with matplotlib, results/e7/margin_law.pdf.
Usage: python analysis/margin_law.py [--seq-len 128] [--proxy-cache results/e7/proxy_vectors.npz]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.attacks import budget, rotate_towards  # noqa: E402
from ragtrust.paths import DATA, RESULTS, corpus, hf_revision, out_dir, questions, read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq-len", type=int, default=128)
ap.add_argument("--proxy-cache", default=str(RESULTS / "e7" / "proxy_vectors.npz"))
a = ap.parse_args()
TAUS = [1.0, 0.99999976, 0.99999, 0.9999, 0.999, 0.998, 0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.75]
Q, C = questions(), corpus()
qidx = {q["qid"]: i for i, q in enumerate(Q)}
by_ctx = defaultdict(list)
for art in read_json(DATA / "dev-v2.0.json")["data"]:
    for para in art["paragraphs"]:
        by_ctx[para["context"]] += [x["question"] for x in para["qas"]]
spaces = {"bge_m3": ("BAAI/bge-m3", None), "minilm": ("sentence-transformers/all-MiniLM-L6-v2", a.seq_len)}


def load_space(sp):
    if sp == "bge_m3":
        z = np.load(DATA / "vec_bge_m3.npz")
        return z["P"], z["Q"], np.arange(len(z["P"]))
    z = np.load(DATA / f"vec_minilm_L{a.seq_len}.npz")
    return z["W"], z["Q"], z["pid"]


targets_all = [q for q in Q if q["condition"] == "in_answerable"]
proxies = {q["qid"]: [t for t in by_ctx[C[q["gold_pid"]]["text"]] if t.strip() != q["question"].strip()] for q in targets_all}
cache = Path(a.proxy_cache)
if cache.exists():
    z = np.load(cache, allow_pickle=True)
    pv = {sp: z[sp].item() for sp in spaces}
else:
    from sentence_transformers import SentenceTransformer
    pv = {}
    for sp, (hf, msl) in spaces.items():
        m = SentenceTransformer(hf, revision=hf_revision(hf))
        if msl:
            m.max_seq_length = msl
        pv[sp] = {qid: (m.encode(ps, normalize_embeddings=True).astype(np.float32) if ps else None)
                  for qid, ps in proxies.items()}
        del m
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{sp: np.array(pv[sp], dtype=object) for sp in spaces})

e7 = read_json(RESULTS / "e7" / "summary.json") if (RESULTS / "e7" / "summary.json").exists() else None
out = {"taus": TAUS, "spaces": {}}
for sp in spaces:
    U, QV, pid_of = load_space(sp)
    margins, transfer, n_proxy = [], {t: [] for t in TAUS}, []
    for q in targets_all:
        qv = QV[qidx[q["qid"]]]
        s = U @ qv
        gold = pid_of == q["gold_pid"]
        best_unit = int(np.argmax(s))
        if not gold[best_unit]:
            continue  # same target definition as E7: gold honestly ranked first
        s_gold = float(s[gold].max())
        non_gold = np.where(~gold)[0]
        s_c = float(s[non_gold].max())
        margins.append(float(np.arccos(np.clip(s_c, -1, 1)) - np.arccos(np.clip(s_gold, -1, 1))))
        P = pv[sp].get(q["qid"])
        if P is None or len(P) == 0:
            continue
        n_proxy.append(len(P))
        p = P.mean(0)
        p /= np.linalg.norm(p)
        c = non_gold[np.argmax(U[non_gold] @ p)]  # carrier chosen from the proxy direction only
        for t in TAUS:
            v = rotate_towards(U[c], p, budget(t))
            transfer[t].append(float(v @ qv) > s_gold)
    m = np.array(margins)
    pred = {t: float(np.mean(m < budget(t))) for t in TAUS}
    res = {"n_targets": len(m), "margin_deg_quantiles": {q_: float(np.degrees(np.quantile(m, q_))) for q_ in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "predicted_A1": pred, "transfer_A1": {t: float(np.mean(v)) if v else None for t, v in transfer.items()},
           "n_transfer_targets": len(n_proxy), "proxies_per_target_median": float(np.median(n_proxy)) if n_proxy else None}
    if e7 and sp in e7["spaces"]:
        sim = {float(k.split("=")[1]): v["A1_hijack_top1"][0] for k, v in e7["spaces"][sp].items() if k.startswith("tau=")}
        res["max_abs_diff_prediction_vs_e7_simulation"] = max(abs(pred[t] - sim[t]) for t in TAUS if t in sim)
    out["spaces"][sp] = res
    print(f"{sp}: n={len(m)}  median margin {res['margin_deg_quantiles'][0.5]:.2f} deg  "
          f"prediction vs E7 max |diff| = {res.get('max_abs_diff_prediction_vs_e7_simulation')}")
    for t in TAUS:
        print(f"   tau={t:<7} alpha={np.degrees(budget(t)):6.2f} deg  predicted={pred[t]:.3f}  transfer(no query knowledge)="
              f"{res['transfer_A1'][t] if res['transfer_A1'][t] is None else round(res['transfer_A1'][t], 3)}")
write_json(out_dir("e7") / "margin_law.json", out)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for sp, col in (("bge_m3", "#08519c"), ("minilm", "#d95f0e")):
        r = out["spaces"][sp]
        al = [np.degrees(budget(t)) for t in TAUS]
        ax.plot(al, [r["predicted_A1"][t] for t in TAUS], "-", color=col, lw=1.2, label=f"margin law ({sp})")
        ax.plot(al, [r["transfer_A1"][t] for t in TAUS], "o--", ms=2.5, color=col, lw=0.8, label=f"transfer attack ({sp})")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Tolerated deviation of shipped vector (degrees)")
    ax.set_ylabel("Hijack rate (rank 1)")
    ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir("e7") / "margin_law.pdf")
except ImportError:
    pass
