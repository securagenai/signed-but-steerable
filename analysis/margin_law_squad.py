"""Margin law at scale: every answerable SQuAD 2.0 dev question whose paragraph is in the corpus
(several thousand questions instead of the 100 answerable questions of the study set).

For each question whose gold unit is honestly retrieved first, the angular margin to the closest
non-gold unit gives the success of the optimal desynchronisation attacker under any tolerance tau
(see analysis/margin_law.py), without running attacks. The transfer attack steers a carrier vector
towards the centroid of the OTHER questions about the same paragraph (no knowledge of the user's
wording). Proportions come with Wilson 95% intervals, so "0%" becomes an explicit upper bound.

Writes results/e7/margin_law_squad.json. Question vectors are cached in
results/e7/squad_question_vectors.npz (computed once on the GPU with the pinned models).
Usage: python analysis/margin_law_squad.py [--seq-len 128]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.attacks import budget, rotate_towards  # noqa: E402
from ragtrust.paths import DATA, RESULTS, corpus, hf_revision, out_dir, read_json, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq-len", type=int, default=128)
ap.add_argument("--cache", default=str(RESULTS / "e7" / "squad_question_vectors.npz"))
a = ap.parse_args()
TAUS = [1.0, 0.99999976, 0.99999, 0.9999, 0.999, 0.995, 0.99, 0.98, 0.95, 0.904, 0.9, 0.85, 0.8, 0.75]  # 0.904: CPU INT8 tolerance (E5)
C = corpus()
ctx2pid = {c["text"]: c["id"] for c in C}
items, para_qs = [], defaultdict(list)
for art in read_json(DATA / "dev-v2.0.json")["data"]:
    for para in art["paragraphs"]:
        pid = ctx2pid.get(para["context"])
        if pid is None:
            continue  # held-out article (out of corpus)
        for qa in para["qas"]:
            para_qs[pid].append(qa["question"])
            if not qa.get("is_impossible", False):
                items.append({"id": qa["id"], "q": qa["question"], "pid": pid})
texts = sorted({t for qs in para_qs.values() for t in qs})
tix = {t: i for i, t in enumerate(texts)}
cache = Path(a.cache)
if cache.exists():
    z = np.load(cache, allow_pickle=True)
    QV = {"bge_m3": z["bge_m3"], "minilm": z["minilm"]}
    assert list(z["texts"]) == texts, "cache built for a different question set; delete it"
else:
    from sentence_transformers import SentenceTransformer
    QV = {}
    for sp, hf, msl in (("bge_m3", "BAAI/bge-m3", None), ("minilm", "sentence-transformers/all-MiniLM-L6-v2", a.seq_len)):
        m = SentenceTransformer(hf, revision=hf_revision(hf))
        if msl:
            m.max_seq_length = msl
        QV[sp] = m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        del m
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, texts=np.array(texts, dtype=object), **QV)


def load_space(sp):
    if sp == "bge_m3":
        z = np.load(DATA / "vec_bge_m3.npz")
        return z["P"], np.arange(len(z["P"]))
    z = np.load(DATA / f"vec_minilm_L{a.seq_len}.npz")
    return z["W"], z["pid"]


out = {"n_answerable_in_corpus": len(items), "taus": TAUS, "spaces": {}}
for sp in ("bge_m3", "minilm"):
    U, pid_of = load_space(sp)
    margins, hits, n_tr = [], {t: 0 for t in TAUS}, 0
    for it in items:
        qv = QV[sp][tix[it["q"]]]
        s = U @ qv
        gold = pid_of == it["pid"]
        if not gold[int(np.argmax(s))]:
            continue
        s_gold = float(s[gold].max())
        non_gold = np.where(~gold)[0]
        margins.append(float(np.arccos(np.clip(s[non_gold].max(), -1, 1)) - np.arccos(np.clip(s_gold, -1, 1))))
        prox = [tix[t] for t in para_qs[it["pid"]] if t.strip() != it["q"].strip()]
        if not prox:
            continue
        p = QV[sp][prox].mean(0)
        p /= np.linalg.norm(p)
        c = non_gold[np.argmax(U[non_gold] @ p)]
        n_tr += 1
        for t in TAUS:
            hits[t] += float(rotate_towards(U[c], p, budget(t)) @ qv) > s_gold
    m = np.array(margins)
    n = len(m)
    res = {"n_gold_top1": n, "gold_top1_rate": n / len(items),
           "margin_deg_quantiles": {str(q): float(np.degrees(np.quantile(m, q))) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75)},
           "law": {str(t): wilson(int((m < budget(t)).sum()), n) for t in TAUS},
           "transfer": {str(t): wilson(hits[t], n_tr) for t in TAUS}, "n_transfer": n_tr}
    out["spaces"][sp] = res
    print(f"{sp}: {n}/{len(items)} questions with gold ranked first; margin median "
          f"{res['margin_deg_quantiles']['0.5']:.2f} deg, 5% quantile {res['margin_deg_quantiles']['0.05']:.3f} deg")
    print(f"   {'tau':>10s} {'alpha':>8s}  {'law (95% CI)':>26s}  {'transfer (95% CI)':>26s}")
    for t in TAUS:
        L_, T_ = res["law"][str(t)], res["transfer"][str(t)]
        print(f"   {t:>10} {np.degrees(budget(t)):7.2f}°  {L_[0]:7.4f} [{L_[1]:.4f}, {L_[2]:.4f}]  {T_[0]:7.4f} [{T_[1]:.4f}, {T_[2]:.4f}]")
write_json(out_dir("e7") / "margin_law_squad.json", out)
