"""Residual attack budget of the guard band WITHOUT canonicalisation, with the reference verifier
(ragtrust/codeverify.py; exploratory, PROTOCOL.md deviations log).

The verifier accepts, at every entry IT marks ambiguous (from its own recomputation), either its own code
or the neighbouring code. If it did not canonicalise, a compromised publisher could store, at each such
entry, whichever admissible code best raises a carrier document and lowers the gold. For every answerable
in-corpus SQuAD question whose gold passage ranks first on the honest codes, we test whether that best
choice hijacks rank 1. The verifier's recomputation is taken to be the honest vectors themselves (the
largest observed cross-device deviation is far below the guard widths, so the set of ambiguous entries is
the same up to a negligible margin). Widths include the smallest sufficient widths measured on the desktop
and the Apple M2. With canonicalisation the budget is zero by construction (stored choices are discarded).
Writes results/e7/guard_budget.json. Usage: python analysis/guard_budget.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.codeverify import PQ, SQ  # noqa: E402
from ragtrust.paths import DATA, RESULTS, corpus, read_json, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

C = corpus()
ctx2pid = {c["text"]: c["id"] for c in C}
items = []
for art in read_json(DATA / "dev-v2.0.json")["data"]:
    for para in art["paragraphs"]:
        pid = ctx2pid.get(para["context"])
        if pid is None:
            continue
        for qa in para["qas"]:
            if not qa.get("is_impossible", False):
                items.append((qa["question"], pid))
qc = np.load(RESULTS / "e7" / "squad_question_vectors.npz", allow_pickle=True)
tix = {t: i for i, t in enumerate(qc["texts"])}
SPACES = {"bge_m3": (np.load(DATA / "vec_bge_m3.npz")["P"].astype(np.float64), qc["bge_m3"].astype(np.float64)),
          "minilm": (np.load(RESULTS / "e7" / "vecs" / "squad_minilm.npz")["D"].astype(np.float64), qc["minilm"].astype(np.float64))}
EPS = {"SQ8": [3e-7, 1e-6, 3e-6, 1e-5], "SQ4": [1e-7, 1e-6, 3e-6, 1e-5], "PQ": [1e-7, 1e-6, 1e-5]}


def options(D, name, eps):
    if name.startswith("SQ"):
        q = SQ(int(name[2:])).fit(D)
        code, amb, alt = q.verifier_view(D, eps)
        R = np.where(q.degenerate, q.lo, q.lo + code * q.step)
        delta = np.where(amb, (alt - code) * q.step, 0.0)
        return R, delta
    q = PQ().fit(D)
    code, amb, alt = q.verifier_view(D, eps)
    R = np.concatenate([q.cb[s][code[:, s]] for s in range(code.shape[1])], 1)
    A = np.concatenate([q.cb[s][alt[:, s]] for s in range(code.shape[1])], 1)
    mask = np.repeat(amb, q.sub, axis=1)
    return R, np.where(mask, A - R, 0.0)


def hijack(R, delta, Qv, golds):
    n_ok = n_hit = 0
    for q, g in zip(Qv, golds):
        s = R @ q
        if int(np.argmax(s)) != g:
            continue
        n_ok += 1
        contrib = delta * q[None, :]
        gain, loss = np.maximum(contrib, 0).sum(1), np.maximum(-contrib, 0).sum(1)
        others = np.delete(np.arange(len(s)), g)
        n_hit += bool(((s[others] + gain[others]) > (s[g] - loss[g])).any())
    return wilson(n_hit, n_ok), n_ok


res = {"verifier": "ragtrust/codeverify.py (verifier-side ambiguity); budget without canonicalisation"}
for sp, (D, QV) in SPACES.items():
    Qv = np.stack([QV[tix[t]] for t, _ in items])
    golds = [p for _, p in items]
    res[sp] = {}
    for name, grid in EPS.items():
        for eps in grid:
            R, delta = options(D, name, eps)
            h, n_ok = hijack(R, delta, Qv, golds)
            res[sp][f"{name}_eps{eps:g}"] = {"hijack": h, "n_gold_top1": n_ok, "docs_with_ambiguous_entries": float((delta != 0).any(1).mean())}
            print(f"{sp:7s} {name:4s} eps={eps:g}: hijack without canonicalisation {h[0]:.4f} [{h[1]:.4f}, {h[2]:.4f}] of {n_ok}", flush=True)
write_json(RESULTS / "e7" / "guard_budget.json", res)
