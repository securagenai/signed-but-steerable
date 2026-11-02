"""Statistics used across the study: AUROC with bootstrap and DeLong tests, split-conformal
thresholds (same procedure as the prior study), paired bootstrap, Wilson intervals, ECE, and the
detection probability of random spot-checks."""
import math
import numpy as np


def _midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); identical to the prior study's pairwise definition."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _midrank(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auroc(pos, neg, B=2000, seed=0):
    """Point estimate and percentile 95% CI; positives and negatives resampled separately (as before)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    v = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [auroc(pos, neg), float(lo), float(hi)]


def delong(pos_a, neg_a, pos_b, neg_b):
    """Paired DeLong test for two scores evaluated on the same positives and negatives (fast
    algorithm of Sun & Xu, 2014). Returns AUC_a, AUC_b, difference, 95% CI of the difference, p."""
    pa, na, pb, nb = (np.asarray(v, float) for v in (pos_a, neg_a, pos_b, neg_b))
    m, n = len(pa), len(na)
    assert len(pb) == m and len(nb) == n, "paired test needs the same samples for both scores"
    X = np.vstack([pa, pb])
    Y = np.vstack([na, nb])
    aucs, v01, v10 = [], [], []
    for r in range(2):
        tx, ty = _midrank(X[r]), _midrank(Y[r])
        tz = _midrank(np.concatenate([X[r], Y[r]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    d = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = math.sqrt(max(var, 1e-15))
    z = d / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(d),
            "ci95": [float(d - 1.96 * se), float(d + 1.96 * se)], "z": float(z), "p": float(p)}


def conformal_tau(answerable_scores, alpha=0.1, floor=-1.0):
    """Split-conformal lower threshold: keeps >= 1-alpha of answerable queries (prior study's rule).
    When floor(alpha(n+1)) < 1 the threshold accepts everything (returns `floor`)."""
    sc = np.sort(np.asarray(answerable_scores, float))
    k = int(np.floor(alpha * (len(sc) + 1)))
    return float(sc[k - 1]) if k >= 1 else floor


def split_conformal(qids, score_of, is_answerable, alpha=0.1, B=2000, seed=0, floor=-1.0):
    """Same procedure as the prior study: B random halves; tau fitted on the answerable queries of the
    calibration half. Returns the list of (tau, calibration-set) pairs and the mean tau."""
    rng = np.random.default_rng(seed)
    qids = list(qids)
    fits = []
    for _ in range(B):
        perm = rng.permutation(len(qids))
        cal = {qids[i] for i in perm[: len(qids) // 2]}
        sc = [score_of(q) for q in cal if is_answerable(q)]
        fits.append((conformal_tau(sc, alpha, floor), cal))
    return fits, float(np.mean([t for t, _ in fits]))


def paired_bootstrap(stat, ids, B=4000, seed=1):
    """stat(list_of_ids) -> float; resamples query ids with replacement (paired across systems)."""
    rng = np.random.default_rng(seed)
    ids = list(ids)
    pt = stat(ids)
    bs = [stat(list(rng.choice(ids, len(ids)))) for _ in range(B)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return [float(pt), float(lo), float(hi)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def ece(prob, label, bins=10):
    prob, label = np.asarray(prob, float), np.asarray(label, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if m.any():
            e += m.mean() * abs(prob[m].mean() - label[m].mean())
    return float(e)


def spotcheck_detection(n_total, n_tampered, n_checked):
    """P(at least one tampered item is among n_checked items drawn without replacement)."""
    if n_tampered <= 0 or n_checked <= 0:
        return 0.0
    if n_tampered >= n_total or n_checked >= n_total:
        return 1.0
    return 1.0 - math.comb(n_total - n_tampered, n_checked) / math.comb(n_total, n_checked)


def summarize(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "min": float(x.min()), "max": float(x.max())}
