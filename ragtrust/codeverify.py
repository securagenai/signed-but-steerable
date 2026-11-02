"""Code-level verification of compressed vector stores: the single reference implementation used by every
measurement in the paper (desktop, Raspberry Pi, Apple M2).

Threat model: the publisher (or whoever wrote the store) may be malicious, so the verifier trusts ONLY its
own recomputation of each embedding from the text. Every decision below is derived from the verifier's
recomputed vector; the publisher's float vectors are never used.

For each stored code entry the verifier computes its own code c, and decides whether the entry is
AMBIGUOUS, i.e. whether honest float noise between platforms could plausibly have produced the other
neighbouring code a:
  SQ (scalar, per dimension):   ambiguous if the recomputed value lies within eps_sq of the rounding
                                boundary between c and its neighbour a (eps_sq in value units).
  PQ (product, per sub-vector): ambiguous if the squared distances from the recomputed sub-vector to the
                                nearest centroid c and the second nearest a differ by less than eps_pq
                                (eps_pq in squared-distance units, NOT comparable with eps_sq).
  Binary (sign, per dimension): ambiguous if |recomputed value| < eps_bin; a is the other sign.
Acceptance rule: a stored vector is accepted iff every entry equals c, or the entry is ambiguous AND equals
a. Anything else (a different code, or a neighbouring code where the verifier sees no ambiguity) rejects.
Canonicalisation: an accepted vector is rewritten with the verifier's own codes c, so the publisher's
choice at ambiguous entries has no effect on retrieval.
"""
import numpy as np


class SQ:
    def __init__(self, bits):
        self.bits, self.n = bits, 2 ** bits - 1

    def fit(self, V):
        self.lo, self.hi = V.min(0).astype(np.float64), V.max(0).astype(np.float64)
        # Degenerate (constant) dimensions carry no information; dividing float noise by a ~0 range would turn
        # it into arbitrary codes. They are encoded as 0 on both sides and are never ambiguous.
        self.degenerate = (self.hi - self.lo) < 1e-9
        self.step = np.where(self.degenerate, 1.0, self.hi - self.lo) / self.n
        return self

    def _t(self, V):
        return np.where(self.degenerate, 0.0, (V - self.lo) / self.step)

    def encode(self, V):
        return np.clip(np.floor(self._t(V) + 0.5), 0, self.n).astype(np.int32)

    def verifier_view(self, V, eps):
        t = self._t(V)
        code = np.clip(np.floor(t + 0.5), 0, self.n).astype(np.int32)
        frac = t - np.floor(t)
        alt = np.where(frac >= 0.5, np.floor(t), np.floor(t) + 1)
        alt = np.clip(alt, 0, self.n).astype(np.int32)
        amb = (np.abs(frac - 0.5) * self.step < eps) & (alt != code) & ~self.degenerate
        return code, amb, alt


class PQ:
    def __init__(self, sub=8, k=256, seed=0):
        self.sub, self.k, self.seed = sub, k, seed

    def fit(self, V, n_train=20000):
        from sklearn.cluster import MiniBatchKMeans
        rng = np.random.default_rng(self.seed)
        tr = V[rng.choice(len(V), min(len(V), n_train), replace=False)]
        self.cb = [MiniBatchKMeans(n_clusters=min(self.k, len(tr)), n_init=1, max_iter=100, batch_size=4096,
                                   random_state=self.seed).fit(tr[:, j:j + self.sub]).cluster_centers_.astype(np.float64)
                   for j in range(0, V.shape[1], self.sub)]
        return self

    def _dist(self, V):
        for c, j in zip(self.cb, range(0, V.shape[1], self.sub)):
            x = V[:, j:j + self.sub].astype(np.float64)
            yield (x * x).sum(1, keepdims=True) - 2 * x @ c.T + (c * c).sum(1)[None]

    def encode(self, V):
        return np.stack([np.argmin(d, 1) for d in self._dist(V)], 1).astype(np.int32)

    def verifier_view(self, V, eps):
        codes, ambs, alts = [], [], []
        for d in self._dist(V):
            o = np.argsort(d, 1)[:, :2]
            r = np.arange(len(d))
            codes.append(o[:, 0]); alts.append(o[:, 1])
            ambs.append((d[r, o[:, 1]] - d[r, o[:, 0]]) < eps)
        return np.stack(codes, 1).astype(np.int32), np.stack(ambs, 1), np.stack(alts, 1).astype(np.int32)


class Binary:
    def fit(self, V):
        return self

    def encode(self, V):
        return (V > 0).astype(np.int8)

    def verifier_view(self, V, eps):
        code = (V > 0).astype(np.int8)
        return code, np.abs(V) < eps, (1 - code).astype(np.int8)


def verify(stored, recomputed, quantizer, eps):
    """Returns accept (per vector), canonical codes, and exception statistics."""
    code, amb, alt = quantizer.verifier_view(recomputed, eps)
    equal = stored == code
    neighbour_ok = amb & (stored == alt)
    accept = (equal | neighbour_ok).all(1)
    exc = neighbour_ok & ~equal                       # entries accepted only thanks to the guard band
    return {"accept": accept, "canonical": code, "exceptions_per_vector": exc.sum(1),
            "ambiguous_entry_share": float(amb.mean()), "exception_entry_share": float(exc.mean())}
