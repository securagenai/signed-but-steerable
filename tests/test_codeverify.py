import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from ragtrust.codeverify import SQ, PQ, Binary, verify
rng = np.random.default_rng(0)
V = rng.normal(size=(400, 32)); V /= np.linalg.norm(V, axis=1, keepdims=True)
noise = rng.normal(size=V.shape) * 1e-7
ok = 0
for name, q, eps in (("SQ8", SQ(8).fit(V), 1e-6), ("PQ", PQ(sub=8, k=64).fit(V), 1e-5), ("binary", Binary(), 1e-6)):
    stored = q.encode(V)                                   # honest publisher
    r = verify(stored, V + noise, q, eps)                  # verifier recomputes with float noise
    assert r["accept"].all(), name + ": honest vectors must be accepted"
    assert np.array_equal(r["canonical"], q.encode(V + noise)), name + ": canonical codes are the verifier's"
    bad = stored.copy()                                    # tamper an entry the verifier sees as unambiguous
    code, amb, alt = q.verifier_view(V + noise, eps)
    i, j = np.argwhere(~amb)[0]
    bad[i, j] = (code[i, j] + 2) % (2 if name == "binary" else 7) if name != "binary" else 1 - code[i, j]
    assert not verify(bad, V + noise, q, eps)["accept"][i], name + ": tampered unambiguous entry must be rejected"
    if amb.any():                                          # neighbouring code at an ambiguous entry: accepted, then canonicalised
        i, j = np.argwhere(amb)[0]
        nb = stored.copy(); nb[i, j] = alt[i, j]
        rr = verify(nb, V + noise, q, eps)
        assert rr["accept"][i] and rr["canonical"][i, j] == code[i, j], name + ": ambiguous neighbour accepted and canonicalised"
    ok += 1
    print(f"{name}: honest accepted, tamper rejected, canonicalisation correct (ambiguous entries {r['ambiguous_entry_share']:.1e})")
print(f"{ok}/3 schemes pass")
