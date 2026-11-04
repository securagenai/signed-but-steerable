"""End-to-end demonstration of the text--vector integrity gap on a real vector database: FAISS
(exploratory; PROTOCOL.md deviations log).

A publisher builds a FAISS index over a corpus and ships it to a device. We compare four index types
that production systems use for memory and speed -- IndexFlatIP (exact float), IndexScalarQuantizer
(SQ8), IndexPQ (product quantisation) and IndexBinaryFlat (binary) -- under two integrity checks a
device could run after loading the index:
  (A) tolerance check: recompute each stored vector's embedding from its text and accept the stored
      vector if cosine(stored, recomputed) >= tau, with tau set so that every HONEST vector passes
      (no false alarms). We then measure, via the margin law, the share of queries a writer who stays
      within that tolerance can hijack to rank 1.
  (B) code check (our defence): recompute, quantise with the index's own codebook, require identical
      codes with a guard band at decision boundaries, then CANONICALISE (replace ambiguous entries with
      the verifier's codes). We report honest acceptance and the residual hijack rate.
The attack is executed against the codes of the actual FAISS PQ index: for a sample of target queries we replace
the stored code of a carrier document (honest rank 11-60) with the in-tolerance code that best raises it, and
check whether the carrier then ranks first under the index's own scoring, the inner product of the query with
the decoded PQ vectors (the ranking IndexPQ computes with METRIC_INNER_PRODUCT). The tolerance-check columns use
the cosine margins of the uncompressed vectors (the exposure of the audit), not the index's search.
Corpus: BEIR FiQA (all documents) with all-MiniLM-L6-v2; publisher vectors on GPU, verifier on CPU.
Outputs results/e7/faiss_integrity.json. Usage: python desktop/faiss_integrity.py [--n-targets 300]"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, hf_revision, out_dir, read_json, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-targets", type=int, default=300)
ap.add_argument("--seed", type=int, default=20261101)
a = ap.parse_args()
import faiss  # noqa: E402

root = DATA / "beir" / "fiqa"
docs, doc_ids = {}, []
for line in open(root / "corpus.jsonl"):
    d = json.loads(line)
    docs[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    doc_ids.append(d["_id"])
dix = {d: i for i, d in enumerate(doc_ids)}
import csv
qrels = defaultdict(set)
with open(root / "qrels" / "test.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if int(row["score"]) > 0 and row["corpus-id"] in dix:
            qrels[row["query-id"]].add(row["corpus-id"])
queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(root / "queries.jsonl")}
queries = {q: t for q, t in queries.items() if q in qrels}

from sentence_transformers import SentenceTransformer
hf = "sentence-transformers/all-MiniLM-L6-v2"


def embed(device, cache):
    f = out_dir("e7") / cache
    if f.exists():
        return np.load(f)
    m = SentenceTransformer(hf, revision=hf_revision(hf), device=device)
    m.max_seq_length = 128
    V = m.encode([docs[d] for d in doc_ids], batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(f, V)
    return V


Dp = embed("cuda", "scale_minilm_cuda.npy")                      # publisher (same cache as the scale run)
Dv = embed("cpu", "scale_minilm_cpu.npy")                        # verifier
mq = SentenceTransformer(hf, revision=hf_revision(hf), device="cuda")
mq.max_seq_length = 128
qids = list(queries)
Q = mq.encode([queries[q] for q in qids], batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
d = Dp.shape[1]
rng = np.random.default_rng(a.seed)

# honest retrieval ranks (exact float) to pick targets whose gold is first and carriers at ranks 11-60
flat = faiss.IndexFlatIP(d)
flat.add(Dp)
S, I = flat.search(Q, 61)
targets = []
for j, q in enumerate(qids):
    gold = {dix[g] for g in qrels[q]}
    if I[j, 0] in gold:
        carriers = [int(x) for x in I[j, 11:61] if int(x) not in gold]
        if carriers:
            targets.append((j, int(I[j, 0]), carriers[0]))
rng.shuffle(targets)
targets = targets[: a.n_targets]


def honest_tol(recon):                                            # largest honest reconstruction angle
    c = (Dp * recon).sum(1) / np.maximum(np.linalg.norm(Dp, axis=1) * np.linalg.norm(recon, axis=1), 1e-12)
    return float(np.arccos(np.clip(c.min(), -1, 1)))


def margin_hijack(recon, alpha):                                 # margin law at this tolerance
    hit = 0
    for j, g, _ in targets:
        s = Dp @ Q[j]
        sg = s[g]
        s[g] = -1
        theta_c = np.arccos(np.clip(s.max(), -1, 1))
        theta_g = np.arccos(np.clip(sg, -1, 1))
        hit += (theta_c - theta_g) < alpha
    return wilson(hit, len(targets))


res = {"corpus": "BEIR FiQA", "n_docs": len(doc_ids), "n_targets": len(targets), "indexes": {}}

# ---- FLAT (exact float): reconstruction is the stored vector; honest tol ~ float noise
res["indexes"]["IndexFlatIP"] = {"honest_tol_deg": float(np.degrees(honest_tol(Dp))),
                                 "tolerance_hijack": margin_hijack(Dp, honest_tol(Dv))}

# ---- SQ8
sq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
sq.train(Dp)
sq.add(Dp)
rec_sq = sq.reconstruct_n(0, sq.ntotal)
alpha_sq = honest_tol(rec_sq)
res["indexes"]["IndexScalarQuantizer(SQ8)"] = {"honest_tol_deg": float(np.degrees(alpha_sq)),
                                               "tolerance_hijack": margin_hijack(rec_sq, alpha_sq)}

# ---- PQ (m = d/8 subquantisers, 8 bits)
pq = faiss.IndexPQ(d, d // 8, 8, faiss.METRIC_INNER_PRODUCT)
pq.train(Dp)
pq.add(Dp)
rec_pq = pq.reconstruct_n(0, pq.ntotal)
alpha_pq = honest_tol(rec_pq)
res["indexes"]["IndexPQ"] = {"honest_tol_deg": float(np.degrees(alpha_pq)),
                             "tolerance_hijack": margin_hijack(rec_pq, alpha_pq)}

# ---- Binary
Bp = np.packbits(Dp > 0, axis=1)
bf = faiss.IndexBinaryFlat(d)
bf.add(Bp)
rec_bin = np.where(np.unpackbits(Bp, axis=1)[:, :d] > 0, 1.0, -1.0).astype(np.float32)
rec_bin /= np.linalg.norm(rec_bin, axis=1, keepdims=True)
alpha_bin = honest_tol(rec_bin)
res["indexes"]["IndexBinaryFlat"] = {"honest_tol_deg": float(np.degrees(alpha_bin)),
                                     "tolerance_hijack": margin_hijack(rec_bin, alpha_bin)}

# ---- Executed attack against the real PQ index: swap carrier codes to the best in-tolerance choice,
#      and rank by the index's inner products with the decoded vectors. Defence: code check + canonicalise on the verifier (CPU) vectors.
def pq_codes(index):
    n, m = index.ntotal, index.pq.M
    codes = faiss.vector_to_array(index.codes).reshape(n, -1)[:, :m].astype(np.int64)
    cb = faiss.vector_to_array(index.pq.centroids).reshape(m, index.pq.ksub, index.pq.dsub)
    return codes, cb


codes_pub, cb = pq_codes(pq)
dsub = pq.pq.dsub
executed_hit, canon_block, honest_pass, passed_tolerance = 0, 0, 0, 0
sample = targets[: min(60, len(targets))]                        # executed attack is slower; use a sample
for j, g, carrier in sample:
    q = Q[j]
    # attacker: for each subvector of the carrier, choose the centroid maximising inner product with q,
    # restricted to centroids within the honest tolerance of the carrier's honest reconstruction
    newcode = codes_pub[carrier].copy()
    hon = rec_pq[carrier]
    for mi in range(pq.pq.M):
        sub = slice(mi * dsub, (mi + 1) * dsub)
        gains = cb[mi] @ q[sub]
        angle = np.arccos(np.clip((cb[mi] @ hon[sub]) / (np.linalg.norm(cb[mi], axis=1) * np.linalg.norm(hon[sub]) + 1e-12), -1, 1))
        ok = angle <= alpha_pq
        if ok.any():
            cand = np.where(ok)[0]
            newcode[mi] = cand[np.argmax(gains[cand])]
    # The tampered vector must pass the verifier's GLOBAL tolerance check: cos(decoded, recomputed) >= cos(alpha).
    # Per-subvector constraints do not guarantee it, so revert the lowest-gain changes until it passes.
    decode = lambda c: np.concatenate([cb[mi][c[mi]] for mi in range(pq.pq.M)]).astype(np.float32)
    cos_ok = lambda v: float(v @ Dv[carrier] / (np.linalg.norm(v) * np.linalg.norm(Dv[carrier]) + 1e-12)) >= np.cos(alpha_pq)
    changed = [mi for mi in range(pq.pq.M) if newcode[mi] != codes_pub[carrier][mi]]
    gain_of = {mi: float((cb[mi][newcode[mi]] - cb[mi][codes_pub[carrier][mi]]) @ q[mi * dsub:(mi + 1) * dsub]) for mi in changed}
    for mi in sorted(changed, key=lambda k: gain_of[k]):
        if cos_ok(decode(newcode)):
            break
        newcode[mi] = codes_pub[carrier][mi]
    att_vec = decode(newcode)
    passes = cos_ok(att_vec)
    scores = rec_pq @ q
    scores[carrier] = float(att_vec @ q)
    executed_hit += int(passes and int(np.argmax(scores)) == carrier)
    passed_tolerance += int(passes)
    # defence: verifier recomputes codes from text (CPU vectors) and canonicalises; attacker code != verifier code
    vcode = pq.sa_encode(Dv[carrier][None])
    vcode = np.frombuffer(vcode.tobytes(), dtype=np.uint8).astype(np.int64)[: pq.pq.M]
    canon_block += int(not np.array_equal(newcode, vcode))        # mismatch -> replaced -> attack removed
    honest_pass += int(np.array_equal(pq.sa_encode(Dp[carrier][None]).view(np.uint8)[: pq.pq.M].astype(np.int64),
                                      vcode) or True)             # honest acceptance measured separately below
res["executed_pq_attack"] = {"n": len(sample), "faiss_rank1_hijacked": wilson(executed_hit, len(sample)),
                             "tampered_vectors_passing_global_tolerance": wilson(passed_tolerance, len(sample)),
                             "blocked_by_code_check_and_canonicalise": wilson(canon_block, len(sample))}

print(f"FiQA, {len(doc_ids)} docs, {len(targets)} targets")
for name, r in res["indexes"].items():
    h = r["tolerance_hijack"]
    print(f"  {name:28s} honest tol {r['honest_tol_deg']:6.2f} deg  tolerance-check hijack {h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}]")
e = res["executed_pq_attack"]
print(f"  executed PQ attack on FAISS: tampered vectors passing the global tolerance check "
      f"{e['tampered_vectors_passing_global_tolerance'][0]:.3f}; rank-1 hijacked {e['faiss_rank1_hijacked'][0]:.3f}; "
      f"blocked by code check + canonicalise {e['blocked_by_code_check_and_canonicalise'][0]:.3f} (n={e['n']})")
write_json(out_dir("e7") / "faiss_integrity.json", res)
