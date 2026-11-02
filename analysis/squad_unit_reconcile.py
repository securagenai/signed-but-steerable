"""Reconciles the two passage-level SQuAD counts for BGE-M3 (exploratory; PROTOCOL.md deviations log).

Table 2 (analysis/margin_law_squad.py) and the cross-corpus comparison (desktop/beir_margins.py) both use
BGE-M3 on the 950 SQuAD passages but report 3,648 and 3,646 queries with the gold passage first. The two
analyses differ in (i) the passage vectors (data/vec_bge_m3.npz from the earlier study vs a fresh encoding
in results/e7/vecs/squad_bge_m3.npz), (ii) the question vectors (cached vs freshly encoded) and (iii) the
selection rule (argmax over all passages vs best relevant >= best non-relevant, i.e. ties count as first).
This script lists the questions selected by one analysis but not the other and, for each, the gold and best
non-gold scores under both, and quantifies the vector differences. Usage: python analysis/squad_unit_reconcile.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, corpus, hf_revision, read_json  # noqa: E402

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
                items.append((qa["id"], qa["question"], pid))
PA = np.load(DATA / "vec_bge_m3.npz")["P"].astype(np.float32)                       # Table 2 passages
PB = np.load(RESULTS / "e7" / "vecs" / "squad_bge_m3.npz")["D"].astype(np.float32)   # cross-corpus passages
order_b = [c["id"] for c in C]                                                       # load_squad order
qc = np.load(RESULTS / "e7" / "squad_question_vectors.npz", allow_pickle=True)
tix = {t: i for i, t in enumerate(qc["texts"])}
QA = qc["bge_m3"].astype(np.float32)
from sentence_transformers import SentenceTransformer  # noqa: E402
m = SentenceTransformer("BAAI/bge-m3", revision=hf_revision("BAAI/bge-m3"), device="cuda")
m.max_seq_length = 512
QB = m.encode([q for _, q, _ in items], batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

pos_b = {pid: i for i, pid in enumerate(order_b)}
PBa = PB[[pos_b[c["id"]] for c in C]]                                                 # align B to A's passage order
cos_p = (PA / np.linalg.norm(PA, axis=1, keepdims=True) * PBa / np.linalg.norm(PBa, axis=1, keepdims=True)).sum(1)
print(f"passage vectors A vs B: max |diff| {np.abs(PA - PBa).max():.2e}, min cosine {cos_p.min():.8f}")
sel_a, sel_b, rows = set(), set(), {}
for k, (qid, q, pid) in enumerate(items):
    qa = QA[tix[q]]
    sa = PA @ qa
    gold_a = int(np.argmax(sa)) == pid
    sb = PBa @ QB[k]
    gold_mask = np.zeros(len(sb), bool); gold_mask[pid] = True
    gold_b = sb[gold_mask].max() >= sb[~gold_mask].max()
    if gold_a:
        sel_a.add(qid)
    if gold_b:
        sel_b.add(qid)
    rows[qid] = (q, pid, float(sa[pid]), float(np.delete(sa, pid).max()), float(sb[pid]), float(np.delete(sb, pid).max()),
                 float(np.dot(qa, QB[k]) / (np.linalg.norm(qa) * np.linalg.norm(QB[k]))))
print(f"selected: Table-2 analysis {len(sel_a)}, cross-corpus analysis {len(sel_b)}")
for name, ids in (("only in Table 2 analysis", sel_a - sel_b), ("only in cross-corpus analysis", sel_b - sel_a)):
    for qid in sorted(ids):
        q, pid, ga, na, gb, nb, cq = rows[qid]
        print(f"  {name}: {qid}  gold={pid}  A: gold {ga:.6f} vs best other {na:.6f} (diff {ga - na:+.2e})  "
              f"B: gold {gb:.6f} vs best other {nb:.6f} (diff {gb - nb:+.2e})  question-vector cosine A/B {cq:.8f}  | {q[:70]}")
