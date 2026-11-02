"""Numerical precision check of the retrieval margins (exploratory; PROTOCOL.md deviations log).

The margins in results/e7/margins/*.npz were computed as arccos of float32 cosines. Near cos = 1 the
float32 cosine has a resolution of about 6e-8, i.e. an angular resolution of about 0.02 degrees, which is
the scale of the smallest tolerances we report (FP32 cross-architecture noise: 0.04 degrees). This script
recomputes every margin from the SAME float32 document vectors (results/e7/vecs/*.npz) and freshly
embedded queries (pinned revisions), but in double precision with the numerically stable formula
angle = 2 asin(||a - b|| / 2) for unit vectors, and compares:
  - stored vs recomputed margins (max difference, share differing by more than 0.01 degrees),
  - margins that are exactly zero (ties: duplicate or identical documents) in each version,
  - the margin law at the small tolerances (FP32 noise, 0.99999, 0.999) in each version.
Writes results/e7/margin_precision_check.json. Usage: python analysis/margin_precision_check.py [--device cuda]"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, corpus, hf_revision, read_json, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--datasets", default="squad,scifact,nfcorpus,arguana,fiqa")
ap.add_argument("--models", default="bge_m3,minilm")
a = ap.parse_args()
MODELS = {"bge_m3": ("BAAI/bge-m3", 512), "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256)}
TAUS = {"fp32_noise": 0.99999976, "0.99999": 0.99999, "0.999": 0.999}
F32_RES_DEG = float(np.degrees(np.sqrt(2 * 6e-8)))      # angular resolution of a float32 cosine near 1


def load_squad():
    C = corpus()
    ctx2pid = {c["text"]: c["id"] for c in C}
    docs = {str(c["id"]): c["text"] for c in C}
    queries, qrels = {}, defaultdict(set)
    for art in read_json(DATA / "dev-v2.0.json")["data"]:
        for para in art["paragraphs"]:
            pid = ctx2pid.get(para["context"])
            if pid is None:
                continue
            for qa in para["qas"]:
                if not qa.get("is_impossible", False):
                    queries[qa["id"]] = qa["question"]
                    qrels[qa["id"]].add(str(pid))
    return docs, queries, qrels


def load_beir(ds):
    root = DATA / "beir" / ds
    docs = {}
    for line in open(root / "corpus.jsonl"):
        d = json.loads(line)
        docs[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(root / "queries.jsonl")}
    qrels = defaultdict(set)
    with open(root / "qrels" / "test.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0 and row["corpus-id"] in docs:
                qrels[row["query-id"]].add(row["corpus-id"])
    return docs, {q: t for q, t in queries.items() if q in qrels}, qrels


def unit64(X):
    X = X.astype(np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def angle64(q, D):                                        # stable angle between unit q and unit rows of D
    return 2 * np.arcsin(np.minimum(np.linalg.norm(D - q[None, :], axis=1) / 2, 1.0))


from sentence_transformers import SentenceTransformer  # noqa: E402

out = {"float32_angular_resolution_deg": F32_RES_DEG, "results": {}}
for ds in a.datasets.split(","):
    docs, queries, qrels = load_squad() if ds == "squad" else load_beir(ds)
    doc_ids = list(docs)
    dix = {d: i for i, d in enumerate(doc_ids)}
    qids = list(queries)
    for mk in a.models.split(","):
        hf, msl = MODELS[mk]
        D32 = np.load(RESULTS / "e7" / "vecs" / f"{ds}_{mk}.npz")["D"]
        stored = np.load(RESULTS / "e7" / "margins" / f"{ds}_{mk}.npz")["margins"]
        m = SentenceTransformer(hf, revision=hf_revision(hf), device=a.device)
        m.max_seq_length = msl
        Q32 = m.encode([queries[q] for q in qids], batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        del m
        D64, Q64 = unit64(D32), unit64(Q32)
        rec32, rec64, zero_cases, pairs = [], [], [], []
        norm = lambda t: " ".join(t.split()).lower()                         # whitespace collapsed, lower-cased
        for j, q in enumerate(qids):
            s = (Q32[j] @ D32.T).astype(np.float32)
            rel = np.zeros(len(doc_ids), bool)
            rel[[dix[d] for d in qrels[q]]] = True
            if q in dix:
                s[dix[q]] = -1.0
                rel[dix[q]] = False
            if not rel.any():
                continue
            br, bn = s[rel].max(), s[~rel].max()
            if br >= bn:                                                          # same selection rule as the original
                rec32.append(float(np.arccos(np.clip(bn, -1, 1)) - np.arccos(np.clip(br, -1, 1))))
                ang = angle64(Q64[j], D64)
                if q in dix:
                    ang[dix[q]] = np.pi
                m64 = float(ang[~rel].min() - ang[rel].min())
                rec64.append(m64)
                g_id = doc_ids[int(np.where(rel)[0][np.argmin(ang[rel])])]
                c_id = doc_ids[int(np.where(~rel)[0][np.argmin(ang[~rel])])]
                pairs.append((q, g_id, c_id, norm(docs[g_id]) == norm(docs[c_id])))
                if m64 <= 1e-12:
                    g_id = doc_ids[int(np.where(rel)[0][np.argmin(ang[rel])])]
                    c_id = doc_ids[int(np.where(~rel)[0][np.argmin(ang[~rel])])]
                    norm = lambda t: " ".join(t.split()).lower()
                    zero_cases.append({"query": q, "gold": g_id, "competitor": c_id,
                                       "identical_text": docs[g_id] == docs[c_id],
                                       "identical_normalised_text": norm(docs[g_id]) == norm(docs[c_id])})
        rec32, rec64 = np.array(rec32), np.array(rec64)
        same_len = len(rec32) == len(stored)
        d_deg = np.degrees(np.abs(rec64 - rec32))
        alpha = np.arccos(TAUS["fp32_noise"])
        exposed = rec64 < alpha
        dup_exposed = sum(1 for k, pr in enumerate(pairs) if exposed[k] and pr[3])     # checked for ALL exposed cases
        exposed_cases = [{"query": pr[0], "gold": pr[1], "competitor": pr[2], "identical_normalised_text": pr[3],
                          "margin_deg": float(np.degrees(rec64[k]))} for k, pr in enumerate(pairs) if exposed[k]]
        r = {"n": int(len(rec64)), "n_stored": int(len(stored)), "same_count_as_stored": same_len,
             "fp32_noise_exposed": int(exposed.sum()),
             "fp32_noise_exposed_with_identical_text_duplicate": dup_exposed,
             "fp32_noise_exposed_excluding_duplicates": {"numerator": int(exposed.sum()) - dup_exposed, "denominator_all": int(len(rec64)),
                                                         "denominator_without_duplicate_queries": int(len(rec64)) - dup_exposed},
             "zero_margin_cases": zero_cases, "fp32_noise_exposed_cases": exposed_cases,
             "reproduces_stored_float32": bool(same_len and np.allclose(rec32, stored, atol=1e-6)),
             "max_abs_diff_deg": float(d_deg.max()), "share_diff_gt_0.01deg": float((d_deg > 0.01).mean()),
             "zero_margins_float32": int((rec32 <= 0).sum()), "zero_margins_float64": int((rec64 <= 1e-12).sum()),
             "negative_margins_float64": int((rec64 < -1e-12).sum()),
             "share_below_f32_resolution_float64": float((np.degrees(rec64) < F32_RES_DEG).mean()),
             "law": {k: {"float32": wilson(int((rec32 < np.arccos(t)).sum()), len(rec32)),
                         "float64": wilson(int((rec64 < np.arccos(t)).sum()), len(rec64))} for k, t in TAUS.items()}}
        out["results"][f"{ds}/{mk}"] = r
        law = "  ".join(f"{k}: {r['law'][k]['float32'][0]*100:.2f}%->{r['law'][k]['float64'][0]*100:.2f}%" for k in TAUS)
        dups = (f"identical normalised text in {dup_exposed}/{int(exposed.sum())} exposed cases "
                f"(zero margins: {sum(z['identical_normalised_text'] for z in zero_cases)}/{len(zero_cases)})")
        ex = r["fp32_noise_exposed_excluding_duplicates"]
        print(f"{ds:9s} {mk:7s} fp32-noise exposed {r['fp32_noise_exposed']}/{r['n']}; {dups}; excluding duplicates "
              f"{ex['numerator']}/{ex['denominator_all']} = {100 * ex['numerator'] / max(1, ex['denominator_all']):.2f}% "
              f"(or {ex['numerator']}/{ex['denominator_without_duplicate_queries']})")
        print(f"{ds:9s} {mk:7s} n={r['n']:5d} reproduces_stored={r['reproduces_stored_float32']}  max|diff|={r['max_abs_diff_deg']:.6f} deg  "
              f">0.01deg: {r['share_diff_gt_0.01deg']*100:.2f}%  zeros f32/f64: {r['zero_margins_float32']}/{r['zero_margins_float64']}  "
              f"neg f64: {r['negative_margins_float64']}  law {law}", flush=True)
# ---- Table 2 units: analysis/margin_law_squad.py (BGE-M3 on passages, MiniLM on 128-token windows, argmax selection)
if "squad" in a.datasets.split(","):
    qc = np.load(RESULTS / "e7" / "squad_question_vectors.npz", allow_pickle=True)
    tix = {t: i for i, t in enumerate(qc["texts"])}
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
    spaces = {"bge_m3": (np.load(DATA / "vec_bge_m3.npz")["P"], None), "minilm": None}
    z = np.load(DATA / "vec_minilm_L128.npz")
    spaces["minilm"] = (z["W"], z["pid"])
    for sp, (U32, pid_of) in spaces.items():
        if pid_of is None:
            pid_of = np.arange(len(U32))
        U64 = unit64(U32)
        Q64 = unit64(qc[sp])
        rec32, rec64 = [], []
        for q, pid in items:
            qv32 = qc[sp][tix[q]].astype(np.float32)
            s32 = (U32.astype(np.float32) @ qv32).astype(np.float32)
            gold = pid_of == pid
            if not gold[int(np.argmax(s32))]:                                     # same selection rule as margin_law_squad.py
                continue
            rec32.append(float(np.arccos(np.clip(s32[~gold].max(), -1, 1)) - np.arccos(np.clip(s32[gold].max(), -1, 1))))
            ang = angle64(Q64[tix[q]], U64)
            rec64.append(float(ang[~gold].min() - ang[gold].min()))
        rec32, rec64 = np.array(rec32), np.array(rec64)
        d_deg = np.degrees(np.abs(rec64 - rec32))
        law = {k: {"float32": wilson(int((rec32 < np.arccos(t)).sum()), len(rec32)),
                   "float64": wilson(int((rec64 < np.arccos(t)).sum()), len(rec64))} for k, t in TAUS.items()}
        out["results"][f"squad_table2_units/{sp}"] = {"n": int(len(rec64)), "max_abs_diff_deg": float(d_deg.max()),
                                                       "zero_margins_float64": int((rec64 <= 1e-12).sum()), "law": law}
        unit = "passages" if sp == "bge_m3" else "128-token windows"
        print(f"squad (Table 2 units, {unit}) {sp:7s} n={len(rec64):5d} max|diff|={d_deg.max():.6f} deg zeros={int((rec64 <= 1e-12).sum())}  law "
              + "  ".join(f"{k}: {law[k]['float32'][0]*100:.2f}%->{law[k]['float64'][0]*100:.2f}%" for k in TAUS), flush=True)
write_json(RESULTS / "e7" / "margin_precision_check.json", out)
