"""Generality of the margin law: retrieval margins on SQuAD (our corpus) and on four
BEIR corpora from other domains, with two embedding models.

For every test query whose top-1 document is relevant, margin = angle(best non-relevant document)
- angle(best relevant document). The fraction of queries with margin < arccos(tau) is the success
rate of the optimal vector-text desynchronisation attacker under a device check with tolerance tau
(analysis/margin_law.py). Corpus vectors are saved for the vector-database quantisation analysis
(analysis/vecdb_quant.py).

Datasets: squad (950 passages, answerable in-corpus SQuAD 2.0 dev questions), scifact, nfcorpus,
arguana, fiqa (BEIR test splits; downloaded once into data/beir/). Models: BGE-M3 (dense, 512 tokens)
and all-MiniLM-L6-v2 (256 tokens), pinned revisions.
Outputs: results/e7/margins/<dataset>_<model>.npz, results/e7/vecs/<dataset>_<model>.npz,
         results/e7/beir_margins.json
Usage: python desktop/beir_margins.py [--datasets squad,scifact,nfcorpus,arguana,fiqa] [--models bge_m3,minilm]"""
import argparse, csv, io, json, sys, time, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, corpus, hf_revision, out_dir, read_json, write_json  # noqa: E402
from ragtrust.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", default="squad,scifact,nfcorpus,arguana,fiqa")
ap.add_argument("--models", default="bge_m3,minilm")
ap.add_argument("--device", default=None)
a = ap.parse_args()
MODELS = {"bge_m3": ("BAAI/bge-m3", 512, 32), "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256, 256)}
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
TAU_REPORT = {"exact": 1.0, "fp32_noise": 0.99999976, "0.99999": 0.99999, "0.999": 0.999, "0.99": 0.99,
              "0.95": 0.95, "cpu_int8": 0.904}


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


def load_beir(name):
    root = DATA / "beir" / name
    if not (root / "corpus.jsonl").exists():
        (DATA / "beir").mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...", flush=True)
        with urllib.request.urlopen(BEIR_URL.format(name), timeout=600) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA / "beir")
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
    queries = {q: t for q, t in queries.items() if q in qrels}
    return docs, queries, qrels


from sentence_transformers import SentenceTransformer  # noqa: E402

summary, MD, VD = {}, out_dir("e7/margins"), out_dir("e7/vecs")
for ds in a.datasets.split(","):
    docs, queries, qrels = load_squad() if ds == "squad" else load_beir(ds)
    doc_ids = list(docs)
    dix = {d: i for i, d in enumerate(doc_ids)}
    qids = list(queries)
    for mk in a.models.split(","):
        hf, msl, bs = MODELS[mk]
        t0 = time.time()
        m = SentenceTransformer(hf, revision=hf_revision(hf), device=a.device)
        m.max_seq_length = msl
        D = m.encode([docs[d] for d in doc_ids], batch_size=bs, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        Qv = m.encode([queries[q] for q in qids], batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        del m
        margins, top1 = [], 0
        for s0 in range(0, len(qids), 512):
            S = Qv[s0:s0 + 512] @ D.T
            for j, q in enumerate(qids[s0:s0 + 512]):
                s = S[j]
                rel = np.zeros(len(doc_ids), bool)
                rel[[dix[d] for d in qrels[q]]] = True
                if q in dix:  # ArguAna: never count the query's own document
                    s[dix[q]] = -1.0
                    rel[dix[q]] = False
                if not rel.any():
                    continue
                best_rel, best_non = s[rel].max(), s[~rel].max()
                if best_rel >= best_non:
                    top1 += 1
                    margins.append(float(np.arccos(np.clip(best_non, -1, 1)) - np.arccos(np.clip(best_rel, -1, 1))))
        mg = np.array(margins)
        np.savez(MD / f"{ds}_{mk}.npz", margins=mg)
        np.savez(VD / f"{ds}_{mk}.npz", D=D)
        law = {k: wilson(int((mg < np.arccos(t)).sum()), len(mg)) for k, t in TAU_REPORT.items()}
        summary[f"{ds}/{mk}"] = {"n_docs": len(doc_ids), "n_queries": len(qids), "n_top1_relevant": int(top1),
                                 "margin_deg_q05": float(np.degrees(np.quantile(mg, 0.05))) if len(mg) else None,
                                 "margin_deg_median": float(np.degrees(np.median(mg))) if len(mg) else None,
                                 "law": law, "seconds": time.time() - t0}
        r = summary[f"{ds}/{mk}"]
        print(f"{ds:9s} {mk:7s} docs {len(doc_ids):6d}  queries {len(qids):5d}  top1-relevant {top1:5d}  "
              f"margin median {r['margin_deg_median']:.2f}° (5%: {r['margin_deg_q05']:.2f}°)  "
              f"fp32-noise {law['fp32_noise'][0]:.4f}  0.999 {law['0.999'][0]:.3f}  cpu-int8 {law['cpu_int8'][0]:.3f}", flush=True)
write_json(RESULTS / "e7" / "beir_margins.json", {"tau": TAU_REPORT, "results": summary})
