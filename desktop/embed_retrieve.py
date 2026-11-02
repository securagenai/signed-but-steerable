"""Desktop retrieval and embedding matrices used by E1, E2, E5, E7.

1. BGE-M3 (the prior study's embedder, pinned revision): passage and question vectors, exact
   inner-product top-20 retrieval, and agreement with the prior study's on-device top-5
   (data/prior/results_v2/retrieval.json, HNSW on the Pi).
2. all-MiniLM-L6-v2: the corpus split into sentence windows
   of at most L tokens (L = window length, default 128), and FP32 vectors of windows and
   questions; the window file is the shared definition used on the Pi.

Outputs: data/retrieval_top20.json, data/vec_bge_m3.npz, data/windows_minilm_L{L}.json,
         data/vec_minilm_L{L}.npz, results/e1/retrieval_agreement.json"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, PRIOR_DATA, corpus, hf_revision, out_dir, questions, read_json, write_json  # noqa: E402
from ragtrust.sentences import sentence_windows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq-len", type=int, default=128)
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--device", default=None, help="torch device for sentence-transformers (cpu/cuda)")
a = ap.parse_args()
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

C, Q = corpus(), questions()
qids = [q["qid"] for q in Q]

# ---------------------------------------------------------------- 1. BGE-M3
name = "BAAI/bge-m3"
m = SentenceTransformer(name, revision=hf_revision(name), device=a.device)
t0 = time.time()
P = m.encode([c["text"] for c in C], batch_size=16, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
t_p = time.time() - t0
QV = m.encode([q["question"] for q in Q], batch_size=32, normalize_embeddings=True).astype(np.float32)
np.savez(DATA / "vec_bge_m3.npz", P=P, Q=QV, qids=np.array(qids))
S = QV @ P.T
top = np.argsort(-S, axis=1)[:, : a.topk]
rows = [{"qid": qid, "ids": [int(i) for i in top[n]], "scores": [float(S[n, i]) for i in top[n]]}
        for n, qid in enumerate(qids)]
write_json(DATA / "retrieval_top20.json", rows, indent=None)
prior = {r["qid"]: r for r in read_json(PRIOR_DATA / "results_v2" / "retrieval.json")}
same5 = np.mean([prior[r["qid"]]["ids"] == r["ids"][:5] for r in rows])
same1 = np.mean([prior[r["qid"]]["ids"][0] == r["ids"][0] for r in rows])
set5 = np.mean([len(set(prior[r["qid"]]["ids"]) & set(r["ids"][:5])) / 5 for r in rows])
dscore = np.mean([abs(prior[r["qid"]]["scores"][0] - r["scores"][0]) for r in rows])
agree = {"identical_top5_lists": float(same5), "identical_top1": float(same1), "top5_set_overlap": float(set5),
         "mean_abs_top1_score_diff": float(dscore), "desktop_passage_embed_s": t_p,
         "note": "desktop exact search vs Pi HNSW (ef=100); small differences come from HNSW and float kernels"}
write_json(out_dir("e1") / "retrieval_agreement.json", agree)
print(json.dumps(agree, indent=1))
del m

# ---------------------------------------------------------------- 2. MiniLM windows
name = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(name, revision=hf_revision(name))
count = lambda s: len(tok(s)["input_ids"])  # includes [CLS] and [SEP]
sents = read_json(DATA / "corpus_sentences.json")
W = []
for s in sents:
    for text, idx in sentence_windows(s["sentences"], count, a.seq_len):
        W.append({"wid": len(W), "pid": s["pid"], "sidx": idx, "text": text, "n_tok": count(text)})
write_json(DATA / f"windows_minilm_L{a.seq_len}.json", W, indent=None)
m = SentenceTransformer(name, revision=hf_revision(name), device=a.device)
m.max_seq_length = a.seq_len
WV = m.encode([w["text"] for w in W], batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
QV2 = m.encode([q["question"] for q in Q], batch_size=64, normalize_embeddings=True).astype(np.float32)
np.savez(DATA / f"vec_minilm_L{a.seq_len}.npz", W=WV, Q=QV2, qids=np.array(qids),
         pid=np.array([w["pid"] for w in W]))
over = sum(w["n_tok"] > a.seq_len for w in W)
print(f"{len(W)} windows (<= {a.seq_len} tokens; {over} single sentences longer than the budget are truncated)")
