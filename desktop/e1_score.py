"""E1 (desktop, FP32 reference): pre-generation answerability scores for the 300 prior-study questions.

For the top-10 BGE-M3 passages of every question the script computes
  bi   cosine(question, sentence)            all-MiniLM-L6-v2, sentence windows truncated at L
  ce   cross-encoder logit(question, sentence) ms-marco-MiniLM-L-6-v2, pairs truncated at L
  cep  cross-encoder logit(question, passage)  same model, passages truncated at 512 (reference)
  qa   best-span minus no-answer score        deepset/minilm-uncased-squad2, 384-token windows
and aggregates them over the top-k passages (k = 3, 5, 10). L defaults to 128, the window length
used on the Pi, so that FP32 scores here and INT8 scores on the Pi see identical tokens.

Outputs (results/e1/): features.jsonl, sentences_top10.jsonl, qa_passages.jsonl, run_info.json"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, corpus, hf_revision, out_dir, questions, read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq-len", type=int, default=128)
ap.add_argument("--kmax", type=int, default=10)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)
dev = torch.device(a.device)
OUT = out_dir("e1")
Q, C = questions(), corpus()
SENT = {s["pid"]: s["sentences"] for s in read_json(DATA / "corpus_sentences.json")}
R = {r["qid"]: r for r in read_json(DATA / "retrieval_top20.json")}
info = {"seq_len": a.seq_len, "kmax": a.kmax, "torch": torch.__version__, "device": a.device, "models": {}}

# candidate sentences (question index, passage id, rank, sentence index)
cand = []
for qi, q in enumerate(Q):
    for rank, pid in enumerate(R[q["qid"]]["ids"][: a.kmax]):
        for si in range(len(SENT[pid])):
            cand.append((qi, pid, rank, si))
print(f"{len(cand)} (question, sentence) pairs over top-{a.kmax} passages")

# ---------------------------------------------------------------- bi-encoder
t0 = time.time()
name = "sentence-transformers/all-MiniLM-L6-v2"
bi = SentenceTransformer(name, revision=hf_revision(name), device=a.device)
bi.max_seq_length = a.seq_len
keys = sorted({(pid, si) for _, pid, _, si in cand})
kidx = {k: i for i, k in enumerate(keys)}
SV = bi.encode([SENT[p][s] for p, s in keys], batch_size=a.batch, normalize_embeddings=True)
QV = bi.encode([q["question"] for q in Q], batch_size=a.batch, normalize_embeddings=True)
bi_s = np.array([float(QV[qi] @ SV[kidx[(pid, si)]]) for qi, pid, _, si in cand])
info["models"][name] = {"revision": hf_revision(name), "seconds": time.time() - t0}
del bi


# ---------------------------------------------------------------- cross-encoder
def ce_scores(model_name, A, B, max_len):
    tok = AutoTokenizer.from_pretrained(model_name, revision=hf_revision(model_name))
    m = AutoModelForSequenceClassification.from_pretrained(model_name, revision=hf_revision(model_name)).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(A), a.batch):
            enc = tok(A[i:i + a.batch], B[i:i + a.batch], truncation="only_second", max_length=max_len,
                      padding=True, return_tensors="pt").to(dev)
            out.append(m(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out)


t0 = time.time()
name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ce_s = ce_scores(name, [Q[qi]["question"] for qi, _, _, _ in cand], [SENT[p][s] for _, p, _, s in cand], a.seq_len)
pp = [(qi, pid, rank) for qi, q in enumerate(Q) for rank, pid in enumerate(R[q["qid"]]["ids"][: a.kmax])]
cep_s = ce_scores(name, [Q[qi]["question"] for qi, _, _ in pp], [C[pid]["text"] for _, pid, _ in pp], 512)
info["models"][name] = {"revision": hf_revision(name), "seconds": time.time() - t0}

# ---------------------------------------------------------------- extractive QA with no-answer score
t0 = time.time()
name = "deepset/minilm-uncased-squad2"
qtok = AutoTokenizer.from_pretrained(name, revision=hf_revision(name))
qa = AutoModelForQuestionAnswering.from_pretrained(name, revision=hf_revision(name)).to(dev).eval()
info["models"][name] = {"revision": hf_revision(name), "num_hidden_layers": qa.config.num_hidden_layers}
MAXANS = 30
band = np.triu(np.ones((384, 384), bool)) & ~np.triu(np.ones((384, 384), bool), MAXANS)


def qa_margin(question, passage):
    enc = qtok(question, passage, truncation="only_second", max_length=384, stride=128, padding="max_length",
               return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="np")
    offs = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    with torch.no_grad():
        o = qa(**{k: torch.tensor(v).to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
    S, E = o.start_logits.float().cpu().numpy(), o.end_logits.float().cpu().numpy()
    best, null, span = -1e9, 1e9, ""
    for w in range(S.shape[0]):
        ctx = np.array([sid == 1 for sid in enc.sequence_ids(w)]) if hasattr(enc, "sequence_ids") else None
        if ctx is None:
            ctx = (enc["token_type_ids"][w] == 1) & (enc["attention_mask"][w] == 1)
        null = min(null, float(S[w, 0] + E[w, 0]))
        M = S[w][:, None] + E[w][None, :]
        M = np.where(band & ctx[:, None] & ctx[None, :], M, -1e9)
        i, j = np.unravel_index(np.argmax(M), M.shape)
        if M[i, j] > best:
            best = float(M[i, j])
            span = passage[offs[w][i][0]:offs[w][j][1]]
    return best - null, span


qa_rows = {}
for qi, q in enumerate(Q):
    items = []
    for rank, pid in enumerate(R[q["qid"]]["ids"][: a.kmax]):
        mgn, span = qa_margin(q["question"], C[pid]["text"])
        items.append({"pid": pid, "rank": rank, "margin": mgn, "span": span})
    qa_rows[q["qid"]] = items
    if qi % 50 == 0:
        print(f"[qa {qi + 1}/{len(Q)}] margin(top1)={items[0]['margin']:.2f}", flush=True)
info["models"][name]["seconds"] = time.time() - t0

# ---------------------------------------------------------------- aggregation
by_q = {}
for (qi, pid, rank, si), b, c in zip(cand, bi_s, ce_s):
    by_q.setdefault(qi, []).append({"pid": pid, "rank": rank, "sidx": si, "bi": float(b), "ce": float(c)})
cep = {}
for (qi, pid, rank), s in zip(pp, cep_s):
    cep.setdefault(qi, []).append((rank, float(s)))
with open(OUT / "features.jsonl", "w") as ff, open(OUT / "sentences_top10.jsonl", "w") as fs, \
        open(OUT / "qa_passages.jsonl", "w") as fq:
    for qi, q in enumerate(Q):
        r = R[q["qid"]]
        gold_rank = r["ids"].index(q["gold_pid"]) if q["gold_pid"] in r["ids"] else None
        feats = {}
        for k in (3, 5, 10):
            if k > a.kmax:
                continue
            S_ = [x for x in by_q[qi] if x["rank"] < k]
            feats[f"k{k}"] = {"top1": r["scores"][0], "bi_max": max(x["bi"] for x in S_),
                              "ce_max": max(x["ce"] for x in S_),
                              "cep_max": max(s for rk, s in cep[qi] if rk < k),
                              "qa_margin": max(x["margin"] for x in qa_rows[q["qid"]] if x["rank"] < k),
                              "n_sentences": len(S_)}
        ff.write(json.dumps({"qid": q["qid"], "condition": q["condition"], "gold_pid": q["gold_pid"],
                             "gold_rank": gold_rank, "f": feats}) + "\n")
        fs.write(json.dumps({"qid": q["qid"], "items": by_q[qi]}) + "\n")
        fq.write(json.dumps({"qid": q["qid"], "items": qa_rows[q["qid"]]}) + "\n")
write_json(OUT / "run_info.json", info)
print("E1 scores written to", OUT)
