"""Exploratory out-of-distribution check of E1 (PROTOCOL.md deviations log): does the extractive-QA
no-answer margin still separate answerable from unanswerable questions outside SQuAD?

Data: MS MARCO v2.1 validation (Bing queries, ~10 retrieved web passages each). Queries whose answer is
"No Answer Present." are natural in-topic unanswerables: the passages are on topic but annotators found
no answer in them. A seeded sample of --n answerable and --n unanswerable queries is scored with the same
models and settings as desktop/e1_score.py, over all passages of each query:
  bi_max     max cosine(query, passage)                 all-MiniLM-L6-v2 (seq 128)
  ce_max     max cross-encoder logit(query, passage)    ms-marco-MiniLM-L-6-v2 (512)
  qa_margin  max best-span minus no-answer score         deepset/minilm-uncased-squad2 (384 windows)
Note the reversal of the E1 caveat: here the cross-encoder is IN distribution (trained on MS MARCO) and
the QA model is OUT of distribution (trained on SQuAD 2.0).
Writes results/e1/ood_msmarco.json. Usage: python desktop/e1_ood_msmarco.py [--n 500] [--device cuda]"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import hf_revision, out_dir, write_json  # noqa: E402
from ragtrust.stats import bootstrap_auroc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=20261101)
a = ap.parse_args()
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
NA = "No Answer Present."
idx_ans = [i for i, x in enumerate(ds["answers"]) if x and x[0] != NA]
idx_un = [i for i, x in enumerate(ds["answers"]) if x and x[0] == NA]
rng = random.Random(a.seed)
pick = [(i, 1) for i in rng.sample(idx_ans, a.n)] + [(i, 0) for i in rng.sample(idx_un, a.n)]
Q = [{"query": ds[i]["query"], "passages": [p for p in ds[i]["passages"]["passage_text"] if p.strip()], "answerable": y,
      "query_id": int(ds[i]["query_id"])} for i, y in pick]
Q = [q for q in Q if q["passages"]]
print(f"{len(Q)} queries ({sum(q['answerable'] for q in Q)} answerable), {sum(len(q['passages']) for q in Q)} passages", flush=True)
info = {"dataset": "microsoft/ms_marco v2.1 validation", "seed": a.seed, "n_per_class": a.n, "device": dev, "models": {}}

t0 = time.time()
name = "sentence-transformers/all-MiniLM-L6-v2"
bi = SentenceTransformer(name, revision=hf_revision(name), device=dev)
bi.max_seq_length = 128
for q in Q:
    v = bi.encode([q["query"]] + q["passages"], normalize_embeddings=True)
    q["bi_max"] = float((v[1:] @ v[0]).max())
info["models"][name] = {"revision": hf_revision(name), "seconds": time.time() - t0}
del bi

t0 = time.time()
name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tok = AutoTokenizer.from_pretrained(name, revision=hf_revision(name))
ce = AutoModelForSequenceClassification.from_pretrained(name, revision=hf_revision(name)).to(dev).eval()
with torch.no_grad():
    for q in Q:
        enc = tok([q["query"]] * len(q["passages"]), q["passages"], truncation="only_second", max_length=512,
                  padding=True, return_tensors="pt").to(dev)
        q["ce_max"] = float(ce(**enc).logits[:, 0].max())
info["models"][name] = {"revision": hf_revision(name), "seconds": time.time() - t0}
del ce

t0 = time.time()
name = "deepset/minilm-uncased-squad2"
qtok = AutoTokenizer.from_pretrained(name, revision=hf_revision(name))
qa = AutoModelForQuestionAnswering.from_pretrained(name, revision=hf_revision(name)).to(dev).eval()
band = np.triu(np.ones((384, 384), bool)) & ~np.triu(np.ones((384, 384), bool), 30)


def qa_margin(question, passage):  # identical to desktop/e1_score.py
    enc = qtok(question, passage, truncation="only_second", max_length=384, stride=128, padding="max_length",
               return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="np")
    enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    with torch.no_grad():
        o = qa(**{k: torch.tensor(v).to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
    S, E = o.start_logits.float().cpu().numpy(), o.end_logits.float().cpu().numpy()
    best, null = -1e9, 1e9
    for w in range(S.shape[0]):
        ctx = np.array([sid == 1 for sid in enc.sequence_ids(w)])
        null = min(null, float(S[w, 0] + E[w, 0]))
        M = np.where(band & ctx[:, None] & ctx[None, :], S[w][:, None] + E[w][None, :], -1e9)
        best = max(best, float(M.max()))
    return best - null


for j, q in enumerate(Q):
    q["qa_margin"] = max(qa_margin(q["query"], p) for p in q["passages"])
    if j % 200 == 0:
        print(f"[qa {j + 1}/{len(Q)}]", flush=True)
info["models"][name] = {"revision": hf_revision(name), "seconds": time.time() - t0}

res = {"info": info, "n": len(Q), "auroc_answerable_vs_unanswerable": {}}
for f in ("bi_max", "ce_max", "qa_margin"):
    pos = [q[f] for q in Q if q["answerable"]]
    neg = [q[f] for q in Q if not q["answerable"]]
    res["auroc_answerable_vs_unanswerable"][f] = bootstrap_auroc(pos, neg)
    print(f"{f:10s} AUROC answerable vs unanswerable {res['auroc_answerable_vs_unanswerable'][f]}", flush=True)
write_json(out_dir("e1") / "ood_msmarco.json", res)
with open(out_dir("e1") / "ood_msmarco_rows.jsonl", "w") as fo:
    for q in Q:
        fo.write(json.dumps({k: q[k] for k in ("query_id", "answerable", "bi_max", "ce_max", "qa_margin")}) + "\n")
