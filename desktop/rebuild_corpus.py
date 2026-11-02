"""Rebuilds the prior study's knowledge-pack corpus (950 SQuAD 2.0 dev passages, 7 of 35 articles held
out) deterministically, with the same seed and procedure as build_pack_v2.py, and verifies it against
the released question set. Writes data/corpus.json (byte-identical to pack_v2/corpus.json, so its
SHA-256 can be compared with the prior manifest) and data/corpus_sentences.json (stable sentence ids).

Usage: python desktop/rebuild_corpus.py [--prior-manifest PATH_TO/pack_v2/manifest.json]"""
import argparse, hashlib, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, PRIOR_DATA, write_json  # noqa: E402
from ragtrust.sentences import split_sentences  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prior-manifest", default=None)
a = ap.parse_args()

SEED, HOLDOUT = 2026, 0.2
squad = json.load(open(DATA / "dev-v2.0.json"))["data"]
rng = random.Random(SEED)
titles = sorted(x["title"] for x in squad)
held = set(rng.sample(titles, max(1, round(len(titles) * HOLDOUT))))
corpus = []
for art in squad:
    for para in art["paragraphs"]:
        if art["title"] not in held:
            corpus.append({"id": len(corpus), "title": art["title"], "text": para["context"]})

qs = json.load(open(PRIOR_DATA / "questions_v2.json"))
problems = []
for q in qs:
    if q["gold_pid"] is None:
        if q["condition"] != "out_of_corpus" or q["title"] not in held:
            problems.append(("ooc", q["qid"]))
        continue
    c = corpus[q["gold_pid"]]
    if c["title"] != q["title"]:
        problems.append(("title", q["qid"]))
    if q["condition"] == "in_answerable" and not any(g in c["text"] for g in q["golds"]):
        problems.append(("answer", q["qid"]))
if len(corpus) != 950 or problems:
    raise SystemExit(f"corpus does not match the prior study: n={len(corpus)}, problems={problems[:5]}")

raw = json.dumps(corpus).encode()  # same serialisation as json.dump(corpus, f) in build_pack_v2.py
(DATA / "corpus.json").write_bytes(raw)
digest = hashlib.sha256(raw).hexdigest()
report = {"n_passages": len(corpus), "held_out_titles": sorted(held), "corpus_sha256": digest,
          "questions_checked": len(qs)}
if a.prior_manifest:
    man = json.load(open(a.prior_manifest))
    report["prior_manifest_corpus_sha256"] = man["files"]["corpus.json"]
    report["byte_identical_to_prior_pack"] = man["files"]["corpus.json"] == digest
sents = [{"pid": c["id"], "sentences": split_sentences(c["text"])} for c in corpus]
write_json(DATA / "corpus_sentences.json", sents, indent=None)
report["n_sentences"] = sum(len(s["sentences"]) for s in sents)
write_json(DATA / "corpus_report.json", report)
print(json.dumps(report, indent=1))
