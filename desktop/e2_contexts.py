"""E2 (desktop): builds the prompt contexts for the generation runs on the Pi.

Config names follow  k{K}_{full|ce|bi}{M}:
  k3_full, k5_full   the prior study's contexts (whole passages), used for the reproducibility check
  k5_ce3             top-5 passages -> 3 sentences with the highest cross-encoder score
  k10_ce3, k10_ce5   top-10 passages -> 3 / 5 sentences
  k5_bi3             top-5 passages -> 3 sentences by bi-encoder cosine
Selected sentences are grouped by source passage (retrieval order) and kept in their original order,
so the prompt keeps the prior study's "[n] text" format and system prompt.

Output: results/e2/contexts/<config>.json  {qid: {"blocks": [...], "pids": [...], "sel": [[pid, sidx], ...]}}"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, corpus, out_dir, questions, read_json, read_jsonl, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="k3_full,k5_full,k5_ce3,k10_ce3,k10_ce5,k5_bi3")
a = ap.parse_args()
Q, C = questions(), corpus()
SENT = {s["pid"]: s["sentences"] for s in read_json(DATA / "corpus_sentences.json")}
R = {r["qid"]: r for r in read_json(DATA / "retrieval_top20.json")}
S = {r["qid"]: r["items"] for r in read_jsonl(out_dir("e1") / "sentences_top10.jsonl")}
D = out_dir("e2/contexts")
summary = {}
for cfg in a.configs.split(","):
    m = re.fullmatch(r"k(\d+)_(full|ce|bi)(\d*)", cfg)
    if not m:
        raise SystemExit(f"bad config name {cfg}")
    k, kind, M = int(m.group(1)), m.group(2), int(m.group(3) or 0)
    out = {}
    for q in Q:
        ids = R[q["qid"]]["ids"][:k]
        if kind == "full":
            out[q["qid"]] = {"blocks": [C[p]["text"] for p in ids], "pids": ids, "sel": None}
            continue
        items = [x for x in S[q["qid"]] if x["rank"] < k]
        best = sorted(items, key=lambda x: (-x[kind], x["rank"], x["sidx"]))[:M]
        blocks, pids, sel = [], [], []
        for p in ids:
            chosen = sorted(x["sidx"] for x in best if x["pid"] == p)
            if chosen:
                blocks.append(" ".join(SENT[p][i] for i in chosen))
                pids.append(p)
                sel += [[p, i] for i in chosen]
        out[q["qid"]] = {"blocks": blocks, "pids": pids, "sel": sel}
    write_json(D / f"{cfg}.json", out, indent=None)
    chars = [sum(len(b) for b in v["blocks"]) for v in out.values()]
    gold = [q["gold_pid"] in out[q["qid"]]["pids"] for q in Q if q["condition"] == "in_answerable"]
    summary[cfg] = {"mean_context_chars": sum(chars) / len(chars), "gold_passage_kept": sum(gold) / len(gold)}
    print(f"{cfg:10s} mean chars {summary[cfg]['mean_context_chars']:7.0f}  gold passage kept {summary[cfg]['gold_passage_kept']:.2f}")
write_json(out_dir("e2") / "contexts_summary.json", summary)
