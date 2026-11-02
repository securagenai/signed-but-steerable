"""E2 (Pi): generation with Qwen2.5-3B (llama-server from the prior study) on pre-built contexts.

The request is identical to the prior study (system prompt, max_tokens=64, temperature 0, seed 42), and
rows carry the same fields as results_v2/gen_*.jsonl, so prior and new configurations are analysed with
the same code. `k3_full` re-runs the prior main configuration as a reproducibility check (expect the
same answers as data/prior/results_v2/gen_q3_k3.jsonl; use --limit to keep it short).

--online-selector recomputes the sentence selection on the device (ort-fp32 / ort-int8) instead
of using the desktop FP32 selection, and records the selector latency and its agreement with the
desktop selection: the end-to-end path whose cost the paper reports.

Usage: python device/e2_generate.py --config k5_ce3 [--limit 300] [--server http://127.0.0.1:8080]
       [--online-selector ort-int8 --selector-model ms-marco-minilm-l6]"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.llm import build_messages, chat  # noqa: E402
from ragtrust.paths import DATA, PRIOR_DATA, done_keys, out_dir, questions, read_json, write_json  # noqa: E402
from ragtrust.power import Telemetry, throttled  # noqa: E402
from ragtrust.textnorm import abstained  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--server", default="http://127.0.0.1:8080")
ap.add_argument("--online-selector", default=None, choices=[None, "ort-fp32", "ort-int8"])
ap.add_argument("--selector-model", default="ms-marco-minilm-l6")
ap.add_argument("--selector-threads", type=int, default=4)
a = ap.parse_args()

ctx = read_json(out_dir("e2") / "contexts" / f"{a.config}.json")
Q = [q for q in questions() if q["qid"] in ctx]
if a.limit:
    Q = Q[: a.limit]
prior_ret = {r["qid"]: r for r in read_json(PRIOR_DATA / "results_v2" / "retrieval.json")}
R = {r["qid"]: r for r in read_json(DATA / "retrieval_top20.json")}
name = a.config + (f"__online_{a.online_selector}" if a.online_selector else "")
OUT = out_dir("e2")
f = OUT / f"gen_{name}.jsonl"
done = done_keys(f)

sel = None
if a.online_selector:
    from ragtrust.encoders import Encoder
    m = re.fullmatch(r"k(\d+)_(ce|bi)(\d+)", a.config)
    if not m:
        raise SystemExit("--online-selector needs a pruned config (k*_ce* or k*_bi*)")
    K, kind, M = int(m.group(1)), m.group(2), int(m.group(3))
    SENT = {s["pid"]: s["sentences"] for s in read_json(DATA / "corpus_sentences.json")}
    sel = Encoder(a.selector_model, a.online_selector, threads=a.selector_threads, batch=64)

with Telemetry(period=0.5) as tel, open(f, "a") as fo:
    for i, q in enumerate(Q):
        if q["qid"] in done:
            continue
        c = ctx[q["qid"]]
        blocks, extra = c["blocks"], {}
        if sel is not None:
            ids = R[q["qid"]]["ids"][:K]
            cand = [(p, si) for p in ids for si in range(len(SENT[p]))]
            sel.reset_timing()
            t0 = time.perf_counter()
            if sel.card["task"] == "embed":
                v = sel.embed([q["question"]] + [SENT[p][si] for p, si in cand])
                s = v[1:] @ v[0]
            else:
                s = sel.score_pairs([q["question"]] * len(cand), [SENT[p][si] for p, si in cand])
            order = sorted(range(len(cand)), key=lambda j: (-float(s[j]), ids.index(cand[j][0]), cand[j][1]))[:M]
            chosen = {cand[j] for j in order}
            blocks = [" ".join(SENT[p][si] for si in sorted(si for pp, si in chosen if pp == p)) for p in ids
                      if any(pp == p for pp, _ in chosen)]
            desk = {tuple(x) for x in c["sel"]}
            extra = {"selector_ms": (time.perf_counter() - t0) * 1e3, "selector_backend": a.online_selector,
                     "selector_stage_ms": dict(sel.timing), "n_candidates": len(cand),
                     "selection_jaccard_vs_desktop": len(chosen & desk) / max(1, len(chosen | desk))}
        msgs = build_messages(q["question"], blocks)
        ta = time.time()
        ans, tim, us, gms = chat(a.server, msgs)
        tb = time.time()
        w = tel.window(ta, tb)
        row = {"qid": q["qid"], "config": name, "k": None, "condition": q["condition"], "golds": q["golds"],
               "gold_pid": q["gold_pid"], "ids": c["pids"], "scores": prior_ret[q["qid"]]["scores"],
               "retrieval_ms": prior_ret[q["qid"]]["retrieval_ms"], "answer": ans,
               "model_abstained": abstained(ans), "generation_ms": gms,
               "prompt_tokens": us.get("prompt_tokens"), "completion_tokens": us.get("completion_tokens"),
               "prompt_tok_per_s": tim.get("prompt_per_second"), "gen_tok_per_s": tim.get("predicted_per_second"),
               "prompt_ms": tim.get("prompt_ms"), "predicted_ms": tim.get("predicted_ms"),
               "mean_power_w": w.get("w_mean"), "cpu_mean": w.get("cpu_mean"), "temp_max": w.get("temp_max"),
               "n_blocks": len(blocks), "ctx_chars": sum(len(b) for b in blocks), **extra}
        fo.write(json.dumps(row) + "\n")
        fo.flush()
        print(f"[{name} {i + 1}/{len(Q)}] {q['condition']:16s} prompt={row['prompt_tokens']} "
              f"gen={gms / 1000:.1f}s -> {ans[:50]!r}", flush=True)
write_json(OUT / f"status_{name}.json", {"config": name, "throttled_at_end": throttled()})
if sel is not None:
    sel.close()
print("CONFIG DONE", name)
