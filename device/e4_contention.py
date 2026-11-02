"""E4 (Pi): does a concurrently running verifier slow down the LLM?

llama-server (Qwen2.5-3B, the prior study's build) streams a fixed-length generation (n_predict tokens,
ignore_eos, no prompt cache) on a fixed ~650-token prompt, `repeats` times per condition, while a
background worker process runs the encoder:
  none                       no background work (reference)
  ort-int8:T / ort-fp32:T    encoder on the CPU with T ONNX Runtime threads
in two modes that mirror the use cases where overlap actually happens:
  saturate   back-to-back batches (background re-indexing of a pack while serving)
  periodic   one batch of 8 every `period` seconds (sentence-level verification during decoding)
Recorded: TTFT, inter-token latency (p50/p95/p99), decode tok/s, worker throughput and latency, CPU
use per process, PMIC power, temperature, throttling. --pin runs llama-server on cores 0-2 and the
worker on core 3 (default: no pinning, the OS scheduler decides).

Note: the pre-generation gate/pruning step runs BEFORE generation and does not overlap with it in a
single-user pipeline; its cost is measured directly in E2, not here.

Usage: python device/e4_contention.py [--conditions none,ort-int8:1,ort-int8:2,ort-int8:4]
       [--modes saturate,periodic] [--model all-minilm-l6-v2] [--repeats 8] [--pin]"""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.llm import SYSTEM_PROMPT, start_server, stop_server, stream_completion  # noqa: E402
from ragtrust.paths import DATA, corpus, out_dir, questions, read_json, read_jsonl, write_json  # noqa: E402
from ragtrust.power import Telemetry, throttled  # noqa: E402
from ragtrust.stats import summarize  # noqa: E402


def worker(model, backend, threads, mode, period, batch, seq_len, cpu, ready, stop, outq):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ragtrust.encoders import Encoder
    enc = Encoder(model, backend, threads=threads, batch=batch, seq_len=seq_len)
    if enc.card["task"] == "embed":
        items = [w["text"] for w in read_json(DATA / f"windows_minilm_L{seq_len}.json")]
        run = lambda xs: enc.embed(xs)
    else:
        Q = {q["qid"]: q["question"] for q in questions()}
        SENT = {s["pid"]: s["sentences"] for s in read_json(DATA / "corpus_sentences.json")}
        items = [(Q[r["qid"]], SENT[x["pid"]][x["sidx"]]) for r in read_jsonl(out_dir("e1") / "sentences_top10.jsonl")
                 for x in r["items"]]
        run = lambda xs: enc.score_pairs([x[0] for x in xs], [x[1] for x in xs])
    run(items[:batch])
    ready.set()
    calls, pos = [], 0
    while not stop.is_set():
        xs = items[pos:pos + batch]
        pos = (pos + batch) % (len(items) - batch)
        t0 = time.time()
        run(xs)
        t1 = time.time()
        calls.append((t0, t1, len(xs)))
        if mode == "periodic":
            time.sleep(max(0.0, period - (t1 - t0)))
    enc.close()
    outq.put(calls)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="none,ort-int8:1,ort-int8:2,ort-int8:4")
    ap.add_argument("--modes", default="saturate,periodic")
    ap.add_argument("--model", default="all-minilm-l6-v2")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    OUT = out_dir("e4")
    C, Q = corpus(), questions()
    retr = {r["qid"]: r for r in read_json(DATA / "retrieval_top20.json")}
    q = next(x for x in Q if x["condition"] == "in_answerable")
    ctx = "\n\n".join(f"[{j + 1}] {C[p]['text']}" for j, p in enumerate(retr[q["qid"]]["ids"][:3]))
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\nQuestion: {q['question']}\nAnswer:"
    srv = start_server(os.environ["LLAMA_SERVER"], os.environ["MODEL_3B"], int(os.environ.get("LLM_THREADS", 4)),
                       port=a.port, log_path=OUT / "server.log", cpus="0-2" if a.pin else None)
    server = f"http://127.0.0.1:{a.port}"
    stream_completion(server, prompt, n_predict=16)  # warm-up
    ctxmp = mp.get_context("spawn")
    rows = []
    with Telemetry(period=0.25, pids={"llama": srv.pid}) as tel:
        for cond in a.conditions.split(","):
            for mode in (["-"] if cond == "none" else a.modes.split(",")):
                be, th = (cond.split(":") + ["4"])[:2] if cond != "none" else (None, None)
                ready, stop, outq, wp = ctxmp.Event(), ctxmp.Event(), ctxmp.Queue(), None
                if be:
                    batch = 32 if mode == "saturate" else 8
                    wp = ctxmp.Process(target=worker, args=(a.model, be, int(th), mode, a.period, batch, a.seq_len,
                                                            3 if a.pin else None, ready, stop, outq))
                    wp.start()
                    if not ready.wait(600):
                        raise RuntimeError(f"worker {cond} did not start")
                    time.sleep(2.0)
                ta = time.time()
                reqs = [stream_completion(server, prompt, n_predict=a.n_predict) for _ in range(a.repeats)]
                tb = time.time()
                calls = []
                if wp:
                    stop.set()
                    calls = outq.get(timeout=300)
                    wp.join(timeout=60)
                itl = [x for r in reqs if r["ok"] for x in r["itl_ms"]]
                inwin = [c for c in calls if c[0] >= ta and c[1] <= tb]
                row = {"condition": cond, "mode": mode, "pinned": a.pin, "model": a.model,
                       "ttft_ms": summarize([r["ttft_ms"] for r in reqs if r["ok"]]),
                       "itl_ms": summarize(itl),
                       "decode_tok_s_server": summarize([r["timings"].get("predicted_per_second") for r in reqs if r["ok"]]),
                       "prefill_tok_s_server": summarize([r["timings"].get("prompt_per_second") for r in reqs if r["ok"]]),
                       "worker_items_per_s": (sum(c[2] for c in inwin) / (tb - ta)) if inwin else None,
                       "worker_call_ms": summarize([(c[1] - c[0]) * 1e3 for c in inwin]),
                       "telemetry": tel.window(ta, tb), "throttled": throttled()}
                rows.append(row)
                print(f"{cond:12s} {mode:9s} ITL p50={row['itl_ms']['median']:.1f} p99={row['itl_ms']['p99']:.1f} ms  "
                      f"decode={row['decode_tok_s_server']['median']:.2f} tok/s  worker={row['worker_items_per_s']}", flush=True)
                write_json(OUT / f"contention{'_pinned' if a.pin else ''}.json", rows)
    stop_server(srv)
