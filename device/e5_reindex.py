"""E5 (Pi): re-indexing the knowledge pack on the device and the fidelity of INT8 embeddings.

The prior study embedded the 950 passages with BGE-M3 on the CPU in 1,725 s, which is why packs shipped
pre-computed vectors (and why E7's desynchronisation attack exists). Here the corpus is embedded as
sentence windows of <= L tokens with all-MiniLM-L6-v2 on ONNX Runtime FP32 and INT8.
Reported per backend: wall time, items/s, CPU cores used, power; cosine fidelity against the device
FP32 reference and against the desktop FP32 vectors; retrieval quality (recall@1/3/5 of the gold
passage for answerable questions; AUROC of the top-1 score, answerable vs unanswerable and vs
out-of-corpus) with symmetric and asymmetric (FP32 queries) indexing. The low quantiles of the honest
cosine define the tolerance an INT8-based vector check must allow (used by E7 and E8).

Usage: python device/e5_reindex.py [--backends ort-fp32,ort-int8] [--batch 64] [--threads 4]"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.encoders import Encoder  # noqa: E402
from ragtrust.paths import DATA, PRIOR_DATA, out_dir, questions, read_json, write_json  # noqa: E402
from ragtrust.power import Telemetry, throttled  # noqa: E402
from ragtrust.stats import auroc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="all-minilm-l6-v2")
ap.add_argument("--backends", default="ort-fp32,ort-int8")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--seq-len", type=int, default=128)
a = ap.parse_args()
import psutil  # noqa: E402

W = read_json(DATA / f"windows_minilm_L{a.seq_len}.json")
pid = np.array([w["pid"] for w in W])
Q = questions()
OUT = out_dir("e5")
desk = np.load(DATA / f"vec_minilm_L{a.seq_len}.npz")
proc = psutil.Process(os.getpid())


def retrieval(Wv, Qv):
    S = Qv @ Wv.T
    n_p = int(pid.max()) + 1
    P = np.full((len(Q), n_p), -1.0, np.float32)
    for j in range(Wv.shape[0]):
        P[:, pid[j]] = np.maximum(P[:, pid[j]], S[:, j])
    order = np.argsort(-P, axis=1)
    res = {}
    A = [i for i, q in enumerate(Q) if q["condition"] == "in_answerable"]
    for k in (1, 3, 5):
        res[f"recall@{k}"] = float(np.mean([Q[i]["gold_pid"] in order[i, :k] for i in A]))
    top1 = P.max(1)
    by = lambda c: [top1[i] for i, q in enumerate(Q) if q["condition"] == c]
    res["auroc_answerable_vs_unanswerable"] = auroc(by("in_answerable"), by("in_unanswerable"))
    res["auroc_answerable_vs_ooc"] = auroc(by("in_answerable"), by("out_of_corpus"))
    return res


summary = {"n_windows": len(W), "seq_len": a.seq_len, "backends": {}, "fidelity": {},
           "prior_bge_m3": {"embed_seconds": read_json(PRIOR_DATA / "pack_v2_build_stats.json")["embed_seconds"]
                            if (PRIOR_DATA / "pack_v2_build_stats.json").exists() else None,
                            "retrieval": read_json(PRIOR_DATA / "results_v2" / "summary_v2.json")["retrieval"]
                            if (PRIOR_DATA / "results_v2" / "summary_v2.json").exists() else None}}
vecs = {}
with Telemetry(period=0.5) as tel:
    for be in a.backends.split(","):
        enc = Encoder(a.model, be, threads=a.threads, batch=a.batch, seq_len=a.seq_len, 
                      cpu_padding="longest")
        enc.embed([w["text"] for w in W[: a.batch]])  # warm-up
        enc.reset_timing()
        c0, t0, ta = proc.cpu_times(), time.perf_counter(), time.time()
        V = enc.embed([w["text"] for w in W])
        wall = time.perf_counter() - t0
        c1, tb = proc.cpu_times(), time.time()
        Qv = enc.embed([q["question"] for q in Q])
        vecs[be] = (V, Qv)
        np.savez(OUT / f"vectors_{be}.npz", W=V, Q=Qv)
        summary["backends"][be] = {"seconds": wall, "windows_per_s": len(W) / wall,
                                   "cpu_cores_used": ((c1.user - c0.user) + (c1.system - c0.system)) / wall,
                                   "stage_ms": dict(enc.timing), "telemetry": tel.window(ta, tb),
                                   "throttled": throttled(), "retrieval_symmetric": retrieval(V, Qv)}
        enc.close()
        print(be, json.dumps(summary["backends"][be]["retrieval_symmetric"]), f"{wall:.1f} s", flush=True)

ref_W, ref_Q = vecs.get("ort-fp32", (desk["W"], desk["Q"]))
for be, (V, Qv) in vecs.items():
    cw = (V * ref_W).sum(1)
    cd = (V * desk["W"]).sum(1)
    summary["fidelity"][be] = {"cos_vs_device_fp32": {"min": float(cw.min()), "p001": float(np.quantile(cw, 0.001)),
                                                      "p01": float(np.quantile(cw, 0.01)), "median": float(np.median(cw))},
                               "cos_vs_desktop_fp32": {"min": float(cd.min()), "median": float(np.median(cd))},
                               "cos_p001": float(np.quantile(cw, 0.001))}
    summary["backends"][be]["retrieval_asymmetric_fp32_queries"] = retrieval(V, ref_Q)
write_json(OUT / "summary.json", summary)
print(json.dumps(summary["fidelity"], indent=1))
