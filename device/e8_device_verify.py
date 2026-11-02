"""E8 (Pi): verification of the update cases built by desktop/e8_build_cases.py.

Stage 1, cryptographic/structural (ragtrust.packs.verify_update): signature with the declared content
  key, file hashes, sequence, expiry, compatibility, parent hash, declared vs actual change set, editor
  threshold for SUPERSEDE/DELETE, transparency-log chain. Timed (median of --reps runs).
Stage 2, vector consistency for accepted packs: the device re-embeds every new chunk and compares with
  the shipped vector (cosine < tau -> flagged). tau = 0.999 with an FP32 recomputation; with an INT8 recomputation it
  must be looser (default: the 0.1% quantile of honest cosines measured in E5, minus 0.01).
Stage 3, semantic screen of ADD chunks: nearest existing chunks (device vectors), most similar sentence
  pairs, NLI contradiction probability (premise = existing sentence, hypothesis = new sentence); a chunk
  is quarantined when the maximum is >= theta. SUPERSEDE chunks are not screened: they were authorised.

Usage: python device/e8_device_verify.py [--vector-backend ort-fp32|ort-int8] [--nli-model nli-minilm2-l6-h768]
                                         [--theta 0.5] [--tau auto]"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust import packs  # noqa: E402
from ragtrust.encoders import Encoder  # noqa: E402
from ragtrust.paths import RESULTS, out_dir, read_json, write_json  # noqa: E402
from ragtrust.sentences import split_sentences  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vector-backend", default="ort-fp32")
ap.add_argument("--embed-model", default="all-minilm-l6-v2")
ap.add_argument("--nli-model", default="nli-minilm2-l6-h768")
ap.add_argument("--nli-backend", default="ort-fp32")
ap.add_argument("--theta", type=float, default=0.5)
ap.add_argument("--tau", default="auto")
ap.add_argument("--neighbors", type=int, default=3)
ap.add_argument("--pairs", type=int, default=5)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
E = out_dir("e8")
K = E / "keys"
trust, state, dev = (read_json(K / f) for f in ("trust_store.json", "device_state.json", "device_profile.json"))
cases = read_json(E / "cases.json")
if a.tau == "auto":
    tau = 0.999
else:
    tau = float(a.tau)
emb = Encoder(a.embed_model, a.vector_backend, batch=64, seq_len=cases["seq_len"], cpu_padding="longest")
nli = Encoder(a.nli_model, a.nli_backend, batch=32)
base = json.loads((E / "packs" / "_base" / "chunks.json").read_text())
t0 = time.perf_counter()
BV = emb.embed([c["text"] for c in base])
base_embed_s = time.perf_counter() - t0
base_by_id = {c["id"]: i for i, c in enumerate(base)}

results, chunk_rows = [], []
for cs in cases["cases"]:
    pdir = E / "packs" / cs["case"]
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        ok, why, info = packs.verify_update(pdir, state, trust, dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    row = {**cs, "accepted_crypto": ok, "reason": why, "verify_ms_median": float(np.median(ts)), "info": info}
    if ok:
        man = json.loads((pdir / "manifest.json").read_bytes())
        chunks = {c["id"]: c for c in json.loads((pdir / "chunks.json").read_text())}
        order = [c["id"] for c in json.loads((pdir / "chunks.json").read_text())]
        new_ids = [o["id"] for o in man["ops"] if o["op"] in ("ADD", "SUPERSEDE")]
        add_ids = [o["id"] for o in man["ops"] if o["op"] == "ADD"]
        t0 = time.perf_counter()
        NV = emb.embed([chunks[i]["text"] for i in new_ids]) if new_ids else np.zeros((0, BV.shape[1]))
        row["reembed_ms"] = (time.perf_counter() - t0) * 1e3
        flagged_vec = []
        if man.get("vectors_shipped") and new_ids:
            SV = np.load(pdir / "vectors.npy")
            pos = {cid: j for j, cid in enumerate(order)}
            cos = [float(SV[pos[cid]] @ NV[j]) for j, cid in enumerate(new_ids)]
            flagged_vec = [cid for cid, c in zip(new_ids, cos) if c < tau]
            row["vector_cos_min"] = min(cos)
        row["vector_flagged"] = len(flagged_vec)
        t0 = time.perf_counter()
        flagged_sem = []
        for cid in add_ids:
            v = NV[new_ids.index(cid)]
            nb = np.argsort(-(BV @ v))[: a.neighbors]
            s_new = split_sentences(chunks[cid]["text"])
            s_old = [s for j in nb for s in split_sentences(base[j]["text"])]
            if not s_new or not s_old:
                continue
            ev = emb.embed(s_new + s_old)
            sim = ev[: len(s_new)] @ ev[len(s_new):].T
            top = np.dstack(np.unravel_index(np.argsort(-sim, axis=None)[: a.pairs], sim.shape))[0]
            _, pc = nli.entail_prob([s_old[j] for _, j in top], [s_new[i] for i, _ in top])
            score = float(pc.max())
            anchor = cs.get("meta", {}).get("anchor_of", {}).get(cid)
            chunk_rows.append({"case": cs["case"], "content_kind": cs["content_kind"], "chunk": cid,
                               "contradiction_max": score, "anchor_in_neighbors": (anchor in [base[j]["id"] for j in nb])
                               if anchor else None})
            if score >= a.theta:
                flagged_sem.append(cid)
        row["semantic_ms"] = (time.perf_counter() - t0) * 1e3
        row["semantic_screened"] = len(add_ids)
        row["semantic_flagged"] = len(flagged_sem)
        row["final_decision"] = "quarantine" if (flagged_vec or flagged_sem) else "install"
    else:
        row["final_decision"] = "reject"
    results.append(row)
    print(f"{cs['case']:24s} crypto={'ACCEPT' if ok else 'reject':7s} final={row['final_decision']:10s} "
          f"vec_flag={row.get('vector_flagged', '-')} sem_flag={row.get('semantic_flagged', '-')}/"
          f"{row.get('semantic_screened', '-')}  ({why})", flush=True)
write_json(E / f"device_results_{a.vector_backend}.json",
           {"tau": tau, "theta": a.theta, "vector_backend": a.vector_backend, "nli_model": a.nli_model,
            "base_embed_s": base_embed_s, "results": results})
with open(E / f"chunk_scores_{a.vector_backend}.jsonl", "w") as f:
    for r in chunk_rows:
        f.write(json.dumps(r) + "\n")
emb.close()
nli.close()
