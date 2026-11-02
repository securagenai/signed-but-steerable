"""E8 semantic screen, offline re-analysis (exploratory; PROTOCOL.md deviations log).

The pre-registered screen (device/e8_device_verify.py) pairs each new sentence with the MOST SIMILAR
existing sentences and takes the maximum NLI contradiction probability. For a silent override (a copy
of an existing passage with one fact changed) the most similar pairs are the unchanged sentences, which
are identical to existing ones and therefore entailed, so the edited sentence may never be scored.
This script recomputes, with the same models in FP32 on the desktop:
  V0  screen as run on the device (sanity check: should reproduce the device AUROC)
  V1  same, but sentences identical to an existing sentence are excluded before pairing
  V2  near-duplicate edit signal: max cosine between a new sentence and any existing sentence, over new
      sentences that are NOT verbatim copies (an edited fact is a near-copy; genuinely new text is not)
and reports AUROC (silent override vs benign addition; hedged override vs benign) with bootstrap 95% CIs,
plus the flag rate on unanchored poison (no existing chunk to contradict).
Writes results/e8/semantic_variants.json. Usage: python analysis/e8_semantic_variants.py"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.encoders import Encoder  # noqa: E402
from ragtrust.paths import out_dir, read_json, write_json  # noqa: E402
from ragtrust.sentences import split_sentences  # noqa: E402
from ragtrust.stats import bootstrap_auroc  # noqa: E402

E = out_dir("e8")
cases = read_json(E / "cases.json")
emb = Encoder("all-minilm-l6-v2", "ort-fp32", batch=64, seq_len=cases["seq_len"], cpu_padding="longest")
nli = Encoder("nli-minilm2-l6-h768", "ort-fp32", batch=32)
base = json.loads((E / "packs" / "_base" / "chunks.json").read_text())
BV = emb.embed([c["text"] for c in base])
NEIGH, PAIRS = 3, 5
norm = lambda s: " ".join(s.lower().split())
kinds = {"silent_override": [], "benign_addition": [], "unanchored_poison": []}
rows = []
for cs in cases["cases"]:
    k = cs.get("content_kind")
    if k not in kinds or cs["case"] == "desync_vectors":
        continue
    pdir = E / "packs" / cs["case"]
    man = json.loads((pdir / "manifest.json").read_bytes())
    chunks = {c["id"]: c for c in json.loads((pdir / "chunks.json").read_text())}
    for op in man["ops"]:
        if op["op"] != "ADD":
            continue
        text = chunks[op["id"]]["text"]
        v = emb.embed([text])[0]
        nb = np.argsort(-(BV @ v))[:NEIGH]
        s_new = split_sentences(text)
        s_old = [s for j in nb for s in split_sentences(base[j]["text"])]
        if not s_new or not s_old:
            continue
        ev = emb.embed(s_new + s_old)
        sim = ev[: len(s_new)] @ ev[len(s_new):].T
        old_set = {norm(s) for s in s_old}
        copied = np.array([norm(s) in old_set for s in s_new])
        out = {"case": cs["case"], "kind": k, "hedged": "hedged" in cs["case"], "chunk": op["id"]}
        for lab, mask in (("V0", np.ones(len(s_new), bool)), ("V1", ~copied)):
            sm = np.where(mask[:, None], sim, -np.inf)
            top = np.dstack(np.unravel_index(np.argsort(-sm, axis=None)[:PAIRS], sm.shape))[0]
            top = [(i, j) for i, j in top if np.isfinite(sm[i, j])]
            if top:
                _, pc = nli.entail_prob([s_old[j] for _, j in top], [s_new[i] for i, _ in top])
                out[lab] = float(pc.max())
            else:
                out[lab] = 0.0
        out["V2"] = float(sim[~copied].max()) if (~copied).any() else 0.0
        out["n_new_sentences"], out["n_copied"] = len(s_new), int(copied.sum())
        rows.append(out)

res = {"n": {k: sum(r["kind"] == k for r in rows) for k in kinds}, "variants": {}}
for lab in ("V0", "V1", "V2"):
    pos = [r[lab] for r in rows if r["kind"] == "silent_override" and not r["hedged"]]
    posh = [r[lab] for r in rows if r["kind"] == "silent_override" and r["hedged"]]
    neg = [r[lab] for r in rows if r["kind"] == "benign_addition"]
    poison = [r[lab] for r in rows if r["kind"] == "unanchored_poison"]
    thr = float(np.quantile(neg, 0.95)) if neg else None  # 5% false-alarm rate on benign additions
    res["variants"][lab] = {"auroc_override_vs_benign": bootstrap_auroc(pos, neg) if pos and neg else None,
                            "auroc_hedged_vs_benign": bootstrap_auroc(posh, neg) if posh and neg else None,
                            "threshold_at_5pct_benign_flags": thr,
                            "TPR_override_at_thr": float(np.mean([x > thr for x in pos])) if pos and thr is not None else None,
                            "TPR_hedged_at_thr": float(np.mean([x > thr for x in posh])) if posh and thr is not None else None,
                            "flag_rate_unanchored_poison_at_thr": float(np.mean([x > thr for x in poison])) if poison and thr is not None else None}
    v = res["variants"][lab]
    print(f"{lab}: AUROC override {v['auroc_override_vs_benign']}  hedged {v['auroc_hedged_vs_benign']}  "
          f"TPR@5%FPR override {v['TPR_override_at_thr']} hedged {v['TPR_hedged_at_thr']}  poison flagged {v['flag_rate_unanchored_poison_at_thr']}")
res["rows"] = rows
write_json(E / "semantic_variants.json", res)
emb.close()
nli.close()
