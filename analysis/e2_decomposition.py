"""Error decomposition of the E2 context-pruning configurations (the numbers reported in the paper, Sect. 5.6).

For each configuration: on answerable questions the shares answered correctly (lenient scoring), abstained and
answered wrongly; and the share of explicit abstentions on in-topic unanswerable and on out-of-corpus questions.
One scoring function (textnorm.score_row, identical to the prior study) is used for every configuration.
Writes results/e2/decomposition.json. Usage: python analysis/e2_decomposition.py"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, write_json  # noqa: E402
from ragtrust.textnorm import score_row  # noqa: E402

CFGS = {"prior_q3_k3": DATA / "prior" / "results_v2" / "gen_q3_k3.jsonl"}
for c in ("k5_ce3", "k5_ce3w", "k3_ce3", "k3_ce3w", "k10_ce3", "k10_ce5", "k5_bi3"):
    CFGS[c] = RESULTS / "e2" / f"gen_{c}.jsonl"
out = {}
for name, path in CFGS.items():
    if not path.exists():
        continue
    rows = [(r.get("condition"), score_row(r)) for r in (json.loads(l) for l in open(path) if l.strip())]
    A = [s for c, s in rows if c == "in_answerable"]
    share = lambda X, f: sum(bool(f(s)) for s in X) / len(X) if X else None
    corr, ab = share(A, lambda s: s["lenient"]), share(A, lambda s: s["model_abstained"])
    out[name] = {"n_answerable": len(A), "correct": corr, "abstained": ab, "wrong": None if corr is None else 1 - corr - ab,
                 "abstained_in_unanswerable": share([s for c, s in rows if c == "in_unanswerable"], lambda s: s["model_abstained"]),
                 "abstained_out_of_corpus": share([s for c, s in rows if c == "out_of_corpus"], lambda s: s["model_abstained"])}
    o = out[name]
    print(f"{name:12s} correct {o['correct']:.2f}  abstained {o['abstained']:.2f}  wrong {o['wrong']:.2f}  "
          f"| unanswerable {o['abstained_in_unanswerable']:.2f}  out of corpus {o['abstained_out_of_corpus']:.2f}")
write_json(RESULTS / "e2" / "decomposition.json", out)
