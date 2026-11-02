"""All figures of the paper from the released results (vector PDF, serif fonts). Re-run after any result changes.

  fig_margin_law.pdf   Hijack rate vs tolerated deviation: margin law on SQuAD and four BEIR corpora for
                       BGE-M3 and MiniLM (results/e7/margins/*.npz) and the transfer attacker on SQuAD
                       (results/e7/margin_law_squad.json); reference lines for FP32 cross-architecture
                       noise, tau = 0.999 and the MiniLM CPU-INT8 threshold.
  fig_vdb.pdf          (a) Exposure of tolerance-based audits per storage format (results/e7/vecdb_quant.json;
                       median and range over corpus-model pairs) with the FAISS FiQA indexes
                       (results/e7/faiss_integrity.json). (b) Tolerance-based exposure versus the residual
                       exposure of code-level verification without canonicalisation (results/e7/guard_budget.json).
Writes results/figures/*.pdf. Usage: python analysis/make_figures.py"""
import glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import RESULTS, out_dir, read_json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "mathtext.fontset": "stix"})
OUT = out_dir("figures")
REF = {"FP32 x86/ARM noise": 0.04, r"$\tau=0.999$": 2.56, "MiniLM CPU INT8 threshold": 25.3}
CORPORA = [("squad", "SQuAD"), ("scifact", "SciFact"), ("nfcorpus", "NFCorpus"), ("fiqa", "FiQA"), ("arguana", "ArguAna")]
COL = {"squad": "#1b4f72", "scifact": "#2e86c1", "nfcorpus": "#28b463", "fiqa": "#d68910", "arguana": "#c0392b"}
made = []

# ---------------------------------------------------------------- Fig. margin law
mdir = RESULTS / "e7" / "margins"
if mdir.exists():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    grid = np.logspace(-2.3, np.log10(60), 300)
    sq = read_json(RESULTS / "e7" / "margin_law_squad.json") if (RESULTS / "e7" / "margin_law_squad.json").exists() else None
    for ax, (model, title) in zip(axes, (("bge_m3", "BGE-M3"), ("minilm", "all-MiniLM-L6-v2"))):
        for key, name in CORPORA:
            f = mdir / f"{key}_{model}.npz"
            if not f.exists():
                continue
            m = np.degrees(np.load(f)["margins"])
            ax.plot(grid, [(m < g).mean() for g in grid], color=COL[key], lw=1.1, label=f"{name} (law)")
        if sq and model in sq["spaces"]:
            tr = sq["spaces"][model]["transfer"]
            pts = sorted((float(np.degrees(np.arccos(float(t)))), v[0]) for t, v in tr.items() if float(t) < 1)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o--", ms=2.2, lw=0.8, color="#555555", label="SQuAD transfer")
        for i, (lab, x) in enumerate(REF.items()):
            ax.axvline(x, color="#999999", lw=0.6, ls=":")
            ax.text(x * 1.08, 0.50, lab, fontsize=5.8, color="#555555", rotation=90, va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
        ax.set_xscale("log")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.set_xlabel("Tolerated deviation of a shipped vector (degrees)")
    axes[0].set_ylabel("Share of queries hijackable (rank 1)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(OUT / "fig_margin_law.pdf")
    made.append("fig_margin_law.pdf")

# ---------------------------------------------------------------- Fig. vector databases
vq = RESULTS / "e7" / "vecdb_quant.json"
if vq.exists():
    V = read_json(vq)
    schemes = ["fp16", "sq8", "sq4", "pq", "binary"]
    labels = ["FP16", "SQ8", "SQ4", "PQ", "Binary"]
    vals = {s: [v[s]["hijack_at_max_tolerance"][0] for v in V.values() if s in v] for s in schemes}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    ax = axes[0]
    x = np.arange(len(schemes))
    med = [np.median(vals[s]) for s in schemes]
    lo = [med[i] - min(vals[s]) for i, s in enumerate(schemes)]
    hi = [max(vals[s]) - med[i] for i, s in enumerate(schemes)]
    ax.bar(x, med, color="#aab7c4", width=0.6, label="5 corpora x 2 models (median, range)")
    ax.errorbar(x, med, yerr=[lo, hi], fmt="none", ecolor="#2c3e50", lw=0.8, capsize=2)
    fj = RESULTS / "e7" / "faiss_integrity.json"
    if fj.exists():
        Fz = read_json(fj)["indexes"]
        fmap = {"sq8": "IndexScalarQuantizer(SQ8)", "pq": "IndexPQ", "binary": "IndexBinaryFlat"}
        fx = [i for i, s in enumerate(schemes) if s in fmap and fmap[s] in Fz]
        ax.plot(fx, [Fz[fmap[schemes[i]]]["tolerance_hijack"][0] for i in fx], "D", ms=4, color="#c0392b", label="FAISS indexes, FiQA")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of queries hijackable")
    ax.set_title("(a) Tolerance-based audit of stored vectors")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    gb = RESULTS / "e7" / "guard_budget.json"
    ax = axes[1]
    if gb.exists():
        G = read_json(gb)
        sch = [("SQ8", "sq8"), ("SQ4", "sq4"), ("PQ", "pq")]
        tol = [V.get("squad/bge_m3", {}).get(k, {}).get("hijack_at_max_tolerance", [np.nan])[0] for _, k in sch]
        cod = [G["bge_m3"][f"{n}_eps1e-05"]["hijack"][0] for n, _ in sch]
        x = np.arange(len(sch))
        ax.bar(x - 0.18, np.maximum(tol, 1e-4), 0.34, color="#aab7c4", label="tolerance-based audit")
        ax.bar(x + 0.18, np.maximum(cod, 1e-4), 0.34, color="#1b4f72", label=r"code-level, guard band $10^{-5}$")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xticks(x, [n for n, _ in sch])
        ax.set_title("(b) SQuAD, BGE-M3: before and after the defence")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_vdb.pdf")
    made.append("fig_vdb.pdf")

print("figures written to", OUT, ":", ", ".join(made))
