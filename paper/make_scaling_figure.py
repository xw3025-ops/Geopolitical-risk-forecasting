"""
Scaling curves from logs/scaling.jsonl (experiment_scaling.py runs):
validation error and test metrics vs number of training dyads, per architecture.

    python paper/make_scaling_figure.py
Output: paper/fig11_scaling.png, paper/scaling_summary.json
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, INK2, GRID, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9893"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 220, "savefig.bbox": "tight"})

runs = [json.loads(l) for l in open("logs/scaling.jsonl")]
latest = {}
for r in runs:  # keep the latest run per (arch, n_external)
    if r["min_active"] == 0.5 and r["sample_seed"] == 0:
        latest[(r["arch"], r["n_external"])] = r

series = {}
for arch, colour, label in [("independent", MUTED, "Independent dyads (no cross-dyad attention)"),
                            ("cross-system", BLUE, "Cross-system attention (MENA + world dyads attend jointly)")]:
    pts = sorted((n, r) for (a, n), r in latest.items() if a == arch)
    if pts:
        series[arch] = dict(colour=colour, label=label, n=[400 + n for n, _ in pts],
                            val=[r["val_mse_ensemble"] for _, r in pts],
                            r2=[r["ensemble"]["r2"] for _, r in pts],
                            corr=[r["ensemble"]["change_corr"] for _, r in pts],
                            esc=[r["ensemble"]["esc_auc"] for _, r in pts])

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
for key, ax, title, better in [("val", axes[0], "(a) Validation error (selection)", "lower is better"),
                               ("corr", axes[1], "(b) Test change correlation", "higher is better"),
                               ("r2", axes[2], "(c) Test R²", "higher is better")]:
    for arch, s in series.items():
        ax.plot(s["n"], s[key], color=s["colour"], lw=2, marker="o", ms=6, mec="white", label=s["label"])
    ax.set_xscale("log")
    ax.set_xticks([400, 800, 1600], ["400", "800", "1,600"])
    ax.minorticks_off()
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlabel(f"training dyads (log scale); {better}", fontsize=7.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
fig.savefig("paper/fig11_scaling.png")

summary = {}
for arch, s in series.items():
    x = np.log2(np.array(s["n"], dtype=float))
    summary[arch] = {"points": [dict(train_dyads=n, val_mse=v, r2=r, change_corr=c, esc_auc=e)
                                for n, v, r, c, e in zip(s["n"], s["val"], s["r2"], s["corr"], s["esc"])]}
    if len(x) >= 3:  # slope per doubling of training dyads
        for key in ("val", "r2", "corr"):
            summary[arch][f"{key}_per_doubling"] = float(np.polyfit(x, s[key], 1)[0])
json.dump(summary, open("paper/scaling_summary.json", "w"), indent=1)
print(json.dumps({a: {k: v for k, v in d.items() if k != "points"} for a, d in summary.items()}, indent=1))
for a, d in summary.items():
    for pt in d["points"]:
        print(f"{a:<13} dyads {pt['train_dyads']:>5}: val {pt['val_mse']:.4f}  R² {pt['r2']:.3f}  chg-corr {pt['change_corr']:.3f}  esc {pt['esc_auc']:.3f}")
