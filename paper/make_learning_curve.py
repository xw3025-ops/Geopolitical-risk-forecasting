"""
Learning curves from logs/scaling.jsonl: MENA-only models (n_external = 0)
trained on 12.5%, 25%, 50% and 100% of the training weeks, for the
independent-dyad and the cross-dyad-attention architectures.

    python paper/make_learning_curve.py
Output: paper/fig12_learning_curve.png, paper/learning_curve.json
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, INK2, GRID, MUTED = "#2a78d6", "#52514e", "#e6e5e1", "#9a9893"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 220, "savefig.bbox": "tight"})

pts = {}
for line in open("logs/scaling.jsonl"):
    r = json.loads(line)
    if r["n_external"] == 0 and r["min_active"] == 0.5 and r["sample_seed"] == 0:
        pts[(r["arch"], r.get("train_frac", 1.0))] = r

curves = {}
for arch, colour, label in [("independent", MUTED, "Independent dyads (no cross-dyad attention)"),
                            ("cross-system", BLUE, "Cross-dyad attention across the regional system")]:
    rs = sorted((f, r) for (a, f), r in pts.items() if a == arch)
    curves[arch] = dict(colour=colour, label=label, weeks=[round(391 * f) for f, _ in rs],
                        val=[r["val_mse_ensemble"] for _, r in rs], corr=[r["ensemble"]["change_corr"] for _, r in rs],
                        esc=[r["ensemble"]["esc_auc"] for _, r in rs], r2=[r["ensemble"]["r2"] for _, r in rs])

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
for key, ax, title, note in [("val", axes[0], "(a) Validation error", "lower is better"),
                             ("corr", axes[1], "(b) Test change correlation", "higher is better"),
                             ("esc", axes[2], "(c) Test escalation AUC", "higher is better")]:
    for c in curves.values():
        ax.plot(c["weeks"], c[key], color=c["colour"], lw=2, marker="o", ms=6, mec="white", label=c["label"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([49, 98, 196, 391], ["49", "98", "196", "391"])
    ax.minorticks_off()
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlabel(f"training weeks (log scale); {note}", fontsize=7.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
fig.savefig("paper/fig12_learning_curve.png")

out = {}
for arch, c in curves.items():
    out[arch] = {k: c[k] for k in ("weeks", "val", "corr", "esc", "r2")}
    out[arch]["val_change_per_doubling"] = list(np.diff(c["val"]))
    out[arch]["corr_change_per_doubling"] = list(np.diff(c["corr"]))
json.dump(out, open("paper/learning_curve.json", "w"), indent=1)
for arch, d in out.items():
    print(arch, "| val per doubling", [round(x, 3) for x in d["val_change_per_doubling"]],
          "| change-corr per doubling", [round(x, 3) for x in d["corr_change_per_doubling"]])
