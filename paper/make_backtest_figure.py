"""
Rolling backtest figure from logs/scaling.jsonl: independent-dyad vs
cross-dyad-attention models (MENA only, release recipe, 3 seeds) on four
test periods, each with its own training/validation window.

    python paper/make_backtest_figure.py
Output: paper/fig13_backtest.png, paper/backtest.json
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, INK, INK2, GRID, MUTED = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9893"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 220, "savefig.bbox": "tight"})

FOLDS = [("2019-07-01", "2019–20\n151 wks"),
         ("2021-07-05", "2021–22\n256 wks"),
         ("2023-01-02", "2023\n334 wks"),
         ("2024-03-18", "2024–26\n391 wks")]
rows = [json.loads(l) for l in open("logs/scaling.jsonl")]


def pick(arch, ts):
    m = [r for r in rows if r["arch"] == arch and r["n_external"] == 0 and r.get("train_frac", 1.0) == 1.0
         and r.get("test_start", "2024-03-18") == ts and r["sample_seed"] == 0]
    return m[-1]


out = {}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
x = np.arange(len(FOLDS))
for ax, key, title in [(axes[0], "change_corr", "(a) Test change correlation"),
                       (axes[1], "esc_auc", "(b) Test escalation AUC (0.5 = chance)")]:
    for off, arch, colour, label in [(-0.18, "independent", MUTED, "Independent dyads"),
                                     (0.18, "cross-system", BLUE, "Cross-dyad attention")]:
        vals = [pick(arch, ts)["ensemble"][key] for ts, _ in FOLDS]
        ax.bar(x + off, vals, width=0.34, color=colour, label=label)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", fontsize=6.8, color=INK)
        out.setdefault(arch, {})[key] = vals
    ax.set_xticks(x, [f[1] for f in FOLDS], fontsize=6.8)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylim(0, 0.42)
axes[1].set_ylim(0.5, 0.72)
axes[1].axhline(0.5, color=MUTED, lw=0.8)  # AUC axis starts at 0.5 = no skill
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
fig.savefig("paper/fig13_backtest.png")
json.dump(out, open("paper/backtest.json", "w"), indent=1)
print(json.dumps(out, indent=1))
