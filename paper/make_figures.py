"""
Figures for the paper section. Reads paper/stats.json, paper/variance_stats.json,
paper/attention_hubs.json, logs/goldstein_experiments.jsonl and saved test
predictions; writes paper/fig*.png and paper/figure_numbers.json.

    python paper/make_figures.py
"""

import datetime as dt
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from honest_common import (DIRECTION_TEST_START, DIRECTION_VAL_START, load_panel, split_by_dates,  # noqa: E402
                           trailing_mean)

OUT = "paper"
# validated reference palette: the first three categorical slots pass all-pairs CVD checks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
BASE = "#b9b7b0"        # naive baselines
LIGHT_BLUE = "#cde2fb"
STEP_BLUE = "#86b6ef"   # earlier transformer recipe

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "savefig.dpi": 220, "savefig.bbox": "tight", "figure.facecolor": "white",
})

S = json.load(open(f"{OUT}/stats.json"))
V = json.load(open(f"{OUT}/variance_stats.json"))
R21 = json.load(open(f"{OUT}/v21_stats.json"))  # release v2.1 (cross-system attention)
R22 = json.load(open(f"{OUT}/v22_stats.json"))  # release v2.2 (+ world training data; surge flag)
EXP = [json.loads(line) for line in open("logs/goldstein_experiments.jsonl")]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc="white", ec=MUTED, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color, lw=1.0))


def dot_row(ax, i, value, color, label_fmt, x0, offset=0.006):
    ax.plot([x0, value], [i, i], color=GRID, lw=1.2, zorder=1)
    ax.scatter([value], [i], s=70, color=color, edgecolor="white", lw=1.2, zorder=3)
    ax.text(value + offset, i, label_fmt.format(value), va="center", fontsize=8, color=INK)


def exp(tag):
    m = [e for e in EXP if e["tag"] == tag]
    return m[-1] if m else None


# ---------------------------------------------------------------- 1. concept
def fig_concept():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    dyads = ["ISR–IRN", "ISR–PSE", "SAU–YEM", "IRN–QAT", "TUR–SYR"]
    ys = np.linspace(0.86, 0.30, len(dyads))
    for ax, title in zip(axes, ["(a) Row-by-row prediction", "(b) System-aware prediction (this study)"]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
    ax = axes[0]
    for y, d in zip(ys, dyads):
        box(ax, 0.02, y - 0.05, 0.26, 0.10, d + "\nfeatures", fs=7.5)
        box(ax, 0.40, y - 0.05, 0.22, 0.10, "model", fc="#f4f3f0", fs=7.5)
        box(ax, 0.74, y - 0.05, 0.24, 0.10, "forecast", fs=7.5)
        arrow(ax, 0.28, y, 0.40, y)
        arrow(ax, 0.62, y, 0.74, y)
    ax.text(0.5, 0.06, "each row scored independently;\ncontext only through hand-made lag features",
            ha="center", va="center", fontsize=7.5, color=INK2)
    ax = axes[1]
    for y, d in zip(ys, dyads):
        box(ax, 0.02, y - 0.05, 0.26, 0.10, d + "\nhistory", fs=7.5)
        box(ax, 0.74, y - 0.05, 0.24, 0.10, "forecast", fs=7.5)
        arrow(ax, 0.28, y, 0.40, y, color=BLUE)
        arrow(ax, 0.62, y, 0.74, y, color=BLUE)
    ax.add_patch(FancyBboxPatch((0.40, 0.23), 0.22, 0.70, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=LIGHT_BLUE, ec=BLUE, lw=1.0))
    ax.text(0.51, 0.58, "self-\nattention\nacross all\n400 dyads", ha="center", va="center", fontsize=8,
            color=INK, fontweight="bold")
    ax.text(0.5, 0.06, "every forecast conditioned on the\nstate of the whole regional system",
            ha="center", va="center", fontsize=7.5, color=INK2)
    save(fig, "fig1_concept")


# ---------------------------------------------------------------- 2. architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    X, W = 0.02, 0.96
    steps = [
        ("GDELT 2.0 event records (BigQuery)  →  weekly directed-dyad panel:  400 MENA dyads (incl. domestic)\n"
         "+ 1,559 dyads linking MENA states to the rest of the world  =  1,959 dyad tokens per week", "#f4f3f0", MUTED),
        ("Input per forecast week: 1,959 dyads × 13 weeks × 24 features\n"
         "log CAMEO root-code counts (20), mean Goldstein, trailing 26/52-week Goldstein means, 52-week activity",
         "white", MUTED),
        ("Token projection (Dense, 64)  +  learned temporal position embedding", "white", MUTED),
        ("TEMPORAL TRANSFORMER × 2:  each dyad attends over its own 13 weeks     [(batch·1959) × 13 × 64]\n"
         "multi-head self-attention (4 heads)  +  SwiGLU feed-forward;  residual connections + LayerNorm",
         LIGHT_BLUE, BLUE),
        ("Latest-week summary vector per dyad  ⊕  country embeddings (actor 1, actor 2; shared table, 8-d)\n"
         "→  dyad token (Dense, 64)", "white", MUTED),
        ("CROSS-SYSTEM TRANSFORMER × 2:  all 1,959 dyad tokens of the week attend to each other     [batch × 1959 × 64]\n"
         "multi-head self-attention (4 heads)  +  SwiGLU feed-forward;  residual connections + LayerNorm",
         LIGHT_BLUE, BLUE),
        ("Dense(1): correction per dyad   +   trailing 52-week mean Goldstein (residual baseline, from input)",
         "white", MUTED),
        ("Output: mean Goldstein over the next 13 weeks, read off the 400 MENA tokens\n"
         "release v2.1 = average of 5 independently trained seeds", "#f4f3f0", MUTED),
    ]
    h = 0.088
    gap = (0.93 - len(steps) * h) / (len(steps) - 1)
    ys = [0.99 - h - i * (h + gap) for i in range(len(steps))]
    for (text, fc, ec), y in zip(steps, ys):
        box(ax, X, y, W, h, text, fc=fc, ec=ec, fs=7)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, X + W / 2, y0, X + W / 2, y1 + h)
    ax.text(0.5, -0.02, f"{R21['params']:,} trainable parameters per seed model", ha="center",
            fontsize=7.5, color=INK2)
    save(fig, "fig2_architecture")


# ---------------------------------------------------------------- 3. evaluation design
def fig_timeline():
    d = dt.date.fromisoformat
    fig, ax = plt.subplots(figsize=(7.2, 1.6))
    segs = [("train (391 weeks)", "2015-05-11", "2022-10-31", BASE),
            ("embargo", "2022-11-07", "2023-01-30", "white"),
            ("validation (45 weeks)", "2023-02-06", "2023-12-11", ORANGE),
            ("embargo", "2023-12-18", "2024-03-11", "white"),
            ("test (119 weeks)", "2024-03-18", "2026-06-22", BLUE)]
    for label, a, b, c in segs:
        x0 = mdates.date2num(d(a))
        x1 = mdates.date2num(d(b)) + 7
        ax.barh(0, x1 - x0, left=x0, height=0.5, color=c, edgecolor=MUTED if c == "white" else "white",
                hatch="////" if c == "white" else None, lw=0.6)
        if label.startswith("validation"):
            ax.text((x0 + x1) / 2, -0.45, label, ha="center", va="center", fontsize=8, color=INK)
        elif label != "embargo":
            ax.text((x0 + x1) / 2, 0, label, ha="center", va="center", fontsize=8,
                    color="white" if c == BLUE else INK)
    ax.text(mdates.date2num(d("2023-07-01")), 0.45, "13-week embargoes (hatched)", fontsize=7.5,
            color=INK2, ha="center")
    ax.set_yticks([])
    ax.xaxis_date()
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.65, 0.65)
    save(fig, "fig3_timeline")


# ---------------------------------------------------------------- 4. benchmark
def fig_benchmark():
    b = S["goldstein_benchmark"]
    lin = json.load(open(f"{OUT}/linear_baselines.json"))
    rows = [("Linear regression, raw weekly inputs", lin["OLS current week"]["r2"], BASE),
            ("Trailing 13-week mean", b["trailing 13-week mean"]["r2"], BASE),
            ("v1.0 transformer recipe (re-run on corrected data)", 0.556, "#b7d3f6"),
            ("Trailing 26-week mean", b["trailing 26-week mean"]["r2"], BASE),
            ("Trailing 52-week mean", b["trailing 52-week mean"]["r2"], BASE),
            ("Gradient boosting, 44 hand-built features", 0.616, ORANGE),
            ("Linear regression, 44 hand-built features", lin["OLS 44 history features"]["r2"], ORANGE),
            ("v2.0: regional attention (400 dyads), 5 seeds", b["transformer v2.0 (5-seed ensemble)"]["r2"], STEP_BLUE),
            ("v2.1: cross-system attention (1,959 dyads), 5 seeds", R21["benchmark"]["v2.1"]["r2"], STEP_BLUE),
            ("v2.2: + 5,122 world dyads as training data, 5 seeds", R22["v2.2"]["r2"], BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (lab, v, c) in enumerate(rows):
        dot_row(ax, i, v, c, "{:.3f}", 0.30)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0.30, 0.67)
    ax.set_xlabel("Test R²  (mean Goldstein over the next 13 weeks; 400 dyads × 119 weeks)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig4_benchmark")


# ---------------------------------------------------------------- 5. improvement path
def fig_path():
    steps = [("v1.0\nrecipe", 0.556),
             ("+ regular-\nisation", exp("lb13_do0.2_lr0.0003_plat_s42")),
             ("+ residual\n(52-wk)", exp("lb13_res52_do0.2_lr0.0003_plat_s42")),
             ("+ long\nmemory", exp("lb13_res52_long_do0.2_lr0.0003_plat_s42")),
             ("+ 5 seeds\n= v2.0", S["goldstein_benchmark"]["transformer v2.0 (5-seed ensemble)"]["r2"]),
             ("+ world dyads\n= v2.1", R21["benchmark"]["v2.1"]["r2"]),
             ("+ world training\ndata = v2.2", R22["v2.2"]["r2"])]
    vals = [v if isinstance(v, float) else (v["te_r2"] if v else np.nan) for _, v in steps]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = np.arange(len(steps))
    ax.plot(x, vals, color=BLUE, lw=2, marker="o", ms=7, mec="white", mew=1.5, zorder=3)
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        delta = "" if i == 0 or np.isnan(vals[i - 1]) else f"\n({v - vals[i - 1]:+.3f})"
        ax.text(i, v + 0.005, f"{v:.3f}{delta}", ha="center", va="bottom", fontsize=8, color=INK)
    for val, lab, c in [(0.600, "trailing 52-week mean 0.600", MUTED), (0.616, "gradient boosting 0.616", ORANGE)]:
        ax.axhline(val, color=c, lw=1, ls="--", zorder=1)
        ax.text(len(steps) - 0.65, val - 0.0015, lab, fontsize=7.5, color=INK2, ha="right", va="top")
    ax.set_xticks(x, [s[0] for s in steps])
    ax.set_ylabel("Test R²")
    ax.set_ylim(0.54, 0.67)
    ax.set_xlim(-0.4, len(steps) - 0.6)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig5_path")
    return {s[0].replace("\n", " "): (None if np.isnan(v) else float(v)) for s, v in zip(steps, vals)}


# ---------------------------------------------------------------- 6. ablation
ABLATIONS = [  # (key, label, tag infix, colour) - bottom to top in the figure
    ("neither", "No cross-dyad attention, no country embeddings\n(purely row-by-row)", "_cd0_noemb", BASE),
    ("no_crossdyad", "No cross-dyad attention\n(country embeddings kept)", "_cd0", STEP_BLUE),
    ("no_embedding", "No country embeddings\n(cross-dyad attention kept)", "_noemb", AQUA),
    ("full", "Full model\n(release architecture)", "", BLUE),
]


def fig_ablation():
    ens = {}
    for key, label, infix, c in ABLATIONS:
        runs = [exp(f"lb13_res52_long{infix}_do0.2_lr0.0003_plat_s{s}") for s in (42, 1, 2)]
        if any(r is None for r in runs):
            print(f"ablation '{key}' incomplete - skipping")
            continue
        preds = [np.load(f"artifacts/experiments/{r['tag']}.npz") for r in runs]
        y, yv = preds[0]["te_true"], preds[0]["va_true"]
        pv = np.mean([p["va_pred"] for p in preds], 0)
        pt = np.mean([p["te_pred"] for p in preds], 0)
        ens[key] = dict(label=label.replace("\n", " "), r2=float(r2_score(y.ravel(), pt.ravel())),
                        mae=float(np.abs(y - pt).mean()),
                        seeds=[r["te_r2"] for r in runs], val_mse_seeds=[r["va_mse"] for r in runs],
                        val_mse_ensemble=float(((yv - pv) ** 2).mean()))
    if not ens:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    rows = [a for a in ABLATIONS if a[0] in ens]
    jitter = [-0.1, 0.0, 0.1]
    for i, (key, label, _, c) in enumerate(rows):
        e = ens[key]
        ax = axes[0]  # validation: used for model selection; lower is better
        ax.scatter(e["val_mse_seeds"], [i + j for j in jitter], s=34, color=c, edgecolor="white", lw=1, zorder=3)
        ax.scatter([e["val_mse_ensemble"]], [i], s=95, marker="D", color=c, edgecolor=INK, lw=0.8, zorder=4)
        ax.text(e["val_mse_ensemble"], i + 0.27, f"{e['val_mse_ensemble']:.3f}", ha="center", fontsize=7.5)
        ax = axes[1]  # test: reported only
        ax.scatter(e["seeds"], [i + j for j in jitter], s=34, color=c, edgecolor="white", lw=1, zorder=3)
        ax.scatter([e["r2"]], [i], s=95, marker="D", color=c, edgecolor=INK, lw=0.8, zorder=4)
        ax.text(e["r2"], i + 0.27, f"{e['r2']:.3f}", ha="center", fontsize=7.5)
    axes[0].set_title("(a) Validation, 2023 (selection)", loc="left", fontsize=8.5)
    axes[0].set_xlabel("Mean squared error (lower is better)")
    axes[0].invert_xaxis()
    axes[1].set_title("(b) Test, 2024–26 (reported)", loc="left", fontsize=8.5)
    axes[1].set_xlabel("R² (higher is better)")
    axes[1].axvline(0.600, color=MUTED, lw=1, ls="--")
    axes[1].text(0.6007, len(rows) - 0.42, "trailing\n52-wk mean", fontsize=6.8, color=INK2, va="top")
    axes[0].set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=7.4)
    axes[0].set_ylim(-0.6, len(rows) - 0.3)
    for ax in axes:
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig6_ablation")
    return ens


# ---------------------------------------------------------------- 7. examples
def fig_examples():
    ex = R21["examples"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for ax, (name, e) in zip(axes.ravel(), ex.items()):
        w = [dt.date.fromisoformat(s) for s in e["weeks"]]
        ax.plot(w, e["trailing52"], color=MUTED, lw=1.2, ls="--", label="trailing 52-week mean")
        ax.plot(w, e["actual"], color=INK, lw=1.5, label="actual")
        ax.plot(w, e["forecast"], color=BLUE, lw=2, label="transformer v2.1")
        red = R21["variance"]["example_reduction"][name]
        ax.set_title(f"{name.replace('-', ' → ')}   (squared error vs 52-week mean: {-red:+.0%})",
                     loc="left", fontsize=8.5)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", labelsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean Goldstein, next 13 weeks")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    save(fig, "fig7_examples")


# ---------------------------------------------------------------- 8. attention hubs
def fig_attention():
    from matplotlib.patches import Patch
    rec = R21["attention"]["received"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, (key, title) in zip(axes, [("crossdyad_block0_h3", "Cross-system block 1, head 4"),
                                       ("crossdyad_block0_h0", "Cross-system block 1, head 1")]):
        top = rec[key][:10][::-1]
        ax.barh(range(10), [t["lift"] for t in top], height=0.62,
                color=[BLUE if t["external"] else AQUA for t in top])
        ax.axvline(1, color=INK2, lw=1, ls="--")
        ax.set_yticks(range(10), [t["dyad"].replace("-", " → ") for t in top], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("Attention from MENA dyads (× uniform)")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend([Patch(color=BLUE), Patch(color=AQUA)], ["MENA–world dyad", "MENA–MENA dyad"],
               loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    save(fig, "fig8_attention")


# ---------------------------------------------------------------- 9. direction
def fig_direction():
    p = load_panel()
    _, _, te = split_by_dates(p["weeks"], p["target4"][0], DIRECTION_VAL_START, DIRECTION_TEST_START, 4)
    z = np.load("artifacts/direction_v2.0_test_predictions.npz", allow_pickle=True)
    y, prob = z["y_true"], z["prob"]
    rule = (p["gold_zero"] > trailing_mean(p["gold_zero"], 13))[:, te].T.astype(int)
    gap = np.abs(p["gold_zero"] - trailing_mean(p["gold_zero"], 13))[:, te].T
    bins = np.digitize(gap, np.quantile(gap, [0.25, 0.5, 0.75]))
    pred = (prob >= 0.5).astype(int)
    q_rule = [float((rule[bins == b] == y[bins == b]).mean()) for b in range(4)]
    q_tr = [float((pred[bins == b] == y[bins == b]).mean()) for b in range(4)]
    acc = float((pred == y).mean())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    rows = [("Majority class", 0.536, BASE), ("Mean-reversion rule", 0.781, BASE),
            ("Gradient boosting", 0.816, ORANGE), ("Transformer v2.0", acc, BLUE)]
    for i, (lab, v, c) in enumerate(rows):
        dot_row(ax, i, v, c, "{:.1%}", 0.5, offset=0.012)
    ax.set_yticks(range(4), [r[0] for r in rows])
    ax.set_xlim(0.5, 0.9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Test accuracy, 4-week direction")
    ax.set_title("(a) Overall", loc="left", fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax = axes[1]
    x = np.arange(4)
    ax.plot(x, q_rule, color=BASE, lw=2, marker="o", ms=7, mec="white", label="mean-reversion rule")
    ax.plot(x, q_tr, color=BLUE, lw=2, marker="o", ms=7, mec="white", label="transformer v2.0")
    ax.set_xticks(x, ["Q1\nnear normal", "Q2", "Q3", "Q4\nfar from normal"], fontsize=7.5)
    ax.set_ylim(0.55, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("(b) By distance of this week from its 13-week mean", loc="left", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig9_direction")
    return dict(q_rule=q_rule, q_transformer=q_tr, accuracy=acc)


# ---------------------------------------------------------------- 10. GDELT limitation
def fig_gdelt():
    g = S["gdelt_example"]
    w = [dt.date.fromisoformat(s) for s in g["weeks"]]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True)
    ax = axes[0]
    ax.plot(w, g["goldstein"], color=INK, lw=1.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Mean Goldstein")
    ax.set_title("(a) Weekly mean Goldstein score, Israel → Palestine", loc="left", fontsize=9)
    ax = axes[1]
    ax.plot(w, g["coop_share"], color=AQUA, lw=1.8, label="cooperative events (CAMEO 01–09)")
    ax.plot(w, g["violent_share"], color=ORANGE, lw=1.8, label="assault / fight / mass violence (CAMEO 18–20)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of events")
    ax.set_title("(b) Event mix", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    for a in axes:
        a.axvline(dt.date(2023, 10, 2), color=INK2, lw=0.9, ls=":")
        a.yaxis.grid(True, color=GRID, lw=0.6)
    axes[0].text(dt.date(2023, 10, 9), -0.6, "week of 7 Oct 2023", fontsize=7.5, color=INK2)
    save(fig, "fig10_gdelt")


if __name__ == "__main__":
    fig_concept()
    fig_architecture()
    fig_timeline()
    fig_benchmark()
    path = fig_path()
    abl = fig_ablation()
    fig_examples()
    fig_attention()
    d = fig_direction()
    fig_gdelt()
    json.dump({"path": path, "ablation": abl, "direction": d}, open(f"{OUT}/figure_numbers.json", "w"), indent=1)
