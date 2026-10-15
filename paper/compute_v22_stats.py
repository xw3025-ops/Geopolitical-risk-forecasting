"""
Numbers for the release v2.2 parts of the paper section (more training data;
the conflict-surge flag), from saved release predictions.

    python paper/compute_v22_stats.py   -> paper/v22_stats.json, paper/fig14_surge_isr.png

goldstein   v2.1 and v2.2 on the release test weeks (forecast_metrics), and
            their validation MSE (selection).
flag        on the test weeks shared by every model (direction v2.0 starts
            later): AUC and weekly top-10 hit rate for conflict surges and
            Goldstein crises, for the v2.0 direction model, v2.1, the surge
            model, a naive "last quarter's peak" forecast and a linear
            regression on the hand-built features.
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())
import target_variants as tv  # noqa: E402
from benchmark_ladder import forecast_metrics  # noqa: E402
from benchmark_targets import make_context  # noqa: E402
from honest_common import build_features, rows, trailing_mean  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = "paper/v22_stats.json"


def main():
    ctx = make_context()
    p = ctx.p
    stats = {}

    # ---- goldstein v2.1 vs v2.2 (release test weeks, 119)
    n52 = trailing_mean(p["gold_zero"], 52)
    for v in ("v2.1", "v2.2"):
        te = np.load(f"artifacts/goldstein_{v}_test_predictions.npz", allow_pickle=True)
        va = np.load(f"artifacts/goldstein_{v}_val_predictions.npz", allow_pickle=True)
        m = forecast_metrics(te["y_true"], te["pred"], n52[:, te["week_idx"]].T)
        m["val_mse"] = float(((va["pred"] - va["y_true"]) ** 2).mean())
        m["seed_r2"] = [float(forecast_metrics(te["y_true"], s, n52[:, te["week_idx"]].T)["r2"])
                        for s in te["pred_per_seed"]]
        stats[v] = m
    stats["world_dyads"] = 5122

    # ---- conflict-surge flag vs the alternatives
    d = np.load("artifacts/direction_v2.0_test_predictions.npz", allow_pickle=True)
    g = np.load("artifacts/goldstein_v2.1_test_predictions.npz", allow_pickle=True)
    s = np.load("artifacts/surge_v2.2_test_predictions.npz", allow_pickle=True)
    W = np.array(sorted(set(ctx.te) & set(d["week_idx"])))
    pick = lambda z, a: z[a][[list(z["week_idx"]).index(w) for w in W]]
    pc, mg = ctx.T["peak-conflict"], ctx.T["mean-gold"]
    normal = pc["normal"][:, W].T
    F, _ = build_features(p)
    c = pc["series"]
    Fc = np.nan_to_num(np.concatenate([F, np.stack([c, trailing_mean(c, 13), trailing_mean(c, 52)], 2)], 2))
    ok = np.isfinite(rows(pc["y"], ctx.tr))
    lin = LinearRegression().fit(rows(Fc, ctx.tr)[ok], rows(pc["y"], ctx.tr)[ok])
    scores = {
        "v2.0 direction model (4-week escalation probability)": pick(d, "prob"),
        "v2.1 Goldstein forecast (fall below normal)": mg["normal"][:, W].T - pick(g, "pred"),
        "Naive: last quarter's peak conflict": tv.peak_past(c, False)[:, W].T - normal,
        "Linear regression, hand-built features*": lin.predict(rows(Fc, W)).reshape(len(W), -1) - normal,
        "v2.2 conflict-surge model": pick(s, "pred") - normal,
    }
    labels = {"conflict_surge": ctx.labels["conflict-surge"][:, W].T,
              "gold_crisis": ctx.labels["gold-crisis"][:, W].T}
    flag = {}
    for name, sc in scores.items():
        flag[name] = {}
        for ln, L in labels.items():
            top = np.argsort(-sc, 1)[:, :10]
            flag[name][ln] = dict(auc=float(roc_auc_score(L.ravel(), sc.ravel())),
                                  top10=float(np.take_along_axis(L, top, 1).mean()))
    stats["flag"] = flag
    stats["flag_weeks"] = int(len(W))
    stats["flag_first_week"] = str(p["weeks"][W[0]].date())
    stats["flag_last_week"] = str(p["weeks"][W[-1]].date())
    stats["surge_rate"] = float(labels["conflict_surge"].mean())
    stats["gold_crisis_rate"] = float(labels["gold_crisis"].mean())
    import joblib
    cal = joblib.load("artifacts/surge_v2.2_preprocessing.joblib")["calibration"]
    sc_all = s["pred"] - s["normal"]
    stats["surge_mean_prob_test"] = float(np.mean(1 / (1 + np.exp(-(cal["a"] * sc_all + cal["b"])))))
    stats["surge_rate_test_all"] = float(ctx.labels["conflict-surge"][:, s["week_idx"]].mean())
    stats["direction_own_auc"] =float(roc_auc_score(pick(d, "y_true").ravel(), pick(d, "prob").ravel()))

    # Israel-Palestine: tracking of the conflict peak vs v2.1's mean-goldstein forecast
    isr = {}
    for a, b in (("ISR", "PSE"), ("PSE", "ISR")):
        j = int(np.where((p["a1"] == a) & (p["a2"] == b))[0][0])
        y, f, n = pc["y"][j, s["week_idx"]], s["pred"][:, j], pc["normal"][j, s["week_idx"]]
        yg, fg, ng = g["y_true"][:, j], g["pred"][:, j], mg["normal"][j, g["week_idx"]]
        isr[f"{a}-{b}"] = dict(surge_corr=float(np.corrcoef(y - n, f - n)[0, 1]),
                               surge_amp=float(np.std(f - n) / np.std(y - n)),
                               gold_corr=float(np.corrcoef(yg - ng, fg - ng)[0, 1]),
                               gold_amp=float(np.std(fg - ng) / np.std(yg - ng)))
    stats["isr_pse"] = isr
    json.dump(stats, open(OUT, "w"), indent=1)

    # figure: Israel -> Palestine, v2.1 (mean goldstein) vs the surge model (peak conflict)
    BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#9a9893", "#e6e5e1"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "savefig.dpi": 220,
                         "savefig.bbox": "tight"})
    j = int(np.where((p["a1"] == "ISR") & (p["a2"] == "PSE"))[0][0])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharex=True)
    for ax, z, t, title, ylab in [
            (axes[0], g, mg, "(a) v2.1: mean Goldstein, next 13 weeks", "Goldstein"),
            (axes[1], s, pc, "(b) v2.2 surge model: peak conflict, next 13 weeks", "log conflict intensity")]:
        wk = p["weeks"][z["week_idx"]]
        ax.plot(wk, z["y_true"][:, j], color=INK, lw=1.8, label="Actual")
        ax.plot(wk, z["pred"][:, j], color=BLUE, lw=1.8, label="Forecast")
        ax.plot(wk, t["normal"][j, z["week_idx"]], color=MUTED, lw=1.3, ls="--", label="Normal")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=7.5)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.savefig("paper/fig14_surge_isr.png")
    print(json.dumps({k: v for k, v in stats.items() if k != "flag"}, indent=1))
    for name, r in flag.items():
        print(f"{name:<55} surge AUC {r['conflict_surge']['auc']:.3f} top10 {r['conflict_surge']['top10']:.0%} | "
              f"gold AUC {r['gold_crisis']['auc']:.3f} top10 {r['gold_crisis']['top10']:.0%}")


if __name__ == "__main__":
    main()
