"""
Figures for the trade-module section (paper/trade_section_draft.md, trade_section.docx).

  fig_trade_1_closure.png     Hormuz tanker transits (PortWatch) and China's imports from Gulf
                              partners, year-on-year, 2025-2026
  fig_trade_2_regression.png  cumulative Gulf-shock x exposure effect across specifications
  fig_trade_3_backtest.png    walk-forward skill of the operational forecast, by year

Run after the src/ pipeline (see compute_trade_stats.py):
    python paper/make_trade_figures.py
Same style constants as paper/make_figures.py.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
P = os.path.join(ROOT, "data", "processed")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
    "savefig.dpi": 220, "savefig.bbox": "tight", "figure.facecolor": "white",
})
CLOSURE = pd.Timestamp("2026-03-02")


def closure_figure():
    pw = pd.read_csv(os.path.join(ROOT, "data", "portwatch", "chokepoints_weekly.csv"), parse_dates=["week_start"])
    h = pw[(pw.chokepoint == "hormuz") & (pw.week_start >= "2025-01-01")].set_index("week_start").n_tanker
    normal = pw[(pw.chokepoint == "hormuz") & (pw.week_start < "2026-01-01")].n_tanker.mean()

    raw = pd.read_csv(os.path.join(ROOT, "data", "imf_imts", "imts_raw.csv"))
    imp = raw[(raw.side == "china") & (raw.indicator == "MG_CIF_USD")].pivot(
        index="period", columns="counterpart", values="value")
    imp.index = pd.PeriodIndex(imp.index, freq="M")
    groups = {"Iraq, Kuwait, Qatar (no bypass)": (["IRQ", "KWT", "QAT"], ORANGE),
              "Saudi Arabia (Red Sea bypass)": (["SAU"], BLUE),
              "Oman (outside the strait)": (["OMN"], AQUA)}
    yoy = {}
    for name, (iso, _) in groups.items():
        s = imp[iso].sum(axis=1)
        yoy[name] = (s / s.shift(12) - 1).loc["2025-01":] * 100

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    a1.plot(h.index, h.values, color=INK2, lw=2)
    a1.axhline(normal, color=MUTED, lw=1, ls="--")
    a1.text(pd.Timestamp("2026-07-01"), normal * 0.95, f"2019–2025 average:\n{normal:.0f} a week", fontsize=8,
            color=INK2, va="top")
    a1.set_ylim(0, normal * 1.35)
    a1.set_ylabel("tankers per week")
    a1.set_title("(a) Tanker transits through the Strait of Hormuz (IMF PortWatch)", loc="left", fontsize=9.5, color=INK)

    nudge = {"Oman (outside the strait)": 7, "Iraq, Kuwait, Qatar (no bypass)": -7}  # points; ends nearly coincide
    for name, (_, color) in groups.items():
        s = yoy[name]
        x = s.index.to_timestamp(how="start") + pd.Timedelta(days=14)
        a2.plot(x, s.values, color=color, lw=2, marker="o", ms=4)
        a2.annotate(name, (x[-1], s.values[-1]), xytext=(6, nudge.get(name, 0)), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    a2.axhline(0, color=MUTED, lw=1)
    a2.set_ylabel("% change vs same month a year earlier")
    a2.set_ylim(-100, 80)
    a2.set_title("(b) China's imports from Gulf partners (IMF IMTS, China-reported)", loc="left", fontsize=9.5,
                 color=INK)
    for ax in (a1, a2):
        ax.axvline(CLOSURE, color=INK, lw=1, ls=":")
    a1.text(CLOSURE + pd.Timedelta(days=5), normal * 1.25, "2 Mar 2026: transits\nfall below 5 a day",
            fontsize=8, color=INK, va="top")
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    a2.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"))
    fig.savefig(os.path.join(OUT, "fig_trade_1_closure.png"))
    plt.close(fig)


def regression_figure():
    rg = pd.read_csv(os.path.join(P, "regression_results.csv"))
    rg = rg[(rg.lag.astype(str) == "0-1-2-3") & (rg.spec == "B")]
    rows = [
        ("Imports, exposure high (main)", (rg.outcome == "imp_dlog12") & (rg.exposure == "high") & (rg["sample"] == "all") & (rg.shock == "gulf")),
        ("Imports, exposure low", (rg.outcome == "imp_dlog12") & (rg.exposure == "low") & (rg["sample"] == "all")),
        ("Exports, exposure high", (rg.outcome == "exp_dlog12") & (rg.exposure == "high") & (rg["sample"] == "all")),
        ("Sample from 2023", rg["sample"] == "start 2023"),
        ("Excluding the USA", rg["sample"] == "excl USA"),
        ("Excluding SYR, SDN, YEM, LBY", rg["sample"] == "excl SYR SDN YEM LBY"),
        ("Excluding IRN, ARE", rg["sample"].str.startswith("excl IRN ARE")),
        ("Import flows ≥ $100m only", rg["sample"].str.startswith("import flows")),
        ("Shock = material-conflict share", rg["sample"] == "shock = conflict_share"),
        ("Shock incl. ISR, YEM", rg["sample"] == "shock incl ISR YEM"),
        ("Oman dependence = 1", rg["sample"] == "Oman dependence = 1"),
        ("Saudi dependence = 0.8", rg["sample"] == "Saudi dependence = 0.8"),
        ("Energy share, no route parameters", rg["sample"] == "energy share, no route parameters"),
        ("Sample ending 2025-12 (no closure)", rg["sample"] == "pre-2026 (no Hormuz shock)"),
    ]
    data = []
    for label, sel in rows:
        r = rg[sel].iloc[0]
        data.append((label, r.beta, r.se_2way, r.p_perm))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    n = len(data)
    for i, (label, b, se, pp) in enumerate(data):
        y = n - 1 - i
        last = i == n - 1
        color = ORANGE if last else BLUE
        ax.plot([b - 1.96 * se, b + 1.96 * se], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(b, y, "o", color=color, ms=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(0.33, y, f"{pp:.3f}", fontsize=8, color=INK2, va="center", ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([d[0] for d in data][::-1], fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-0.95, 0.36)
    ax.set_xlabel("cumulative effect, lags 0–3 (log points per SD of Gulf conflict, fully exposed partner)")
    ax.text(0.33, n - 0.45, "permutation p", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_title("Gulf conflict × Hormuz exposure: effect on China's trade growth (95% CI, two-way clustered)",
                 loc="left", fontsize=9.5, color=INK)
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(OUT, "fig_trade_2_regression.png"))
    plt.close(fig)


def backtest_figure():
    meta = json.load(open(os.path.join(ROOT, "models", "trade_baseline", "meta.json")))
    bt = pd.DataFrame(meta["backtest"]["imf"])
    bt = bt[bt.year != "all"]
    years = sorted(bt.year.unique())
    x = np.arange(len(years))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for j, (flow, color, label) in enumerate((("imp", BLUE, "Imports"), ("exp", ORANGE, "Exports"))):
        v = bt[bt.flow == flow].set_index("year").reindex(years).skill_vs_zero.to_numpy()
        ax.bar(x + (j - 0.5) * (w + 0.02), v, width=w, color=color, label=label)
        for xi, vi in zip(x + (j - 0.5) * (w + 0.02), v):
            if abs(vi) < 0.005:  # a zero bar is invisible: say so
                ax.text(xi, 0.004, "0.00", ha="center", va="bottom", fontsize=7.5, color=INK2)
    allr = {f: next(r for r in meta["backtest"]["imf"] if r["year"] == "all" and r["flow"] == f)["skill_vs_zero"]
            for f in ("imp", "exp")}
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("skill vs 'no change'\n(1 − MSE / MSE of no change)")
    ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"Operational forecast, walk-forward by year (IMF data, 100-day lag): "
                 f"all years imports {allr['imp']:+.3f}, exports {allr['exp']:+.3f}",
                 loc="left", fontsize=9.5, color=INK)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(OUT, "fig_trade_3_backtest.png"))
    plt.close(fig)


if __name__ == "__main__":
    closure_figure()
    regression_figure()
    backtest_figure()
    print("wrote paper/fig_trade_1_closure.png, fig_trade_2_regression.png, fig_trade_3_backtest.png")
