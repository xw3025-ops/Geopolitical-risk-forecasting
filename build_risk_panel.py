"""
Step 1 of the v2 pipeline: build a clean dyad-week panel that fixes the data
problems found in the audit (see README "Known data issues").

Differences from build_goldstein_panel.py:
  - Weeks with zero events get goldstein = NaN ("no data"), not 0 ("neutral").
    n_events is kept as an explicit column.
  - Dyads whose goldstein source is missing (LBY/SDN in the local CSVs) are
    no longer padded with 0. Their goldstein is reconstructed from the CAMEO
    root-code counts using per-root weights fitted on the covered dyads
    (held-out R^2 ~0.98), and flagged with goldstein_imputed = 1.
  - goldstein_sum = goldstein * n_events is stored so downstream targets and
    features can be EVENT-WEIGHTED (a 5,000-event week outweighs a 1-event week).
  - No target columns and no trimming: every week is kept, including the most
    recent ones a live forecast needs. Targets for any horizon are computed
    in risk_features.py.
  - The week grid is every Monday between the first and last week, so a week
    with no rows at all in the source still appears (all zeros/NaN).

Sources (pick one):
  default        the committed local CSVs (20 MENA countries)
  --gdelt PATH   the CSV written by build_gdelt_panel.py (MENA + CHN + USA +
                 host countries, pulled directly from public GDELT; it already
                 contains goldstein for every dyad, so nothing is imputed
                 unless a source row is missing it)

Output: raw data/risk_panel.csv (+ raw data/goldstein_root_weights.json)

Run:
    python build_risk_panel.py
    python build_risk_panel.py --gdelt "raw data/gdelt_panel.csv"
"""

import argparse
import json

import numpy as np
import pandas as pd

from risk_config import GOLDSTEIN_WEIGHTS_PATH, PANEL_START, RISK_PANEL_PATH

COUNTS_SRC = "raw data/bq-results-20260927-170349-1790530001905.csv"
GOLDSTEIN_SRC = "raw data/results-20260927-135201 - results-20260927-135201.csv"
KEYS = ["week_start", "Actor1CountryCode", "Actor2CountryCode"]
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
EXTRA_NUMERIC = ["n_mentions", "avg_tone"]  # present only in the GDELT pull


def load_local():
    counts = pd.read_csv(COUNTS_SRC, parse_dates=["week_start"])[KEYS + COUNT_COLS]
    gold = pd.read_csv(GOLDSTEIN_SRC, parse_dates=["week_start"])
    countries = sorted(set(counts.Actor1CountryCode) | set(counts.Actor2CountryCode))
    # The goldstein file has ERI/SOM instead of LBY/SDN; keep the counts universe.
    gold = gold[gold.Actor1CountryCode.isin(countries) & gold.Actor2CountryCode.isin(countries)]
    return counts.merge(gold, on=KEYS, how="left"), countries


def load_gdelt(path):
    src = pd.read_csv(path, parse_dates=["week_start"])
    countries = sorted(set(src.Actor1CountryCode) | set(src.Actor2CountryCode))
    return src, countries


def fit_root_weights(panel):
    """Per-CAMEO-root goldstein weights: avg_goldstein ~ sum_r share_r * w_r,
    fitted with sqrt(n_events) row weights on dyad-weeks that have both."""
    rows = panel[(panel.n_events > 0) & panel.goldstein.notna()]
    shares = rows[COUNT_COLS].to_numpy(float) / rows.n_events.to_numpy(float)[:, None]
    w = np.sqrt(rows.n_events.to_numpy(float))
    coef, *_ = np.linalg.lstsq(shares * w[:, None], rows.goldstein.to_numpy() * w, rcond=None)
    pred = shares @ coef
    resid = rows.goldstein.to_numpy() - pred
    n = rows.n_events.to_numpy(float)
    r2 = 1 - (n * resid**2).sum() / (n * (rows.goldstein - np.average(rows.goldstein, weights=n)) ** 2).sum()
    return coef, {"weighted_r2_in_sample": round(float(r2), 4),
                  "mae": round(float(np.abs(resid).mean()), 4), "n_rows": int(len(rows))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdelt", help="CSV from build_gdelt_panel.py instead of the local CSVs")
    parser.add_argument("--out", default=RISK_PANEL_PATH)
    args = parser.parse_args()

    src, countries = load_gdelt(args.gdelt) if args.gdelt else load_local()
    extra = [c for c in EXTRA_NUMERIC if c in src.columns]
    weeks = pd.date_range(PANEL_START, src.week_start.max(), freq="W-MON")
    print(f"{len(countries)} countries, {len(weeks)} weeks "
          f"({weeks[0].date()}..{weeks[-1].date()}) -> {len(countries) ** 2 * len(weeks):,} dyad-weeks")

    grid = pd.MultiIndex.from_product([weeks, countries, countries], names=KEYS).to_frame(index=False)
    panel = grid.merge(src[KEYS + COUNT_COLS + ["avg_goldstein"] + extra], on=KEYS, how="left")
    panel[COUNT_COLS] = panel[COUNT_COLS].fillna(0).astype("int64")
    panel["n_events"] = panel[COUNT_COLS].sum(axis=1)
    panel["goldstein"] = panel.avg_goldstein.where(panel.n_events > 0)
    panel = panel.drop(columns="avg_goldstein")

    coef, stats = fit_root_weights(panel)
    need = (panel.n_events > 0) & panel.goldstein.isna()
    shares = panel.loc[need, COUNT_COLS].to_numpy(float) / panel.loc[need, "n_events"].to_numpy(float)[:, None]
    panel.loc[need, "goldstein"] = shares @ coef
    panel["goldstein_imputed"] = need.astype("int8")
    panel["goldstein_sum"] = (panel.goldstein * panel.n_events).fillna(0.0)

    imputed_dyads = panel[need].groupby(["Actor1CountryCode", "Actor2CountryCode"]).size()
    print(f"Goldstein root weights fitted on {stats['n_rows']:,} dyad-weeks "
          f"(in-sample weighted R^2 {stats['weighted_r2_in_sample']}): "
          + ", ".join(f"{i + 1:02d}:{w:+.1f}" for i, w in enumerate(coef)))
    print(f"Imputed goldstein for {int(need.sum()):,} dyad-weeks across {len(imputed_dyads)} dyads; "
          f"{(panel.n_events == 0).mean():.1%} of dyad-weeks have no events (goldstein = NaN)")

    with open(GOLDSTEIN_WEIGHTS_PATH, "w") as f:
        json.dump({"cameo_root_weights": {f"{i + 1:02d}": round(float(w), 4) for i, w in enumerate(coef)},
                   "fit": stats, "source": args.gdelt or "local CSVs"}, f, indent=2)

    panel = panel.sort_values(KEYS).reset_index(drop=True)
    panel.to_csv(args.out, index=False, float_format="%.5g")
    print(f"Wrote {len(panel):,} rows to {args.out}")


if __name__ == "__main__":
    main()
