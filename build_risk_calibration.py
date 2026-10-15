"""
Calibrate the model-based parts of the country risk index (risk_index.py)
and record the evidence behind them. Uses only HELD-OUT predictions of the
release models (goldstein v2.2; direction v2.0 for the evidence only) (the validation and test weeks they were never trained
on), saved by experiment_goldstein.py / release_v2.0_direction.py.

1. Evidence: which signal should drive LIKELIHOOD? Each candidate is
   calibrated on the 2023 validation weeks and scored on the 2024-26 test
   weeks (ROC-AUC, Brier), incl. the direction model's escalation probability.
2. Likelihood bands on the test weeks: does "level 4" really mean higher
   realized deterioration than "level 2"?
3. Final calibration: the chosen logistic model refit on ALL held-out weeks
   (2023-02..2026-06), plus the hostility quintile thresholds.

Event ("material deterioration"): over the next 13 weeks a country's
activity-weighted mean goldstein (all its dyads) falls >= DELTA below its
trailing 52-week normal.

Run:
    python build_risk_calibration.py
Output:
    artifacts/risk_v1_calibration.json
"""

import glob
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from honest_common import DIRECTION_TEST_START, DIRECTION_VAL_START, MENA, load_panel, split_by_dates
from risk_index import (CALIBRATION_FILE, LIKELIHOOD_BANDS, aggregate, country_matrix, country_signals,
                        dyad_context, level_from_thresholds)

DELTA = 0.5
GOLDSTEIN_PREDS = {"va": "artifacts/goldstein_v2.2_val_predictions.npz",   # release v2.2 ensemble,
                   "te": "artifacts/goldstein_v2.2_test_predictions.npz"}  # held-out weeks only
DIRECTION_RUNS = "artifacts/experiments/direction_v2.0_s*.npz"


def ensemble(pattern, key):
    files = sorted(glob.glob(pattern))
    assert files, f"no files match {pattern}"
    return np.mean([np.load(f)[key] for f in files], axis=0)


def main():
    p = load_panel()
    ctx = dyad_context(p)
    M = country_matrix(p["a1"], p["a2"])
    _, d_va, d_te = split_by_dates(p["weeks"], p["target4"][0], DIRECTION_VAL_START, DIRECTION_TEST_START, 4)

    def build(week_idx, forecast, esc=None):
        """Country-level signals and outcome for these weeks; forecast: (weeks, D)."""
        sl = lambda a: a[:, week_idx].T
        s = country_signals(forecast, sl(ctx["normal"]), sl(ctx["recent"]), sl(ctx["weight"]), M)
        realized = aggregate(sl(p["target13"]) - sl(ctx["normal"]), sl(ctx["weight"]), M)
        out = {k: v.ravel() for k, v in s.items()}
        out["y"] = (realized <= -DELTA).astype(int).ravel()
        if esc is not None:
            out["esc"] = aggregate(esc, sl(ctx["weight"]), M).ravel()
        return out

    g = {s: (np.load(f)["week_idx"], np.load(f)["pred"]) for s, f in GOLDSTEIN_PREDS.items()}
    d = {"va": (d_va, ensemble(DIRECTION_RUNS, "va_prob")), "te": (d_te, ensemble(DIRECTION_RUNS, "te_prob"))}

    # ---- 1. evidence (overlap of both models' held-out weeks, so all candidates see the same rows)
    both = {}
    for s in ("va", "te"):
        wk = np.intersect1d(g[s][0], d[s][0])
        both[s] = build(wk, g[s][1][np.searchsorted(g[s][0], wk)], d[s][1][np.searchsorted(d[s][0], wk)])
        print(f"{s}: {p['weeks'][wk[0]].date()}..{p['weeks'][wk[-1]].date()} ({len(wk)} weeks x 20 countries), "
              f"deterioration rate {both[s]['y'].mean():.1%}")
    y_te = both["te"]["y"]
    print(f"\n1. Which signal predicts material deterioration (>= {DELTA} below normal, next 13 weeks)?")
    print(f"   calibrated on 2023, scored on the test weeks; base-rate Brier "
          f"{brier_score_loss(y_te, np.full(len(y_te), both['va']['y'].mean())):.4f}")
    print(f"   {'escalation probability, used as-is':<44} AUC {roc_auc_score(y_te, both['te']['esc']):.3f}"
          f"   <- worse than random")
    for name, cols in [("goldstein forecast vs normal", ["forecast_change"]),
                       ("recent 4 weeks vs normal", ["recent_change"]),
                       ("forecast + escalation probability", ["forecast_change", "esc"]),
                       ("forecast + recent  [chosen]", ["forecast_change", "recent_change"])]:
        lr = LogisticRegression().fit(np.column_stack([both["va"][c] for c in cols]), both["va"]["y"])
        pt = lr.predict_proba(np.column_stack([both["te"][c] for c in cols]))[:, 1]
        print(f"   {name:<44} AUC {roc_auc_score(y_te, pt):.3f}   Brier {brier_score_loss(y_te, pt):.4f}   "
              f"coefs {np.round(lr.coef_[0], 2)}")
    print(f"   corr(escalation probability, this week vs normal) = "
          f"{np.corrcoef(both['te']['esc'], both['te']['recent_change'])[0, 1]:.2f}: it mostly flags countries "
          f"that just had a GOOD week")

    # ---- 2. bands on the test weeks, calibration fit on 2023 only
    cols = ["forecast_change", "recent_change"]
    lr = LogisticRegression().fit(np.column_stack([both["va"][c] for c in cols]), both["va"]["y"])
    pt = lr.predict_proba(np.column_stack([both["te"][c] for c in cols]))[:, 1]
    lv = level_from_thresholds(pt, LIKELIHOOD_BANDS)
    print("\n2. Likelihood levels on the test weeks (calibration fit on 2023 only):")
    edges = [0] + LIKELIHOOD_BANDS + [1]
    for level in range(1, 6):
        m = lv == level
        txt = (f"{m.mean():5.1%} of country-weeks   predicted {pt[m].mean():5.1%}   "
               f"actually deteriorated {y_te[m].mean():5.1%}") if m.any() else "none"
        print(f"   level {level} (P {edges[level - 1]:.0%}-{edges[level]:.0%}): {txt}")

    # ---- 3. final calibration on all held-out goldstein weeks
    allw = {s: build(g[s][0], g[s][1]) for s in ("va", "te")}
    X = np.vstack([np.column_stack([allw[s][c] for c in cols]) for s in ("va", "te")])
    y = np.concatenate([allw[s]["y"] for s in ("va", "te")])
    final = LogisticRegression().fit(X, y)
    hostility = np.concatenate([allw[s]["hostility"] for s in ("va", "te")])
    thresholds = list(np.quantile(hostility, [0.2, 0.4, 0.6, 0.8]))
    calib = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": f"country activity-weighted mean goldstein over the next 13 weeks >= {DELTA} below its 52-week normal",
        "fit_weeks": f"{p['weeks'][g['va'][0][0]].date()}..{p['weeks'][g['te'][0][-1]].date()} (held-out only)",
        "fit_rows": int(len(y)), "event_rate": float(y.mean()),
        "likelihood": {"intercept": float(final.intercept_[0]),
                       "coef_forecast_change": float(final.coef_[0][0]),
                       "coef_recent_change": float(final.coef_[0][1])},
        "likelihood_bands": LIKELIHOOD_BANDS,
        "hostility_thresholds": [float(t) for t in thresholds],
    }
    with open(f"artifacts/{CALIBRATION_FILE}", "w") as fh:
        json.dump(calib, fh, indent=2)
    print(f"\n3. Final calibration ({len(y)} held-out country-weeks, event rate {y.mean():.1%}):")
    print(f"   P(deterioration) = logistic({final.intercept_[0]:+.3f} {final.coef_[0][0]:+.3f} x forecast_change "
          f"{final.coef_[0][1]:+.3f} x recent_change)")
    print(f"   hostility quintile thresholds: {np.round(thresholds, 3).tolist()}")
    print(f"   saved artifacts/{CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
