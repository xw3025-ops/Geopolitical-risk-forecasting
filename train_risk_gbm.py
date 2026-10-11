"""
Step 4 of the v2 pipeline: LightGBM risk models with an honest walk-forward
backtest, calibrated probabilities, and saved artifacts that predict_risk.py
can load without refitting anything.

Six models (task x horizon, H = 4 and 13 weeks):
  dyad_worsen_hH            P(the TONE of relationship A -> B gets >= 1
                            Goldstein point worse than its own 26-week normal
                            over the next H weeks) - diplomatic friction.
  dyad_violence_hH          P(violent events A -> B over the next H weeks
                            reach >= 2x their 26-week normal, volume-adjusted)
                            - conflict intensity. Catches wars that the tone
                            target misses (see risk_features.py).
                            Both dyad tasks include domestic A -> A dyads and,
                            when the panel has China, CHN <-> host dyads, all
                            graded (nothing is masked out).
  country_escalate_h4 / h13 P(UCDP organised-violence deaths in the country
                            over the next 4 / 13 weeks reach max(10 / 25,
                            2 x the previous 4 / 13 weeks)).
                            This is the host-country security risk.

Backtest: one fold per calendar year, TEST_YEARS. For the fold testing
year Y, the model only trains on weeks ending at least H weeks before Jan 1
of Y (embargo: no training target can overlap the test period). The last
CALIB_WEEKS of that training window are held out to fit a Platt-scaling
calibrator, so the probabilities mean what they say. Each model is an
average of N_SEEDS LightGBM boosters.

Every fold is compared against simple baselines on the same rows:
  dyad:    "recent worsening" (Goldstein last 4w minus last 26w, negated) or
           "recent violent surge" (violent events last 4w vs 26w), and the
           dyad's historical event rate in the training window
  country: "recent UCDP deaths" (lagged 13-week deaths) and the country's
           historical event rate

Outputs:
  models/risk_gbm/<model>/seed*.txt, calibrator + metadata in meta.json
  outputs/backtest_<model>.csv (every out-of-sample prediction)

Run:
    python train_risk_gbm.py            # backtest + final models
    python train_risk_gbm.py --no-backtest
"""

import argparse
import datetime as dt
import hashlib
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from risk_config import MODEL_DIR, OUTPUT_DIR, RISK_PANEL_PATH, UCDP_COUNTRY_WEEK_PATH
from risk_features import HORIZONS, build_frames, load_panel, load_ucdp

TEST_YEARS = range(2019, 2027)
CALIB_WEEKS = 52
N_SEEDS = 3
MIN_TRAIN_START = "2015-06-01"  # GDELT 2.0 starts Feb 2015; allow 13+ weeks of history

PARAMS = {
    "dyad": dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
    "country": dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
}
TOP_K = {"dyad": 20, "country": 5}
RECENT_BASELINE = {  # (description, feature, sign) - the obvious analyst heuristic per target
    "y_worsen": ("recent worsening (4w vs 26w)", "g_trend_4_26", -1),
    "y_violence": ("recent violent surge (4w vs 26w)", "violent_surge_4_26", 1),
    "y_escalate": ("recent UCDP deaths (13w)", "ucdp_log_deaths_13", 1),
}

# Known episodes to spot-check in the backtest: (task, key, week just before).
EPISODES = [
    ("dyad", "PSE_ISR", "2023-09-25", "Hamas attack, 7 Oct 2023"),
    ("dyad", "ISR_PSE", "2023-09-25", "Israeli response, Oct 2023"),
    ("dyad", "ISR_LBN", "2024-08-26", "Israel-Hezbollah escalation, Sep 2024"),
    ("dyad", "ISR_IRN", "2025-05-26", "Israel-Iran war, Jun 2025"),
    ("dyad", "USA_YEM", "2023-12-25", "US/UK strikes on Houthis, Jan 2024"),
    ("country", "SDN", "2023-03-27", "Sudan civil war, 15 Apr 2023"),
    ("country", "PSE", "2023-09-25", "Gaza war, Oct 2023"),
    ("country", "LBN", "2024-08-26", "Lebanon escalation, Sep 2024"),
    ("country", "IRN", "2025-05-26", "Israel-Iran war, Jun 2025"),
]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_specs(dyad_frame, country_frame, dyad_features, country_features):
    specs = []
    for h in HORIZONS:
        for target in ("worsen", "violence"):
            specs.append(dict(name=f"dyad_{target}_h{h}", task="dyad", h=h, frame=dyad_frame, key="dyad",
                              label=f"y_{target}_{h}", features=dyad_features, categorical=["actor1", "actor2"]))
        specs.append(dict(name=f"country_escalate_h{h}", task="country", h=h, frame=country_frame, key="country",
                          label=f"y_escalate_{h}", features=country_features, categorical=["country"]))
    return specs


def prepare_X(frame, features, categories):
    X = frame[features].copy()
    for col, cats in categories.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return X


class CalibratedEnsemble:
    def __init__(self, boosters, a, b):
        self.boosters, self.a, self.b = boosters, a, b

    def margin(self, X):
        return np.mean([bst.predict(X, raw_score=True) for bst in self.boosters], axis=0)

    def proba(self, X):
        return 1 / (1 + np.exp(-(self.a * self.margin(X) + self.b)))


def fit_calibrated(spec, rows, categories):
    """Fit boosters on all but the last CALIB_WEEKS weeks of `rows`, then
    Platt-scale on those last weeks."""
    calib_start = rows.week_start.max() - pd.Timedelta(weeks=CALIB_WEEKS - 1)
    fit, cal = rows[rows.week_start < calib_start], rows[rows.week_start >= calib_start]
    X_fit, y_fit = prepare_X(fit, spec["features"], categories), fit[spec["label"]].astype(int)
    boosters = []
    for seed in range(N_SEEDS):
        model = lgb.LGBMClassifier(**PARAMS[spec["task"]], random_state=seed, verbose=-1)
        model.fit(X_fit, y_fit)
        boosters.append(model.booster_)
    ens = CalibratedEnsemble(boosters, 1.0, 0.0)
    m_cal = ens.margin(prepare_X(cal, spec["features"], categories))
    if cal[spec["label"]].nunique() == 2:
        lr = LogisticRegression(C=1e4).fit(m_cal[:, None], cal[spec["label"]].astype(int))
        ens.a, ens.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return ens, (fit.week_start.min(), fit.week_start.max(), cal.week_start.min(), cal.week_start.max())


def precision_at_k(df, score, k):
    top = df.sort_values(score, ascending=False).groupby("week_start").head(k)
    return top.y.mean()


def metrics(df, score, k):
    return {"auc": roc_auc_score(df.y, df[score]), "ap": average_precision_score(df.y, df[score]),
            f"prec@{k}": precision_at_k(df, score, k)}


def backtest(spec, categories):
    frame, label, h = spec["frame"], spec["label"], spec["h"]
    graded = frame[frame[label].notna() & (frame.week_start >= MIN_TRAIN_START)]
    preds = []
    for year in TEST_YEARS:
        test_start = pd.Timestamp(f"{year}-01-01")
        train = graded[graded.week_start < test_start - pd.Timedelta(weeks=h)]
        test = graded[(graded.week_start >= test_start) & (graded.week_start < pd.Timestamp(f"{year + 1}-01-01"))]
        if test.empty or test[label].nunique() < 2:
            continue
        ens, _ = fit_calibrated(spec, train, categories)
        out = test[["week_start", spec["key"]]].copy()
        out["y"] = test[label].to_numpy()
        out["fold"] = year
        out["p_model"] = ens.proba(prepare_X(test, spec["features"], categories))
        hist_rate = train.groupby(spec["key"])[label].mean()
        out["s_hist_rate"] = test[spec["key"]].map(hist_rate).fillna(train[label].mean()).to_numpy()
        _, feature, sign = RECENT_BASELINE[label.rsplit("_", 1)[0]]
        out["s_recent"] = (sign * test[feature]).fillna(0).to_numpy()
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def report_backtest(spec, preds):
    k = TOP_K[spec["task"]]
    recent_name = RECENT_BASELINE[spec["label"].rsplit("_", 1)[0]][0]
    rows = []
    for fold, df in preds.groupby("fold"):
        m = metrics(df, "p_model", k)
        rows.append({"fold": fold, "n": len(df), "base_rate": df.y.mean(), **m,
                     "auc_recent": roc_auc_score(df.y, df.s_recent),
                     "auc_hist": roc_auc_score(df.y, df.s_hist_rate)})
    per_fold = pd.DataFrame(rows)
    print(f"\n=== {spec['name']}  (walk-forward, {spec['h']}-week embargo, {N_SEEDS} seeds) ===")
    print(per_fold.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    pooled = {}
    print(f"{'pooled out-of-sample':<34}{'ROC-AUC':>9}{'avg prec':>10}{f'prec@{k}/wk':>13}")
    for name, col in (("LightGBM (calibrated)", "p_model"), (recent_name, "s_recent"),
                      ("historical rate per " + spec["key"], "s_hist_rate")):
        m = metrics(preds, col, k)
        pooled[col] = m
        print(f"{name:<34}{m['auc']:>9.3f}{m['ap']:>10.3f}{m[f'prec@{k}']:>13.3f}")
    print(f"base rate {preds.y.mean():.3f};  Brier: model {brier_score_loss(preds.y, preds.p_model):.4f} "
          f"vs constant {brier_score_loss(preds.y, np.full(len(preds), preds.y.mean())):.4f}")
    bins = pd.cut(preds.p_model, [0, .05, .1, .2, .4, 1.0], include_lowest=True)
    calib = preds.groupby(bins, observed=True).agg(n=("y", "size"), mean_pred=("p_model", "mean"), observed=("y", "mean"))
    print("calibration (predicted vs observed):  " + "  ".join(
        f"{iv}: {r.mean_pred:.2f}/{r.observed:.2f} (n={r.n})" for iv, r in calib.iterrows()))

    for task, key, week, label in EPISODES:
        if task != spec["task"]:
            continue
        hit = preds[(preds[spec["key"]] == key) & (preds.week_start == pd.Timestamp(week))]
        if hit.empty:
            continue
        same_week = preds[preds.week_start == pd.Timestamp(week)]
        rank = int((same_week.p_model > hit.p_model.iloc[0]).sum()) + 1
        print(f"  episode check - {label}: {key} week {week}: P={hit.p_model.iloc[0]:.2f}, "
              f"rank {rank}/{len(same_week)}, outcome={'yes' if hit.y.iloc[0] else 'no'}")
    return per_fold, pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=RISK_PANEL_PATH)
    parser.add_argument("--ucdp", default=UCDP_COUNTRY_WEEK_PATH)
    parser.add_argument("--no-backtest", action="store_true")
    args = parser.parse_args()

    panel, ucdp = load_panel(args.panel), load_ucdp(args.ucdp)
    dyad_frame, country_frame, dyad_features, country_features = build_frames(panel, ucdp)
    print(f"Panel {panel.week_start.min().date()}..{panel.week_start.max().date()}: "
          f"{dyad_frame.dyad.nunique()} dyads, {country_frame.country.nunique()} countries; "
          f"{len(dyad_features)} dyad features, {len(country_features)} country features")
    hashes = {"panel": file_sha256(args.panel), "ucdp": file_sha256(args.ucdp)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for spec in task_specs(dyad_frame, country_frame, dyad_features, country_features):
        categories = {c: sorted(spec["frame"][c].unique().tolist()) for c in spec["categorical"]}
        backtest_summary = None
        if not args.no_backtest:
            preds = backtest(spec, categories)
            preds.to_csv(os.path.join(OUTPUT_DIR, f"backtest_{spec['name']}.csv"), index=False)
            per_fold, pooled = report_backtest(spec, preds)
            backtest_summary = {"per_fold": per_fold.to_dict(orient="records"), "pooled": pooled}

        graded = spec["frame"][spec["frame"][spec["label"]].notna() & (spec["frame"].week_start >= MIN_TRAIN_START)]
        ens, (fit_start, fit_end, cal_start, cal_end) = fit_calibrated(spec, graded, categories)
        out_dir = os.path.join(MODEL_DIR, spec["name"])
        os.makedirs(out_dir, exist_ok=True)
        for i, bst in enumerate(ens.boosters):
            bst.save_model(os.path.join(out_dir, f"seed{i}.txt"))
        meta = {
            "name": spec["name"], "task": spec["task"], "horizon_weeks": spec["h"], "label": spec["label"],
            "key": spec["key"], "features": spec["features"], "categories": categories,
            "calibrator": {"a": ens.a, "b": ens.b}, "n_seeds": N_SEEDS, "params": PARAMS[spec["task"]],
            "fit_weeks": [str(fit_start.date()), str(fit_end.date())],
            "calibration_weeks": [str(cal_start.date()), str(cal_end.date())],
            "data_sha256": hashes, "data_paths": {"panel": args.panel, "ucdp": args.ucdp},
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "backtest": backtest_summary,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"Saved {out_dir} (fit {fit_start.date()}..{fit_end.date()}, "
              f"calibrated on {cal_start.date()}..{cal_end.date()})")


if __name__ == "__main__":
    main()
