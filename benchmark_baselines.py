"""
Honest benchmark for the goldstein forecasting task - runs on the LOCAL CSV
panel only (no BigQuery, no TensorFlow), so anyone can reproduce it.

Why this exists: every model in this repo was compared against two
baselines only - "always predict the train mean" and "persistence (this
week's goldstein)". Both are very weak for a smoothed 13-week-average
target, so they made the transformers look much better than they are. This
script adds the baselines a forecaster would actually try first, fixes three
evaluation problems, and adds a risk-oriented task closer to how the output
is used (flagging relationships about to deteriorate).

Evaluation fixes vs. the training scripts:
  1. Same test weeks as release_v1.0_goldstein.py (2024-03-18..2026-06-22),
     but with a 13-week EMBARGO before validation/test: a training row's
     target averages the 13 weeks AFTER it, so without a gap the last 13
     training weeks peek into the evaluation period.
  2. The 76 dyads involving LBY or SDN have avg_goldstein padded to 0 for
     every week (the goldstein source never covered them), so their target
     is a constant 0 - trivially "predictable" and inflating R^2. Results
     are reported both on all 400 dyads (comparable to the release numbers)
     and on the 324 dyads with real data.
  3. Weeks with zero events have avg_goldstein = 0, which reads as
     "neutral" but means "no data". Features here treat those as missing
     (NaN) instead; tree models handle NaN natively.

Models compared on the release v1.0 goldstein regression target
(target_avg_goldstein_next13w):
  - train mean, per-dyad train mean, persistence, trailing 13/26-week mean
  - HistGradientBoostingRegressor on hand-built lag features (trailing
    goldstein means/volatility/trend, CAMEO conflict shares, country- and
    region-level context, reverse-dyad history)

Risk task ("deterioration alert"): label = 1 if the next-13-week mean
goldstein ends up >= ALERT_DROP points BELOW the dyad's own trailing
26-week mean, i.e. the relationship gets materially worse than its recent
normal. Graded with ROC-AUC, average precision, and precision of the top-K
alerts each week (what an analyst reviewing a weekly watchlist would see).

Also prints a diagnostic for the escalation/de-escalation label used by the
v9/v10/v12/release direction models.

Reference numbers (from the docstrings of the release scripts, not re-run):
  release_v1.0_goldstein   test R^2 = 0.646, MAE = 0.600
  release_v1.0_direction   test accuracy = 77.63% (4-week horizon, BigQuery panel)

Run:
    python benchmark_baselines.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
HORIZON_WEEKS = 13
LOOKBACK_WEEKS = 13  # matches the spatio-temporal models, so the test weeks line up exactly
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
EMBARGO_WEEKS = HORIZON_WEEKS
NO_GOLDSTEIN_COVERAGE = ["LBY", "SDN"]  # padded to 0 in build_goldstein_panel.py

ALERT_DROP = 1.0  # goldstein points below the trailing 26-week mean
ALERT_TOP_K = 20  # dyads on the weekly watchlist

COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
# CAMEO quad classes by root code: 01-05 verbal cooperation, 06-09 material
# cooperation, 10-14 verbal conflict, 15-20 material conflict; 18-20 are
# assault / fight / mass violence.
MATERIAL_CONFLICT = [f"count_{i:02d}" for i in range(15, 21)]
VIOLENT = ["count_18", "count_19", "count_20"]
VERBAL_CONFLICT = [f"count_{i:02d}" for i in range(10, 15)]


def to_matrix(df, col, dyads, weeks):
    """(n_weeks, n_dyads) wide frame - rolling() then works per dyad column."""
    return df.pivot(index="week_start", columns="dyad", values=col).reindex(index=weeks, columns=dyads)


def safe_share(num, den):
    return num / den.where(den > 0)


def build_features(df):
    df = df.copy()
    df["dyad"] = df.Actor1CountryCode + "_" + df.Actor2CountryCode
    df["n_events"] = df[COUNT_COLS].sum(axis=1)
    df["mat_conf"] = df[MATERIAL_CONFLICT].sum(axis=1)
    df["violent"] = df[VIOLENT].sum(axis=1)
    df["verb_conf"] = df[VERBAL_CONFLICT].sum(axis=1)
    # Zero-event weeks: goldstein 0 means "no data", not "neutral".
    df["g_obs"] = df.avg_goldstein.where(df.n_events > 0)

    weeks = np.sort(df.week_start.unique())
    dyads = np.sort(df.dyad.unique())
    W = {c: to_matrix(df, c, dyads, weeks) for c in
         ["g_obs", "avg_goldstein", "n_events", "mat_conf", "violent", "verb_conf", TARGET_COL]}

    feats = {}
    g = W["g_obs"]
    feats["g_now"] = g
    for w in (4, 13, 26, 52):
        feats[f"g_mean_{w}"] = g.rolling(w, min_periods=1).mean()
        feats[f"log_events_{w}"] = np.log1p(W["n_events"].rolling(w, min_periods=1).mean())
        feats[f"mat_conf_share_{w}"] = safe_share(W["mat_conf"].rolling(w, min_periods=1).sum(),
                                                  W["n_events"].rolling(w, min_periods=1).sum())
        feats[f"violent_share_{w}"] = safe_share(W["violent"].rolling(w, min_periods=1).sum(),
                                                 W["n_events"].rolling(w, min_periods=1).sum())
    feats["verb_conf_share_4"] = safe_share(W["verb_conf"].rolling(4, min_periods=1).sum(),
                                            W["n_events"].rolling(4, min_periods=1).sum())
    feats["g_std_13"] = g.rolling(13, min_periods=2).std()
    feats["g_min_13"] = g.rolling(13, min_periods=1).min()
    feats["g_trend_4_26"] = feats["g_mean_4"] - feats["g_mean_26"]
    feats["event_surge_4_52"] = feats["log_events_4"] - feats["log_events_52"]
    feats["frac_weeks_active_13"] = (W["n_events"] > 0).rolling(13, min_periods=1).mean()
    # Same definition as the target (zero-event weeks count as 0), so the
    # model can learn the target's own bookkeeping.
    feats["raw_mean_13"] = W["avg_goldstein"].rolling(13, min_periods=1).mean()
    feats["raw_mean_26"] = W["avg_goldstein"].rolling(26, min_periods=1).mean()

    # Reverse dyad (B->A) history.
    a1 = pd.Series([d.split("_")[0] for d in dyads], index=dyads)
    a2 = pd.Series([d.split("_")[1] for d in dyads], index=dyads)
    reverse = [f"{a2[d]}_{a1[d]}" for d in dyads]
    feats["rev_g_mean_13"] = feats["g_mean_13"][reverse].set_axis(dyads, axis=1)
    feats["rev_mat_conf_share_4"] = feats["mat_conf_share_4"][reverse].set_axis(dyads, axis=1)

    # Country context: material-conflict share over every dyad a country is in.
    countries = sorted(set(a1) | set(a2))
    country_conf, country_events = {}, {}
    for c in countries:
        cols = dyads[(a1 == c).to_numpy() | (a2 == c).to_numpy()]
        ev = W["n_events"][cols].sum(axis=1).rolling(4, min_periods=1).sum()
        country_conf[c] = safe_share(W["mat_conf"][cols].sum(axis=1).rolling(4, min_periods=1).sum(), ev)
        country_events[c] = np.log1p(ev)
    for role, actor in (("a1", a1), ("a2", a2)):
        feats[f"{role}_country_conf_share_4"] = pd.DataFrame(
            {d: country_conf[actor[d]] for d in dyads}, index=weeks)
        feats[f"{role}_country_log_events_4"] = pd.DataFrame(
            {d: country_events[actor[d]] for d in dyads}, index=weeks)
    region_conf = safe_share(W["mat_conf"].sum(axis=1).rolling(4, min_periods=1).sum(),
                             W["n_events"].sum(axis=1).rolling(4, min_periods=1).sum())
    feats["region_conf_share_4"] = pd.DataFrame({d: region_conf for d in dyads}, index=weeks)

    long = pd.DataFrame({
        "week_idx": np.repeat(np.arange(len(weeks)), len(dyads)),
        "week_start": np.repeat(weeks, len(dyads)),
        "dyad": np.tile(dyads, len(weeks)),
    })
    long["actor1"] = long.dyad.map(a1)
    long["actor2"] = long.dyad.map(a2)
    for name, mat in feats.items():
        long[name] = mat.to_numpy().ravel()
    long["target"] = W[TARGET_COL].to_numpy().ravel()
    return long, list(feats), weeks


def make_splits(long, weeks):
    """Replicate the release scripts' split on prediction weeks, then embargo."""
    n_valid_target = int(long.groupby("week_idx").target.apply(lambda s: s.notna().all()).sum())
    first = LOOKBACK_WEEKS - 1
    n_pred = n_valid_target - LOOKBACK_WEEKS + 1
    n_train = int(n_pred * TRAIN_FRAC)
    n_val = int(n_pred * VAL_FRAC)
    val_start = first + n_train
    test_start = val_start + n_val
    test_end = first + n_pred  # exclusive
    train = (long.week_idx >= first) & (long.week_idx < val_start - EMBARGO_WEEKS)
    test = (long.week_idx >= test_start) & (long.week_idx < test_end)
    print(f"Train weeks {pd.Timestamp(weeks[first]).date()}..{pd.Timestamp(weeks[val_start - EMBARGO_WEEKS - 1]).date()} "
          f"({EMBARGO_WEEKS}-week embargo before validation), "
          f"test weeks {pd.Timestamp(weeks[test_start]).date()}..{pd.Timestamp(weeks[test_end - 1]).date()} "
          f"({test_end - test_start} weeks)")
    return train.to_numpy(), test.to_numpy()


def within_dyad_r2(frame, pred_col):
    """R^2 after removing each dyad's own test-period mean: does the model
    track CHANGES over time, or just know which dyads are usually hostile?"""
    t = frame.target - frame.groupby("dyad").target.transform("mean")
    p = frame[pred_col] - frame.groupby("dyad")[pred_col].transform("mean")
    return r2_score(t, p)


def report_regression(test_df, preds, real_mask):
    print(f"\n{'model':<34}{'R2 all400':>10}{'MAE all400':>11}{'R2 real':>9}{'MAE real':>10}{'within-dyad R2':>16}")
    for name, col in preds:
        r = test_df[real_mask]
        print(f"{name:<34}{r2_score(test_df.target, test_df[col]):>10.3f}"
              f"{mean_absolute_error(test_df.target, test_df[col]):>11.3f}"
              f"{r2_score(r.target, r[col]):>9.3f}{mean_absolute_error(r.target, r[col]):>10.3f}"
              f"{within_dyad_r2(r, col):>16.3f}")
    print(f"{'release_v1.0_goldstein (reported)':<34}{0.646:>10.3f}{0.600:>11.3f}{'n/a':>9}{'n/a':>10}{'n/a':>16}")


def precision_at_k(frame, score_col, k):
    top = frame.sort_values(score_col, ascending=False).groupby("week_idx").head(k)
    return top.label.mean()


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
    long, feature_cols, weeks = build_features(raw)
    train, test = make_splits(long, weeks)
    real = ~(long.actor1.isin(NO_GOLDSTEIN_COVERAGE) | long.actor2.isin(NO_GOLDSTEIN_COVERAGE)).to_numpy()
    print(f"{len(feature_cols)} features; {long.dyad.nunique()} dyads, "
          f"{long[real].dyad.nunique()} with real goldstein coverage")

    tr, te = long[train].copy(), long[test].copy()
    test_real = ~(te.actor1.isin(NO_GOLDSTEIN_COVERAGE) | te.actor2.isin(NO_GOLDSTEIN_COVERAGE)).to_numpy()

    # ---------------- Regression: release v1.0 goldstein target ----------------
    te["p_train_mean"] = tr.target.mean()
    te["p_dyad_mean"] = te.dyad.map(tr.groupby("dyad").target.mean())
    te["p_persistence"] = te.g_now.fillna(0.0)  # current week, zero-event weeks as 0 (as in benchmark_persistence.py)
    te["p_trail_13"] = te.raw_mean_13
    te["p_trail_26"] = te.raw_mean_26

    X_cols = feature_cols + ["actor1", "actor2"]
    for c in ("actor1", "actor2"):
        cats = pd.Categorical(long[c]).categories
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c], categories=cats)
    gbr = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features="from_dtype", random_state=0,
    )
    gbr.fit(tr[X_cols], tr.target)
    te["p_gbm"] = gbr.predict(te[X_cols])

    print("\n=== Regression: next-13-week average goldstein (release v1.0 goldstein task) ===")
    report_regression(te, [
        ("train mean", "p_train_mean"),
        ("persistence (this week)", "p_persistence"),
        ("per-dyad train mean", "p_dyad_mean"),
        ("trailing 13-week mean", "p_trail_13"),
        ("trailing 26-week mean", "p_trail_26"),
        ("gradient boosting (lag features)", "p_gbm"),
    ], test_real)

    # ---------------- Risk task: deterioration alert ----------------
    long["label"] = (long.target - long.raw_mean_26 <= -ALERT_DROP).astype(float)
    long.loc[long.target.isna(), "label"] = np.nan
    tr_a = long[train & real].copy()
    te_a = long[test & real].copy()
    for c in ("actor1", "actor2"):
        cats = pd.Categorical(long[c]).categories
        tr_a[c] = pd.Categorical(tr_a[c], categories=cats)
        te_a[c] = pd.Categorical(te_a[c], categories=cats)
    gbc = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features="from_dtype", random_state=0,
    )
    gbc.fit(tr_a[X_cols], tr_a.label.astype(int))
    te_a["s_gbm"] = gbc.predict_proba(te_a[X_cols])[:, 1]
    te_a["s_momentum"] = -te_a.g_trend_4_26.fillna(0.0)  # recent 4 weeks worse than 26-week normal
    te_a["s_conflict_share"] = te_a.mat_conf_share_4.fillna(0.0)

    print(f"\n=== Deterioration alert: next-13w mean >= {ALERT_DROP} below trailing 26w mean "
          f"({int(real.sum() / len(weeks))} real dyads) ===")
    print(f"Base rate in test: {te_a.label.mean():.1%} of dyad-weeks")
    print(f"{'score':<34}{'ROC-AUC':>9}{'avg prec':>10}{f'prec@top{ALERT_TOP_K}/wk':>16}")
    for name, col in [("recent momentum (4w vs 26w)", "s_momentum"),
                      ("material-conflict share (4w)", "s_conflict_share"),
                      ("gradient boosting classifier", "s_gbm")]:
        print(f"{name:<34}{roc_auc_score(te_a.label, te_a[col]):>9.3f}"
              f"{average_precision_score(te_a.label, te_a[col]):>10.3f}"
              f"{precision_at_k(te_a, col, ALERT_TOP_K):>16.3f}")

    # ---------------- Diagnostic: the direction label ----------------
    # v9/v10/v12/release_direction label = 1 if the future average goldstein
    # is above THIS WEEK's single value. A single noisy week vs a multi-week
    # average is mostly regression to the mean, which a one-line rule catches.
    g_raw = raw.assign(dyad=raw.Actor1CountryCode + "_" + raw.Actor2CountryCode)
    g_raw = to_matrix(g_raw, "avg_goldstein", np.sort(long.dyad.unique()), weeks).to_numpy().T
    n_weeks = g_raw.shape[1]
    print("\n=== Direction-label diagnostic (rule: 'this week below its own trailing 13w mean => rise') ===")
    for h in (4, 13):
        fut = np.full_like(g_raw, np.nan)
        for k in range(n_weeks - h):
            fut[:, k] = g_raw[:, k + 1:k + 1 + h].mean(axis=1)
        trail = pd.DataFrame(g_raw.T).rolling(LOOKBACK_WEEKS).mean().to_numpy().T
        ks = np.arange(LOOKBACK_WEEKS - 1, n_weeks - h)
        n_tr, n_va = int(len(ks) * TRAIN_FRAC), int(len(ks) * VAL_FRAC)
        kt = ks[n_tr + n_va:]
        label = fut[:, kt] > g_raw[:, kt]
        rule = g_raw[:, kt] < trail[:, kt]
        majority = max(label.mean(), 1 - label.mean())
        print(f"  {h:>2}-week horizon: rule accuracy {(rule == label).mean():.1%}  "
              f"(majority class {majority:.1%}; release_v1.0_direction reported 77.6% at 4 weeks)")


if __name__ == "__main__":
    main()
