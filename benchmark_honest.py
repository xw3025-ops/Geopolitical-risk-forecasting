"""
Honest benchmark for the v2.0 release models - local files only (no
BigQuery calls, no TensorFlow), so anyone can reproduce it.

Why this exists: every earlier model in this repo was compared against two
baselines only - "always predict the train mean" and "persistence (this
week's goldstein)". Both are very weak for a smoothed multi-week-average
target, so they made the transformers look much better than they are. This
script scores the release transformers against the baselines a forecaster
would actually try first, plus gradient-boosted trees on hand-built lag
features (a BENCHMARK only - not a release candidate), all on exactly the
same rows.

Same data and evaluation rules as the v2.0 release scripts (honest_common.py):
  - all 20 MENA countries (400 dyads) with real data from the BigQuery
    panel (v1.0 goldstein's CSV lacked LBY and SDN and had them as constant 0s)
  - embargo before validation and test equal to the target horizon (13 / 4
    weeks), so no training row's target overlaps the evaluation period
  - test weeks identical to v1.0: goldstein 2024-03-18..2026-06-22,
    direction 2024-05-27..2026-08-24
  - for the tree-model features, zero-event weeks are missing (NaN), not
    "neutral 0"; targets keep the v1.0 definitions (zeros included)

Tasks
  A. GOLDSTEIN - mean goldstein over the next 13 weeks (regression).
  B. DETERIORATION ALERT - derived from A's forecasts: label = 1 if the
     next-13-week mean goldstein (event weeks only) ends up >= ALERT_DROP
     points BELOW the dyad's trailing 26-week mean (event weeks only). Rows
     need >= 4 future and >= 8 trailing event weeks to be labelled. Graded
     with ROC-AUC, average precision, and precision of the top-K alerts each
     week (what an analyst reviewing a weekly watchlist would see).
  C. DIRECTION - 1 = ESCALATION if the mean goldstein over the next 4 weeks
     is LOWER than this week's (v2.0 label; v1.0 had it inverted). Plus a
     diagnostic of how much of it is plain mean reversion.

Transformer rows appear once release_v2.0_goldstein.py /
release_v2.0_direction.py have been run (they save their test predictions
to artifacts/).

Run:
    python benchmark_honest.py
"""

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

from honest_common import (DIRECTION_HORIZON, DIRECTION_TEST_START, DIRECTION_VAL_START,
                           GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START,
                           build_features, forward_nanmean, load_panel, rows, split_by_dates,
                           trailing_count, trailing_mean, trailing_nanmean, trailing_nanstd)

GOLDSTEIN_PREDS = "artifacts/goldstein_v2.0_test_predictions.npz"
DIRECTION_PREDS = "artifacts/direction_v2.0_test_predictions.npz"

ALERT_DROP = 1.0
ALERT_MIN_FUTURE, ALERT_MIN_TRAILING = 4, 8
TOP_K = 10

GBM_PARAMS = dict(learning_rate=0.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=False, random_state=0)


def fit_gbm(cls, X_tr, y_tr, X_va, y_va, score):
    """Fit with a generous max_iter, pick the iteration count that scores
    best on validation (lower = better), refit with exactly that many."""
    model = cls(**GBM_PARAMS).fit(X_tr, y_tr)
    staged = model.staged_predict_proba(X_va) if cls is HistGradientBoostingClassifier \
        else model.staged_predict(X_va)
    best = int(np.argmin([score(y_va, s) for s in staged])) + 1
    return cls(**{**GBM_PARAMS, "max_iter": best}).fit(X_tr, y_tr), best


def load_transformer(path, p, week_idx, key):
    """Saved release test predictions, checked to line up with these rows."""
    if not os.path.exists(path):
        print(f"  ({path} not found - run the release script to add the transformer row)")
        return None
    z = np.load(path, allow_pickle=True)
    assert np.array_equal(z["week_idx"], week_idx), f"{path}: test weeks differ"
    assert np.array_equal(z["a1"], p["a1"]) and np.array_equal(z["a2"], p["a2"]), f"{path}: dyads differ"
    return z[key]


def header(title):
    print(f"\n{'=' * 104}\n{title}\n{'=' * 104}")


def main():
    p = load_panel()
    D = len(p["a1"])
    F, feat_names = build_features(p)
    wk = p["weeks"]
    groups = {"all 400 dyads": np.ones(D, bool)}
    print(f"{D} dyads, {len(wk)} weeks ({wk[0].date()}..{wk[-1].date()}), {len(feat_names)} GBM features")

    # ================================================================ A
    header("A. GOLDSTEIN - mean goldstein over the next 13 weeks")
    y_all = p["target13"]
    tr, va, te = split_by_dates(wk, y_all[0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START, GOLDSTEIN_HORIZON)
    print(f"train {wk[tr[0]].date()}..{wk[tr[-1]].date()} ({len(tr)} wks) | "
          f"val {wk[va[0]].date()}..{wk[va[-1]].date()} ({len(va)}) | "
          f"test {wk[te[0]].date()}..{wk[te[-1]].date()} ({len(te)}) | {GOLDSTEIN_HORIZON}-week embargo")
    y_te = y_all[:, te].T  # (weeks, dyads)

    preds = {
        "train mean": np.full_like(y_te, np.nanmean(y_all[:, tr])),
        "per-dyad train mean": np.tile(np.nanmean(y_all[:, tr], axis=1), (len(te), 1)),
        "persistence (this week)": p["gold_zero"][:, te].T,
        "trailing 13-week mean": trailing_mean(p["gold_zero"], 13)[:, te].T,
        "trailing 26-week mean": trailing_mean(p["gold_zero"], 26)[:, te].T,
        "trailing 52-week mean": trailing_mean(p["gold_zero"], 52)[:, te].T,
    }
    gbm_g, n_iter = fit_gbm(HistGradientBoostingRegressor, rows(F, tr), rows(y_all, tr),
                            rows(F, va), rows(y_all, va), mean_squared_error)
    preds[f"gradient boosting ({n_iter} trees)"] = gbm_g.predict(rows(F, te)).reshape(len(te), D)
    t_pred = load_transformer(GOLDSTEIN_PREDS, p, te, "pred")
    if t_pred is not None:
        assert np.allclose(load_transformer(GOLDSTEIN_PREDS, p, te, "y_true"), y_te, equal_nan=True)
        preds["RELEASE v2.0 transformer"] = t_pred

    print(f"{'':38}" + "".join(f"  {g:<24}" for g in groups))
    for name, pred in preds.items():
        out = f"  {name:<36}"
        for mask in groups.values():
            yt, pt = y_te[:, mask].ravel(), pred[:, mask].ravel()
            out += f"  R^2 {r2_score(yt, pt):6.3f}  MAE {mean_absolute_error(yt, pt):.3f}"
        print(out)
    print("  for reference, release v1.0 (docstring): R^2 0.646  MAE 0.600 on all 400 - but with LBY/SDN as\n"
          "  constant-0 targets and no embargo, so not comparable to the rows above")

    # ================================================================ B
    header(f"B. DETERIORATION ALERT - next 13 wks >= {ALERT_DROP} below the dyad's trailing 26-wk normal")
    fut, fut_n = forward_nanmean(p["gold_obs"], 13)
    normal = trailing_nanmean(p["gold_obs"], 26)
    labelled = ((fut_n >= ALERT_MIN_FUTURE) & (trailing_count(p["gold_obs"], 26) >= ALERT_MIN_TRAILING)
                & ~np.isnan(fut))
    alert = (fut <= normal - ALERT_DROP).astype("int32")

    def lab_rows(week_idx):
        m = rows(labelled, week_idx)
        return m, rows(alert, week_idx)[m]

    m_tr, a_tr = lab_rows(tr)
    m_va, a_va = lab_rows(va)
    m_te, a_te = lab_rows(te)
    print(f"labelled test rows: {m_te.sum()} of {m_te.size}; alert base rate: test {a_te.mean():.1%}, "
          f"train {a_tr.mean():.1%}")
    gbm_a, n_iter_a = fit_gbm(HistGradientBoostingClassifier, rows(F, tr)[m_tr], a_tr,
                              rows(F, va)[m_va], a_va, log_loss)

    lab_te, alert_te = labelled[:, te].T, alert[:, te].T
    zmean26_te = trailing_mean(p["gold_zero"], 26)[:, te].T
    scores = {  # all (test weeks, dyads); higher = more alarming
        "recent decline (26wk - 4wk mean)": (normal - trailing_nanmean(p["gold_obs"], 4))[:, te].T,
        "volatility (13wk std)": trailing_nanstd(p["gold_obs"], 13)[:, te].T,
        "gradient boosting: normal - forecast": zmean26_te - preds[f"gradient boosting ({n_iter} trees)"],
    }
    s = np.full(lab_te.shape, np.nan)
    s[lab_te] = gbm_a.predict_proba(rows(F, te)[m_te])[:, 1]
    scores[f"gradient boosting on the alert label"] = s
    if t_pred is not None:
        scores["RELEASE v2.0 transformer: normal - forecast"] = zmean26_te - t_pred
    for name, s_te in scores.items():
        s_te = np.nan_to_num(s_te, nan=-1e9)
        auc = roc_auc_score(alert_te[lab_te], s_te[lab_te])
        ap = average_precision_score(alert_te[lab_te], s_te[lab_te])
        topk = [alert_te[w, lab_te[w]][np.argsort(-s_te[w, lab_te[w]])[:TOP_K]].mean()
                for w in range(len(te)) if lab_te[w].sum() >= TOP_K]
        print(f"  {name:<44} ROC-AUC {auc:.3f}   avg precision {ap:.3f}   "
              f"precision@top-{TOP_K}/week {np.mean(topk):.1%}")
    print(f"  (random ranking: avg precision = precision@top-{TOP_K} = base rate {a_te.mean():.1%})")

    # ================================================================ C
    header("C. DIRECTION - escalation = mean goldstein over the next 4 weeks LOWER than this week's")
    label = p["escalation"]
    tr, va, te = split_by_dates(wk, p["target4"][0], DIRECTION_VAL_START, DIRECTION_TEST_START,
                                DIRECTION_HORIZON)
    print(f"train {wk[tr[0]].date()}..{wk[tr[-1]].date()} | val {wk[va[0]].date()}..{wk[va[-1]].date()} | "
          f"test {wk[te[0]].date()}..{wk[te[-1]].date()} ({len(te)} wks) | {DIRECTION_HORIZON}-week embargo")
    l_te = label[:, te].T
    print(f"escalation rate: train {label[:, tr].mean():.1%}, test {l_te.mean():.1%}")
    majority = int(label[:, tr].mean() >= 0.5)
    # escalation if this week is ABOVE its recent level (=> likely to fall back)
    reversion = (p["gold_zero"] > trailing_mean(p["gold_zero"], 13)).astype("int32")

    gbm_d, n_iter_d = fit_gbm(HistGradientBoostingClassifier, rows(F, tr), rows(label, tr),
                              rows(F, va), rows(label, va), log_loss)
    prob_d = gbm_d.predict_proba(rows(F, te))[:, 1].reshape(len(te), D)
    dir_preds = {
        "majority class": (np.full(l_te.shape, majority), None),
        "mean reversion: this wk > 13wk mean": (reversion[:, te].T, None),
        f"gradient boosting ({n_iter_d} trees)": ((prob_d >= 0.5).astype(int), prob_d),
    }
    t_prob = load_transformer(DIRECTION_PREDS, p, te, "prob")
    if t_prob is not None:
        assert np.array_equal(load_transformer(DIRECTION_PREDS, p, te, "y_true"), l_te)
        dir_preds["RELEASE v2.0 transformer"] = ((t_prob >= 0.5).astype(int), t_prob)

    print(f"{'':38}" + "".join(f"  {g:<24}" for g in groups))
    for name, (pred, prob) in dir_preds.items():
        out = f"  {name:<36}"
        for mask in groups.values():
            acc = accuracy_score(l_te[:, mask].ravel(), pred[:, mask].ravel())
            out += (f"  acc {acc:.1%}  AUC {roc_auc_score(l_te[:, mask].ravel(), prob[:, mask].ravel()):.3f}"
                    if prob is not None else f"  acc {acc:.1%}           ")
        print(out)
    print("  for reference, release v1.0 (docstring): acc 77.63% - but on the inverted label (goldstein\n"
          "  rising) and without the embargo")

    print("\n  Diagnostic - how much of the direction label is mean reversion:")
    print("    The label compares a 4-week AVERAGE to ONE week's value. One week is noisy; the average\n"
          "    sits near normal. So 'this week was unusually good' => 'next weeks look lower' wins often\n"
          "    without anticipating any real turn.")
    empty = ~p["has_events"][:, te].T
    rv = reversion[:, te].T
    print(f"    rule accuracy when this week had events: {(rv[~empty] == l_te[~empty]).mean():.1%} | "
          f"when it had none ({empty.mean():.1%} of rows): {(rv[empty] == l_te[empty]).mean():.1%}")
    gap = np.abs(p["gold_zero"] - trailing_mean(p["gold_zero"], 13))[:, te].T
    bins = np.digitize(gap, np.quantile(gap, [0.25, 0.5, 0.75]))
    print("    rule accuracy by distance of this week from its 13-wk mean (quartiles, near -> far): " +
          "  ".join(f"Q{b + 1} {(rv[bins == b] == l_te[bins == b]).mean():.1%}" for b in range(4)))
    for name, (pred, _) in dir_preds.items():
        if name.startswith(("gradient", "RELEASE")):
            print(f"    {name} agrees with the rule on {(pred == rv).mean():.1%} of test rows; "
                  f"accuracy on the near-normal quartile Q1: {(pred[bins == 0] == l_te[bins == 0]).mean():.1%}")


if __name__ == "__main__":
    main()
