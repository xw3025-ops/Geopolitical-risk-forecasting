"""
Re-compute the test-set R^2 of the SAVED release v1.0 goldstein model
(model_release_v1.0_goldstein.weights.h5) instead of trusting the number in
release_v1.0_goldstein.py's docstring (R^2 = 0.646, MAE = 0.600).

How: run release_v1.0_goldstein.py's own main() - same data, preprocessing,
split and architecture, line for line - but with training replaced by
loading the saved weights, and saving disabled. Its own printout then shows
the release model's metrics. The predictions are also captured to report
the R^2 variants the release script does not:

  - R^2 on the 324 dyads with real goldstein data (the 76 LBY/SDN dyads have
    a constant 0 target, padded by build_goldstein_panel.py)
  - within-dyad R^2 (each dyad's test-period mean removed from both truth
    and prediction): does the model track CHANGES, or only which dyads are
    usually hostile?
  - the same numbers for the trailing 26-week mean baseline

Run:
    python evaluate_release_v1_goldstein.py
"""

import importlib.util

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

SCRIPT = "release_v1.0_goldstein.py"
WEIGHTS = "model_release_v1.0_goldstein.weights.h5"
NO_GOLDSTEIN_COVERAGE = ("LBY", "SDN")

spec = importlib.util.spec_from_file_location("release_v1_goldstein", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

captured = {}


def load_instead_of_fit(self, x, y, **kwargs):
    self.load_weights(WEIGHTS)
    captured["inputs"] = x
    print(f"[evaluate] loaded {WEIGHTS} instead of training")


def capture_r2(y_true, y_pred):
    captured["y_true"], captured["y_pred"] = np.asarray(y_true), np.asarray(y_pred)
    return r2_score(y_true, y_pred)


keras.Model.fit = load_instead_of_fit
keras.Model.save_weights = lambda self, *a, **k: None
release.r2_score = capture_r2
release.main()

# ---------------- extra R^2 variants ----------------
# Rebuild the dyad order and the matching test-week raw goldstein exactly as
# the release script does (sorted by Actor1, Actor2, week_start).
df = release.pd.read_csv(release.DATA_PATH, parse_dates=["week_start"])
df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
n_dyads, W = release.N_DYADS_PER_WEEK, release.LOOKBACK_WEEKS
n_weeks = df.week_start.nunique()
a1 = df.Actor1CountryCode.to_numpy().reshape(n_dyads, n_weeks)[:, 0]
a2 = df.Actor2CountryCode.to_numpy().reshape(n_dyads, n_weeks)[:, 0]
gold = df.avg_goldstein.to_numpy().reshape(n_dyads, n_weeks)
target = df[release.TARGET_COL].to_numpy().reshape(n_dyads, n_weeks)
n_pred = int((~np.isnan(target[0])).sum()) - W + 1
n_train, n_val = int(n_pred * release.TRAIN_FRAC), int(n_pred * release.VAL_FRAC)
test_week_idx = np.arange(n_train + n_val, n_pred) + W - 1  # original week index of each test row

y_true = captured["y_true"].reshape(len(test_week_idx), n_dyads)
y_pred = captured["y_pred"].reshape(len(test_week_idx), n_dyads)
assert np.allclose(y_true, target[:, test_week_idx].T, equal_nan=False), "dyad/week alignment mismatch"
trail26 = np.stack([gold[:, k - 25:k + 1].mean(axis=1) for k in test_week_idx])  # (test weeks, dyads)

real = ~(np.isin(a1, NO_GOLDSTEIN_COVERAGE) | np.isin(a2, NO_GOLDSTEIN_COVERAGE))


def within(t, p):
    return r2_score((t - t.mean(axis=0)).ravel(), (p - p.mean(axis=0)).ravel())


test_weeks = df.week_start.drop_duplicates().sort_values().to_numpy()[test_week_idx]
print(f"\n=== Release v1.0 goldstein, test weeks {str(test_weeks[0])[:10]}..{str(test_weeks[-1])[:10]} "
      f"({len(test_week_idx)} weeks) ===")
print(f"{'':<28}{'R2 all 400':>11}{'MAE all':>9}{'R2 real 324':>13}{'MAE real':>10}{'within-dyad R2':>16}")
for name, p in (("release v1.0 transformer", y_pred), ("trailing 26-week mean", trail26)):
    print(f"{name:<28}{r2_score(y_true.ravel(), p.ravel()):>11.3f}{mean_absolute_error(y_true.ravel(), p.ravel()):>9.3f}"
          f"{r2_score(y_true[:, real].ravel(), p[:, real].ravel()):>13.3f}"
          f"{mean_absolute_error(y_true[:, real].ravel(), p[:, real].ravel()):>10.3f}"
          f"{within(y_true[:, real], p[:, real]):>16.3f}")
print("(docstring claim: R^2 0.646, MAE 0.600 on all 400 dyads)")
