"""
Literal "regress y x" baseline: plain multivariate OLS (sklearn
LinearRegression, equivalent to Stata's `regress`), using the exact same
features (current week's count_01..20 + avg_goldstein, 21 regressors),
target, and chronological train/test split as release_v1.0_goldstein - the
only difference is the model itself: no embeddings, no attention, no
nonlinearity at all, just y = b0 + b1*x1 + ... + b21*x21.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]
N_DYADS_PER_WEEK = 400
LOOKBACK_WEEKS = 13
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10

df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
n_total_weeks = df.week_start.nunique()

numeric = np.concatenate([
    df[COUNT_COLS].to_numpy(dtype="float32"),  # no log1p - keep it a plain OLS on raw counts
    df[RAW_COLS].to_numpy(dtype="float32"),
], axis=1)
n_numeric = numeric.shape[1]
numeric = numeric.reshape(N_DYADS_PER_WEEK, n_total_weeks, n_numeric)
target_all = df[TARGET_COL].to_numpy(dtype="float32").reshape(N_DYADS_PER_WEEK, n_total_weeks)

windows = sliding_window_view(numeric, window_shape=LOOKBACK_WEEKS, axis=1)
windows = np.moveaxis(windows, -1, 2)

n_valid_target = int((~np.isnan(target_all[0])).sum())
n_pred_weeks = n_valid_target - LOOKBACK_WEEKS + 1
windows_valid = windows[:, :n_pred_weeks]
y_valid = target_all[:, LOOKBACK_WEEKS - 1: LOOKBACK_WEEKS - 1 + n_pred_weeks]

# Only the CURRENT (most recent) week's features - the same thing you'd type
# into Stata as `regress target count_01 count_02 ... avg_goldstein`
current_week = windows_valid[:, :, -1, :]  # (dyads, n_pred_weeks, n_numeric)

X_all = np.transpose(current_week, (1, 0, 2)).reshape(-1, n_numeric)
y_all = np.transpose(y_valid, (1, 0)).reshape(-1)

n_train = int(n_pred_weeks * TRAIN_FRAC)
n_val = int(n_pred_weeks * VAL_FRAC)

X_train = np.transpose(current_week, (1, 0, 2))[:n_train].reshape(-1, n_numeric)
y_train = y_valid[:, :n_train].T.reshape(-1)  # align shape: (dyads, weeks) -> weeks-major
# redo cleanly week-major like the DNN pipeline:
cw_week_major = np.transpose(current_week, (1, 0, 2))  # (n_pred_weeks, dyads, n_numeric)
y_week_major = np.transpose(y_valid, (1, 0))            # (n_pred_weeks, dyads)

X_train = cw_week_major[:n_train].reshape(-1, n_numeric)
y_train = y_week_major[:n_train].reshape(-1)
X_test = cw_week_major[n_train + n_val:].reshape(-1, n_numeric)
y_test = y_week_major[n_train + n_val:].reshape(-1)

ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred = ols.predict(X_test)

print(f"Plain OLS ('regress y x' in Stata), {n_numeric} regressors, {len(X_train)} training rows")
print(f"Test MSE: {np.mean((y_pred - y_test)**2):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Test R^2: {r2_score(y_test, y_pred):.4f}")
print()
print("For comparison, the DNN (release_v1.0_goldstein): R^2=0.646, MAE=0.600, MSE=0.775")
