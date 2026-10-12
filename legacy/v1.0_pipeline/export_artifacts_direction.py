"""
Regenerate and save the preprocessing artifacts (StandardScaler, country
LabelEncoder, relation matrix, feature-column lists) used by
release_v1.0_direction.py, WITHOUT retraining the network. See
export_artifacts_goldstein.py for why this is necessary.

Run once:
    python export_artifacts_direction.py
Output:
    artifacts/direction_preprocessing.joblib
"""

import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

DATA_PATH = "raw data/bq_panel_22_with_target_4w.parquet"
TARGET_COL = "target_avg_goldstein_next4w"
EXCLUDE_COLS = {"week_start", "Actor1CountryCode", "Actor2CountryCode",
                "action_lat_mean", "action_long_mean", TARGET_COL}

MENA = [
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
]
EXTRA = ["USA", "CHN"]

N_DYADS_PER_WEEK = 484
LOOKBACK_WEEKS = 13
TRAIN_FRAC = 0.70


def classify_relation(a1_i, a2_i, a1_j, a2_j):
    if a1_i == a1_j and a2_i == a2_j:
        return 0
    if a1_i == a2_j and a2_i == a1_j:
        return 1
    if a1_i == a1_j:
        return 2
    if a2_i == a2_j:
        return 3
    if a2_i == a1_j:
        return 4
    if a1_i == a2_j:
        return 5
    return 6


def build_relation_matrix(actor1_per_dyad, actor2_per_dyad):
    n = len(actor1_per_dyad)
    mat = np.zeros((n, n), dtype="int32")
    for i in range(n):
        for j in range(n):
            mat[i, j] = classify_relation(
                actor1_per_dyad[i], actor2_per_dyad[i], actor1_per_dyad[j], actor2_per_dyad[j]
            )
    return mat


df = pd.read_parquet(DATA_PATH)
df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
n_total_weeks = df.week_start.nunique()

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
log_cols = [c for c in feature_cols if str(df[c].dtype).startswith("Int")]
raw_cols = [c for c in feature_cols if c not in log_cols]

country_encoder = LabelEncoder()
country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())

numeric = np.concatenate([
    np.log1p(df[log_cols].to_numpy(dtype="float32")),
    df[raw_cols].to_numpy(dtype="float32"),
], axis=1)
n_numeric = numeric.shape[1]
numeric = numeric.reshape(N_DYADS_PER_WEEK, n_total_weeks, n_numeric)
target_all = df[TARGET_COL].to_numpy(dtype="float32").reshape(N_DYADS_PER_WEEK, n_total_weeks)

dyad_actor1 = df.Actor1CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
dyad_actor2 = df.Actor2CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
relation_matrix = build_relation_matrix(dyad_actor1, dyad_actor2)

windows = sliding_window_view(numeric, window_shape=LOOKBACK_WEEKS, axis=1)
windows = np.moveaxis(windows, -1, 2)

n_valid_target = int((~np.isnan(target_all[0])).sum())
n_pred_weeks = n_valid_target - LOOKBACK_WEEKS + 1
windows_valid = windows[:, :n_pred_weeks]
numeric_seq = np.transpose(windows_valid, (1, 0, 2, 3))

n_train = int(n_pred_weeks * TRAIN_FRAC)
num_train = numeric_seq[:n_train]

scaler = StandardScaler()
scaler.fit(num_train.reshape(-1, n_numeric))

os.makedirs("artifacts", exist_ok=True)
joblib.dump({
    "scaler": scaler,
    "country_encoder": country_encoder,
    "log_cols": log_cols,
    "raw_cols": raw_cols,
    "relation_matrix": relation_matrix,
    "n_dyads_per_week": N_DYADS_PER_WEEK,
    "lookback_weeks": LOOKBACK_WEEKS,
    "mena_countries": MENA,
    "extra_countries": EXTRA,
}, "artifacts/direction_preprocessing.joblib")

print("Saved artifacts/direction_preprocessing.joblib")
print(f"  scaler fit on {num_train.shape[0] * num_train.shape[1]} rows, {n_numeric} features "
      f"({len(log_cols)} log1p, {len(raw_cols)} raw)")
print(f"  country_encoder knows {len(country_encoder.classes_)} countries: {list(country_encoder.classes_)}")
