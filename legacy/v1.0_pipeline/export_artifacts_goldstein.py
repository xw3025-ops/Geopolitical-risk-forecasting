"""
Regenerate and save the preprocessing artifacts (StandardScaler, country
LabelEncoder) used by release_v1.0_goldstein.py, WITHOUT retraining the
network - the model's weights are already saved, but the scaler/encoder
objects were only ever held in memory during training and then discarded.

Why this matters: a saved neural network alone isn't a complete, usable
model - whatever scaling/encoding was applied to its training inputs must be
applied IDENTICALLY to any new data you ever feed it, or its predictions are
meaningless. This script reproduces the exact same data-loading steps as
release_v1.0_goldstein.py, stopping right after scaler.fit()/encoder.fit()
(before any neural-network code runs), and saves those two fitted objects
plus the exact feature-column list/order they expect.

Run once:
    python export_artifacts_goldstein.py
Output:
    artifacts/goldstein_preprocessing.joblib
"""

import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]
N_DYADS_PER_WEEK = 400
LOOKBACK_WEEKS = 13
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10

MENA = [
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
]

df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
n_total_weeks = df.week_start.nunique()

country_encoder = LabelEncoder()
country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())

counts = np.log1p(df[COUNT_COLS].to_numpy(dtype="float32"))
raw = df[RAW_COLS].to_numpy(dtype="float32")
numeric = np.concatenate([counts, raw], axis=1)
n_numeric = numeric.shape[1]
numeric = numeric.reshape(N_DYADS_PER_WEEK, n_total_weeks, n_numeric)
target_all = df["target_avg_goldstein_next13w"].to_numpy(dtype="float32").reshape(N_DYADS_PER_WEEK, n_total_weeks)

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
    "count_cols": COUNT_COLS,
    "raw_cols": RAW_COLS,
    "n_dyads_per_week": N_DYADS_PER_WEEK,
    "lookback_weeks": LOOKBACK_WEEKS,
    "mena_countries": MENA,
}, "artifacts/goldstein_preprocessing.joblib")

print("Saved artifacts/goldstein_preprocessing.joblib")
print(f"  scaler fit on {num_train.shape[0] * num_train.shape[1]} rows, {n_numeric} features")
print(f"  country_encoder knows {len(country_encoder.classes_)} countries: {list(country_encoder.classes_)}")
