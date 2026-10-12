"""
Sanity check: for the week BEFORE a real-world shock (using only data
available up to that point - the shock itself is NOT yet in the input
window), what did release_v1.0_goldstein predict for the next 13 weeks,
and what actually happened (the real historical target)?

Checks:
  1. Week of 2023-09-25 (the week immediately before the Oct 7, 2023 Hamas
     attack) - ISR<->PSE
  2. Week of 2026-02-16 (the week immediately before the IRN<->ISR
     escalation spike that begins the week of 2026-02-23) - ISR<->IRN
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder

from model_architectures import build_goldstein_model

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]
N_DYADS_PER_WEEK = 400
LOOKBACK_WEEKS = 13
TRAIN_FRAC = 0.70

CHECKS = [
    (pd.Timestamp("2023-09-25"), [("ISR", "PSE"), ("PSE", "ISR")], "week before the Oct 7, 2023 Hamas attack"),
    (pd.Timestamp("2026-02-16"), [("ISR", "IRN"), ("IRN", "ISR")], "week before the 2026 IRN<->ISR escalation spike"),
]

df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
n_total_weeks = df.week_start.nunique()
all_weeks = sorted(df.week_start.unique())

country_encoder = LabelEncoder()
country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())

counts = np.log1p(df[COUNT_COLS].to_numpy(dtype="float32"))
raw = df[RAW_COLS].to_numpy(dtype="float32")
numeric = np.concatenate([counts, raw], axis=1)
n_numeric = numeric.shape[1]
numeric = numeric.reshape(N_DYADS_PER_WEEK, n_total_weeks, n_numeric)
target_all = df[TARGET_COL].to_numpy(dtype="float32").reshape(N_DYADS_PER_WEEK, n_total_weeks)

dyad_actor1 = df.Actor1CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
dyad_actor2 = df.Actor2CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
a1_idx_per_dyad = country_encoder.transform(dyad_actor1).astype("int32")
a2_idx_per_dyad = country_encoder.transform(dyad_actor2).astype("int32")

windows = sliding_window_view(numeric, window_shape=LOOKBACK_WEEKS, axis=1)
windows = np.moveaxis(windows, -1, 2)  # (dyads, n_total_weeks-W+1, W, n_numeric)

n_valid_target = int((~np.isnan(target_all[0])).sum())
n_pred_weeks = n_valid_target - LOOKBACK_WEEKS + 1
windows_valid = windows[:, :n_pred_weeks]
y_valid = target_all[:, LOOKBACK_WEEKS - 1: LOOKBACK_WEEKS - 1 + n_pred_weeks]
numeric_seq = np.transpose(windows_valid, (1, 0, 2, 3))

n_train = int(n_pred_weeks * TRAIN_FRAC)
scaler = StandardScaler()
scaler.fit(numeric_seq[:n_train].reshape(-1, n_numeric))

model = build_goldstein_model(
    n_dyads=N_DYADS_PER_WEEK, lookback_weeks=LOOKBACK_WEEKS, n_numeric=n_numeric,
    n_countries=len(country_encoder.classes_),
)
model.load_weights("model_release_v1.0_goldstein.weights.h5")

for target_week, dyad_pairs, label in CHECKS:
    print(f"\n{'='*70}\n{label}  (week: {target_week.date()})\n{'='*70}")
    target_week_orig_idx = all_weeks.index(target_week)
    window_idx = target_week_orig_idx - (LOOKBACK_WEEKS - 1)

    if window_idx < 0:
        print("Not enough history before this week - skipping.")
        continue
    has_actual = window_idx < n_pred_weeks

    week_numeric = windows[:, window_idx]  # (400, 13, n_numeric) - raw (pre-scaling)
    week_numeric_scaled = scaler.transform(
        week_numeric.reshape(-1, n_numeric)
    ).reshape(week_numeric.shape).astype("float32")[np.newaxis, ...]

    a1_batch = a1_idx_per_dyad[np.newaxis, :]
    a2_batch = a2_idx_per_dyad[np.newaxis, :]
    pred = model.predict(
        {"numeric_seq": week_numeric_scaled, "actor1_idx": a1_batch, "actor2_idx": a2_batch}, verbose=0
    )[0]

    actual = target_all[:, target_week_orig_idx] if has_actual else np.full(N_DYADS_PER_WEEK, np.nan)

    for a1, a2 in dyad_pairs:
        mask = (dyad_actor1 == a1) & (dyad_actor2 == a2)
        idx = np.where(mask)[0][0]
        print(f"{a1}->{a2}: predicted_avg_goldstein_next13w = {pred[idx]:+.4f}   "
              f"actual = {actual[idx]:+.4f}" if has_actual else
              f"{a1}->{a2}: predicted_avg_goldstein_next13w = {pred[idx]:+.4f}   actual = N/A (future not observed yet)")
