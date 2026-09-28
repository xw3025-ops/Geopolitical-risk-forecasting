"""
Plot predicted vs actual target_avg_goldstein_next13w for the v3 model
(count features + current-week goldstein + country embeddings,
model_goldstein.keras) on its held-out test set.

Rebuilds the exact same test split/preprocessing as train_goldstein.py (the
scaler and label encoder aren't saved, so they're refit deterministically the
same way - fitting only on train data, same as during training).

Run:
    python plot_v3_predictions.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from tensorflow import keras

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10

df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

country_encoder = LabelEncoder()
country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())

weeks = sorted(df.week_start.unique())
n = len(weeks)
train_weeks = set(weeks[: int(n * TRAIN_FRAC)])
val_weeks = set(weeks[int(n * TRAIN_FRAC): int(n * (TRAIN_FRAC + VAL_FRAC))])
test_weeks = set(weeks[int(n * (TRAIN_FRAC + VAL_FRAC)):])

train_df = df[df.week_start.isin(train_weeks)]
test_df = df[df.week_start.isin(test_weeks)]


def prep(split_df, scaler=None, fit=False):
    counts = np.log1p(split_df[COUNT_COLS].to_numpy(dtype="float32"))
    raw = split_df[RAW_COLS].to_numpy(dtype="float32")
    numeric = np.concatenate([counts, raw], axis=1)
    if fit:
        numeric = scaler.fit_transform(numeric)
    else:
        numeric = scaler.transform(numeric)
    a1 = country_encoder.transform(split_df.Actor1CountryCode).astype("int32")
    a2 = country_encoder.transform(split_df.Actor2CountryCode).astype("int32")
    y = split_df[TARGET_COL].to_numpy(dtype="float32")
    return numeric, a1, a2, y


scaler = StandardScaler()
_ = prep(train_df, scaler=scaler, fit=True)
num_test, a1_test, a2_test, y_test = prep(test_df, scaler=scaler, fit=False)

model = keras.models.load_model("model_goldstein.keras")
y_pred = model.predict(
    {"numeric": num_test, "actor1_idx": a1_test, "actor2_idx": a2_test}, verbose=0
).ravel()

r2 = r2_score(y_test, y_pred)

# --- styling per dataviz skill: sequential blue ramp (100->700), light surface ---
blue_ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seq_blue", blue_ramp)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
ORANGE = "#eb6834"  # categorical slot 2, for the y=x reference line

fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

lo, hi = -8, 8
hb = ax.hexbin(
    y_test, y_pred, gridsize=60, cmap=cmap, mincnt=1,
    extent=(lo, hi, lo, hi), linewidths=0.0,
)

ax.plot([lo, hi], [lo, hi], color=ORANGE, linewidth=1.8, linestyle="--",
        label="Perfect prediction (y = x)", zorder=5)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")

ax.set_xlabel("Actual avg goldstein, next 13 weeks", color=INK_SECONDARY, fontsize=11)
ax.set_ylabel("Predicted avg goldstein, next 13 weeks", color=INK_SECONDARY, fontsize=11)
ax.set_title(f"v3 model: predicted vs actual (test set, R² = {r2:.3f})",
             color=INK_PRIMARY, fontsize=13, pad=12)

ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
for spine in ax.spines.values():
    spine.set_color(BASELINE)
ax.tick_params(colors=INK_MUTED, labelsize=9)

legend = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

cbar = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Number of dyad-weeks", color=INK_SECONDARY, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=INK_MUTED, labelsize=8)
plt.setp(cbar.ax.get_yticklabels(), color=INK_MUTED)

plt.tight_layout()
plt.savefig("v3_predicted_vs_actual.png", facecolor=SURFACE)
print("Saved v3_predicted_vs_actual.png")
print(f"Test R^2 (recomputed): {r2:.4f}")
