"""
Run the trained spatio-temporal model (model_goldstein_spatiotemporal.keras)
on the week containing Oct 7, 2023 (week_start = 2023-10-02), and compare its
predictions to what's already known to have actually happened (since this is
historical data, the real target_avg_goldstein_next13w is available).

Rebuilds the EXACT same preprocessing pipeline as
train_goldstein_spatiotemporal.py (same sort order, same sliding window, same
scaler fit only on the train split) so the model sees data prepared the same
way it was trained on.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]

N_DYADS_PER_WEEK = 400
LOOKBACK_WEEKS = 13
COUNTRY_EMBED_DIM = 8
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_TEMPORAL_BLOCKS = 2
NUM_CROSSDYAD_BLOCKS = 2
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10


def transformer_block(x, name):
    attn_out = keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=0.1, name=f"{name}_mha"
    )(x, x)
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + attn_out)
    ffn_out = keras.layers.Dense(FF_DIM, activation="relu")(x)
    ffn_out = keras.layers.Dense(D_MODEL)(ffn_out)
    ffn_out = keras.layers.Dropout(0.1)(ffn_out)
    x = keras.layers.LayerNormalization(name=f"{name}_ln2")(x + ffn_out)
    return x


def build_model(n_countries, n_numeric):
    numeric_seq_input = keras.layers.Input(
        shape=(N_DYADS_PER_WEEK, LOOKBACK_WEEKS, n_numeric), name="numeric_seq"
    )
    actor1_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor1_idx")
    actor2_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor2_idx")

    token = keras.layers.Dense(D_MODEL, activation="relu", name="token_projection")(numeric_seq_input)

    positions = keras.layers.Lambda(
        lambda t: tf.range(LOOKBACK_WEEKS), output_shape=(LOOKBACK_WEEKS,)
    )(numeric_seq_input)
    temporal_pos_embedding = keras.layers.Embedding(
        input_dim=LOOKBACK_WEEKS, output_dim=D_MODEL, name="temporal_position_embedding"
    )(positions)
    token = keras.layers.Lambda(
        lambda t: t[0] + t[1], name="add_temporal_position"
    )([token, temporal_pos_embedding])

    merged = keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, LOOKBACK_WEEKS, D_MODEL]), name="merge_batch_dyad"
    )(token)

    x = merged
    for i in range(NUM_TEMPORAL_BLOCKS):
        x = transformer_block(x, name=f"temporal_block{i}")

    temporal_summary = keras.layers.Lambda(lambda t: t[:, -1, :], name="take_last_week")(x)
    temporal_summary = keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, N_DYADS_PER_WEEK, D_MODEL]), name="split_batch_dyad"
    )(temporal_summary)

    country_embedding = keras.layers.Embedding(
        input_dim=n_countries, output_dim=COUNTRY_EMBED_DIM, name="country_embedding"
    )
    actor1_vec = country_embedding(actor1_input)
    actor2_vec = country_embedding(actor2_input)

    dyad_token = keras.layers.Concatenate(axis=-1)([temporal_summary, actor1_vec, actor2_vec])
    dyad_token = keras.layers.Dense(D_MODEL, activation="relu", name="dyad_token_projection")(dyad_token)

    y = dyad_token
    for i in range(NUM_CROSSDYAD_BLOCKS):
        y = transformer_block(y, name=f"crossdyad_block{i}")

    output = keras.layers.Dense(1, name="per_dyad_output")(y)
    output = keras.layers.Reshape((N_DYADS_PER_WEEK,), name="squeeze")(output)

    return keras.Model(inputs=[numeric_seq_input, actor1_input, actor2_input], outputs=output)

TARGET_WEEK = pd.Timestamp("2023-10-02")  # the week containing Oct 7, 2023

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
windows = np.moveaxis(windows, -1, 2)  # (400, n_total_weeks-W+1, W, n_numeric)

n_valid_target = int((~np.isnan(target_all[0])).sum())
n_pred_weeks = n_valid_target - LOOKBACK_WEEKS + 1
windows_valid = windows[:, :n_pred_weeks]
y_valid = target_all[:, LOOKBACK_WEEKS - 1: LOOKBACK_WEEKS - 1 + n_pred_weeks]

numeric_seq = np.transpose(windows_valid, (1, 0, 2, 3))  # (n_pred_weeks, 400, W, n_numeric)
y_all = np.transpose(y_valid, (1, 0))  # (n_pred_weeks, 400)

n_train = int(n_pred_weeks * TRAIN_FRAC)
n_val = int(n_pred_weeks * VAL_FRAC)
print(f"{n_pred_weeks} usable prediction weeks: train=0..{n_train-1}, "
      f"val={n_train}..{n_train+n_val-1}, test={n_train+n_val}..{n_pred_weeks-1}")

# Fit scaler on TRAIN portion only, exactly as during training.
scaler = StandardScaler()
scaler.fit(numeric_seq[:n_train].reshape(-1, n_numeric))

# Prediction-week-index i corresponds to original week index (i + LOOKBACK_WEEKS - 1)
target_week_orig_idx = all_weeks.index(TARGET_WEEK)
pred_week_idx = target_week_orig_idx - (LOOKBACK_WEEKS - 1)
assert 0 <= pred_week_idx < n_pred_weeks, "target week out of usable range"

split_name = "train" if pred_week_idx < n_train else ("val" if pred_week_idx < n_train + n_val else "test")
print(f"Week {TARGET_WEEK.date()} = original week index {target_week_orig_idx}, "
      f"prediction-week index {pred_week_idx} -> falls in the {split_name.upper()} split")

week_numeric = numeric_seq[pred_week_idx]  # (400, 13, n_numeric)
week_numeric_scaled = scaler.transform(week_numeric.reshape(-1, n_numeric)).reshape(week_numeric.shape).astype("float32")
week_numeric_scaled = week_numeric_scaled[np.newaxis, ...]  # add batch dim -> (1, 400, 13, n_numeric)

a1_batch = a1_idx_per_dyad[np.newaxis, :]
a2_batch = a2_idx_per_dyad[np.newaxis, :]

n_countries = len(country_encoder.classes_)
model = build_model(n_countries, n_numeric)
model.load_weights("model_goldstein_spatiotemporal.keras")
pred = model.predict(
    {"numeric_seq": week_numeric_scaled, "actor1_idx": a1_batch, "actor2_idx": a2_batch}, verbose=0
)[0]  # (400,)

actual = y_all[pred_week_idx]  # (400,)

dyad_labels = [f"{a}<->{b}" for a, b in zip(dyad_actor1, dyad_actor2)]
result = pd.DataFrame({
    "dyad": dyad_labels,
    "actor1": dyad_actor1,
    "actor2": dyad_actor2,
    "predicted_next13w_avg_goldstein": pred,
    "actual_next13w_avg_goldstein": actual,
})
result["abs_error"] = (result.predicted_next13w_avg_goldstein - result.actual_next13w_avg_goldstein).abs()

print("\n=== ISR<->PSE dyads specifically ===")
mask = ((result.actor1 == "ISR") & (result.actor2 == "PSE")) | ((result.actor1 == "PSE") & (result.actor2 == "ISR"))
print(result[mask].to_string(index=False))

print("\n=== 10 most negative PREDICTED outcomes that week ===")
print(result.sort_values("predicted_next13w_avg_goldstein").head(10).to_string(index=False))

print("\n=== 10 most negative ACTUAL outcomes that week (what really happened) ===")
print(result.sort_values("actual_next13w_avg_goldstein").head(10).to_string(index=False))

print(f"\nOverall correlation (predicted vs actual, all 400 dyads this week): "
      f"{np.corrcoef(result.predicted_next13w_avg_goldstein, result.actual_next13w_avg_goldstein)[0,1]:.3f}")
print(f"Mean absolute error, this week only: {result.abs_error.mean():.4f}")
