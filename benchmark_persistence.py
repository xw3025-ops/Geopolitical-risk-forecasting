"""
Benchmark the spatio-temporal transformer against two naive baselines on the
exact same test split:

  1. mean baseline    - always predict the train-set average target
  2. persistence      - predict "next 13 weeks' average = this week's own
                         goldstein score" (the classic naive time-series
                         forecast - no model at all, just carry the last
                         known value forward)
  3. the trained model (model_goldstein_spatiotemporal.keras)

Same preprocessing pipeline as train_goldstein_spatiotemporal.py.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
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

numeric_seq = np.transpose(windows_valid, (1, 0, 2, 3))  # (n_pred_weeks, 400, W, n_numeric) - raw, pre-scaling
y_all = np.transpose(y_valid, (1, 0))  # (n_pred_weeks, 400)

n_train = int(n_pred_weeks * TRAIN_FRAC)
n_val = int(n_pred_weeks * VAL_FRAC)
n_test = n_pred_weeks - n_train - n_val
print(f"Test set: {n_test} weeks x {N_DYADS_PER_WEEK} dyads = {n_test * N_DYADS_PER_WEEK} predictions")

num_train_raw = numeric_seq[:n_train]
num_test_raw = numeric_seq[n_train + n_val:]
y_train = y_all[:n_train]
y_test = y_all[n_train + n_val:]

# ---------------- Baseline 1: always predict the train-set mean ----------------
mean_pred = np.full_like(y_test, y_train.mean())
mean_mse = np.mean((mean_pred - y_test) ** 2)
mean_mae = mean_absolute_error(y_test.ravel(), mean_pred.ravel())
mean_r2 = r2_score(y_test.ravel(), mean_pred.ravel())

# ---------------- Baseline 2: persistence (carry forward this week's own goldstein) ----------------
# avg_goldstein is the last of the RAW_COLS, which sit last in the numeric
# feature vector; the most recent week is the last timestep in the window.
persistence_pred = num_test_raw[:, :, -1, -1]  # (n_test, 400) - raw avg_goldstein, current week
persistence_mse = np.mean((persistence_pred - y_test) ** 2)
persistence_mae = mean_absolute_error(y_test.ravel(), persistence_pred.ravel())
persistence_r2 = r2_score(y_test.ravel(), persistence_pred.ravel())

# ---------------- The trained model ----------------
scaler = StandardScaler()
scaler.fit(num_train_raw.reshape(-1, n_numeric))
num_test_scaled = scaler.transform(num_test_raw.reshape(-1, n_numeric)).reshape(num_test_raw.shape).astype("float32")

a1_test = np.tile(a1_idx_per_dyad, (n_test, 1))
a2_test = np.tile(a2_idx_per_dyad, (n_test, 1))

n_countries = len(country_encoder.classes_)
model = build_model(n_countries, n_numeric)
model.load_weights("model_goldstein_spatiotemporal.keras")

model_pred = model.predict(
    {"numeric_seq": num_test_scaled, "actor1_idx": a1_test, "actor2_idx": a2_test}, verbose=0
)
model_mse = np.mean((model_pred - y_test) ** 2)
model_mae = mean_absolute_error(y_test.ravel(), model_pred.ravel())
model_r2 = r2_score(y_test.ravel(), model_pred.ravel())

print(f"\n{'method':<38}{'MSE':>10}{'MAE':>10}{'R^2':>10}")
print(f"{'Always predict train mean':<38}{mean_mse:>10.4f}{mean_mae:>10.4f}{mean_r2:>10.4f}")
print(f"{'Persistence (carry forward this wk)':<38}{persistence_mse:>10.4f}{persistence_mae:>10.4f}{persistence_r2:>10.4f}")
print(f"{'Spatio-temporal transformer':<38}{model_mse:>10.4f}{model_mae:>10.4f}{model_r2:>10.4f}")
