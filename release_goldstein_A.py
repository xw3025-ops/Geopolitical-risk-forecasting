"""
RELEASE CANDIDATE A - goldstein regression. SwiGLU spatio-temporal
transformer (v11 architecture: temporal self-attention + country embeddings
+ plain cross-dyad self-attention, no structural bias), now using the FULL
BigQuery news/media feature set (goldstein stats, tone/sentiment, media
coverage volume, actor diversity, CAMEO category rollups) instead of the
old count_01..20 + avg_goldstein set - but still restricted to just the
original 20 MENA countries (400 dyads, all graded). This is "step 1": news
features added, before USA/China are introduced in release_goldstein_B.

Target: target_avg_goldstein_next13w (continuous, regression, MSE loss)

Run:
    python release_goldstein_A.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

DATA_PATH = "raw data/bq_panel_22_with_target_13w.parquet"
TARGET_COL = "target_avg_goldstein_next13w"
EXCLUDE_COLS = {"week_start", "Actor1CountryCode", "Actor2CountryCode",
                "action_lat_mean", "action_long_mean", TARGET_COL}

MENA = {
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
}

N_DYADS_PER_WEEK = 400  # MENA-only for this release candidate
LOOKBACK_WEEKS = 13
COUNTRY_EMBED_DIM = 8
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_TEMPORAL_BLOCKS = 2
NUM_CROSSDYAD_BLOCKS = 2
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10


def swiglu_ffn(x, name):
    gate = keras.layers.Dense(FF_DIM, name=f"{name}_gate")(x)
    gate = keras.layers.Activation("swish", name=f"{name}_swish")(gate)
    value = keras.layers.Dense(FF_DIM, name=f"{name}_value")(x)
    hidden = keras.layers.Multiply(name=f"{name}_gated")([gate, value])
    out = keras.layers.Dense(D_MODEL, name=f"{name}_out")(hidden)
    out = keras.layers.Dropout(0.1)(out)
    return out


def transformer_block(x, name):
    attn_out = keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=0.1, name=f"{name}_mha"
    )(x, x)
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + attn_out)
    ffn_out = swiglu_ffn(x, name=f"{name}_ffn")
    x = keras.layers.LayerNormalization(name=f"{name}_ln2")(x + ffn_out)
    return x


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df[df.Actor1CountryCode.isin(MENA) & df.Actor2CountryCode.isin(MENA)].reset_index(drop=True)
    df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)

    n_total_weeks = df.week_start.nunique()
    per_dyad_counts = df.groupby(["Actor1CountryCode", "Actor2CountryCode"]).size()
    assert (per_dyad_counts == n_total_weeks).all(), "expected a fully dense panel"
    assert len(per_dyad_counts) == N_DYADS_PER_WEEK
    print(f"{n_total_weeks} total weeks, {len(per_dyad_counts)} dyads (MENA-only)")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    log_cols = [c for c in feature_cols if str(df[c].dtype).startswith("Int")]
    raw_cols = [c for c in feature_cols if c not in log_cols]
    print(f"{len(feature_cols)} feature columns ({len(log_cols)} log1p counts, {len(raw_cols)} raw continuous)")

    country_encoder = LabelEncoder()
    country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())
    n_countries = len(country_encoder.classes_)

    numeric = np.concatenate([
        np.log1p(df[log_cols].to_numpy(dtype="float32")),
        df[raw_cols].to_numpy(dtype="float32"),
    ], axis=1)
    n_numeric = numeric.shape[1]
    numeric = numeric.reshape(N_DYADS_PER_WEEK, n_total_weeks, n_numeric)
    target_all = df[TARGET_COL].to_numpy(dtype="float32").reshape(N_DYADS_PER_WEEK, n_total_weeks)

    dyad_actor1 = df.Actor1CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
    dyad_actor2 = df.Actor2CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
    a1_idx_per_dyad = country_encoder.transform(dyad_actor1).astype("int32")
    a2_idx_per_dyad = country_encoder.transform(dyad_actor2).astype("int32")

    windows = sliding_window_view(numeric, window_shape=LOOKBACK_WEEKS, axis=1)
    windows = np.moveaxis(windows, -1, 2)

    n_valid_target = int((~np.isnan(target_all[0])).sum())
    n_pred_weeks = n_valid_target - LOOKBACK_WEEKS + 1
    windows_valid = windows[:, :n_pred_weeks]
    y_valid = target_all[:, LOOKBACK_WEEKS - 1: LOOKBACK_WEEKS - 1 + n_pred_weeks]
    print(f"{n_valid_target} weeks have a valid target; {n_pred_weeks} usable prediction weeks")

    numeric_seq = np.transpose(windows_valid, (1, 0, 2, 3))
    y_all = np.transpose(y_valid, (1, 0))

    n_train = int(n_pred_weeks * TRAIN_FRAC)
    n_val = int(n_pred_weeks * VAL_FRAC)
    n_test = n_pred_weeks - n_train - n_val
    print(f"Split: train={n_train}, val={n_val}, test={n_test} weeks")

    def split(arr):
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    num_train, num_val, num_test = split(numeric_seq)
    y_train, y_val, y_test = split(y_all)

    scaler = StandardScaler()
    scaler.fit(num_train.reshape(-1, n_numeric))
    num_train = scaler.transform(num_train.reshape(-1, n_numeric)).reshape(num_train.shape).astype("float32")
    num_val = scaler.transform(num_val.reshape(-1, n_numeric)).reshape(num_val.shape).astype("float32")
    num_test = scaler.transform(num_test.reshape(-1, n_numeric)).reshape(num_test.shape).astype("float32")

    a1_train = np.tile(a1_idx_per_dyad, (n_train, 1))
    a1_val = np.tile(a1_idx_per_dyad, (n_val, 1))
    a1_test = np.tile(a1_idx_per_dyad, (n_test, 1))
    a2_train = np.tile(a2_idx_per_dyad, (n_train, 1))
    a2_val = np.tile(a2_idx_per_dyad, (n_val, 1))
    a2_test = np.tile(a2_idx_per_dyad, (n_test, 1))

    numeric_seq_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK, LOOKBACK_WEEKS, n_numeric), name="numeric_seq")
    actor1_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor1_idx")
    actor2_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor2_idx")

    token = keras.layers.Dense(D_MODEL, activation="relu", name="token_projection")(numeric_seq_input)

    positions = keras.layers.Lambda(
        lambda t: tf.range(LOOKBACK_WEEKS), output_shape=(LOOKBACK_WEEKS,)
    )(numeric_seq_input)
    temporal_pos_embedding = keras.layers.Embedding(
        input_dim=LOOKBACK_WEEKS, output_dim=D_MODEL, name="temporal_position_embedding"
    )(positions)
    token = keras.layers.Lambda(lambda t: t[0] + t[1], name="add_temporal_position")(
        [token, temporal_pos_embedding]
    )

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

    model = keras.Model(inputs=[numeric_seq_input, actor1_input, actor2_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    model.fit(
        {"numeric_seq": num_train, "actor1_idx": a1_train, "actor2_idx": a2_train}, y_train,
        validation_data=({"numeric_seq": num_val, "actor1_idx": a1_val, "actor2_idx": a2_val}, y_val),
        epochs=100,
        batch_size=8,
        callbacks=[early_stop],
    )

    y_pred = model.predict(
        {"numeric_seq": num_test, "actor1_idx": a1_test, "actor2_idx": a2_test}, verbose=0
    )
    y_pred_flat = y_pred.ravel()
    y_test_flat = y_test.ravel()

    test_mse = np.mean((y_pred_flat - y_test_flat) ** 2)
    test_mae = mean_absolute_error(y_test_flat, y_pred_flat)
    test_r2 = r2_score(y_test_flat, y_pred_flat)

    baseline_pred = np.full_like(y_test_flat, y_train.ravel().mean())
    baseline_mse = np.mean((baseline_pred - y_test_flat) ** 2)
    baseline_mae = mean_absolute_error(y_test_flat, baseline_pred)

    print(f"\nTest MSE:  {test_mse:.4f}   (always-predict-train-mean baseline: {baseline_mse:.4f})")
    print(f"Test MAE:  {test_mae:.4f}   (baseline: {baseline_mae:.4f})")
    print(f"Test R^2:  {test_r2:.4f}")

    model.save_weights("model_release_goldstein_A.weights.h5")
    print("Saved trained weights to model_release_goldstein_A.weights.h5")


if __name__ == "__main__":
    main()
