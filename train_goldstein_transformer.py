"""
Train a Transformer (Keras) to predict each dyad's average Goldstein score
over the NEXT 13 weeks, letting each dyad attend to every OTHER dyad in the
same week via self-attention - unlike train_goldstein.py, where every
dyad-week row is predicted completely independently.

Why this can help: a dyad's future isn't just a function of its own history
- e.g. a spike in IRN<->SAU might carry information relevant to predicting
IRN<->IRQ too. Self-attention lets the model learn which other dyads (if
any) are informative for a given dyad, rather than us having to hand-engineer
"regional context" features.

How the data is reshaped: the panel is fully dense (exactly 400 dyads every
week - 20 countries x 20, including self-pairs), so each week becomes one
sequence of 400 tokens (one per dyad). A batch is a handful of weeks.

Per-dyad token = [log1p(counts), current avg_goldstein] (21 numeric values)
                 + Actor1 country embedding + Actor2 country embedding,
                 projected to a d_model-dim vector.
Those 400 token vectors for a week go through 2 Transformer encoder blocks
(multi-head self-attention + feed-forward, each with a residual connection
and layer norm - the standard Transformer encoder design), then each token's
final vector is mapped to a single number: that dyad's predicted target.

Target (y): target_avg_goldstein_next13w (continuous, regression, MSE loss)

Run:
    python train_goldstein_transformer.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]
FEATURE_COLS = COUNT_COLS + RAW_COLS

N_DYADS_PER_WEEK = 400  # 20 countries x 20, fully dense panel
COUNTRY_EMBED_DIM = 8
D_MODEL = 64
NUM_HEADS = 4
NUM_TRANSFORMER_BLOCKS = 2
FF_DIM = 128

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with no complete 13-week-forward "
          f"window (kept {len(df)} complete rows)")
    df = df.sort_values(["week_start", "Actor1CountryCode", "Actor2CountryCode"]).reset_index(drop=True)
    counts_per_week = df.groupby("week_start").size()
    assert (counts_per_week == N_DYADS_PER_WEEK).all(), \
        f"Expected {N_DYADS_PER_WEEK} dyads/week everywhere, found: {counts_per_week[counts_per_week != N_DYADS_PER_WEEK]}"
    return df


def reshape_to_weeks(df, feature_cols, country_encoder):
    weeks = df.week_start.unique()  # already sorted
    n_weeks = len(weeks)

    counts = np.log1p(df[COUNT_COLS].to_numpy(dtype="float32"))
    raw = df[RAW_COLS].to_numpy(dtype="float32")
    numeric = np.concatenate([counts, raw], axis=1).reshape(n_weeks, N_DYADS_PER_WEEK, -1)

    a1 = country_encoder.transform(df.Actor1CountryCode).astype("int32").reshape(n_weeks, N_DYADS_PER_WEEK)
    a2 = country_encoder.transform(df.Actor2CountryCode).astype("int32").reshape(n_weeks, N_DYADS_PER_WEEK)
    y = df[TARGET_COL].to_numpy(dtype="float32").reshape(n_weeks, N_DYADS_PER_WEEK)
    return weeks, numeric, a1, a2, y


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


def main():
    df = load_data()

    country_encoder = LabelEncoder()
    country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())
    n_countries = len(country_encoder.classes_)

    weeks, numeric, a1, a2, y = reshape_to_weeks(df, FEATURE_COLS, country_encoder)
    n_weeks = len(weeks)
    n_train = int(n_weeks * TRAIN_FRAC)
    n_val = int(n_weeks * VAL_FRAC)
    print(f"{n_weeks} weeks total -> train={n_train}, val={n_val}, test={n_weeks - n_train - n_val}")

    num_train, num_val, num_test = numeric[:n_train], numeric[n_train:n_train + n_val], numeric[n_train + n_val:]
    a1_train, a1_val, a1_test = a1[:n_train], a1[n_train:n_train + n_val], a1[n_train + n_val:]
    a2_train, a2_val, a2_test = a2[:n_train], a2[n_train:n_train + n_val], a2[n_train + n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]

    # Fit the scaler on train numeric features flattened across (weeks, dyads)
    n_numeric = num_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(num_train.reshape(-1, n_numeric))
    num_train = scaler.transform(num_train.reshape(-1, n_numeric)).reshape(num_train.shape).astype("float32")
    num_val = scaler.transform(num_val.reshape(-1, n_numeric)).reshape(num_val.shape).astype("float32")
    num_test = scaler.transform(num_test.reshape(-1, n_numeric)).reshape(num_test.shape).astype("float32")

    numeric_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK, n_numeric), name="numeric")
    actor1_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor1_idx")
    actor2_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor2_idx")

    country_embedding = keras.layers.Embedding(
        input_dim=n_countries, output_dim=COUNTRY_EMBED_DIM, name="country_embedding"
    )
    actor1_vec = country_embedding(actor1_input)  # (batch, 400, 8)
    actor2_vec = country_embedding(actor2_input)  # (batch, 400, 8)

    token = keras.layers.Concatenate(axis=-1)([numeric_input, actor1_vec, actor2_vec])  # (batch, 400, n_numeric+16)
    x = keras.layers.Dense(D_MODEL, activation="relu", name="token_projection")(token)

    for i in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_block(x, name=f"block{i}")

    output = keras.layers.Dense(1, name="per_dyad_output")(x)  # (batch, 400, 1)
    output = keras.layers.Reshape((N_DYADS_PER_WEEK,), name="squeeze")(output)  # (batch, 400)

    model = keras.Model(inputs=[numeric_input, actor1_input, actor2_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    model.fit(
        {"numeric": num_train, "actor1_idx": a1_train, "actor2_idx": a2_train}, y_train,
        validation_data=(
            {"numeric": num_val, "actor1_idx": a1_val, "actor2_idx": a2_val}, y_val
        ),
        epochs=100,
        batch_size=8,  # 8 WEEKS per batch = 8*400=3200 dyad-predictions per batch
        callbacks=[early_stop],
    )

    y_pred = model.predict(
        {"numeric": num_test, "actor1_idx": a1_test, "actor2_idx": a2_test}, verbose=0
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
    print(f"Test R^2:  {test_r2:.4f}   (0 = no better than predicting the mean, 1 = perfect)")

    model.save("model_goldstein_transformer.keras")
    print("Saved trained model to model_goldstein_transformer.keras")


if __name__ == "__main__":
    main()
