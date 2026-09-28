"""
Train a DNN (Keras) to predict each dyad's average Goldstein score over the
NEXT 13 weeks, using raw data/panel_with_goldstein_target.csv
(built by build_goldstein_panel.py).

Inputs (X):
  - count_01 ... count_20, avg_goldstein (current week's own goldstein
    score)                                          (21 numeric features)
  - Actor1CountryCode, Actor2CountryCode - each country mapped to a learned
    embedding vector (see below), instead of being ignored entirely.
Target (y):  target_avg_goldstein_next13w         (continuous, regression)

Why embeddings for the countries: a DNN can't take a string like "IRN"
directly, and one-hot encoding 20 countries would work but treats every
country as equally different from every other. An embedding instead maps
each country to a small vector of numbers that the network tunes during
training - similar countries (in terms of how they affect the target) end up
with similar vectors. The same embedding table is shared for both
Actor1CountryCode and Actor2CountryCode, since "being Iran" should mean the
same thing regardless of which side of the dyad it's on.

Note on direction: IRN->ISR and ISR->IRN still produce different predictions
despite the shared embedding table, because actor1_vec and actor2_vec occupy
fixed, separately-weighted positions in the concatenated vector below - the
dense layer right after can (and does) treat those two positions
differently. An explicit "role" embedding (analogous to a Transformer's
positional embedding) was tried on top of this and measured slightly worse
(R^2 0.507 vs 0.520), so it was reverted - the position-based signal alone
was already sufficient.

Run:
    python train_goldstein.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]  # can be negative - skip log1p for this one
FEATURE_COLS = COUNT_COLS + RAW_COLS
EMBEDDING_DIM = 8  # 20 countries - a small vector each is plenty

# Chronological split: model must predict the future, so train on the
# earliest weeks and evaluate on the most recent ones (never split randomly
# on time-series data - that leaks future info into training).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10  # remainder (0.20) goes to test


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with no complete 13-week-forward "
          f"window (kept {len(df)} complete rows)")
    return df


def chronological_split(df):
    weeks = sorted(df.week_start.unique())
    n = len(weeks)
    train_weeks = set(weeks[: int(n * TRAIN_FRAC)])
    val_weeks = set(weeks[int(n * TRAIN_FRAC): int(n * (TRAIN_FRAC + VAL_FRAC))])
    test_weeks = set(weeks[int(n * (TRAIN_FRAC + VAL_FRAC)):])

    train = df[df.week_start.isin(train_weeks)]
    val = df[df.week_start.isin(val_weeks)]
    test = df[df.week_start.isin(test_weeks)]
    print(f"Split by week: train={train.week_start.nunique()} weeks "
          f"({len(train)} rows), val={val.week_start.nunique()} weeks "
          f"({len(val)} rows), test={test.week_start.nunique()} weeks "
          f"({len(test)} rows)")
    return train, val, test


def main():
    df = load_data()

    # Shared vocabulary for both actor columns - fit on the union of both so
    # the same integer always means the same country in either role.
    country_encoder = LabelEncoder()
    country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())
    n_countries = len(country_encoder.classes_)
    print(f"{n_countries} countries in the shared embedding vocabulary")

    train_df, val_df, test_df = chronological_split(df)

    def prep(split_df):
        counts = np.log1p(split_df[COUNT_COLS].to_numpy(dtype="float32"))
        raw = split_df[RAW_COLS].to_numpy(dtype="float32")  # goldstein - already ~[-10,10], no log1p
        numeric = np.concatenate([counts, raw], axis=1)
        actor1_idx = country_encoder.transform(split_df.Actor1CountryCode).astype("int32")
        actor2_idx = country_encoder.transform(split_df.Actor2CountryCode).astype("int32")
        y = split_df[TARGET_COL].to_numpy(dtype="float32")
        return numeric, actor1_idx, actor2_idx, y

    num_train, a1_train, a2_train, y_train = prep(train_df)
    num_val, a1_val, a2_val, y_val = prep(val_df)
    num_test, a1_test, a2_test, y_test = prep(test_df)

    scaler = StandardScaler()
    num_train = scaler.fit_transform(num_train)
    num_val = scaler.transform(num_val)
    num_test = scaler.transform(num_test)

    numeric_input = keras.layers.Input(shape=(num_train.shape[1],), name="numeric")
    actor1_input = keras.layers.Input(shape=(1,), name="actor1_idx")
    actor2_input = keras.layers.Input(shape=(1,), name="actor2_idx")

    country_embedding = keras.layers.Embedding(
        input_dim=n_countries, output_dim=EMBEDDING_DIM, name="country_embedding"
    )
    actor1_vec = keras.layers.Flatten()(country_embedding(actor1_input))
    actor2_vec = keras.layers.Flatten()(country_embedding(actor2_input))

    x = keras.layers.Concatenate()([numeric_input, actor1_vec, actor2_vec])
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    output = keras.layers.Dense(1)(x)  # linear output - regression, not classification

    model = keras.Model(inputs=[numeric_input, actor1_input, actor2_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        {"numeric": num_train, "actor1_idx": a1_train, "actor2_idx": a2_train}, y_train,
        validation_data=(
            {"numeric": num_val, "actor1_idx": a1_val, "actor2_idx": a2_val}, y_val
        ),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
    )

    y_pred = model.predict(
        {"numeric": num_test, "actor1_idx": a1_test, "actor2_idx": a2_test}, verbose=0
    ).ravel()
    test_mse = np.mean((y_pred - y_test) ** 2)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = np.mean((baseline_pred - y_test) ** 2)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"\nTest MSE:  {test_mse:.4f}   (always-predict-train-mean baseline: {baseline_mse:.4f})")
    print(f"Test MAE:  {test_mae:.4f}   (baseline: {baseline_mae:.4f})")
    print(f"Test R^2:  {test_r2:.4f}   (0 = no better than predicting the mean, 1 = perfect)")

    model.save("model_goldstein.keras")
    print("Saved trained model to model_goldstein.keras")


if __name__ == "__main__":
    main()
