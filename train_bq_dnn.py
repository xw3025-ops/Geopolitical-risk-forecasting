"""
Train a DNN (Keras) to predict each dyad's average Goldstein score over the
NEXT 13 weeks, using the full BigQuery-derived panel
(raw data/bq_panel_with_target.parquet, built by build_bq_panel.py).

Inputs (X): ~70 numeric signal columns - CAMEO event-type counts, CAMEO
            category rollups (cooperation/conflict/violent), goldstein
            stats, news tone/sentiment, media coverage volume (mentions/
            sources/articles), actor diversity, event-geography counts, and
            a binary flag for INTRA_REGION vs EXTERNAL_LINK dyads.
            Excluded: action_lat_mean/action_long_mean (0 there would be a
            real, misleading coordinate rather than "nothing happened"),
            top_event_root (mostly null, categorical - skipped for now).
Target (y): target_avg_goldstein_next13w  (continuous, regression)

Run:
    python train_bq_dnn.py
"""

import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

DATA_PATH = "raw data/bq_panel_with_target.parquet"
TARGET_COL = "target_avg_goldstein_next13w"
EXCLUDE_COLS = {
    "week_start", "Actor1CountryCode", "Actor2CountryCode", "dyad_scope",
    "top_event_root", "action_lat_mean", "action_long_mean", TARGET_COL,
}

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10  # remainder (0.20) goes to test


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["is_external_link"] = (df["dyad_scope"] == "EXTERNAL_LINK").astype("float32")
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"{len(feature_cols)} feature columns")
    return df, feature_cols


def chronological_split(df):
    weeks = sorted(df.week_start.unique())
    n = len(weeks)
    train_weeks = set(weeks[: int(n * TRAIN_FRAC)])
    val_weeks = set(weeks[int(n * TRAIN_FRAC): int(n * (TRAIN_FRAC + VAL_FRAC))])
    test_weeks = set(weeks[int(n * (TRAIN_FRAC + VAL_FRAC)):])

    train_mask = df.week_start.isin(train_weeks)
    val_mask = df.week_start.isin(val_weeks)
    test_mask = df.week_start.isin(test_weeks)
    print(f"Split by week: train={len(train_weeks)} weeks ({train_mask.sum()} rows), "
          f"val={len(val_weeks)} weeks ({val_mask.sum()} rows), "
          f"test={len(test_weeks)} weeks ({test_mask.sum()} rows)")
    return train_mask, val_mask, test_mask


def main():
    df, feature_cols = load_data()
    train_mask, val_mask, test_mask = chronological_split(df)

    X_all = np.log1p(np.clip(df[feature_cols].to_numpy(dtype="float32"), a_min=0, a_max=None))
    y_all = df[TARGET_COL].to_numpy(dtype="float32")
    del df
    gc.collect()

    X_train, y_train = X_all[train_mask.to_numpy()], y_all[train_mask.to_numpy()]
    X_val, y_val = X_all[val_mask.to_numpy()], y_all[val_mask.to_numpy()]
    X_test, y_test = X_all[test_mask.to_numpy()], y_all[test_mask.to_numpy()]
    del X_all, y_all
    gc.collect()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),  # linear output - regression
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,  # bigger batch than before - keeps epoch time reasonable at 4.7M rows
        callbacks=[early_stop],
    )

    y_pred = model.predict(X_test, batch_size=4096, verbose=0).ravel()
    test_mse = np.mean((y_pred - y_test) ** 2)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = np.mean((baseline_pred - y_test) ** 2)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"\nTest MSE:  {test_mse:.4f}   (always-predict-train-mean baseline: {baseline_mse:.4f})")
    print(f"Test MAE:  {test_mae:.4f}   (baseline: {baseline_mae:.4f})")
    print(f"Test R^2:  {test_r2:.4f}   (0 = no better than predicting the mean, 1 = perfect)")

    model.save("model_bq_goldstein.keras")
    print("Saved trained model to model_bq_goldstein.keras")


if __name__ == "__main__":
    main()
