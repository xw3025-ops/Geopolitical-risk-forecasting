"""
Train a DNN (Keras) to predict each dyad's average Goldstein score over the
NEXT 13 weeks, using raw data/panel_with_goldstein_target.csv
(built by build_goldstein_panel.py).

Inputs (X):  count_01 ... count_20                (20 numeric features)
Target (y):  target_avg_goldstein_next13w         (continuous, regression)

Run:
    python train_goldstein.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
FEATURE_COLS = [f"count_{i:02d}" for i in range(1, 21)]

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
    train_df, val_df, test_df = chronological_split(df)

    def prep(split_df):
        X = np.log1p(split_df[FEATURE_COLS].to_numpy(dtype="float32"))
        y = split_df[TARGET_COL].to_numpy(dtype="float32")
        return X, y

    X_train, y_train = prep(train_df)
    X_val, y_val = prep(val_df)
    X_test, y_test = prep(test_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),  # linear output - regression, not classification
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
        batch_size=64,
        callbacks=[early_stop],
    )

    y_pred = model.predict(X_test, verbose=0).ravel()
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
