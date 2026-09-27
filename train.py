"""
Train a DNN (Keras) to predict each dyad's dominant event category 13 weeks
in advance, using raw data/panel_with_target.csv (built by build_panel.py).

Inputs (X):  count_01 ... count_20, total_count   (21 numeric features)
Target (y):  target_top_label_13w_ahead           (21 classes, incl. "00 Null")

Run:
    python train.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow import keras

DATA_PATH = "raw data/panel_with_target.csv"
TARGET_COL = "target_top_label_13w_ahead"
FEATURE_COLS = [f"count_{i:02d}" for i in range(1, 21)] + ["total_count"]

# Chronological split: model must predict the future, so train on the
# earliest weeks and evaluate on the most recent ones (never split randomly
# on time-series data - that leaks future info into training).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10  # remainder (0.20) goes to test


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with no 13-week-ahead label "
          f"(kept {len(df)} complete rows)")
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

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET_COL])
    n_classes = len(label_encoder.classes_)
    print(f"{n_classes} target classes: {list(label_encoder.classes_)}")

    def prep(split_df):
        X = np.log1p(split_df[FEATURE_COLS].to_numpy(dtype="float32"))
        y = label_encoder.transform(split_df[TARGET_COL])
        return X, y

    X_train, y_train = prep(train_df)
    X_val, y_val = prep(val_df)
    X_test, y_test = prep(test_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Classes are heavily imbalanced (e.g. "04 CONSULT" is ~50% of rows,
    # the rarest class has ~10 rows total), so weight the loss inversely
    # to class frequency or the model will just learn to predict the
    # majority class.
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.3f}  (test loss: {test_loss:.3f})")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    ))

    model.save("model.keras")
    print("Saved trained model to model.keras")


if __name__ == "__main__":
    main()
