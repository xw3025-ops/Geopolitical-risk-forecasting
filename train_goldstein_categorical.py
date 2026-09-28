"""
Train a DNN (Keras) to predict each dyad's average Goldstein score over the
NEXT 13 weeks as a CATEGORICAL variable (rounded to the nearest integer),
using cross-entropy loss - a classification version of train_goldstein.py,
which instead treats the target as continuous (regression, MSE loss).

Inputs (X): same as train_goldstein.py - count_01..count_20, current-week
            avg_goldstein, and Actor1/Actor2 country embeddings.
Target (y): target_avg_goldstein_next13w, ROUNDED to the nearest integer.
            In principle this ranges -10..10, but since it's a 13-week
            AVERAGE, extremes get smoothed out - the data only actually
            spans about -7..7, and is heavily skewed toward 0/1/2. Only the
            integer values that actually appear in the data become classes
            (fitting 21 classes when several never occur would be
            meaningless - there's no data to learn them from).

Run:
    python train_goldstein_categorical.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, mean_absolute_error
from tensorflow import keras

DATA_PATH = "raw data/panel_with_goldstein_target.csv"
TARGET_COL = "target_avg_goldstein_next13w"
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
RAW_COLS = ["avg_goldstein"]  # can be negative - skip log1p for this one
FEATURE_COLS = COUNT_COLS + RAW_COLS
EMBEDDING_DIM = 8  # 20 countries - a small vector each is plenty

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10  # remainder (0.20) goes to test


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with no complete 13-week-forward "
          f"window (kept {len(df)} complete rows)")
    df["target_class_value"] = df[TARGET_COL].round().clip(-10, 10).astype(int)
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

    country_encoder = LabelEncoder()
    country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())
    n_countries = len(country_encoder.classes_)

    class_encoder = LabelEncoder()
    class_encoder.fit(df["target_class_value"])
    n_classes = len(class_encoder.classes_)
    print(f"{n_countries} countries, {n_classes} target classes: "
          f"{list(class_encoder.classes_)}")

    train_df, val_df, test_df = chronological_split(df)

    def prep(split_df):
        counts = np.log1p(split_df[COUNT_COLS].to_numpy(dtype="float32"))
        raw = split_df[RAW_COLS].to_numpy(dtype="float32")
        numeric = np.concatenate([counts, raw], axis=1)
        actor1_idx = country_encoder.transform(split_df.Actor1CountryCode).astype("int32")
        actor2_idx = country_encoder.transform(split_df.Actor2CountryCode).astype("int32")
        y = class_encoder.transform(split_df["target_class_value"]).astype("int32")
        return numeric, actor1_idx, actor2_idx, y

    num_train, a1_train, a2_train, y_train = prep(train_df)
    num_val, a1_val, a2_val, y_val = prep(val_df)
    num_test, a1_test, a2_test, y_test = prep(test_df)

    scaler = StandardScaler()
    num_train = scaler.fit_transform(num_train)
    num_val = scaler.transform(num_val)
    num_test = scaler.transform(num_test)

    # Classes are heavily imbalanced (class "0" is ~37% of rows, the rarest
    # class has 2 examples total), so weight the loss inversely to class
    # frequency, same as we did for the original CAMEO-category classifier.
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train).tolist(), class_weights))

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
    output = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=[numeric_input, actor1_input, actor2_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
        class_weight=class_weight_dict,
        callbacks=[early_stop],
    )

    y_pred_probs = model.predict(
        {"numeric": num_test, "actor1_idx": a1_test, "actor2_idx": a2_test}, verbose=0
    )
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_acc = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {test_acc:.3f}")
    print(classification_report(
        y_test, y_pred, labels=range(n_classes),
        target_names=[str(c) for c in class_encoder.classes_], zero_division=0,
    ))

    # Bonus: decode predicted class back to its integer value and compare to
    # the true CONTINUOUS target, as an MAE, so this is roughly comparable
    # to the regression model's reported MAE.
    y_pred_value = class_encoder.inverse_transform(y_pred)
    y_true_continuous = test_df[TARGET_COL].to_numpy()
    decoded_mae = mean_absolute_error(y_true_continuous, y_pred_value)
    print(f"MAE of predicted class (as a number) vs true continuous target: {decoded_mae:.4f}")
    print("(compare to train_goldstein.py's regression MAE - not apples-to-apples, "
          "since this model can only output whole integers)")

    model.save("model_goldstein_categorical.keras")
    print("Saved trained model to model_goldstein_categorical.keras")


if __name__ == "__main__":
    main()
