"""
One configurable training run of the goldstein transformer, for picking the
release v2.0 configuration. Same data, splits and embargo as
release_v2.0_goldstein.py (honest_common.py); the knobs are:

  --lookback N     weeks of history each dyad's temporal attention sees (13 / 26)
  --residual       predict the CORRECTION to a trailing mean of goldstein (a
                   forecast known at prediction time) instead of the raw
                   target: output = baseline + net
  --baseline-weeks window of that trailing mean (26 / 52)
  --has-events     add a 0/1 "had any events this week" input feature
  --long-features  add long-memory inputs to every weekly token: trailing
                   26- and 52-week mean goldstein and log 52-week mean event
                   count - history the 13/26-week attention window can't see
  --dropout P      dropout in every attention / feed-forward sublayer
  --lr X           Adam learning rate
  --plateau        halve the learning rate when val loss stalls for 3 epochs
  --seed S

Configs are compared on VALIDATION loss only; test numbers are recorded but
not used for choosing. Each run appends one JSON line to
logs/goldstein_experiments.jsonl and saves its val/test predictions to
artifacts/experiments/<tag>.npz (so seed ensembles can be scored later).

Run:
    python experiment_goldstein.py --lookback 26 --residual --seed 1
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

from honest_common import (GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA,
                           goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean,
                           windows_at)
from model_architectures import build_goldstein_model


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=13)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--baseline-weeks", type=int, default=26)
    ap.add_argument("--has-events", action="store_true")
    ap.add_argument("--long-features", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--plateau", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="TF intra-op threads (0 = all)")
    ap.add_argument("--tag", default=None)
    return ap.parse_args(argv)


def config(**overrides):
    """An args namespace with the CLI defaults, for calling from other scripts."""
    ap_args = parse_args([])
    for k, v in overrides.items():
        setattr(ap_args, k, v)
    return ap_args


def make_tag(args):
    return args.tag or (f"lb{args.lookback}{f'_res{args.baseline_weeks}' if args.residual else ''}"
                        f"{'_ev' if args.has_events else ''}{'_long' if args.long_features else ''}"
                        f"_do{args.dropout}_lr{args.lr}{'_plat' if args.plateau else ''}_s{args.seed}")


def numeric_inputs(p, args):
    """(D, T, F) per-week input features for this config."""
    return goldstein_numeric_inputs(p, long_features=args.long_features, has_events=args.has_events)


def prepare_data(args, p=None):
    """Splits, scaled model inputs and targets for train / val / test.
    The scaler is fit on the training split only."""
    p = p if p is not None else load_panel(MENA)
    numeric = numeric_inputs(p, args)
    n_numeric = numeric.shape[-1]
    y_all = p["target13"]
    base_all = trailing_mean(p["gold_zero"], args.baseline_weeks)
    tr, va, te = split_by_dates(p["weeks"], y_all[0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START,
                                GOLDSTEIN_HORIZON, lookback=args.lookback)
    idx = {"tr": tr, "va": va, "te": te}
    X = {k: windows_at(numeric, w, args.lookback) for k, w in idx.items()}
    Y = {k: y_all[:, w].T for k, w in idx.items()}
    B = {k: base_all[:, w].T for k, w in idx.items()}

    scaler = StandardScaler().fit(X["tr"].reshape(-1, n_numeric))
    for k in X:
        X[k] = scaler.transform(X[k].reshape(-1, n_numeric)).reshape(X[k].shape).astype("float32")

    enc = LabelEncoder().fit(MENA)
    a1 = enc.transform(p["a1"]).astype("int32")
    a2 = enc.transform(p["a2"]).astype("int32")

    def inputs(k):
        d = {"numeric_seq": X[k], "actor1_idx": np.tile(a1, (len(X[k]), 1)),
             "actor2_idx": np.tile(a2, (len(X[k]), 1))}
        if args.residual:
            d["baseline"] = B[k]
        return d

    return dict(p=p, idx=idx, X=X, Y=Y, B=B, inputs=inputs, scaler=scaler, encoder=enc,
                n_numeric=n_numeric, n_dyads=len(p["a1"]))


def build(args, data):
    return build_goldstein_model(n_dyads=data["n_dyads"], lookback_weeks=args.lookback,
                                 n_numeric=data["n_numeric"], n_countries=len(data["encoder"].classes_),
                                 dropout=args.dropout, residual_baseline=args.residual)


def train(args, data):
    keras.utils.set_random_seed(args.seed)
    model = build(args, data)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="mse")
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    if args.plateau:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3))
    hist = model.fit(data["inputs"]("tr"), data["Y"]["tr"],
                     validation_data=(data["inputs"]("va"), data["Y"]["va"]),
                     epochs=args.epochs, batch_size=8, verbose=2, callbacks=callbacks)
    return model, hist


def main():
    args = parse_args()
    tag = make_tag(args)
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    data = prepare_data(args)
    t0 = time.time()
    model, hist = train(args, data)
    Y, B, idx = data["Y"], data["B"], data["idx"]

    pred = {k: model.predict(data["inputs"](k), verbose=0) for k in ("va", "te")}
    res = dict(**{**vars(args), "tag": tag}, n_numeric=data["n_numeric"], train_weeks=len(idx["tr"]),
               best_epoch=int(np.argmin(hist.history["val_loss"])) + 1,
               epochs_run=len(hist.history["val_loss"]), minutes=round((time.time() - t0) / 60, 1))
    for k in ("va", "te"):
        yt, pt, bt = Y[k].ravel(), pred[k].ravel(), B[k].ravel()
        res[f"{k}_mse"] = float(mean_squared_error(yt, pt))
        res[f"{k}_r2"] = float(r2_score(yt, pt))
        res[f"{k}_mae"] = float(mean_absolute_error(yt, pt))
        res[f"{k}_r2_trailing{args.baseline_weeks}"] = float(r2_score(yt, bt))
    print(json.dumps(res))

    os.makedirs("logs", exist_ok=True)
    os.makedirs("artifacts/experiments", exist_ok=True)
    with open("logs/goldstein_experiments.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    np.savez(f"artifacts/experiments/{tag}.npz", va_week_idx=idx["va"], te_week_idx=idx["te"],
             va_pred=pred["va"], te_pred=pred["te"], va_true=Y["va"], te_true=Y["te"])
    model.save_weights(f"artifacts/experiments/{tag}.weights.h5")


if __name__ == "__main__":
    main()
