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

import keras
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from honest_common import (GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA,
                           direction_feature_columns, direction_numeric_inputs, forward_mean, goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean,
                           windows_at)
from model_architectures import build_goldstein_model


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=13)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--baseline-weeks", type=int, default=26)
    ap.add_argument("--has-events", action="store_true")
    ap.add_argument("--long-features", action="store_true")
    ap.add_argument("--crossdyad-blocks", type=int, default=2,
                    help="0 = ablation without cross-dyad attention (each dyad predicted independently)")
    ap.add_argument("--no-country-embedding", action="store_true",
                    help="ablation: replace the country embeddings with zeros (no country identity)")
    ap.add_argument("--target", choices=["goldstein", "conflict"], default="goldstein",
                    help="goldstein: mean Goldstein next 13 weeks (release). conflict: mean of "
                         "log(1 + weekly sum of negative Goldstein scores) next 13 weeks")
    ap.add_argument("--rich-features", action="store_true",
                    help="all numeric GDELT table fields (tone, media volume, shares, diversity, ...) "
                         "instead of CAMEO counts + Goldstein only")
    ap.add_argument("--acled", action="store_true",
                    help="add both countries' weekly ACLED activity (event types, fatalities) to every dyad token")
    ap.add_argument("--news", action="store_true",
                    help="add Guardian news-text features (news_features.py): pair co-coverage + both countries")
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
                        f"{f'_cd{args.crossdyad_blocks}' if args.crossdyad_blocks != 2 else ''}"
                        f"{'_noemb' if args.no_country_embedding else ''}"
                        f"{'' if args.target == 'goldstein' else '_T' + args.target}"
                        f"{'_rich' if args.rich_features else ''}{'_acled' if args.acled else ''}{'_news' if args.news else ''}"
                        f"_do{args.dropout}_lr{args.lr}{'_plat' if args.plateau else ''}_s{args.seed}")


def conflict_series(p):
    """(D, T) weekly conflict intensity: log(1 + sum of |negative Goldstein|).
    Uses only negative events, so cooperative events cannot mask escalation,
    and keeps volume, which is where escalation actually shows up."""
    D, T = p["gold_zero"].shape
    neg = p["df"]["negative_goldstein_intensity"].to_numpy(dtype="float64").reshape(D, T)
    return np.log1p(neg).astype("float32")


ACLED_LOCAL = "raw data/acled_country_week.parquet"
ACLED_COLS = ["n_battles", "n_explosions_remote_violence", "n_violence_against_civilians", "n_protests",
              "n_riots", "n_strategic_developments", "fatalities"]


def acled_inputs(p):
    """(D, T, 16): for actor 1 and actor 2, log1p of their country's weekly ACLED
    counts by event type and fatalities, plus a 0/1 'ACLED already covers this
    country' flag (coverage starts later than 2015 for some countries; before
    that the counts are unknown, not zero)."""
    import pandas as pd
    a = pd.read_parquet(ACLED_LOCAL)
    a["week_start"] = pd.to_datetime(a["week_start"])
    weeks = p["weeks"]
    per_country = {}
    for c, g in a.groupby("country"):
        g = g.set_index("week_start").reindex(weeks, fill_value=0)
        vals = np.log1p(g[ACLED_COLS].to_numpy(dtype="float64"))
        first = np.argmax(g["events"].to_numpy() > 0) if (g["events"] > 0).any() else len(weeks)
        covered = (np.arange(len(weeks)) >= first).astype("float64")[:, None]
        per_country[c] = np.concatenate([vals, covered], axis=1)  # (T, 8)
    empty = np.zeros((len(weeks), len(ACLED_COLS) + 1))
    x1 = np.stack([per_country.get(c, empty) for c in p["a1"]])
    x2 = np.stack([per_country.get(c, empty) for c in p["a2"]])
    return np.concatenate([x1, x2], axis=2).astype("float32")


NEWS_LOCAL = "raw data/news_features.npz"


def news_inputs(p):
    """(D, T, 24): Guardian co-coverage of the pair (8) + coverage of actor 1's and
    actor 2's countries (8 each): log article count, conflict-vs-cooperation
    score, 6 topic dimensions."""
    z = np.load(NEWS_LOCAL)
    assert len(z["weeks"]) == len(p["weeks"]), "news features out of date - rerun news_features.py"
    ci = {c: i for i, c in enumerate(z["countries"])}
    country = z["country"]
    return np.concatenate([z["dyad"], country[[ci[c] for c in p["a1"]]], country[[ci[c] for c in p["a2"]]]],
                          axis=2).astype("float32")


def target_and_series(p, args):
    """-> (y_all: target per dyad-week, series: the weekly quantity it averages)"""
    if args.target == "conflict":
        c = conflict_series(p)
        return forward_mean(c, GOLDSTEIN_HORIZON), c
    return p["target13"], p["gold_zero"]


def numeric_inputs(p, args):
    """(D, T, F) per-week input features for this config."""
    if args.rich_features:  # every numeric column of table 1.1 (+ long-memory features)
        log_cols, raw_cols = direction_feature_columns(p["df"])
        x = direction_numeric_inputs(p, log_cols, raw_cols)
    else:
        x = goldstein_numeric_inputs(p, long_features=args.long_features, has_events=args.has_events)
    if args.acled:
        x = np.concatenate([x, acled_inputs(p)], axis=2).astype("float32")
    if args.news:
        x = np.concatenate([x, news_inputs(p)], axis=2).astype("float32")
    if args.target == "conflict":
        c = conflict_series(p)
        x = np.concatenate([x, c[:, :, None], trailing_mean(c, 26)[:, :, None],
                            trailing_mean(c, 52)[:, :, None]], axis=2).astype("float32")
    return x


def prepare_data(args, p=None):
    """Splits, scaled model inputs and targets for train / val / test.
    The scaler is fit on the training split only."""
    p = p if p is not None else load_panel(MENA)
    numeric = numeric_inputs(p, args)
    n_numeric = numeric.shape[-1]
    y_all, series = target_and_series(p, args)
    base_all = trailing_mean(series, args.baseline_weeks)
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START,
                                GOLDSTEIN_HORIZON, lookback=args.lookback)
    if args.target != "goldstein":  # computed locally from the panel, which ends earlier than the SQL target
        te = te[~np.isnan(y_all[:, te]).any(axis=0)]
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

    N52 = {k: trailing_mean(series, 52)[:, w].T for k, w in idx.items()}
    return dict(p=p, idx=idx, X=X, Y=Y, B=B, N52=N52, inputs=inputs, scaler=scaler, encoder=enc,
                n_numeric=n_numeric, n_dyads=len(p["a1"]))


def build(args, data):
    return build_goldstein_model(n_dyads=data["n_dyads"], lookback_weeks=args.lookback,
                                 n_numeric=data["n_numeric"], n_countries=len(data["encoder"].classes_),
                                 dropout=args.dropout, residual_baseline=args.residual,
                                 crossdyad_blocks=args.crossdyad_blocks,
                                 country_embedding=not args.no_country_embedding)


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
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(args.threads)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        else:
            import torch
            torch.set_num_threads(args.threads)

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
    from benchmark_ladder import forecast_metrics
    esc = 1.0 if args.target == "goldstein" else 0.5
    worse_up = args.target == "conflict"
    res["test_metrics"] = forecast_metrics(Y["te"], pred["te"], data["N52"]["te"], esc, worse_up)
    res["test_metrics_trailing52"] = forecast_metrics(Y["te"], data["N52"]["te"], data["N52"]["te"], esc, worse_up)
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
