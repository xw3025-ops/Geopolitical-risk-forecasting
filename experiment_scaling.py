"""
Scaling experiments: does forecast quality keep improving with more training
data, and does attention across the system help more as the system grows?

Data: the 400 MENA dyads (GDELT 2.0, honest_common) plus N "external" dyads -
MENA countries' relations with the rest of the world, from the same BigQuery
table - drawn at random (fixed --sample-seed) from the pool of external dyads
active in >= --min-active of weeks, so that only QUANTITY varies with N.
External dyads are used for training only; validation and test are the 400
MENA dyads and the usual weeks, always.

Architectures (--arch):
  independent    no cross-dyad attention; each dyad is forecast from its own
                 history (+ country embeddings). External dyads enter as extra
                 samples (chunks of 400).
  cross-system   one sequence per week = 400 MENA dyads + all N external dyads;
                 two cross-dyad attention blocks over the whole sequence, so a
                 MENA forecast can attend to e.g. Israel-US or Iran-Russia.
Both use the release recipe otherwise (temporal transformer, residual on the
52-week mean, long-memory inputs, dropout 0.2, lr 3e-4 with halving).

Every run appends one line to logs/scaling.jsonl.

    KERAS_BACKEND=torch python experiment_scaling.py --arch cross-system --n-external 1200 --seeds 42 1 2
"""

import argparse
import json
import os
import time

import keras
import numpy as np
import pandas as pd
from keras import ops
from sklearn.preprocessing import LabelEncoder, StandardScaler

from benchmark_ladder import forecast_metrics
from honest_common import (COUNT_COLS, GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA,
                           goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean, windows_at)
from model_architectures import build_goldstein_model

WIDE = "raw data/bq_panel_with_target.parquet"
CHUNK = 400
LOOKBACK = 13
LOG = "logs/scaling.jsonl"


def external_pool(weeks, min_active):
    cols = ["week_start", "Actor1CountryCode", "Actor2CountryCode", "dyad_scope", "goldstein_mean",
            "total_count", "negative_goldstein_intensity", "target_avg_goldstein_next13w"] + COUNT_COLS
    df = pd.read_parquet(WIDE, columns=cols)
    df = df[df.dyad_scope == "EXTERNAL_LINK"]
    df["week_start"] = pd.to_datetime(df.week_start)
    act = df.groupby(["Actor1CountryCode", "Actor2CountryCode"]).total_count.apply(lambda s: (s > 0).mean())
    keep = sorted(act[act >= min_active].index)
    df = df.set_index(["Actor1CountryCode", "Actor2CountryCode"]).loc[keep].reset_index()
    df["key"] = df.Actor1CountryCode + "|" + df.Actor2CountryCode
    full = pd.MultiIndex.from_product([[f"{a}|{b}" for a, b in keep], weeks])
    df = df.set_index(["key", "week_start"]).reindex(full)
    D, T = len(keep), len(weeks)
    return dict(a1=np.array([a for a, _ in keep]), a2=np.array([b for _, b in keep]),
                counts=np.nan_to_num(df[COUNT_COLS].to_numpy(dtype="float32")).reshape(D, T, 20),
                gold_zero=np.nan_to_num(df["goldstein_mean"].to_numpy(dtype="float32")).reshape(D, T),
                neg=np.nan_to_num(df["negative_goldstein_intensity"].to_numpy(dtype="float32")).reshape(D, T),
                target13=df["target_avg_goldstein_next13w"].to_numpy(dtype="float32").reshape(D, T))


def subset(pool, idx):
    return {k: v[idx] for k, v in pool.items()}


def masked_mse(y_true, y_pred):
    """MSE over finite targets only (NaN = not graded)."""
    mask = ops.isfinite(y_true)
    err = ops.where(mask, ops.square(y_pred - ops.where(mask, y_true, 0.0)), 0.0)
    return ops.sum(err) / ops.maximum(ops.sum(ops.cast(mask, "float32")), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["independent", "cross-system"], required=True)
    ap.add_argument("--n-external", type=int, default=1200)
    ap.add_argument("--min-active", type=float, default=0.50)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--val-start", default=GOLDSTEIN_VAL_START, help="rolling backtest: validation start (Monday)")
    ap.add_argument("--test-start", default=GOLDSTEIN_TEST_START, help="rolling backtest: test start (Monday)")
    ap.add_argument("--test-end", default=None, help="rolling backtest: last test week (default: last available)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="learning curve: train on a random fraction of the training weeks (val/test unchanged)")
    args = ap.parse_args()

    p = load_panel(MENA)
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], args.val_start, args.test_start,
                                GOLDSTEIN_HORIZON, lookback=LOOKBACK)
    if args.test_end:  # leave the embargoed gap after a fold's test period unused
        te = te[p["weeks"][te] <= pd.Timestamp(args.test_end)]
    if args.train_frac < 1.0:
        keep_w = np.random.default_rng(args.sample_seed).choice(len(tr), int(round(len(tr) * args.train_frac)), replace=False)
        tr = np.sort(tr[keep_w])
    pool = external_pool(p["weeks"], args.min_active)
    N = args.n_external
    if N > len(pool["a1"]):
        raise SystemExit(f"only {len(pool['a1'])} external dyads in the pool")
    if args.arch == "independent" and N % CHUNK:
        raise SystemExit(f"--n-external must be a multiple of {CHUNK} for the independent architecture")
    ext = subset(pool, np.sort(np.random.default_rng(args.sample_seed).choice(len(pool["a1"]), N, replace=False)))
    tr_ok = tr[~np.isnan(ext["target13"][:, tr]).any(axis=0)] if N else tr
    print(f"{args.arch}: 400 MENA + {N} external dyads (pool {len(pool['a1'])}); train weeks {len(tr_ok)}", flush=True)

    enc = LabelEncoder().fit(np.concatenate([p["a1"], p["a2"], ext["a1"], ext["a2"]]))
    srcs = {"mena": p, "ext": ext}
    X = {k: goldstein_numeric_inputs(v, long_features=True) for k, v in srcs.items()}
    B = {k: trailing_mean(v["gold_zero"], 52) for k, v in srcs.items()}
    A = {k: (enc.transform(v["a1"]).astype("int32"), enc.transform(v["a2"]).astype("int32")) for k, v in srcs.items()}

    def block(k, dsel, weeks_idx, grade=True):
        y = srcs[k]["target13"][dsel][:, weeks_idx].T
        return dict(numeric_seq=windows_at(X[k][dsel], weeks_idx, LOOKBACK), baseline=B[k][dsel][:, weeks_idx].T,
                    actor1_idx=np.tile(A[k][0][dsel], (len(weeks_idx), 1)),
                    actor2_idx=np.tile(A[k][1][dsel], (len(weeks_idx), 1)),
                    y=y if grade else np.full_like(y, np.nan))

    def cat(parts, axis):
        return {key: np.concatenate([q[key] for q in parts], axis=axis) for key in parts[0]}

    mena_all = np.arange(400)
    if args.arch == "independent":
        n_tok = CHUNK
        train = cat([block("mena", mena_all, tr)] +
                    [block("ext", np.arange(c * CHUNK, (c + 1) * CHUNK), tr_ok) for c in range(N // CHUNK)], axis=0)
        val, test = block("mena", mena_all, va), block("mena", mena_all, te)
        crossdyad = 0
    else:
        n_tok = 400 + N
        ext_all = np.arange(N)
        parts = lambda w, grade_ext: [block("mena", mena_all, w)] + ([block("ext", ext_all, w, grade_ext)] if N else [])
        train = cat(parts(tr_ok, True), axis=1)
        val, test = cat(parts(va, False), axis=1), cat(parts(te, False), axis=1)  # graded on MENA only
        crossdyad = 2

    n_num = train["numeric_seq"].shape[-1]
    scaler = StandardScaler().fit(train["numeric_seq"].reshape(-1, n_num))
    for d in (train, val, test):
        d["numeric_seq"] = scaler.transform(d["numeric_seq"].reshape(-1, n_num)).reshape(d["numeric_seq"].shape).astype("float32")
        d["baseline"] = d["baseline"].astype("float32")
    split = lambda d: ({k: d[k] for k in ("numeric_seq", "actor1_idx", "actor2_idx", "baseline")}, d["y"].astype("float32"))
    (xtr, ytr), (xva, yva), (xte, _) = split(train), split(val), split(test)
    y_te = p["target13"][:, te].T
    n52 = B["mena"][:, te].T
    y_va_mena = p["target13"][:, va].T

    preds_te, preds_va, seeds_out = [], [], {}
    t0 = time.time()
    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        m = build_goldstein_model(n_dyads=n_tok, lookback_weeks=LOOKBACK, n_numeric=n_num,
                                  n_countries=len(enc.classes_), dropout=0.2, residual_baseline=True,
                                  crossdyad_blocks=crossdyad)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss=masked_mse)
        h = m.fit(xtr, ytr, validation_data=(xva, yva), epochs=100, batch_size=8, shuffle=True, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        pt = m.predict(xte, verbose=0)[:, :400]
        pv = m.predict(xva, verbose=0)[:, :400]
        preds_te.append(pt)
        preds_va.append(pv)
        met = forecast_metrics(y_te, pt, n52)
        seeds_out[str(seed)] = dict(val_mse=float(((pv - y_va_mena) ** 2).mean()),
                                    best_epoch=int(np.argmin(h.history["val_loss"]) + 1), **met)
        print(f"  seed {seed}: best epoch {seeds_out[str(seed)]['best_epoch']}, val MSE {seeds_out[str(seed)]['val_mse']:.4f} | "
              f"R² {met['r2']:.3f} dir {met['dir_acc']:.1%} chg-corr {met['change_corr']:.3f} esc-AUC {met['esc_auc']:.3f}",
              flush=True)
    ens = forecast_metrics(y_te, np.mean(preds_te, 0), n52)
    val_ens = float(((np.mean(preds_va, 0) - y_va_mena) ** 2).mean())
    print(f"ENSEMBLE {len(args.seeds)}: val MSE {val_ens:.4f} | R² {ens['r2']:.3f} vs52 {ens['mse_vs_52']:+.1%} "
          f"dir {ens['dir_acc']:.1%} chg-corr {ens['change_corr']:.3f} esc-AUC {ens['esc_auc']:.3f}  "
          f"({(time.time() - t0) / 60:.1f} min)")
    rec = dict(arch=args.arch, n_external=N, min_active=args.min_active, sample_seed=args.sample_seed,
               train_frac=args.train_frac, train_weeks=int(len(tr)), val_start=args.val_start,
               test_start=args.test_start, test_end=args.test_end, test_weeks=int(len(te)),
               seeds=args.seeds, train_dyad_weeks=int(np.isfinite(ytr).sum()), val_mse_ensemble=val_ens,
               val_mse_mean=float(np.mean([s["val_mse"] for s in seeds_out.values()])), ensemble=ens, per_seed=seeds_out)
    os.makedirs("logs", exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
