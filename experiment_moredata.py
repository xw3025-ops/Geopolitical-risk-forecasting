"""
Experiment 5: more training EXAMPLES instead of more features.

Trains the release recipe without cross-dyad attention (dyads are modelled
independently, so dyads from outside the region can be added freely) on the
400 MENA dyads PLUS the MENA-vs-rest-of-world dyads from the BigQuery table
(EXTERNAL_LINK scope, dyads active in >= MIN_ACTIVE of weeks). External dyads
are used in TRAINING weeks only; validation and test are exactly the 400 MENA
dyads and weeks used everywhere else, so results compare directly with the
MENA-only model (no cross-dyad attention, test R^2 0.646 for 3 seeds).

Runs on the GPU when KERAS_BACKEND=torch.

    KERAS_BACKEND=torch python experiment_moredata.py --seeds 42 1 2
"""

import argparse
import json
import os

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from benchmark_ladder import forecast_metrics
from honest_common import (COUNT_COLS, GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA,
                           goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean, windows_at)
from model_architectures import build_goldstein_model

WIDE = "raw data/bq_panel_with_target.parquet"
CHUNK = 400          # dyads per training sample (same width as the MENA panel)
LOOKBACK = 13


def external_panel(weeks, min_active):
    """EXTERNAL_LINK dyads as arrays aligned to `weeks` (train weeks are all inside the wide panel)."""
    cols = ["week_start", "Actor1CountryCode", "Actor2CountryCode", "dyad_scope", "goldstein_mean",
            "total_count", "target_avg_goldstein_next13w"] + COUNT_COLS
    df = pd.read_parquet(WIDE, columns=cols)
    df = df[df.dyad_scope == "EXTERNAL_LINK"]
    df["week_start"] = pd.to_datetime(df.week_start)
    act = df.groupby(["Actor1CountryCode", "Actor2CountryCode"]).total_count.apply(lambda s: (s > 0).mean())
    keep = act[act >= min_active].index
    n_keep = (len(keep) // CHUNK) * CHUNK          # whole chunks only: drop the least active remainder
    keep = act.loc[keep].sort_values(ascending=False).index[:n_keep]
    df = df.set_index(["Actor1CountryCode", "Actor2CountryCode"]).loc[keep].reset_index()
    df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"])
    D = len(keep)
    full = pd.MultiIndex.from_product([[f"{a}|{b}" for a, b in keep], weeks])
    df["key"] = df.Actor1CountryCode + "|" + df.Actor2CountryCode
    df = df.set_index(["key", "week_start"]).reindex(full)
    T = len(weeks)
    g = lambda c: df[c].to_numpy(dtype="float32").reshape(D, T)
    counts = np.nan_to_num(df[COUNT_COLS].to_numpy(dtype="float32")).reshape(D, T, 20)
    p = dict(a1=np.array([a for a, _ in keep]), a2=np.array([b for _, b in keep]), counts=counts,
             gold_zero=np.nan_to_num(g("goldstein_mean")), target13=g("target_avg_goldstein_next13w"))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--min-active", type=float, default=0.25)
    ap.add_argument("--d-model", type=int, default=64, help="model width (feed-forward = 2x)")
    args = ap.parse_args()
    import model_architectures as ma  # width is a module-level setting read when the model is built
    ma.D_MODEL, ma.FF_DIM = args.d_model, 2 * args.d_model

    p = load_panel(MENA)
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START,
                                GOLDSTEIN_HORIZON, lookback=LOOKBACK)
    ext = external_panel(p["weeks"], args.min_active)
    tr_ext = tr[~np.isnan(ext["target13"][:, tr]).any(axis=0)]
    n_chunks = len(ext["a1"]) // CHUNK
    print(f"MENA dyads 400 + external dyads {len(ext['a1'])} ({n_chunks} chunks of {CHUNK}); "
          f"training samples per week {1 + n_chunks}", flush=True)

    feats = {"mena": goldstein_numeric_inputs(p, long_features=True), "ext": goldstein_numeric_inputs(ext, long_features=True)}
    base = {"mena": trailing_mean(p["gold_zero"], 52), "ext": trailing_mean(ext["gold_zero"], 52)}
    enc = LabelEncoder().fit(np.concatenate([p["a1"], p["a2"], ext["a1"], ext["a2"]]))
    actors = {"mena": (enc.transform(p["a1"]), enc.transform(p["a2"])), "ext": (enc.transform(ext["a1"]), enc.transform(ext["a2"]))}

    # training samples: (week, chunk) -> chunk of 400 dyads
    Xs, Ys, Bs, A1, A2 = [], [], [], [], []
    def add(src, dsel, weeks_idx):
        Xs.append(windows_at(feats[src][dsel], weeks_idx, LOOKBACK))
        Ys.append((p if src == "mena" else ext)["target13"][dsel][:, weeks_idx].T)
        Bs.append(base[src][dsel][:, weeks_idx].T)
        a1, a2 = actors[src]
        A1.append(np.tile(a1[dsel], (len(weeks_idx), 1)))
        A2.append(np.tile(a2[dsel], (len(weeks_idx), 1)))
    add("mena", np.arange(400), tr)
    for c in range(n_chunks):
        add("ext", np.arange(c * CHUNK, (c + 1) * CHUNK), tr_ext)
    X_tr, Y_tr, B_tr = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Bs)
    A1_tr, A2_tr = np.concatenate(A1), np.concatenate(A2)
    n_num = X_tr.shape[-1]
    scaler = StandardScaler().fit(X_tr.reshape(-1, n_num))
    sc = lambda X: scaler.transform(X.reshape(-1, n_num)).reshape(X.shape).astype("float32")
    X_tr = sc(X_tr)
    print(f"training samples: {len(X_tr)} (MENA-only would be {len(tr)})", flush=True)

    def mena_inputs(weeks_idx):
        a1, a2 = actors["mena"]
        return {"numeric_seq": sc(windows_at(feats["mena"], weeks_idx, LOOKBACK)),
                "actor1_idx": np.tile(a1, (len(weeks_idx), 1)).astype("int32"),
                "actor2_idx": np.tile(a2, (len(weeks_idx), 1)).astype("int32"),
                "baseline": base["mena"][:, weeks_idx].T.astype("float32")}
    va_in, te_in = mena_inputs(va), mena_inputs(te)
    y_va, y_te = p["target13"][:, va].T, p["target13"][:, te].T
    n52_te = base["mena"][:, te].T

    preds, out = [], {}
    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        m = build_goldstein_model(n_dyads=CHUNK, lookback_weeks=LOOKBACK, n_numeric=n_num,
                                  n_countries=len(enc.classes_), dropout=0.2, residual_baseline=True,
                                  crossdyad_blocks=0)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mse")
        h = m.fit({"numeric_seq": X_tr, "actor1_idx": A1_tr.astype("int32"), "actor2_idx": A2_tr.astype("int32"),
                   "baseline": B_tr.astype("float32")}, Y_tr, validation_data=(va_in, y_va),
                  epochs=100, batch_size=8, shuffle=True, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        pt = m.predict(te_in, verbose=0)
        preds.append(pt)
        met = forecast_metrics(y_te, pt, n52_te)
        out[f"seed{seed}"] = dict(val_mse=float(min(h.history["val_loss"])), best_epoch=int(np.argmin(h.history["val_loss"]) + 1), **met)
        print(f"seed {seed}: best epoch {out[f'seed{seed}']['best_epoch']}, val MSE {out[f'seed{seed}']['val_mse']:.4f} | "
              f"R² {met['r2']:.3f} dir {met['dir_acc']:.1%} chg-corr {met['change_corr']:.3f} esc-AUC {met['esc_auc']:.3f}", flush=True)
    ens = forecast_metrics(y_te, np.mean(preds, 0), n52_te)
    out["ensemble"] = ens
    print(f"ENSEMBLE of {len(preds)}: R² {ens['r2']:.3f} vs52 {ens['mse_vs_52']:+.1%} dir {ens['dir_acc']:.1%} "
          f"chg-corr {ens['change_corr']:.3f} esc-AUC {ens['esc_auc']:.3f}")
    os.makedirs("logs", exist_ok=True)
    json.dump(out, open(f"logs/moredata_min{args.min_active}_d{args.d_model}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
