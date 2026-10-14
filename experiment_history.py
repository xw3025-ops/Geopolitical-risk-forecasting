"""
Experiment 6: more training HISTORY. Prepends GDELT 1.0 weekly dyad data
(fetch_gdelt1_history.py) from --start (default 2008, when GDELT 1.0 coverage
approaches GDELT 2.0's) to the GDELT 2.0 panel, and trains the release
architecture (full model, release recipe) on the longer history. Validation
and test weeks and dyads are exactly those used everywhere else.

    KERAS_BACKEND=torch python experiment_history.py --seeds 42 1 2 [--start 2008-01-07]
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
                           forward_mean, goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean,
                           windows_at)
from model_architectures import build_goldstein_model

HISTORY = "raw data/gdelt1_history.parquet"
LOOKBACK = 13
LONG_MEMORY = 52 + LOOKBACK - 1  # weeks of history a training row needs


def combined_panel(start):
    p = load_panel(MENA)
    h = pd.read_parquet(HISTORY)
    h = h[h.week_start >= pd.Timestamp(start)]
    hweeks = pd.date_range(pd.Timestamp(start), p["weeks"][0] - pd.Timedelta(weeks=1), freq="W-MON")
    full = pd.MultiIndex.from_arrays([np.repeat(p["a1"], len(hweeks)), np.repeat(p["a2"], len(hweeks)),
                                      np.tile(hweeks, len(p["a1"]))])
    h = h.set_index(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reindex(full)
    D, Th = len(p["a1"]), len(hweeks)
    hc = np.nan_to_num(h[COUNT_COLS].to_numpy(dtype="float32")).reshape(D, Th, 20)
    hg = np.nan_to_num(h["goldstein_mean"].to_numpy(dtype="float32")).reshape(D, Th)
    c = dict(a1=p["a1"], a2=p["a2"], counts=np.concatenate([hc, p["counts"]], 1),
             gold_zero=np.concatenate([hg, p["gold_zero"]], 1))
    tgt = forward_mean(c["gold_zero"], GOLDSTEIN_HORIZON)
    tgt[:, Th:] = p["target13"]  # GDELT 2.0 weeks: the exact target used everywhere else
    c["target13"] = tgt
    return p, c, Th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--start", default="2008-01-07")
    ap.add_argument("--crossdyad-blocks", type=int, default=2)
    args = ap.parse_args()

    p, c, Th = combined_panel(args.start)
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START,
                                GOLDSTEIN_HORIZON, lookback=LOOKBACK)
    tr_all = np.r_[np.arange(LONG_MEMORY, Th), tr + Th]   # history weeks + the usual training weeks
    va, te = va + Th, te + Th
    print(f"history from {args.start}: {Th} extra weeks; training weeks {len(tr_all)} (usual {len(tr)})", flush=True)

    X = goldstein_numeric_inputs(c, long_features=True)
    base = trailing_mean(c["gold_zero"], 52)
    n_num = X.shape[-1]
    Xtr = windows_at(X, tr_all, LOOKBACK)
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_num))
    sc = lambda A: scaler.transform(A.reshape(-1, n_num)).reshape(A.shape).astype("float32")
    enc = LabelEncoder().fit(MENA)
    a1, a2 = enc.transform(c["a1"]).astype("int32"), enc.transform(c["a2"]).astype("int32")

    def inputs(w, Xw=None):
        return {"numeric_seq": sc(windows_at(X, w, LOOKBACK)) if Xw is None else Xw,
                "actor1_idx": np.tile(a1, (len(w), 1)), "actor2_idx": np.tile(a2, (len(w), 1)),
                "baseline": base[:, w].T.astype("float32")}
    tr_in, va_in, te_in = inputs(tr_all, sc(Xtr)), inputs(va), inputs(te)
    y_tr, y_va, y_te = (c["target13"][:, w].T for w in (tr_all, va, te))
    assert not np.isnan(y_tr).any()

    preds, out = [], {}
    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        m = build_goldstein_model(n_dyads=400, lookback_weeks=LOOKBACK, n_numeric=n_num, n_countries=len(MENA),
                                  dropout=0.2, residual_baseline=True, crossdyad_blocks=args.crossdyad_blocks)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mse")
        h = m.fit(tr_in, y_tr, validation_data=(va_in, y_va), epochs=100, batch_size=8, shuffle=True, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        pt = m.predict(te_in, verbose=0)
        preds.append(pt)
        met = forecast_metrics(y_te, pt, base[:, te].T)
        out[f"seed{seed}"] = dict(val_mse=float(min(h.history["val_loss"])),
                                  best_epoch=int(np.argmin(h.history["val_loss"]) + 1), **met)
        print(f"seed {seed}: best epoch {out[f'seed{seed}']['best_epoch']}, val MSE {out[f'seed{seed}']['val_mse']:.4f} | "
              f"R² {met['r2']:.3f} dir {met['dir_acc']:.1%} chg-corr {met['change_corr']:.3f} esc-AUC {met['esc_auc']:.3f}",
              flush=True)
    ens = forecast_metrics(y_te, np.mean(preds, 0), base[:, te].T)
    out["ensemble"] = ens
    print(f"ENSEMBLE of {len(preds)}: R² {ens['r2']:.3f} vs52 {ens['mse_vs_52']:+.1%} dir {ens['dir_acc']:.1%} "
          f"chg-corr {ens['change_corr']:.3f} esc-AUC {ens['esc_auc']:.3f}")
    os.makedirs("logs", exist_ok=True)
    json.dump(out, open(f"logs/history_{args.start}_cd{args.crossdyad_blocks}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
