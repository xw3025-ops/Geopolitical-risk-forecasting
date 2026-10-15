"""
RELEASE v2.2 - GOLDSTEIN (regression, 13-week horizon): v2.1 + WORLD TRAINING DATA.

Identical to release v2.1 (cross-system attention over 1,959 dyad tokens: the
400 MENA<->MENA dyads + 1,559 MENA<->world dyads; residual on the 52-week
mean; long-memory inputs; dropout 0.2; Adam 3e-4 with halving; 5 seeds;
13-week embargo) except for MORE TRAINING DATA: each training week also
contributes 3 extra sequences of the same 1,959-token shape filled with the
5,122 dyads between countries OUTSIDE the region (fetch_world_dyads.py; active
in >= 50% of weeks; shuffled once with a fixed seed, the last sequence padded
with ungraded repeats). The model learns how relationships evolve from
roughly 3.6x as many dyads.

World dyads are used in training only. Validation, test and live forecasts
are the MENA system exactly as in v2.1 (same 1,959 tokens, same inputs), so
the live pipeline needs no extra data; only the country encoder is larger.

Selected only if the 5-seed ensemble beats v2.1 on VALIDATION (v2.1: 0.8644).

    KERAS_BACKEND=torch python release_v2.2_goldstein.py
Outputs:
    model_release_v2.2_goldstein_s<seed>.weights.h5
    artifacts/goldstein_v2.2_preprocessing.joblib
    artifacts/goldstein_v2.2_test_predictions.npz, artifacts/goldstein_v2.2_val_predictions.npz
"""

import joblib
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from benchmark_ladder import forecast_metrics
from experiment_scaling import external_pool, masked_mse
from experiment_targets import load_world
from honest_common import (COUNT_COLS, GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA,
                           forward_mean, goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean,
                           windows_at)
from model_architectures import build_goldstein_model

SEEDS = [42, 1, 2, 3, 4]
MIN_ACTIVE = 0.50
LOOKBACK = 13
WORLD_SEQUENCES = 3
V21_VAL_MSE = 0.8644
CONFIG = dict(lookback=LOOKBACK, residual=True, baseline_weeks=52, long_features=True, has_events=False,
              dropout=0.2, lr=3e-4, plateau=True, crossdyad_blocks=2, min_active_external=MIN_ACTIVE,
              world_training_sequences=WORLD_SEQUENCES)
ARTIFACTS = "artifacts/goldstein_v2.2_preprocessing.joblib"


def main():
    p = load_panel(MENA)
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START,
                                GOLDSTEIN_HORIZON, lookback=LOOKBACK)
    ext = external_pool(p["weeks"], MIN_ACTIVE)
    world = load_world(p["weeks"])
    N, Dw = len(ext["a1"]), len(world["a1"])
    n_tok = 400 + N
    print(f"cross-system: {n_tok} tokens per week; + {Dw} world dyads in {WORLD_SEQUENCES} extra training "
          f"sequences per week", flush=True)

    enc = LabelEncoder().fit(np.concatenate([p["a1"], p["a2"], ext["a1"], ext["a2"], world["a1"], world["a2"]]))
    X = np.concatenate([goldstein_numeric_inputs(p, long_features=True),
                        goldstein_numeric_inputs(ext, long_features=True)], axis=0)
    base = np.concatenate([trailing_mean(p["gold_zero"], 52), trailing_mean(ext["gold_zero"], 52)], axis=0)
    a1 = enc.transform(np.concatenate([p["a1"], ext["a1"]])).astype("int32")
    a2 = enc.transform(np.concatenate([p["a2"], ext["a2"]])).astype("int32")
    y_all = np.concatenate([p["target13"], ext["target13"]], axis=0)
    n_num = X.shape[-1]

    Xtr = windows_at(X, tr, LOOKBACK)
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_num))  # fit on the MENA system, as in v2.1
    sc = lambda A: scaler.transform(A.reshape(-1, n_num)).reshape(A.shape).astype("float32")

    def inputs(w, Xw=None):
        return {"numeric_seq": sc(windows_at(X, w, LOOKBACK)) if Xw is None else sc(Xw),
                "actor1_idx": np.tile(a1, (len(w), 1)), "actor2_idx": np.tile(a2, (len(w), 1)),
                "baseline": base[:, w].T.astype("float32")}

    def graded_mena_only(w):
        y = y_all[:, w].T.astype("float32").copy()
        y[:, 400:] = np.nan
        return y

    x_tr, x_va, x_te = inputs(tr, Xtr), inputs(va), inputs(te)
    y_tr, y_va = y_all[:, tr].T.astype("float32"), graded_mena_only(va)
    del Xtr

    # world training sequences
    Xw = goldstein_numeric_inputs(world, long_features=True)
    yw = forward_mean(world["gold_zero"], GOLDSTEIN_HORIZON)
    bw = trailing_mean(world["gold_zero"], 52)
    aw1, aw2 = enc.transform(world["a1"]).astype("int32"), enc.transform(world["a2"]).astype("int32")
    order = np.resize(np.random.default_rng(0).permutation(Dw), WORLD_SEQUENCES * n_tok)
    pad = np.arange(len(order)) >= Dw
    xs, ys = [x_tr], [y_tr]
    for c in range(WORLD_SEQUENCES):
        sel, padded = order[c * n_tok:(c + 1) * n_tok], pad[c * n_tok:(c + 1) * n_tok]
        y = yw[sel][:, tr].T.astype("float32").copy()
        y[:, padded] = np.nan
        xs.append({"numeric_seq": np.nan_to_num(sc(windows_at(Xw[sel], tr, LOOKBACK))),
                   "actor1_idx": np.tile(aw1[sel], (len(tr), 1)), "actor2_idx": np.tile(aw2[sel], (len(tr), 1)),
                   "baseline": np.nan_to_num(bw[sel][:, tr].T).astype("float32")})
        ys.append(y)
    x_tr = {k: np.concatenate([x[k] for x in xs], 0) for k in x_tr}
    y_tr = np.concatenate(ys, 0)
    del xs, ys, Xw
    print(f"training sequences: {len(y_tr)} ({len(tr)} weeks x {1 + WORLD_SEQUENCES}); "
          f"graded cells {np.isfinite(y_tr).sum()}", flush=True)

    y_te_mena, y_va_mena = p["target13"][:, te].T, p["target13"][:, va].T
    n52 = trailing_mean(p["gold_zero"], 52)[:, te].T

    preds = {"va": [], "te": []}
    files = []
    for seed in SEEDS:
        keras.utils.set_random_seed(seed)
        m = build_goldstein_model(n_dyads=n_tok, lookback_weeks=LOOKBACK, n_numeric=n_num,
                                  n_countries=len(enc.classes_), dropout=0.2, residual_baseline=True,
                                  crossdyad_blocks=2)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss=masked_mse)
        m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=100, batch_size=8, shuffle=True, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        out = f"model_release_v2.2_goldstein_s{seed}.weights.h5"
        m.save_weights(out)
        files.append(out)
        pv, pt = m.predict(x_va, verbose=0)[:, :400], m.predict(x_te, verbose=0)[:, :400]
        preds["va"].append(pv)
        preds["te"].append(pt)
        met = forecast_metrics(y_te_mena, pt, n52)
        print(f"seed {seed}: val MSE {((pv - y_va_mena) ** 2).mean():.4f} | test R² {met['r2']:.3f} "
              f"dir {met['dir_acc']:.1%} chg-corr {met['change_corr']:.3f} esc-AUC {met['esc_auc']:.3f}", flush=True)
    ens = {k: np.mean(v, 0) for k, v in preds.items()}
    met = forecast_metrics(y_te_mena, ens["te"], n52)
    val_mse = float(((ens["va"] - y_va_mena) ** 2).mean())
    print(f"ENSEMBLE {len(SEEDS)}: val MSE {val_mse:.4f} (v2.1 {V21_VAL_MSE}) | test R² {met['r2']:.3f} "
          f"vs52 {met['mse_vs_52']:+.1%} dir {met['dir_acc']:.1%} chg-corr {met['change_corr']:.3f} "
          f"esc-AUC {met['esc_auc']:.3f}")
    print("SELECTED over v2.1 on validation" if val_mse < V21_VAL_MSE else "NOT better than v2.1 on validation")

    joblib.dump({
        "config": CONFIG, "seeds": SEEDS, "weight_files": files, "scaler": scaler, "country_encoder": enc,
        "count_cols": COUNT_COLS, "raw_cols": ["goldstein_mean"],
        "long_feature_cols": ["gold_zmean26", "gold_zmean52", "log_events_mean52"],
        "n_dyads_per_week": n_tok, "n_mena_dyads": 400, "lookback_weeks": LOOKBACK,
        "history_weeks_needed": LOOKBACK + 51, "mena_countries": MENA,
        "external_dyads": list(zip(ext["a1"].tolist(), ext["a2"].tolist())),
        "val_mse_ensemble": val_mse,
    }, ARTIFACTS)
    for k, path in (("te", "artifacts/goldstein_v2.2_test_predictions.npz"),
                    ("va", "artifacts/goldstein_v2.2_val_predictions.npz")):
        w = te if k == "te" else va
        np.savez(path, week_idx=w, a1=p["a1"], a2=p["a2"], y_true=p["target13"][:, w].T,
                 pred=ens[k], pred_per_seed=np.stack(preds[k]))
    print(f"saved {', '.join(files)}, {ARTIFACTS} and test/val predictions")


if __name__ == "__main__":
    main()
