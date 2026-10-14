"""
RELEASE v2.1 - GOLDSTEIN (regression, 13-week horizon): CROSS-SYSTEM attention.

Each forecast week is one sequence of 1,959 dyad tokens: the 400 MENA<->MENA
dyads plus the 1,559 dyads linking MENA countries to the rest of the world
(EXTERNAL_LINK scope of BigQuery table 1.1, dyads active in >= 50% of weeks).
The temporal transformer encodes each dyad's own 13-week history; two
cross-dyad attention blocks then let every token attend to every other, so a
MENA forecast can draw on, e.g., Israel-US or Iran-Russia relations. External
dyads are trained on but never graded at validation/test: evaluation is the
400 MENA dyads and weeks used for every model in this project.

Selected over v2.0 (regional attention, 400 tokens) on validation loss
(experiment_scaling.py; logs/scaling.jsonl): ensemble val MSE 0.859 vs 0.874.
Everything else is the v2.0 recipe: residual on the 52-week mean, long-memory
inputs, dropout 0.2, Adam 3e-4 with halving, 5-seed ensemble, 13-week embargo.

Train on the GPU (Windows: Keras on PyTorch); the saved weights load on any
Keras backend, so the TensorFlow Cloud Function serves them unchanged.

    KERAS_BACKEND=torch python release_v2.1_goldstein.py
Outputs:
    model_release_v2.1_goldstein_s<seed>.weights.h5
    artifacts/goldstein_v2.1_preprocessing.joblib
    artifacts/goldstein_v2.1_test_predictions.npz, artifacts/goldstein_v2.1_val_predictions.npz
"""

import joblib
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from benchmark_ladder import forecast_metrics
from experiment_scaling import external_pool, masked_mse
from honest_common import (COUNT_COLS, GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA,
                           goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean, windows_at)
from model_architectures import build_goldstein_model

SEEDS = [42, 1, 2, 3, 4]
MIN_ACTIVE = 0.50
LOOKBACK = 13
CONFIG = dict(lookback=LOOKBACK, residual=True, baseline_weeks=52, long_features=True, has_events=False,
              dropout=0.2, lr=3e-4, plateau=True, crossdyad_blocks=2, min_active_external=MIN_ACTIVE)
ARTIFACTS = "artifacts/goldstein_v2.1_preprocessing.joblib"


def main():
    p = load_panel(MENA)
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START,
                                GOLDSTEIN_HORIZON, lookback=LOOKBACK)
    ext = external_pool(p["weeks"], MIN_ACTIVE)
    N = len(ext["a1"])
    tr_ok = tr[~np.isnan(ext["target13"][:, tr]).any(axis=0)]
    assert np.array_equal(tr_ok, tr), "external targets missing in training weeks"
    print(f"cross-system: 400 MENA + {N} external dyads = {400 + N} tokens per week", flush=True)

    enc = LabelEncoder().fit(np.concatenate([p["a1"], p["a2"], ext["a1"], ext["a2"]]))
    X = np.concatenate([goldstein_numeric_inputs(p, long_features=True),
                        goldstein_numeric_inputs(ext, long_features=True)], axis=0)
    base = np.concatenate([trailing_mean(p["gold_zero"], 52), trailing_mean(ext["gold_zero"], 52)], axis=0)
    a1 = enc.transform(np.concatenate([p["a1"], ext["a1"]])).astype("int32")
    a2 = enc.transform(np.concatenate([p["a2"], ext["a2"]])).astype("int32")
    y_all = np.concatenate([p["target13"], ext["target13"]], axis=0)
    n_num = X.shape[-1]

    Xtr = windows_at(X, tr, LOOKBACK)
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_num))
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
    y_te_mena, y_va_mena = p["target13"][:, te].T, p["target13"][:, va].T
    n52 = trailing_mean(p["gold_zero"], 52)[:, te].T

    preds = {"va": [], "te": []}
    files = []
    for seed in SEEDS:
        keras.utils.set_random_seed(seed)
        m = build_goldstein_model(n_dyads=400 + N, lookback_weeks=LOOKBACK, n_numeric=n_num,
                                  n_countries=len(enc.classes_), dropout=0.2, residual_baseline=True,
                                  crossdyad_blocks=2)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss=masked_mse)
        m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=100, batch_size=8, shuffle=True, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        out = f"model_release_v2.1_goldstein_s{seed}.weights.h5"
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
    print(f"ENSEMBLE {len(SEEDS)}: val MSE {((ens['va'] - y_va_mena) ** 2).mean():.4f} | test R² {met['r2']:.3f} "
          f"vs52 {met['mse_vs_52']:+.1%} dir {met['dir_acc']:.1%} chg-corr {met['change_corr']:.3f} esc-AUC {met['esc_auc']:.3f}")

    joblib.dump({
        "config": CONFIG, "seeds": SEEDS, "weight_files": files, "scaler": scaler, "country_encoder": enc,
        "count_cols": COUNT_COLS, "raw_cols": ["goldstein_mean"],
        "long_feature_cols": ["gold_zmean26", "gold_zmean52", "log_events_mean52"],
        "n_dyads_per_week": 400 + N, "n_mena_dyads": 400, "lookback_weeks": LOOKBACK,
        "history_weeks_needed": LOOKBACK + 51, "mena_countries": MENA,
        "external_dyads": list(zip(ext["a1"].tolist(), ext["a2"].tolist())),
    }, ARTIFACTS)
    for k, path in (("te", "artifacts/goldstein_v2.1_test_predictions.npz"),
                    ("va", "artifacts/goldstein_v2.1_val_predictions.npz")):
        w = te if k == "te" else va
        np.savez(path, week_idx=w, a1=p["a1"], a2=p["a2"], y_true=p["target13"][:, w].T,
                 pred=ens[k], pred_per_seed=np.stack(preds[k]))
    print(f"saved {', '.join(files)}, {ARTIFACTS} and test/val predictions")


if __name__ == "__main__":
    main()
