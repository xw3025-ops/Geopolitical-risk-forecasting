"""
RELEASE v2.2 - CONFLICT-SURGE FLAG (13-week horizon). Replaces the v2.0
direction model, whose 4-week label (next month's mean goldstein below this
week's) mostly captures bounce-back after unusual weeks and does not rank
crises (test AUC 0.49 on conflict surges, logs/exp/flag_vs_direction.json).

Target: the HIGHEST 4-week mean conflict intensity in the next 13 weeks, where
weekly conflict intensity = log(1 + sum of |negative goldstein|)
(target_variants.py, "peak-conflict"). Hostile events only, so aid, talks and
ceasefire diplomacy cannot mask fighting; a quarter is long enough for
organisations to act.

Model: the v2.1 cross-system transformer (1,959 tokens: 400 MENA dyads +
1,559 MENA-world dyads), residual on the dyad's typical quarterly peak (52-week
mean of its past-quarter peaks), inputs = goldstein inputs + conflict
intensity and summed goldstein with their 52-week means
(target_variants.surge_numeric_inputs), squared error weighted by each dyad's
training-period volatility, 5 seeds. No world data: it did not help this
target (logs/targets.jsonl).

Flag: score = forecast - normal. A logistic calibration fitted on VALIDATION
turns the score into P(conflict surge), where a surge = the next quarter's peak
exceeds the dyad's normal by more than the worst 10% of training-period gaps.

    KERAS_BACKEND=torch python release_v2.2_surge.py
Outputs:
    model_release_v2.2_surge_s<seed>.weights.h5
    artifacts/surge_v2.2_preprocessing.joblib
    artifacts/surge_v2.2_test_predictions.npz, artifacts/surge_v2.2_val_predictions.npz
"""

import joblib
import keras
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import target_variants as tv
from benchmark_targets import make_context, neg_intensity, summary_line
from experiment_scaling import external_pool
from experiment_targets import weighted_masked_mse
from honest_common import MENA, windows_at
from model_architectures import build_goldstein_model

SEEDS = [42, 1, 2, 3, 4]
LOOKBACK = 13
TARGET = "peak-conflict"
CONFIG = dict(target=TARGET, lookback=LOOKBACK, residual=True, dropout=0.2, lr=3e-4, plateau=True,
              crossdyad_blocks=2, min_active_external=0.50, loss_weight="training-period volatility",
              inputs="surge_numeric_inputs")
ARTIFACTS = "artifacts/surge_v2.2_preprocessing.joblib"


def main():
    ctx = make_context()
    p, tr, va, te = ctx.p, ctx.tr, ctx.va, ctx.te
    p["neg"] = neg_intensity(p).astype("float32")
    ext = external_pool(p["weeks"], 0.50)
    N = len(ext["a1"])
    n_tok = 400 + N
    tm, txt = ctx.T[TARGET], tv.build(TARGET, ext["gold_zero"], ext["neg"])
    y_all = np.concatenate([tm["y"], txt["y"]], 0)
    base = np.concatenate([tm["normal"], txt["normal"]], 0)
    X = np.concatenate([tv.surge_numeric_inputs(p), tv.surge_numeric_inputs(ext)], 0)
    n_num = X.shape[-1]
    enc = LabelEncoder().fit(np.concatenate([p["a1"], p["a2"], ext["a1"], ext["a2"]]))
    a1 = enc.transform(np.concatenate([p["a1"], ext["a1"]])).astype("int32")
    a2 = enc.transform(np.concatenate([p["a2"], ext["a2"]])).astype("int32")

    ch = (y_all - base)[:, tr]
    sd = np.sqrt(np.nanmean((ch - np.nanmean(ch, 1, keepdims=True)) ** 2, 1))
    w_tok = np.nan_to_num(np.clip(sd / np.nanmean(sd), 0.2, 5.0), nan=1.0).astype("float32")

    Xtr = windows_at(X, tr, LOOKBACK)
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_num))
    sc = lambda A: np.nan_to_num(scaler.transform(A.reshape(-1, n_num)).reshape(A.shape)).astype("float32")

    def inputs(w, Xw=None):
        return {"numeric_seq": sc(windows_at(X, w, LOOKBACK) if Xw is None else Xw),
                "actor1_idx": np.tile(a1, (len(w), 1)), "actor2_idx": np.tile(a2, (len(w), 1)),
                "baseline": np.nan_to_num(base[:, w].T).astype("float32")}

    def targets(w, grade_ext):
        y = y_all[:, w].T.astype("float32").copy()
        if not grade_ext:
            y[:, 400:] = np.nan
        return np.concatenate([y, np.tile(w_tok, (len(w), 1))], 1)

    x_tr, x_va, x_te = inputs(tr, Xtr), inputs(va), inputs(te)
    y_tr, y_va = targets(tr, True), targets(va, False)
    del Xtr

    preds, files = {"va": [], "te": []}, []
    for seed in SEEDS:
        keras.utils.set_random_seed(seed)
        m = build_goldstein_model(n_dyads=n_tok, lookback_weeks=LOOKBACK, n_numeric=n_num,
                                  n_countries=len(enc.classes_), dropout=0.2, residual_baseline=True,
                                  crossdyad_blocks=2)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss=weighted_masked_mse(n_tok))
        m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=100, batch_size=8, shuffle=True, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        out = f"model_release_v2.2_surge_s{seed}.weights.h5"
        m.save_weights(out)
        files.append(out)
        preds["va"].append(m.predict(x_va, verbose=0)[:, :400])
        preds["te"].append(m.predict(x_te, verbose=0)[:, :400])
        print(f"seed {seed}: test conflict-surge AUC "
              f"{ctx.evaluate(TARGET, preds['te'][-1])['auc_conflict-surge']:.3f}", flush=True)
    ens = {k: np.mean(v, 0) for k, v in preds.items()}

    # calibration on validation: P(surge) = sigmoid(a * score + b), score = forecast - normal
    score = {k: ens[k] - tm["normal"][:, w].T for k, w in (("va", va), ("te", te))}
    lab = {k: ctx.labels["conflict-surge"][:, w].T for k, w in (("va", va), ("te", te))}
    cal = LogisticRegression().fit(score["va"].reshape(-1, 1), lab["va"].ravel())
    prob_te = cal.predict_proba(score["te"].reshape(-1, 1))[:, 1].reshape(score["te"].shape)
    top = np.argsort(-prob_te, 1)[:, :10]
    hit = np.take_along_axis(lab["te"], top, 1).mean()
    met = ctx.evaluate(TARGET, ens["te"])
    print("TEST" + summary_line("surge v2.2 (5 seeds)", met))
    print(f"calibrated: mean P {prob_te.mean():.3f} vs surge rate {lab['te'].mean():.3f}; "
          f"AUC {roc_auc_score(lab['te'].ravel(), prob_te.ravel()):.3f}; top-10 weekly alerts hit {hit:.0%}")

    joblib.dump({
        "config": CONFIG, "seeds": SEEDS, "weight_files": files, "scaler": scaler, "country_encoder": enc,
        "n_dyads_per_week": n_tok, "n_mena_dyads": 400, "lookback_weeks": LOOKBACK,
        "history_weeks_needed": 66,  # normal = 52-week mean of past-13-week peaks of 4-week means
        "mena_countries": MENA,
        "external_dyads": list(zip(ext["a1"].tolist(), ext["a2"].tolist())),
        "surge_threshold": ctx.esc[TARGET], "calibration": dict(a=float(cal.coef_[0, 0]), b=float(cal.intercept_[0])),
        "test_metrics": dict(met, top10_hit=float(hit)),
    }, ARTIFACTS)
    for k, w in (("te", te), ("va", va)):
        np.savez(f"artifacts/surge_v2.2_{'test' if k == 'te' else 'val'}_predictions.npz", week_idx=w,
                 a1=p["a1"], a2=p["a2"], y_true=tm["y"][:, w].T, normal=tm["normal"][:, w].T,
                 pred=ens[k], pred_per_seed=np.stack(preds[k]))
    print(f"saved {', '.join(files)}, {ARTIFACTS} and test/val predictions")


if __name__ == "__main__":
    main()
