"""
RELEASE v2.0 - DIRECTION (escalation / de-escalation, 4-week horizon): an
average of several identically-configured SwiGLU spatio-temporal
transformers with structural-bias cross-dyad attention (v1.0's architecture,
built by model_architectures.build_direction_model). Universe: 20 MENA +
USA + CHN = 484 dyads; USA/CHN dyads are attention context only - the loss
and metrics grade the 400 MENA<->MENA dyads. Full BigQuery feature set.

Changes from v1.0:
  1. LABEL DIRECTION FIXED. Escalation means goldstein FALLS: label = 1 if
     the mean goldstein over the next 4 weeks is LOWER than this week's
     (honest_common.escalation_label). v1.0 had it inverted. Flat = 0.
  2. 4-week EMBARGO before validation and test (a row's target averages
     weeks t+1..t+4). Test weeks unchanged from v1.0 (2024-05-27..2026-08-24).
  3. The recipe that made release v2.0 goldstein beat its baselines, applied
     directly (no separate sweep): long-memory inputs on every weekly token
     (trailing 26/52-week mean goldstein, log 52-week mean event count),
     dropout 0.2, Adam lr 3e-4 halved when val loss stalls, and an average
     over SEEDS models.

Benchmarked against simple baselines and gradient boosting by
benchmark_honest.py, which reads this script's saved test predictions.

Run (seeds can train in parallel, then combine):
    python release_v2.0_direction.py --seed 42      # one model -> artifacts/experiments/
    python release_v2.0_direction.py --ensemble     # average the trained seeds
Outputs:
    model_release_v2.0_direction_s<seed>.weights.h5   (one per seed)
    artifacts/direction_v2.0_preprocessing.joblib
    artifacts/direction_v2.0_test_predictions.npz
"""

import argparse
import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

from honest_common import (DIRECTION_HORIZON, DIRECTION_TEST_START, DIRECTION_VAL_START, EXTRA,
                           LOOKBACK_WEEKS, MENA, direction_feature_columns, direction_numeric_inputs,
                           load_panel, split_by_dates, windows_at)
from model_architectures import build_direction_model

SEEDS = [42, 1, 2]
DROPOUT = 0.2
LR = 3e-4
MAX_EPOCHS = 40
PATIENCE = 5

ARTIFACTS_OUT = "artifacts/direction_v2.0_preprocessing.joblib"
PREDICTIONS_OUT = "artifacts/direction_v2.0_test_predictions.npz"
EXPERIMENT_DIR = "artifacts/experiments"

LONG_FEATURE_COLS = ["gold_zmean26", "gold_zmean52", "log_events_mean52"]


def build_relation_matrix(a1, a2):
    """Structural relation between every pair of dyads, same precedence as
    v1.0: self, reverse, same_source, same_target, forward_chain,
    backward_chain, unrelated."""
    A1i, A2i = a1[:, None], a2[:, None]
    A1j, A2j = a1[None, :], a2[None, :]
    conds = [(A1i == A1j) & (A2i == A2j), (A1i == A2j) & (A2i == A1j),
             A1i == A1j, A2i == A2j, A2i == A1j, A1i == A2j]
    return np.select(conds, np.arange(len(conds)), default=len(conds)).astype("int32")


def prepare_data():
    countries = MENA + EXTRA
    p = load_panel(countries)
    D = len(p["a1"])
    log_cols, raw_cols = direction_feature_columns(p["df"])
    numeric = direction_numeric_inputs(p, log_cols, raw_cols)
    n_numeric = numeric.shape[-1]

    tr, va, te = split_by_dates(p["weeks"], p["target4"][0], DIRECTION_VAL_START, DIRECTION_TEST_START,
                                DIRECTION_HORIZON)
    idx = {"tr": tr, "va": va, "te": te}
    X = {k: windows_at(numeric, w) for k, w in idx.items()}
    label = {k: p["escalation"][:, w].T for k, w in idx.items()}  # (weeks, dyads)

    scaler = StandardScaler().fit(X["tr"].reshape(-1, n_numeric))
    for k in X:
        X[k] = scaler.transform(X[k].reshape(-1, n_numeric)).reshape(X[k].shape).astype("float32")

    enc = LabelEncoder().fit(countries)
    a1 = enc.transform(p["a1"]).astype("int32")
    a2 = enc.transform(p["a2"]).astype("int32")

    def inputs(k):
        return {"numeric_seq": X[k], "actor1_idx": np.tile(a1, (len(X[k]), 1)),
                "actor2_idx": np.tile(a2, (len(X[k]), 1))}

    graded = np.isin(p["a1"], MENA) & np.isin(p["a2"], MENA)
    return dict(p=p, idx=idx, inputs=inputs, label=label, scaler=scaler, encoder=enc,
                log_cols=log_cols, raw_cols=raw_cols, n_numeric=n_numeric, n_dyads=D,
                graded=graded, relation_matrix=build_relation_matrix(p["a1"], p["a2"]))


def build(data):
    return build_direction_model(n_dyads=data["n_dyads"], lookback_weeks=LOOKBACK_WEEKS,
                                 n_numeric=data["n_numeric"], n_countries=len(data["encoder"].classes_),
                                 relation_matrix=data["relation_matrix"])


def train_seed(seed, data):
    keras.utils.set_random_seed(seed)
    mask_tf = tf.constant(data["graded"].astype("float32"))

    def masked_cce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        per_dyad = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return tf.reduce_sum(per_dyad * mask_tf[tf.newaxis, :], axis=-1) / tf.reduce_sum(mask_tf)

    def masked_accuracy(y_true, y_pred):
        correct = tf.cast(tf.equal(tf.argmax(y_pred, -1), tf.argmax(y_true, -1)), tf.float32)
        return tf.reduce_sum(correct * mask_tf[tf.newaxis, :], axis=-1) / tf.reduce_sum(mask_tf)

    model = build(data)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss=masked_cce, metrics=[masked_accuracy])
    onehot = {k: keras.utils.to_categorical(v, 2) for k, v in data["label"].items()}
    model.fit(data["inputs"]("tr"), onehot["tr"], validation_data=(data["inputs"]("va"), onehot["va"]),
              epochs=MAX_EPOCHS, batch_size=8, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                       restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)])
    return model


def scores(prob, true):
    """accuracy, ROC-AUC, log loss on graded dyads; prob/true: (weeks, graded dyads)"""
    p = np.clip(prob.ravel(), 1e-7, 1 - 1e-7)
    t = true.ravel()
    return ((p >= 0.5) == t).mean(), roc_auc_score(t, p), -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))


def run_seed(seed):
    data = prepare_data()
    g = data["graded"]
    print(f"{data['n_dyads']} dyads ({int(g.sum())} graded), {data['n_numeric']} features, seed {seed}")
    model = train_seed(seed, data)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    prob = {k: model.predict(data["inputs"](k), verbose=0)[:, :, 1][:, g] for k in ("va", "te")}
    for k in prob:
        acc, auc, ll = scores(prob[k], data["label"][k][:, g])
        print(f"  {k}: accuracy {acc:.4f}   ROC-AUC {auc:.4f}   log loss {ll:.4f}")
    model.save_weights(f"{EXPERIMENT_DIR}/direction_v2.0_s{seed}.weights.h5")
    np.savez(f"{EXPERIMENT_DIR}/direction_v2.0_s{seed}.npz", va_prob=prob["va"], te_prob=prob["te"])


def run_ensemble():
    data = prepare_data()
    g, wk, idx = data["graded"], data["p"]["weeks"], data["idx"]
    print(f"train {wk[idx['tr'][0]].date()}..{wk[idx['tr'][-1]].date()} | "
          f"val {wk[idx['va'][0]].date()}..{wk[idx['va'][-1]].date()} | "
          f"test {wk[idx['te'][0]].date()}..{wk[idx['te'][-1]].date()} ({len(idx['te'])} weeks) | "
          f"{DIRECTION_HORIZON}-week embargo")
    probs = {"va": [], "te": []}
    weight_files = []
    for seed in SEEDS:
        z = np.load(f"{EXPERIMENT_DIR}/direction_v2.0_s{seed}.npz")
        for k in probs:
            probs[k].append(z[f"{k}_prob"])
        model = build(data)
        model.load_weights(f"{EXPERIMENT_DIR}/direction_v2.0_s{seed}.weights.h5")
        out = f"model_release_v2.0_direction_s{seed}.weights.h5"
        model.save_weights(out)
        weight_files.append(out)

    true = {k: data["label"][k][:, g] for k in probs}
    print(f"\n{int(g.sum())} MENA<->MENA dyads:")
    rows = [(f"seed {s}", probs["va"][i], probs["te"][i]) for i, s in enumerate(SEEDS)]
    ens = {k: np.mean(v, axis=0) for k, v in probs.items()}
    rows.append((f"ENSEMBLE of {len(SEEDS)}", ens["va"], ens["te"]))
    for name, pv, pt in rows:
        va_acc, _, va_ll = scores(pv, true["va"])
        te_acc, te_auc, _ = scores(pt, true["te"])
        print(f"  {name:<14} val acc {va_acc:.4f}  log loss {va_ll:.4f}   |   "
              f"test acc {te_acc:.4f}  ROC-AUC {te_auc:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({
        "seeds": SEEDS,
        "weight_files": weight_files,
        "scaler": data["scaler"],
        "country_encoder": data["encoder"],
        "log_cols": data["log_cols"],
        "raw_cols": data["raw_cols"],
        "long_feature_cols": LONG_FEATURE_COLS,
        "relation_matrix": data["relation_matrix"],
        "n_dyads_per_week": data["n_dyads"],
        "lookback_weeks": LOOKBACK_WEEKS,
        "history_weeks_needed": LOOKBACK_WEEKS + 51,
        "mena_countries": MENA,
        "extra_countries": EXTRA,
        "label": "1 = escalation = next-4-week mean goldstein LOWER than this week's",
    }, ARTIFACTS_OUT)
    p = data["p"]
    np.savez(PREDICTIONS_OUT, week_idx=idx["te"], a1=p["a1"][g], a2=p["a2"][g],
             y_true=true["te"], prob=ens["te"], prob_per_seed=np.stack(probs["te"]))
    print(f"\nSaved {', '.join(weight_files)}, {ARTIFACTS_OUT}, {PREDICTIONS_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    if a.ensemble:
        run_ensemble()
    elif a.seed is not None:
        run_seed(a.seed)
    else:
        ap.error("pass --seed S to train one model, or --ensemble to combine the trained seeds")
