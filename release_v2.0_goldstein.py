"""
RELEASE v2.0 - GOLDSTEIN (regression, 13-week horizon): an average of
SEEDS identically-configured SwiGLU spatio-temporal transformers (same
architecture family as v1.0: temporal self-attention over each dyad's
trailing weeks -> country embeddings -> cross-dyad self-attention).

Data and evaluation (honest_common.py):
  - all 20 MENA countries (400 dyads) with real data from the BigQuery panel
    (v1.0 trained on a CSV that lacked LBY and SDN entirely)
  - 13-week EMBARGO before validation and test, so no training row's
    13-week-ahead target overlaps the evaluation period
  - test weeks identical to v1.0 (2024-03-18..2026-06-22)

What changed from v1.0's model, each chosen on VALIDATION loss in
experiment_goldstein.py (see logs/goldstein_experiments.jsonl):
  - residual output: the network predicts the correction to the dyad's
    trailing 52-week mean goldstein, output = baseline + network. The
    trailing mean is the strongest simple forecast; the network only has to
    learn when and how far the future departs from it.
  - long-memory inputs: every weekly token also carries the trailing 26- and
    52-week mean goldstein and the log 52-week mean event count - history
    the attention window itself can't see.
  - more regularization: dropout 0.2 (was 0.1), Adam lr 3e-4 (was 1e-3),
    learning rate halved when validation loss stalls for 3 epochs.
  - seed ensemble: SEEDS models averaged; single runs vary by ~0.01 R^2.

Target (y): target_avg_goldstein_next13w (continuous, regression, MSE loss)
Benchmarked against simple baselines and gradient boosting by
benchmark_honest.py, which reads this script's saved test predictions.

Run:
    python release_v2.0_goldstein.py
    (reuses artifacts/experiments/<tag>.weights.h5 for any seed already
    trained by experiment_goldstein.py with this exact config)
Outputs:
    model_release_v2.0_goldstein_s<seed>.weights.h5   (one per seed)
    artifacts/goldstein_v2.0_preprocessing.joblib
    artifacts/goldstein_v2.0_test_predictions.npz
"""

import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import experiment_goldstein as eg
from honest_common import COUNT_COLS, MENA

CONFIG = dict(lookback=13, residual=True, baseline_weeks=52, long_features=True, has_events=False,
              dropout=0.2, lr=3e-4, plateau=True)
SEEDS = [42, 1, 2, 3, 4]

ARTIFACTS_OUT = "artifacts/goldstein_v2.0_preprocessing.joblib"
PREDICTIONS_OUT = "artifacts/goldstein_v2.0_test_predictions.npz"


def main():
    args = eg.config(**CONFIG)
    data = eg.prepare_data(args)
    Y, B, idx, wk = data["Y"], data["B"], data["idx"], data["p"]["weeks"]
    print(f"{data['n_dyads']} dyads, {data['n_numeric']} input features, lookback {args.lookback} weeks")
    print(f"train {wk[idx['tr'][0]].date()}..{wk[idx['tr'][-1]].date()} ({len(idx['tr'])} weeks) | "
          f"val {wk[idx['va'][0]].date()}..{wk[idx['va'][-1]].date()} ({len(idx['va'])}) | "
          f"test {wk[idx['te'][0]].date()}..{wk[idx['te'][-1]].date()} ({len(idx['te'])}) | 13-week embargo")

    preds = {"va": [], "te": []}
    weight_files = []
    for seed in SEEDS:
        args.seed = seed
        cached = f"artifacts/experiments/{eg.make_tag(args)}.weights.h5"
        model = eg.build(args, data)
        if os.path.exists(cached):
            model.load_weights(cached)
            print(f"seed {seed}: loaded {cached}")
        else:
            print(f"seed {seed}: training")
            model, _ = eg.train(args, data)
        out = f"model_release_v2.0_goldstein_s{seed}.weights.h5"
        model.save_weights(out)
        weight_files.append(out)
        for k in preds:
            preds[k].append(model.predict(data["inputs"](k), verbose=0))

    def report(name, pv, pt):
        print(f"  {name:<24} val MSE {mean_squared_error(Y['va'].ravel(), pv.ravel()):.4f}   "
              f"test R^2 {r2_score(Y['te'].ravel(), pt.ravel()):.4f}   "
              f"MAE {mean_absolute_error(Y['te'].ravel(), pt.ravel()):.4f}")

    print(f"\nTest: {len(idx['te'])} weeks x {data['n_dyads']} dyads")
    for seed, pv, pt in zip(SEEDS, preds["va"], preds["te"]):
        report(f"seed {seed}", pv, pt)
    ens = {k: np.mean(v, axis=0) for k, v in preds.items()}
    report(f"ENSEMBLE of {len(SEEDS)}", ens["va"], ens["te"])
    report(f"trailing {args.baseline_weeks}-week mean", B["va"], B["te"])

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({
        "config": CONFIG,
        "seeds": SEEDS,
        "weight_files": weight_files,
        "scaler": data["scaler"],
        "country_encoder": data["encoder"],
        "count_cols": COUNT_COLS,
        "raw_cols": ["goldstein_mean"],
        "long_feature_cols": ["gold_zmean26", "gold_zmean52", "log_events_mean52"] if args.long_features else [],
        "n_dyads_per_week": data["n_dyads"],
        "lookback_weeks": args.lookback,
        "history_weeks_needed": max(args.lookback + (51 if args.long_features else 0), args.baseline_weeks),
        "mena_countries": MENA,
    }, ARTIFACTS_OUT)
    np.savez(PREDICTIONS_OUT, week_idx=idx["te"], a1=data["p"]["a1"], a2=data["p"]["a2"],
             y_true=Y["te"], pred=ens["te"], pred_per_seed=np.stack(preds["te"]))
    print(f"\nSaved {', '.join(weight_files)}, {ARTIFACTS_OUT}, {PREDICTIONS_OUT}")


if __name__ == "__main__":
    main()
