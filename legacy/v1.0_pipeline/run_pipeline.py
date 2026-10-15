"""
The full live-prediction pipeline: load the freshly-pulled data (see
pull_latest_data.py), preprocess it with the SAME scaler/encoder fitted
during training (see export_artifacts_*.py), run both release v1.0 models,
and write one combined, tabular results file.

Run (after pull_latest_data.py has been run at least once):
    python run_pipeline.py
Output:
    predictions/latest_predictions.csv
"""

import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from model_architectures import build_goldstein_model, build_direction_model

LATEST_DATA_PATH = "raw data/latest_pull.parquet"
GOLDSTEIN_ARTIFACTS = "artifacts/goldstein_preprocessing.joblib"
DIRECTION_ARTIFACTS = "artifacts/direction_preprocessing.joblib"
GOLDSTEIN_WEIGHTS = "model_release_v1.0_goldstein.weights.h5"
DIRECTION_WEIGHTS = "model_release_v1.0_direction.weights.h5"
OUT_DIR = "predictions"


def latest_window(df, countries, lookback_weeks):
    """Filter to `countries`, sort into dyad-major order, and return the
    last `lookback_weeks` weeks for every dyad as (n_dyads, lookback, ...),
    plus the per-dyad actor codes and the as-of week."""
    sub = df[df.Actor1CountryCode.isin(countries) & df.Actor2CountryCode.isin(countries)]
    sub = sub.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
    weeks = sorted(sub.week_start.unique())
    n_weeks = len(weeks)
    n_dyads = len(countries) ** 2
    per_dyad_counts = sub.groupby(["Actor1CountryCode", "Actor2CountryCode"]).size()
    assert (per_dyad_counts == n_weeks).all(), "expected a fully dense panel in the pulled buffer"
    assert len(per_dyad_counts) == n_dyads
    as_of_week = weeks[-1]
    return sub, n_dyads, n_weeks, as_of_week


def build_numeric(sub, log_cols, raw_cols, n_dyads, n_weeks, lookback_weeks):
    numeric = np.concatenate([
        np.log1p(sub[log_cols].to_numpy(dtype="float32")),
        sub[raw_cols].to_numpy(dtype="float32"),
    ], axis=1)
    n_numeric = numeric.shape[1]
    numeric = numeric.reshape(n_dyads, n_weeks, n_numeric)
    window = numeric[:, -lookback_weeks:, :]  # most recent `lookback_weeks` weeks
    return window, n_numeric


def main():
    df = pd.read_parquet(LATEST_DATA_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])

    # ---------------- Goldstein (regression, MENA-only) ----------------
    g_art = joblib.load(GOLDSTEIN_ARTIFACTS)
    sub, n_dyads, n_weeks, as_of_week_g = latest_window(df, g_art["mena_countries"], g_art["lookback_weeks"])

    dyad_actor1 = sub.Actor1CountryCode.to_numpy().reshape(n_dyads, n_weeks)[:, 0]
    dyad_actor2 = sub.Actor2CountryCode.to_numpy().reshape(n_dyads, n_weeks)[:, 0]

    # the old training pipeline called this column "avg_goldstein"; the
    # BigQuery live-pull source calls the same quantity "goldstein_mean"
    sub_renamed = sub.rename(columns={"goldstein_mean": "avg_goldstein"})
    window, n_numeric_g = build_numeric(
        sub_renamed, g_art["count_cols"], g_art["raw_cols"], n_dyads, n_weeks, g_art["lookback_weeks"]
    )
    window_scaled = g_art["scaler"].transform(window.reshape(-1, n_numeric_g)).reshape(window.shape).astype("float32")

    a1_idx = g_art["country_encoder"].transform(dyad_actor1).astype("int32")[np.newaxis, :]
    a2_idx = g_art["country_encoder"].transform(dyad_actor2).astype("int32")[np.newaxis, :]

    goldstein_model = build_goldstein_model(
        n_dyads=n_dyads, lookback_weeks=g_art["lookback_weeks"], n_numeric=n_numeric_g,
        n_countries=len(g_art["country_encoder"].classes_),
    )
    goldstein_model.load_weights(GOLDSTEIN_WEIGHTS)
    goldstein_pred = goldstein_model.predict(
        {"numeric_seq": window_scaled[np.newaxis, ...], "actor1_idx": a1_idx, "actor2_idx": a2_idx}, verbose=0
    )[0]

    goldstein_df = pd.DataFrame({
        "actor1": dyad_actor1, "actor2": dyad_actor2,
        "predicted_avg_goldstein_next13w": goldstein_pred,
    })

    # ---------------- Direction (classification, 22-country universe) ----------------
    d_art = joblib.load(DIRECTION_ARTIFACTS)
    all_countries = d_art["mena_countries"] + d_art["extra_countries"]
    sub_d, n_dyads_d, n_weeks_d, as_of_week_d = latest_window(df, all_countries, d_art["lookback_weeks"])

    dyad_actor1_d = sub_d.Actor1CountryCode.to_numpy().reshape(n_dyads_d, n_weeks_d)[:, 0]
    dyad_actor2_d = sub_d.Actor2CountryCode.to_numpy().reshape(n_dyads_d, n_weeks_d)[:, 0]

    window_d, n_numeric_d = build_numeric(
        sub_d, d_art["log_cols"], d_art["raw_cols"], n_dyads_d, n_weeks_d, d_art["lookback_weeks"]
    )
    window_d_scaled = d_art["scaler"].transform(
        window_d.reshape(-1, n_numeric_d)
    ).reshape(window_d.shape).astype("float32")

    a1_idx_d = d_art["country_encoder"].transform(dyad_actor1_d).astype("int32")[np.newaxis, :]
    a2_idx_d = d_art["country_encoder"].transform(dyad_actor2_d).astype("int32")[np.newaxis, :]

    direction_model = build_direction_model(
        n_dyads=n_dyads_d, lookback_weeks=d_art["lookback_weeks"], n_numeric=n_numeric_d,
        n_countries=len(d_art["country_encoder"].classes_), relation_matrix=d_art["relation_matrix"],
    )
    direction_model.load_weights(DIRECTION_WEIGHTS)
    direction_probs = direction_model.predict(
        {"numeric_seq": window_d_scaled[np.newaxis, ...], "actor1_idx": a1_idx_d, "actor2_idx": a2_idx_d}, verbose=0
    )[0]  # (n_dyads_d, 2)

    direction_df = pd.DataFrame({
        "actor1": dyad_actor1_d, "actor2": dyad_actor2_d,
        # v1.0 label 1 = goldstein ROSE = relations IMPROVE (it used to be called "escalation")
        "predicted_direction": np.where(direction_probs[:, 1] >= 0.5, "improve", "worsen_or_flat"),
        "improve_probability": direction_probs[:, 1],
    })
    # Only the original MENA<->MENA dyads were ever graded - USA/CHN dyads
    # are context-only and their own predictions were never trained to be
    # meaningful, so they're dropped here.
    mena_set = set(g_art["mena_countries"])
    direction_df = direction_df[
        direction_df.actor1.isin(mena_set) & direction_df.actor2.isin(mena_set)
    ].reset_index(drop=True)

    # ---------------- Combine ----------------
    assert as_of_week_g == as_of_week_d, "goldstein and direction pulls disagree on the most recent week"
    results = goldstein_df.merge(direction_df, on=["actor1", "actor2"], how="inner")
    results.insert(0, "as_of_week", as_of_week_g.date())
    results.insert(1, "generated_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    results = results.sort_values("improve_probability", ascending=True).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "latest_predictions.csv")
    results.to_csv(out_path, index=False)

    print(f"As-of week: {as_of_week_g.date()}")
    print(f"Wrote {len(results)} dyad predictions to {out_path}")
    print("\nMost likely to WORSEN or stay flat (lowest improve probability):")
    print(results.head(10).to_string(index=False))
    print("\nMost likely to IMPROVE:")
    print(results.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
