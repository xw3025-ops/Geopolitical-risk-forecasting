"""
Live inference for release v2.0 - the ONE implementation used by both the
Cloud Function (cloud_function/main.py) and local runs (run_pipeline.py).

Flow:
  1. as_of = the last COMPLETE week (Monday..Sunday). The week in progress is
     never used: a half-finished week has too few events and would be read
     as a sudden drop in activity.
  2. Pull HISTORY_WEEKS complete weeks of dyad-week data from BigQuery table
     geopolitical-risk-forecasting.1.1 (via the BigQuery API), same columns
     and null handling as the training pulls (build_bq_panel_22*.py).
  3. Build model inputs with the same functions the release scripts trained
     with (honest_common.panel_from_df / *_numeric_inputs), scale with the
     saved training scalers, run every seed model of each release and
     average them.
  4. Return one row per MENA<->MENA dyad (400).

Output columns
  model_version, as_of_week (Monday of the last complete week), generated_at_utc,
  actor1, actor2,
  goldstein_forecast_next13w   - mean goldstein over the 13 weeks after as_of_week
  goldstein_trailing52w        - the dyad's recent normal, for context
  escalation_probability_next4w, direction_next4w
                               - escalation = mean goldstein over the next 4
                                 weeks LOWER than as_of_week's (relations worsening)
"""

import os
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from honest_common import (COUNT_COLS, EXTRA, MENA, direction_numeric_inputs, goldstein_numeric_inputs,
                           panel_from_df, trailing_mean)
from model_architectures import build_direction_model, build_goldstein_model

MODEL_VERSION = "v2.1"  # goldstein v2.1 (cross-system attention) + direction v2.0
GOLDSTEIN_ARTIFACTS = "goldstein_v2.1_preprocessing.joblib"
DIRECTION_ARTIFACTS = "direction_v2.0_preprocessing.joblib"

SOURCE_TABLE = "geopolitical-risk-forecasting.1.1"
HISTORY_WEEKS = 70  # both models need 64 (13-week window + 52-week averages); 6 spare


# ---------------------------------------------------------------- dates

def last_complete_week(today=None):
    """Monday of the most recent Monday..Sunday week that has fully ended
    (UTC). On any day of week W, that's the Monday of week W-1."""
    today = today or datetime.now(timezone.utc).date()
    this_monday = today - timedelta(days=today.weekday())
    return this_monday - timedelta(days=7)


# ---------------------------------------------------------------- data

def pull_panel(bq_client, as_of_week, n_weeks=HISTORY_WEEKS, table=SOURCE_TABLE):
    """The n_weeks complete weeks ending with as_of_week, for the 22-country
    universe, as a dense dyad-week frame (USA/CHN<->USA/CHN pairs, which the
    table never contains, zero-padded exactly as in training)."""
    from google.cloud import bigquery

    schema = bq_client.get_table(table).schema
    numeric_cols = [f.name for f in schema if f.field_type in ("FLOAT", "FLOAT64", "INTEGER", "INT64")]
    first_week = as_of_week - timedelta(weeks=n_weeks - 1)
    query = f"""
    SELECT week_start, Actor1CountryCode, Actor2CountryCode,
           {', '.join(f'COALESCE({c}, 0) AS {c}' for c in numeric_cols)}
    FROM `{table}`
    WHERE week_start BETWEEN @first_week AND @as_of_week
      AND (dyad_scope = 'INTRA_REGION'
           OR (Actor1CountryCode IN UNNEST(@extra) AND Actor2CountryCode IN UNNEST(@mena))
           OR (Actor2CountryCode IN UNNEST(@extra) AND Actor1CountryCode IN UNNEST(@mena)))
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("first_week", "DATE", first_week),
        bigquery.ScalarQueryParameter("as_of_week", "DATE", as_of_week),
        bigquery.ArrayQueryParameter("mena", "STRING", MENA),
        bigquery.ArrayQueryParameter("extra", "STRING", EXTRA),
    ])
    df = bq_client.query(query, job_config=job_config).to_dataframe()
    df["week_start"] = pd.to_datetime(df["week_start"])
    return pad_extra_pairs(df, numeric_cols)


def pad_extra_pairs(df, numeric_cols):
    weeks = sorted(df.week_start.unique())
    pads = []
    for a1 in EXTRA:
        for a2 in EXTRA:
            block = pd.DataFrame({"week_start": weeks, "Actor1CountryCode": a1, "Actor2CountryCode": a2})
            for c in numeric_cols:
                block[c] = 0
            pads.append(block)
    return pd.concat([df] + pads, ignore_index=True)


def pull_external(bq_client, as_of_week, pairs, n_weeks=HISTORY_WEEKS, table=SOURCE_TABLE):
    """The MENA<->rest-of-world dyads the v2.1 goldstein model attends over
    (`pairs` = [(actor1, actor2), ...] from its artifacts), same weeks and
    null handling as pull_panel. Only the columns its inputs use."""
    from google.cloud import bigquery

    cols = COUNT_COLS + ["goldstein_mean", "total_count"]
    first_week = as_of_week - timedelta(weeks=n_weeks - 1)
    query = f"""
    SELECT week_start, Actor1CountryCode, Actor2CountryCode,
           {', '.join(f'COALESCE({c}, 0) AS {c}' for c in cols)}
    FROM `{table}`
    WHERE week_start BETWEEN @first_week AND @as_of_week
      AND dyad_scope = 'EXTERNAL_LINK'
      AND CONCAT(Actor1CountryCode, '|', Actor2CountryCode) IN UNNEST(@pairs)
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("first_week", "DATE", first_week),
        bigquery.ScalarQueryParameter("as_of_week", "DATE", as_of_week),
        bigquery.ArrayQueryParameter("pairs", "STRING", [f"{a}|{b}" for a, b in pairs]),
    ])
    df = bq_client.query(query, job_config=job_config).to_dataframe()
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def external_panel(ext_df, pairs, weeks):
    """Dense arrays for the external dyads over `weeks` (a pair-week with no
    row counts as no events, as in training)."""
    full = pd.MultiIndex.from_product([[f"{a}|{b}" for a, b in pairs], pd.DatetimeIndex(weeks)])
    e = ext_df.assign(key=ext_df.Actor1CountryCode + "|" + ext_df.Actor2CountryCode)
    e = e.set_index(["key", "week_start"]).reindex(full)
    D, T = len(pairs), len(weeks)
    return dict(a1=np.array([a for a, _ in pairs]), a2=np.array([b for _, b in pairs]),
                counts=np.nan_to_num(e[COUNT_COLS].to_numpy(dtype="float32")).reshape(D, T, 20),
                gold_zero=np.nan_to_num(e["goldstein_mean"].to_numpy(dtype="float32")).reshape(D, T))


def check_panel(df, as_of_week, n_weeks=HISTORY_WEEKS):
    """Fail loudly on an incomplete pull; warn if the as-of week looks
    under-ingested (far fewer events than the weeks before it)."""
    weeks = pd.DatetimeIndex(sorted(df.week_start.unique()))
    if len(weeks) != n_weeks or weeks[-1] != pd.Timestamp(as_of_week):
        raise ValueError(f"expected {n_weeks} weeks ending {as_of_week}, got {len(weeks)} "
                         f"({weeks[0].date()}..{weeks[-1].date()})")
    n_dyads = (len(MENA) + len(EXTRA)) ** 2
    per_week = df.groupby("week_start").size()
    if (per_week != n_dyads).any():
        raise ValueError(f"panel not dense: dyads per week {per_week.min()}..{per_week.max()}, "
                         f"expected {n_dyads}")
    mena = df[df.Actor1CountryCode.isin(MENA) & df.Actor2CountryCode.isin(MENA)]
    events = mena.groupby("week_start").total_count.sum()
    ratio = events.iloc[-1] / events.iloc[-9:-1].median()
    warning = None
    if ratio < 0.6:
        warning = (f"as-of week {as_of_week} has only {ratio:.0%} of the usual event volume - "
                   f"GDELT may still be ingesting it")
    return {"as_of_events": int(events.iloc[-1]), "as_of_vs_median8": float(ratio), "warning": warning}


# ---------------------------------------------------------------- models

def _find(model_dir, name):
    """model_dir/name, or model_dir/artifacts/name (the repo layout)."""
    for path in (os.path.join(model_dir, name), os.path.join(model_dir, "artifacts", name)):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found in {model_dir}")


def load_models(model_dir):
    """Rebuild every seed model of both releases and load their weights."""
    g = joblib.load(_find(model_dir, GOLDSTEIN_ARTIFACTS))
    d = joblib.load(_find(model_dir, DIRECTION_ARTIFACTS))
    cfg = g["config"]
    g_models = []
    for f in g["weight_files"]:
        m = build_goldstein_model(n_dyads=g.get("n_dyads_per_week", len(MENA) ** 2),
                                  lookback_weeks=g["lookback_weeks"], n_numeric=g["scaler"].n_features_in_,
                                  n_countries=len(g["country_encoder"].classes_), dropout=cfg["dropout"],
                                  residual_baseline=cfg["residual"], crossdyad_blocks=cfg.get("crossdyad_blocks", 2))
        m.load_weights(_find(model_dir, os.path.basename(f)))
        g_models.append(m)
    d_models = []
    for f in d["weight_files"]:
        m = build_direction_model(n_dyads=d["n_dyads_per_week"], lookback_weeks=d["lookback_weeks"],
                                  n_numeric=d["scaler"].n_features_in_, n_countries=len(d["country_encoder"].classes_),
                                  relation_matrix=d["relation_matrix"])
        m.load_weights(_find(model_dir, os.path.basename(f)))
        d_models.append(m)
    return {"goldstein": (g, g_models), "direction": (d, d_models)}


def _window_inputs(p, numeric, art):
    lb = art["lookback_weeks"]
    X = numeric[:, -lb:][np.newaxis]  # (1, dyads, lookback, features) - window ends at as_of_week
    n = X.shape[-1]
    X = art["scaler"].transform(X.reshape(-1, n)).reshape(X.shape).astype("float32")
    enc = art["country_encoder"]
    return {"numeric_seq": X,
            "actor1_idx": enc.transform(p["a1"]).astype("int32")[np.newaxis],
            "actor2_idx": enc.transform(p["a2"]).astype("int32")[np.newaxis]}


def predict(df, models, ext_df=None, history_weeks=HISTORY_WEEKS):
    """Forecasts as of the LAST week in `df` (which must be complete).
    `ext_df` (pull_external) is required by the cross-system goldstein model."""
    g_art, g_models = models["goldstein"]
    d_art, d_models = models["direction"]
    need = max(g_art["history_weeks_needed"], d_art["history_weeks_needed"])
    if df.week_start.nunique() < need:
        raise ValueError(f"need >= {need} weeks of history, got {df.week_start.nunique()}")
    cfg = g_art["config"]

    # goldstein: 400 MENA dyads (+ the external dyads they attend over, v2.1)
    p = panel_from_df(df, MENA)
    tokens = dict(a1=p["a1"], a2=p["a2"])
    numeric = goldstein_numeric_inputs(p, cfg["long_features"], cfg["has_events"])
    baseline = trailing_mean(p["gold_zero"], cfg["baseline_weeks"])[:, -1]
    if g_art.get("external_dyads"):
        if ext_df is None:
            raise ValueError("this goldstein release attends over external dyads - pass ext_df (pull_external)")
        e = external_panel(ext_df, g_art["external_dyads"], p["weeks"])
        numeric = np.concatenate([numeric, goldstein_numeric_inputs(e, cfg["long_features"], cfg["has_events"])], 0)
        baseline = np.concatenate([baseline, trailing_mean(e["gold_zero"], cfg["baseline_weeks"])[:, -1]])
        tokens = dict(a1=np.concatenate([p["a1"], e["a1"]]), a2=np.concatenate([p["a2"], e["a2"]]))
    inputs = _window_inputs(tokens, numeric, g_art)
    if cfg["residual"]:
        inputs["baseline"] = baseline[np.newaxis].astype("float32")
    gold = np.mean([m.predict(inputs, verbose=0)[0, :len(p["a1"])] for m in g_models], axis=0)

    # direction: 484 dyads in, keep the 400 MENA<->MENA
    p22 = panel_from_df(df, MENA + EXTRA)
    inputs = _window_inputs(p22, direction_numeric_inputs(p22, d_art["log_cols"], d_art["raw_cols"]), d_art)
    prob = np.mean([m.predict(inputs, verbose=0)[0, :, 1] for m in d_models], axis=0)
    direction = pd.DataFrame({"actor1": p22["a1"], "actor2": p22["a2"], "escalation_probability_next4w": prob})

    out = pd.DataFrame({"actor1": p["a1"], "actor2": p["a2"],
                        "goldstein_forecast_next13w": gold,
                        "goldstein_trailing52w": trailing_mean(p["gold_zero"], 52)[:, -1]})
    out = out.merge(direction, on=["actor1", "actor2"], how="left")
    assert out.escalation_probability_next4w.notna().all()
    out["direction_next4w"] = np.where(out.escalation_probability_next4w >= 0.5, "escalation", "de-escalation")
    out.insert(0, "model_version", MODEL_VERSION)
    out.insert(1, "as_of_week", p["weeks"][-1].date())
    out.insert(2, "generated_at_utc", datetime.now(timezone.utc).replace(microsecond=0))
    return out.sort_values("escalation_probability_next4w", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- model files

RELEASE_FILES = [
    *(f"model_release_v2.1_goldstein_s{s}.weights.h5" for s in (42, 1, 2, 3, 4)),
    *(f"model_release_v2.0_direction_s{s}.weights.h5" for s in (42, 1, 2)),
    GOLDSTEIN_ARTIFACTS,
    DIRECTION_ARTIFACTS,
    "risk_v1_calibration.json",  # country risk index (risk_index.py)
]


def download_release(bucket_name, local_dir, version=MODEL_VERSION):
    """Fetch gs://<bucket>/models/<version>/<file> for every release file
    into local_dir (skipping files already there - the directory is
    per-version, so a new release never reuses stale files)."""
    from google.cloud import storage

    os.makedirs(local_dir, exist_ok=True)
    bucket = storage.Client().bucket(bucket_name)
    for name in RELEASE_FILES:
        path = os.path.join(local_dir, name)
        if not os.path.exists(path):
            bucket.blob(f"models/{version}/{name}").download_to_filename(path)
    return local_dir
