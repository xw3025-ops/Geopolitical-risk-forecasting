"""
Run the release v2.0 live pipeline locally - same code as the Cloud Function
(live_pipeline.py), without writing to BigQuery/GCS.

    python run_pipeline.py                       # live: last complete week, pulled from BigQuery
    python run_pipeline.py --download            # ...with model files fetched from GCS first
    python run_pipeline.py --source local --as-of 2026-08-24
                                                 # replay a past week from the local panel

Live mode needs Google Cloud credentials (`gcloud auth application-default
login`) and the google-cloud-* packages in requirements.txt. It does NOT
rebuild table 1.1 from GDELT (only the Cloud Function does), so it forecasts
from whatever complete weeks the table currently holds.

Output: predictions/forecast_v2.0_<as_of_week>.csv
"""

import argparse
import os

import pandas as pd

import live_pipeline as lp
from honest_common import PANEL_PATH, TARGET4_COL

PROJECT = "geopolitical-risk-forecasting"
BUCKET = "geopolitical-risk-forecasting-models"


def local_frame(as_of_week, n_weeks=lp.HISTORY_WEEKS):
    """The n_weeks weeks ending as_of_week from the local training panel,
    without targets - what a live pull would have returned that week."""
    df = pd.read_parquet(PANEL_PATH).drop(columns=[TARGET4_COL])
    df["week_start"] = pd.to_datetime(df["week_start"])
    first = pd.Timestamp(as_of_week) - pd.Timedelta(weeks=n_weeks - 1)
    return df[(df.week_start >= first) & (df.week_start <= pd.Timestamp(as_of_week))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bigquery", "local"], default="bigquery")
    ap.add_argument("--as-of", help="Monday of the as-of week (local mode; default: latest in the panel)")
    ap.add_argument("--models", default=".", help="directory with the release files (default: repo)")
    ap.add_argument("--download", action="store_true", help=f"fetch release files from gs://{BUCKET} first")
    args = ap.parse_args()

    model_dir = args.models
    if args.download:
        model_dir = lp.download_release(BUCKET, os.path.join("models", lp.MODEL_VERSION))

    if args.source == "bigquery":
        from google.cloud import bigquery
        as_of = lp.last_complete_week()
        print(f"Pulling {lp.HISTORY_WEEKS} complete weeks ending {as_of} from {lp.SOURCE_TABLE}...")
        df = lp.pull_panel(bigquery.Client(project=PROJECT), as_of)
    else:
        weeks = pd.to_datetime(pd.read_parquet(PANEL_PATH, columns=["week_start"]).week_start)
        as_of = pd.Timestamp(args.as_of or weeks.max()).date()
        df = local_frame(as_of)

    check = lp.check_panel(df, as_of)
    if check["warning"]:
        print("WARNING:", check["warning"])
    results = lp.predict(df, lp.load_models(model_dir))

    os.makedirs("predictions", exist_ok=True)
    out = f"predictions/forecast_{lp.MODEL_VERSION}_{as_of}.csv"
    results.to_csv(out, index=False)
    cols = ["actor1", "actor2", "escalation_probability_next4w", "goldstein_forecast_next13w",
            "goldstein_trailing52w"]
    print(f"\nAs-of week {as_of} (data through {pd.Timestamp(as_of) + pd.Timedelta(days=6):%Y-%m-%d}); "
          f"wrote {len(results)} forecasts to {out}")
    print("\nMost likely to escalate in the next 4 weeks:")
    print(results[cols].head(10).to_string(index=False, float_format="%.3f"))
    worst = results.assign(change=results.goldstein_forecast_next13w - results.goldstein_trailing52w)
    print("\nLargest expected 13-week deterioration vs the dyad's 52-week normal:")
    print(worst.nsmallest(10, "change")[cols + ["change"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
