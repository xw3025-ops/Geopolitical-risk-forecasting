"""
One-time (or occasional full-refresh) pull of ACLED event counts for the 20
MENA countries, weekly, from late 2017 to today, into BigQuery table
geopolitical-risk-forecasting.risk.acled_country_week (+ a local parquet copy).

The risk index needs this history for its fixed ACLED reference scale
(2018-2025 quintiles); after the backfill, the Cloud Function only refreshes
the last few weeks on each run.

Credentials come from environment variables, never from code:
    PowerShell:  $env:ACLED_EMAIL = "you@columbia.edu"; $env:ACLED_PASSWORD = "..."
    Git Bash:    export ACLED_EMAIL=you@columbia.edu ACLED_PASSWORD=...
plus `gcloud auth application-default login` for BigQuery.

Run:
    python backfill_acled.py [--start 2017-12-04] [--local-only]
"""

import argparse
import os
from datetime import date

import pandas as pd

from risk_index import (ACLED_TABLE, acled_country_weeks, latest_complete_acled_week, store_acled_weeks)

PROJECT = "geopolitical-risk-forecasting"
LOCAL_COPY = "raw data/acled_country_week.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-12-04", help="a Monday; 3 weeks before the 2018 reference window")
    ap.add_argument("--local-only", action="store_true", help="write the parquet copy only, not BigQuery")
    args = ap.parse_args()
    try:  # local runs: credentials from the git-ignored .env file
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    email, password = os.environ.get("ACLED_EMAIL"), os.environ.get("ACLED_PASSWORD")
    if not email or not password:
        raise SystemExit("set ACLED_EMAIL and ACLED_PASSWORD environment variables first")

    print(f"Pulling ACLED {args.start}..{date.today()} for 20 countries (this takes a few minutes)...")
    weekly, max_date = acled_country_weeks(email, password, args.start, date.today().isoformat())
    complete = latest_complete_acled_week(max_date)
    weekly = weekly[pd.to_datetime(weekly.week_start) <= pd.Timestamp(complete)]
    print(f"Latest ACLED event date {max_date}; keeping complete weeks through {complete}: "
          f"{len(weekly)} country-weeks, {int(weekly.events.sum())} events")
    print(weekly.groupby("country").events.sum().sort_values(ascending=False).to_string())

    os.makedirs(os.path.dirname(LOCAL_COPY), exist_ok=True)
    weekly.to_parquet(LOCAL_COPY, index=False)
    print(f"Saved {LOCAL_COPY}")
    if not args.local_only:
        from google.cloud import bigquery
        store_acled_weeks(bigquery.Client(project=PROJECT), weekly)
        print(f"Wrote {ACLED_TABLE}")


if __name__ == "__main__":
    main()
