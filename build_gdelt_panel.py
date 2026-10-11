"""
Step 2 of the v2 pipeline: pull a dyad-week panel straight from the PUBLIC
GDELT 2.0 events table for every ordered pair among GDELT_COUNTRIES
(risk_config.py: 20 MENA + CHN + USA + Belt and Road host countries).

Why a new pull instead of the project table `geopolitical-risk-forecasting.1.1`:
  - That table only covers MENA <-> MENA and MENA <-> world, so CHN <-> host
    dyads outside MENA (PAK, KAZ, IDN, ETH, ...) and host domestic events are
    missing.
  - The SQL that built it is not in this repository, so it cannot be
    reproduced. This script is the full, versioned definition.
  - One query produces counts AND goldstein for every dyad, so there is no
    second source with a different country list (the cause of the LBY/SDN
    gap in the CSVs).

Output schema matches the CSV exports, so it feeds straight into
build_risk_panel.py:
  week_start (Monday), Actor1CountryCode, Actor2CountryCode,
  count_01..count_20 (events per CAMEO root code), avg_goldstein,
  n_mentions, avg_tone
The current (incomplete) week is excluded.

Cost: the query scans 7 columns of gdelt-bq.gdeltv2.events_partitioned from
START_DATE. Always run --dry-run first; it prints the bytes that would be
billed to YOUR project (on-demand pricing is about USD 6.25 per TiB). A hard
cap (--max-gb) makes BigQuery refuse the query rather than overspend.

Setup (once):
    gcloud auth application-default login      # with the account that has
                                               # access to your billing project
Run:
    python build_gdelt_panel.py --project geopolitical-risk-forecasting --dry-run
    python build_gdelt_panel.py --project geopolitical-risk-forecasting
    python build_risk_panel.py --gdelt "raw data/gdelt_panel.csv"

NOTE: not executed by its author (no access to the billing project); the
dry run is the first thing to check.
"""

import argparse

import pandas as pd
from google.cloud import bigquery

from risk_config import GDELT_COUNTRIES, GDELT_PANEL_PATH, PANEL_START

GDELT_TABLE = "gdelt-bq.gdeltv2.events_partitioned"
USD_PER_TIB = 6.25

COUNT_SELECTS = ",\n    ".join(f"COUNTIF(EventRootCode = '{i:02d}') AS count_{i:02d}" for i in range(1, 21))
QUERY = f"""
WITH events AS (
  SELECT
    DATE_TRUNC(PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)), WEEK(MONDAY)) AS week_start,
    Actor1CountryCode, Actor2CountryCode, EventRootCode, GoldsteinScale, NumMentions, AvgTone
  FROM `{GDELT_TABLE}`
  WHERE _PARTITIONTIME >= TIMESTAMP(@start)
    AND SQLDATE >= @start_int
    AND Actor1CountryCode IN UNNEST(@countries)
    AND Actor2CountryCode IN UNNEST(@countries)
)
SELECT
  week_start, Actor1CountryCode, Actor2CountryCode,
  {COUNT_SELECTS},
  AVG(GoldsteinScale) AS avg_goldstein,
  SUM(NumMentions) AS n_mentions,
  AVG(AvgTone) AS avg_tone
FROM events
WHERE week_start >= @start AND week_start < DATE_TRUNC(CURRENT_DATE(), WEEK(MONDAY))
GROUP BY week_start, Actor1CountryCode, Actor2CountryCode
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project billed for the query")
    parser.add_argument("--start", default=PANEL_START)
    parser.add_argument("--dry-run", action="store_true", help="only estimate bytes scanned")
    parser.add_argument("--max-gb", type=float, default=300, help="refuse to run above this many GB billed")
    parser.add_argument("--out", default=GDELT_PANEL_PATH)
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    params = [
        bigquery.ScalarQueryParameter("start", "DATE", args.start),
        bigquery.ScalarQueryParameter("start_int", "INT64", int(args.start.replace("-", ""))),
        bigquery.ArrayQueryParameter("countries", "STRING", GDELT_COUNTRIES),
    ]
    dry = client.query(QUERY, job_config=bigquery.QueryJobConfig(
        query_parameters=params, dry_run=True, use_query_cache=False))
    gb = dry.total_bytes_processed / 1e9
    print(f"{len(GDELT_COUNTRIES)} countries ({len(GDELT_COUNTRIES) ** 2} ordered dyads) from {args.start}: "
          f"query would scan {gb:,.1f} GB (~USD {gb / 1024 * USD_PER_TIB:,.2f} on-demand)")
    if args.dry_run:
        return
    if gb > args.max_gb:
        raise SystemExit(f"Refusing: {gb:.1f} GB > --max-gb {args.max_gb}. Raise the cap deliberately if intended.")

    job = client.query(QUERY, job_config=bigquery.QueryJobConfig(
        query_parameters=params, maximum_bytes_billed=int(args.max_gb * 1e9)))
    df = job.to_dataframe()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["week_start", "Actor1CountryCode", "Actor2CountryCode"])
    print(f"Downloaded {len(df):,} dyad-weeks, {df.week_start.min().date()}..{df.week_start.max().date()}; "
          f"{(df.Actor1CountryCode + df.Actor2CountryCode).nunique()} dyads with at least one event")
    missing = sorted(set(GDELT_COUNTRIES) - set(df.Actor1CountryCode) - set(df.Actor2CountryCode))
    if missing:
        print(f"WARNING: no events at all for {missing} - check the CAMEO country codes in risk_config.py")
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
