"""
Pull 1979 - Feb 2015 history for the 400 MENA dyads from GDELT 1.0
(gdelt-bq.full.events, public), aggregated in BigQuery to the same weekly
dyad features as table 1.1: count_01..count_20, total_count, goldstein_mean.
GDELT 2.0 (our main source) starts 2015-02-19; this file ends the week before.

Free-tier check first: a dry run reports the scan size (22 GB at time of
writing) and the script stops if it exceeds MAX_GB.

    python fetch_gdelt1_history.py
Output:
    raw data/gdelt1_history.parquet
"""

import pandas as pd
from google.cloud import bigquery

from honest_common import MENA

PROJECT = "geopolitical-risk-forecasting"
OUT = "raw data/gdelt1_history.parquet"
MAX_GB = 40

QUERY = f"""
WITH ev AS (
  SELECT
    DATE_TRUNC(PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)), WEEK(MONDAY)) AS week_start,
    Actor1CountryCode, Actor2CountryCode, EventRootCode, GoldsteinScale
  FROM `gdelt-bq.full.events`
  WHERE SQLDATE >= 19790101 AND SQLDATE < 20150216
    AND Actor1CountryCode IN UNNEST(@mena) AND Actor2CountryCode IN UNNEST(@mena)
    AND EventRootCode IS NOT NULL
)
SELECT week_start, Actor1CountryCode, Actor2CountryCode,
  {", ".join(f"COUNTIF(EventRootCode = '{i:02d}') AS count_{i:02d}" for i in range(1, 21))},
  COUNT(*) AS total_count,
  AVG(GoldsteinScale) AS goldstein_mean
FROM ev
GROUP BY week_start, Actor1CountryCode, Actor2CountryCode
"""


def main():
    bq = bigquery.Client(project=PROJECT)
    cfg = dict(query_parameters=[bigquery.ArrayQueryParameter("mena", "STRING", MENA)])
    dry = bq.query(QUERY, job_config=bigquery.QueryJobConfig(dry_run=True, **cfg))
    gb = dry.total_bytes_processed / 1e9
    print(f"dry run: {gb:.1f} GB")
    if gb > MAX_GB:
        raise SystemExit(f"scan of {gb:.0f} GB exceeds the {MAX_GB} GB limit set for this pull - not running")
    job = bq.query(QUERY, job_config=bigquery.QueryJobConfig(**cfg))
    df = job.to_dataframe()
    df["week_start"] = pd.to_datetime(df.week_start.astype(str))  # plain dates, readable without db-dtypes
    print(f"billed {job.total_bytes_billed / 1e9:.1f} GB; {len(df)} dyad-weeks, "
          f"{df.week_start.min()}..{df.week_start.max()}, {int(df.total_count.sum())} events")
    df.to_parquet(OUT, index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
