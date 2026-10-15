"""
World dyads: weekly GDELT 2.0 panel for country pairs OUTSIDE the MENA region
(neither actor is one of the 20 MENA countries), for use as extra TRAINING
data only. Same weekly aggregation as table 1.1 for the columns the models
use (CAMEO root counts, mean goldstein, negative goldstein intensity), same
weeks (2015-02-16 .. last complete week), dyads active in >= MIN_ACTIVE of
weeks (as for the MENA-world dyads in v2.1).

Cost: one query, ~23 GB scanned (5 columns of gdelt-bq.gdeltv2.events), well
inside BigQuery's free 1 TB/month.

    python fetch_world_dyads.py
Output: raw data/world_dyads_panel.parquet (dense dyad x week grid; weeks
without events have zero counts and NaN goldstein_mean, as in table 1.1)
"""

import pandas as pd
from google.cloud import bigquery

from honest_common import COUNT_COLS, MENA

PROJECT = "geopolitical-risk-forecasting"
LAST_WEEK = "2026-10-05"  # Monday of the last complete week when pulled
MIN_ACTIVE = 0.50
OUT = "raw data/world_dyads_panel.parquet"

mena = ",".join(f"'{c}'" for c in MENA)
counts = ",\n  ".join(f"COUNTIF(EventRootCode = '{i:02d}') AS count_{i:02d}" for i in range(1, 21))
QUERY = f"""
WITH ev AS (
  SELECT DATE_TRUNC(PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)), WEEK(MONDAY)) AS week_start,
         Actor1CountryCode AS a1, Actor2CountryCode AS a2, EventRootCode, GoldsteinScale
  FROM `gdelt-bq.gdeltv2.events`
  WHERE SQLDATE >= 20150216
    AND SQLDATE <= CAST(FORMAT_DATE('%Y%m%d', DATE_ADD(DATE '{LAST_WEEK}', INTERVAL 6 DAY)) AS INT64)
    AND EventRootCode IS NOT NULL
    AND Actor1CountryCode IS NOT NULL AND Actor2CountryCode IS NOT NULL
    AND Actor1CountryCode NOT IN ({mena}) AND Actor2CountryCode NOT IN ({mena})
),
weekly AS (
  SELECT week_start, a1, a2,
  {counts},
  COUNT(*) AS total_count,
  AVG(GoldsteinScale) AS goldstein_mean,
  SUM(IF(GoldsteinScale < 0, -GoldsteinScale, 0)) AS negative_goldstein_intensity
  FROM ev WHERE week_start >= DATE '2015-02-16' GROUP BY 1, 2, 3
),
n_weeks AS (SELECT DATE_DIFF(DATE '{LAST_WEEK}', DATE '2015-02-16', WEEK) + 1 AS n),
active AS (
  SELECT a1, a2 FROM weekly, n_weeks GROUP BY a1, a2, n_weeks.n HAVING COUNT(*) >= {MIN_ACTIVE} * n_weeks.n
)
SELECT w.* FROM weekly w JOIN active USING (a1, a2)
"""


def main():
    bq = bigquery.Client(project=PROJECT)
    dry = bq.query(QUERY, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    print(f"scanning {dry.total_bytes_processed / 1e9:.0f} GB", flush=True)
    job = bq.query(QUERY)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    print(f"billed {job.total_bytes_billed / 1e9:.0f} GB; {len(df)} rows, "
          f"{df.groupby(['a1', 'a2']).ngroups} dyads", flush=True)
    df["week_start"] = pd.to_datetime(df.week_start)
    weeks = pd.date_range("2015-02-16", LAST_WEEK, freq="7D")
    keys = sorted(df.groupby(["a1", "a2"]).groups)
    full = pd.MultiIndex.from_tuples([(a, b, w) for a, b in keys for w in weeks], names=["a1", "a2", "week_start"])
    df = df.set_index(["a1", "a2", "week_start"]).reindex(full).reset_index()
    for c in COUNT_COLS + ["total_count", "negative_goldstein_intensity"]:
        df[c] = df[c].fillna(0).astype("float32")
    df["goldstein_mean"] = df["goldstein_mean"].astype("float32")
    df.to_parquet(OUT, index=False)
    print(f"saved {OUT}: {len(keys)} dyads x {len(weeks)} weeks")


if __name__ == "__main__":
    main()
