"""
Pull the big pre-aggregated dyad-week table from BigQuery
(geopolitical-risk-forecasting.1.1), compute the forward-looking target
(average goldstein_mean over the NEXT 13 weeks per dyad) server-side with a
SQL window function, and save the result locally as parquet.

Both scopes are included:
  INTRA_REGION  - the 20-country MENA panel we've used before
  EXTERNAL_LINK - each MENA country's dyads with ~200 other countries

Both scopes are already fully dense (no missing dyad-weeks to pad - verified:
row counts match distinct-dyads x weeks exactly), so this script only needs
to compute the target, not fill gaps.

Nulls (e.g. goldstein_mean when a dyad-week had zero events) are coalesced to
0, matching our earlier convention: 0 = no significant developments.

Output: raw data/bq_panel_with_target.parquet
"""

import pandas as pd
from google.cloud import bigquery

PROJECT = "geopolitical-risk-forecasting"
TABLE = "geopolitical-risk-forecasting.1.1"
OUT = "raw data/bq_panel_with_target.parquet"
HORIZON_WEEKS = 13

client = bigquery.Client(project=PROJECT)
table = client.get_table(TABLE)

float_cols = [f.name for f in table.schema if f.field_type == "FLOAT"]
int_cols = [f.name for f in table.schema if f.field_type == "INTEGER"]

# action_lat_mean/action_long_mean are real coordinates - coalescing NULL to
# 0 would fabricate a fake location (0,0 is a real point), so we still fill
# them for query simplicity but exclude them from training features later.
coalesce_selects = [f"COALESCE({c}, 0) AS {c}" for c in float_cols + int_cols]
passthrough_selects = ["week_start", "Actor1CountryCode", "Actor2CountryCode",
                        "dyad_scope", "top_event_root"]

query = f"""
WITH filled AS (
  SELECT
    {', '.join(passthrough_selects)},
    {', '.join(coalesce_selects)}
  FROM `{TABLE}`
),
windowed AS (
  SELECT
    *,
    AVG(goldstein_mean) OVER (
      PARTITION BY Actor1CountryCode, Actor2CountryCode
      ORDER BY week_start
      ROWS BETWEEN 1 FOLLOWING AND {HORIZON_WEEKS} FOLLOWING
    ) AS target_avg_goldstein_next{HORIZON_WEEKS}w,
    COUNT(*) OVER (
      PARTITION BY Actor1CountryCode, Actor2CountryCode
      ORDER BY week_start
      ROWS BETWEEN 1 FOLLOWING AND {HORIZON_WEEKS} FOLLOWING
    ) AS n_future_weeks
  FROM filled
)
SELECT * EXCEPT(n_future_weeks)
FROM windowed
WHERE n_future_weeks = {HORIZON_WEEKS}
"""

print("Running query in BigQuery (server-side window function)...")
df = client.query(query).to_dataframe()
print(f"Downloaded {len(df)} rows, {len(df.columns)} columns")
print(df.dyad_scope.value_counts())

# BigQuery's DATE type loads as a 'dbdate' pandas extension dtype, which
# pyarrow can't round-trip through parquet without db_dtypes imported first -
# convert to plain datetime64 here so the saved file loads normally anywhere.
df["week_start"] = pd.to_datetime(df["week_start"])

df.to_parquet(OUT, index=False)
print(f"Saved to {OUT}")
