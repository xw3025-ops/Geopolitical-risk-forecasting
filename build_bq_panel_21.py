"""
Pull a 21-country dyad-week panel (the 20 MENA countries + USA) from
BigQuery, with the full rich feature set (CAMEO counts, category rollups,
goldstein stats, news tone/sentiment, media coverage volume, actor
diversity, geography) - not just the smaller count_01..20 + avg_goldstein
set used by the CSV-based pipeline.

Sources, from geopolitical-risk-forecasting.1.1:
  - dyad_scope = INTRA_REGION: the 400 MENA<->MENA dyads (unchanged from before)
  - dyad_scope = EXTERNAL_LINK, restricted to the USA<->MENA rows: 40 dyads
    (20 USA->MENA + 20 MENA->USA), confirmed fully dense (606 weeks each)

USA<->USA (the 441st dyad) does not exist in this table at all - it's
neither INTRA_REGION nor EXTERNAL_LINK by the table's own construction
(EXTERNAL_LINK always has exactly one MENA-side actor). It's zero-padded,
same convention as LBY/SDN's missing goldstein data earlier: 0 = no data
available, not "nothing happened."

Target: forward 13-week average of goldstein_mean per dyad, computed with a
SQL window function (same approach as build_bq_panel.py), then zero-filled
for USA<->USA and recombined.

Output: raw data/bq_panel_21_with_target.parquet
"""

import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT = "geopolitical-risk-forecasting"
TABLE = "geopolitical-risk-forecasting.1.1"
OUT = "raw data/bq_panel_21_with_target.parquet"
HORIZON_WEEKS = 13

MENA = [
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
]
COUNTRIES_21 = MENA + ["USA"]

client = bigquery.Client(project=PROJECT)
table = client.get_table(TABLE)

float_cols = [f.name for f in table.schema if f.field_type == "FLOAT"]
int_cols = [f.name for f in table.schema if f.field_type == "INTEGER"]
coalesce_selects = [f"COALESCE({c}, 0) AS {c}" for c in float_cols + int_cols]
passthrough_selects = ["week_start", "Actor1CountryCode", "Actor2CountryCode"]

query = f"""
WITH filled AS (
  SELECT
    {', '.join(passthrough_selects)},
    {', '.join(coalesce_selects)}
  FROM `{TABLE}`
  WHERE dyad_scope = 'INTRA_REGION'
     OR (Actor1CountryCode = 'USA' AND Actor2CountryCode IN UNNEST(@mena))
     OR (Actor2CountryCode = 'USA' AND Actor1CountryCode IN UNNEST(@mena))
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

job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ArrayQueryParameter("mena", "STRING", MENA)]
)

print("Running query in BigQuery (server-side window function)...")
df = client.query(query, job_config=job_config).to_dataframe()
print(f"Downloaded {len(df)} rows (MENA<->MENA + USA<->MENA), {len(df.columns)} columns")

df["week_start"] = pd.to_datetime(df["week_start"])

# ---- Zero-pad the USA<->USA self-dyad across every week present ----
weeks = sorted(df.week_start.unique())
numeric_cols = float_cols + int_cols
usa_self = pd.DataFrame({"week_start": weeks})
usa_self["Actor1CountryCode"] = "USA"
usa_self["Actor2CountryCode"] = "USA"
for c in numeric_cols:
    usa_self[c] = 0
target_col = f"target_avg_goldstein_next{HORIZON_WEEKS}w"
usa_self[target_col] = 0.0  # all-zero series -> forward average is trivially 0

print(f"Padding USA<->USA with {len(usa_self)} all-zero rows (no source data exists for this dyad)")

panel = pd.concat([df, usa_self], ignore_index=True)
panel = panel.sort_values(["week_start", "Actor1CountryCode", "Actor2CountryCode"]).reset_index(drop=True)

expected_dyads = len(COUNTRIES_21) ** 2  # 441
counts_per_week = panel.groupby("week_start").size()
print(f"Dyads per week: min={counts_per_week.min()}, max={counts_per_week.max()}, "
      f"expected={expected_dyads}")

panel.to_parquet(OUT, index=False)
print(f"Saved {len(panel)} rows to {OUT}")
