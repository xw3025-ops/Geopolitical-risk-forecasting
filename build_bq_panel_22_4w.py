"""
Pull a 22-country dyad-week panel (20 MENA + USA + CHN) from BigQuery, with
the full rich feature set (CAMEO counts, category rollups, goldstein stats,
news tone/sentiment, media coverage volume, actor diversity). Extends
build_bq_panel_21.py (which added just USA) by also adding China.

Sources, from geopolitical-risk-forecasting.1.1:
  - dyad_scope = INTRA_REGION: the 400 MENA<->MENA dyads
  - dyad_scope = EXTERNAL_LINK, restricted to USA<->MENA and CHN<->MENA rows:
    80 dyads (20 USA->MENA + 20 MENA->USA + 20 CHN->MENA + 20 MENA->CHN),
    confirmed fully dense (606 weeks each)

The 4 dyads among {USA, CHN} themselves (USA<->USA, CHN<->CHN, USA<->CHN,
CHN<->USA) don't exist in this table at all - EXTERNAL_LINK always has
exactly one MENA-side actor by construction. They're zero-padded, same
convention as LBY/SDN's missing goldstein data and USA<->USA earlier: 0 = no
data available, not "nothing happened."

Target: forward HORIZON_WEEKS-week average of goldstein_mean per dyad,
computed with a SQL window function.

Output path depends on HORIZON_WEEKS (see OUT below).
"""

import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT = "geopolitical-risk-forecasting"
TABLE = "geopolitical-risk-forecasting.1.1"
HORIZON_WEEKS = 4  # 4-week horizon, for direction classification
OUT = f"raw data/bq_panel_22_with_target_{HORIZON_WEEKS}w.parquet"

MENA = [
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
]
EXTRA = ["USA", "CHN"]
COUNTRIES_22 = MENA + EXTRA

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
     OR (Actor1CountryCode IN UNNEST(@extra) AND Actor2CountryCode IN UNNEST(@mena))
     OR (Actor2CountryCode IN UNNEST(@extra) AND Actor1CountryCode IN UNNEST(@mena))
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
    query_parameters=[
        bigquery.ArrayQueryParameter("mena", "STRING", MENA),
        bigquery.ArrayQueryParameter("extra", "STRING", EXTRA),
    ]
)

print(f"Running query in BigQuery (horizon={HORIZON_WEEKS} weeks)...")
df = client.query(query, job_config=job_config).to_dataframe()
print(f"Downloaded {len(df)} rows (MENA<->MENA + USA/CHN<->MENA), {len(df.columns)} columns")

df["week_start"] = pd.to_datetime(df["week_start"])

# ---- Zero-pad the 4 dyads entirely among {USA, CHN} ----
weeks = sorted(df.week_start.unique())
numeric_cols = float_cols + int_cols
pad_rows = []
for a1 in EXTRA:
    for a2 in EXTRA:
        block = pd.DataFrame({"week_start": weeks})
        block["Actor1CountryCode"] = a1
        block["Actor2CountryCode"] = a2
        for c in numeric_cols:
            block[c] = 0
        pad_rows.append(block)
pad_df = pd.concat(pad_rows, ignore_index=True)
target_col = f"target_avg_goldstein_next{HORIZON_WEEKS}w"
pad_df[target_col] = 0.0

print(f"Padding {len(pad_df)} all-zero rows for the 4 USA/CHN<->USA/CHN dyads "
      f"(no source data exists for these)")

panel = pd.concat([df, pad_df], ignore_index=True)
panel = panel.sort_values(["week_start", "Actor1CountryCode", "Actor2CountryCode"]).reset_index(drop=True)

expected_dyads = len(COUNTRIES_22) ** 2  # 484
counts_per_week = panel.groupby("week_start").size()
print(f"Dyads per week: min={counts_per_week.min()}, max={counts_per_week.max()}, "
      f"expected={expected_dyads}")

panel.to_parquet(OUT, index=False)
print(f"Saved {len(panel)} rows to {OUT}")
