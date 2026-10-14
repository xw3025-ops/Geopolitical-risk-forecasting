"""
Pull the most recent weeks of dyad-week data for the 22-country universe
(20 MENA + USA + CHN) from BigQuery - for LIVE prediction, not training.

Unlike build_bq_panel_22.py, this does NOT compute a forward-looking target
(there's no future data yet for the most recent weeks - that's the whole
point of predicting it) and does NOT filter out incomplete rows. It just
pulls a buffer of the last N_WEEKS_BUFFER calendar weeks, which is more than
enough to cover the 13-week lookback both models need.

Run:
    python pull_latest_data.py
Output:
    raw data/latest_pull.parquet
"""

import pandas as pd
from google.cloud import bigquery

PROJECT = "geopolitical-risk-forecasting"
TABLE = "geopolitical-risk-forecasting.1.1"
OUT = "raw data/latest_pull.parquet"

MENA = [
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
]
EXTRA = ["USA", "CHN"]

N_WEEKS_BUFFER = 16  # >= the 13-week lookback, with a little slack

client = bigquery.Client(project=PROJECT)
table = client.get_table(TABLE)

float_cols = [f.name for f in table.schema if f.field_type == "FLOAT"]
int_cols = [f.name for f in table.schema if f.field_type == "INTEGER"]
coalesce_selects = [f"COALESCE({c}, 0) AS {c}" for c in float_cols + int_cols]
passthrough_selects = ["week_start", "Actor1CountryCode", "Actor2CountryCode"]

query = f"""
SELECT
  {', '.join(passthrough_selects)},
  {', '.join(coalesce_selects)}
FROM `{TABLE}`
WHERE (dyad_scope = 'INTRA_REGION'
    OR (Actor1CountryCode IN UNNEST(@extra) AND Actor2CountryCode IN UNNEST(@mena))
    OR (Actor2CountryCode IN UNNEST(@extra) AND Actor1CountryCode IN UNNEST(@mena)))
  AND week_start >= (
    SELECT DATE_SUB(MAX(week_start), INTERVAL {N_WEEKS_BUFFER} WEEK) FROM `{TABLE}`
  )
"""

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("mena", "STRING", MENA),
        bigquery.ArrayQueryParameter("extra", "STRING", EXTRA),
    ]
)

print(f"Pulling the latest {N_WEEKS_BUFFER} weeks from BigQuery...")
df = client.query(query, job_config=job_config).to_dataframe()
df["week_start"] = pd.to_datetime(df["week_start"])
print(f"Downloaded {len(df)} rows, weeks {df.week_start.min().date()} to {df.week_start.max().date()}")

# Zero-pad the 4 dyads entirely among {USA, CHN} (never exist in this table)
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

panel = pd.concat([df, pad_df], ignore_index=True)
panel = panel.sort_values(["week_start", "Actor1CountryCode", "Actor2CountryCode"]).reset_index(drop=True)

expected_dyads = (len(MENA) + len(EXTRA)) ** 2
counts_per_week = panel.groupby("week_start").size()
print(f"Dyads per week: min={counts_per_week.min()}, max={counts_per_week.max()}, expected={expected_dyads}")
if (counts_per_week != expected_dyads).any():
    print("WARNING: some weeks are missing dyads - likely the most recent "
          "week is still partially ingested upstream. Check before trusting predictions.")

panel.to_parquet(OUT, index=False)
print(f"Saved {len(panel)} rows to {OUT}")
