"""Run sql/gdelt_country_month.sql on BigQuery -> data/gdelt/gdelt_country_month.csv
Needs: pip install google-cloud-bigquery db-dtypes
Auth once: gcloud auth application-default login   (and set your project)
Run:  python src/fetch_gdelt.py <your-gcp-project-id>
"""

import sys

from google.cloud import bigquery

from config import ROOT

project = (
    sys.argv[1]
    if len(sys.argv) > 1
    else sys.exit("usage: python src/fetch_gdelt.py <gcp-project-id>")
)
client = bigquery.Client(project=project)
sql = (ROOT / "sql" / "gdelt_country_month.sql").read_text()

est = client.query(
    sql, job_config=bigquery.QueryJobConfig(dry_run=True)
).total_bytes_processed
print(f"Estimated scan: {est / 1e9:.1f} GB (free tier: 1 TB/month)")

df = client.query(sql).to_dataframe()
out = ROOT / "data" / "gdelt" / "gdelt_country_month.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"{len(df):,} rows -> {out}")
print(df.groupby("iso3").ym.agg(["min", "max", "count"]))
