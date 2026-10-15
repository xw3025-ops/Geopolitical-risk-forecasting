"""
Cloud Function "trade-refresh" (deployed by cloud_function/deploy_trade.sh, separate from the
forecasting function so a trade-data failure can never affect the forecasts). Each run reloads:

  trade.imf_imts_raw      IMF IMTS monthly China <-> 20 MENA + USA goods trade (both reporting
                          sides), 2017-01 onward. Full replace: the IMF revises past months
                          (STATUS 'r'), so appending would keep stale values.
  trade.portwatch_daily   IMF PortWatch daily transits for Hormuz, Bab el-Mandeb, Suez,
                          Cape of Good Hope and Malacca, 2019-01-01 onward. Full replace.

Both APIs are free and keyless, and the tables are small (~10k and ~15k rows), so a full reload
costs nothing within BigQuery's free tier. The download code is src/trade_sources.py (copied into
the function by deploy_trade.sh), the same code the local fetch scripts use.

Local test without BigQuery:  python -c "import trade_refresh"  (from a folder holding this file,
src/config.py and src/trade_sources.py, which is how deploy_trade.sh assembles it).
"""

import traceback

import functions_framework
from google.cloud import bigquery

from trade_sources import imf_imts, portwatch

PROJECT = "geopolitical-risk-forecasting"
DATASET = f"{PROJECT}.trade"


def _replace(bq, df, table):
    job = bq.load_table_from_dataframe(df, f"{DATASET}.{table}", job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"))
    job.result()
    return len(df)


def refresh_trade_tables(bq):
    bq.create_dataset(DATASET, exists_ok=True)
    imf = imf_imts()
    imf = imf.astype({c: "string" for c in imf.columns if c not in ("value", "SCALE")})
    n_imf = _replace(bq, imf, "imf_imts_raw")
    pw = portwatch()
    n_pw = _replace(bq, pw, "portwatch_daily")
    last_month = imf[imf.side == "china"].period.max()
    return (f"trade tables refreshed: imf_imts_raw {n_imf} rows (China-reported through {last_month}), "
            f"portwatch_daily {n_pw} rows (through {pw.date.max()})")


@functions_framework.http
def refresh_trade(request):
    try:
        msg = "OK: " + refresh_trade_tables(bigquery.Client(project=PROJECT))
        print(msg)
        return msg, 200
    except Exception:
        print(traceback.format_exc())
        return "TRADE REFRESH FAILED - see logs", 500
