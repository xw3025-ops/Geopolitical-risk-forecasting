"""
Cloud Function entry point for the live forecasting pipeline (release v2.2)
and the country risk index (risk-v1).

Every run (triggered by Cloud Scheduler; HTTP, not public):
  1. as_of = Monday of the last COMPLETE week (Mon..Sun, UTC). The week in
     progress is never used.
  2. Rebuild BigQuery table geopolitical-risk-forecasting.1.1 from GDELT's
     public BigQuery dataset, through the as-of week (gdelt_refresh.py).
  3. Pull the last 70 complete weeks from that table (BigQuery API), run the
     release v2.2 goldstein (13-week mean) and conflict-surge (13-week peak) ensembles
     (live_pipeline.py - the same code verify_live_pipeline.py checks against
     the release test predictions).
  4. Write the 400 dyad forecasts:
       BigQuery  predictions.forecasts_v22  (new columns: the surge flag replaces direction)
       GCS       predictions/<version>/latest.csv, predictions/<version>/<as_of>.csv
  5. Country risk index (risk_index.py): refresh the last 8 weeks of ACLED
     event counts (ACLED API) into risk.acled_country_week, fetch trade
     exposure (World Bank API), combine with the forecasts:
       BigQuery  predictions.risk_country_v1
       GCS       predictions/<version>/risk_latest.csv, predictions/<version>/risk_<as_of>.csv
     A failure here is reported but does not undo step 4.

Re-running a week replaces that week's rows (no duplicates).

Website (?site=1, its own weekly Scheduler job, after this run and trade-refresh):
  build the site bundle (build_site_data.build_bundle: forecasts recomputed from
  table 1.1 with the same release models, ACLED events, IMF trade and PortWatch
  from BigQuery trade.*, Guardian/NYT news, DeepSeek briefs) and publish it to
  gs://$SITE_BUCKET/site/latest.json, which the website reads on every visit.
  Ignores the every-2-weeks rule; trade and shipping refresh weekly.

Cadence: Cloud Scheduler fires weekly; the function works only every other
week (counted from REFRESH_ANCHOR_MONDAY). Call with ?force=1 to run
regardless (e.g. right after a deploy).

Secrets (Secret Manager, attached by deploy.sh when present): ACLED_EMAIL / ACLED_PASSWORD;
for the website briefs also DEEPSEEK_API_KEY, GUARDIAN_API_KEY, NYT_API_KEY.
"""

import json
import os
import traceback
from datetime import date, datetime, timedelta, timezone

import functions_framework
from google.cloud import bigquery, storage

import live_pipeline as lp
import risk_index as ri
from gdelt_refresh import refresh_gdelt_table

PROJECT = "geopolitical-risk-forecasting"
FORECASTS_TABLE = "geopolitical-risk-forecasting.predictions.forecasts_v22"  # v2.0/v2.1 rows stay in forecasts_v2
RISK_TABLE = "geopolitical-risk-forecasting.predictions.risk_country_v1"
BUCKET = "geopolitical-risk-forecasting-models"
MODEL_VERSION = os.environ.get("MODEL_VERSION", lp.MODEL_VERSION)
MODEL_DIR = f"/tmp/models/{MODEL_VERSION}"  # per-version: a new release never reuses cached files
ACLED_REFRESH_WEEKS = 8  # ACLED revises recent weeks; re-pull this many each run

REFRESH_ANCHOR_MONDAY = date(2026, 10, 5)  # a run week; every 2nd week from here runs

_models = None  # loaded once per warm container


def is_run_week(today):
    """True in every 2nd calendar week counted from the anchor - on any
    weekday, so the Scheduler day can change without touching this."""
    this_monday = date.fromordinal(today.toordinal() - today.weekday())
    return ((this_monday - REFRESH_ANCHOR_MONDAY).days // 7) % 2 == 0


def replace_week(bq, table, df, as_of, version_col, version):
    """Delete this as-of week's rows for this version, then append df."""
    try:
        bq.query(f"DELETE FROM `{table}` WHERE as_of_week = @w AND {version_col} = @v",
                 job_config=bigquery.QueryJobConfig(query_parameters=[
                     bigquery.ScalarQueryParameter("w", "DATE", as_of),
                     bigquery.ScalarQueryParameter("v", "STRING", version)])).result()
    except Exception as e:  # first run: the table doesn't exist yet - the load below creates it
        if "Not found" not in str(e):
            raise
    bq.load_table_from_dataframe(df, table, job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND")).result()


def upload_csv(df, prefix, as_of):
    bucket = storage.Client().bucket(BUCKET)
    csv = df.to_csv(index=False)
    for name in (f"{prefix}latest.csv", f"{prefix}{as_of}.csv"):
        bucket.blob(f"predictions/{MODEL_VERSION}/{name}").upload_from_string(csv, content_type="text/csv")


def run_risk(bq, df, results, as_of):
    if not (os.environ.get("ACLED_EMAIL") and os.environ.get("ACLED_PASSWORD")):
        return "risk index skipped: ACLED secrets not configured in Secret Manager"
    calib = ri.load_calibration(MODEL_DIR)
    start = as_of - timedelta(weeks=ACLED_REFRESH_WEEKS - 1)
    recent, max_date = ri.acled_country_weeks(os.environ["ACLED_EMAIL"], os.environ["ACLED_PASSWORD"],
                                              start.isoformat(), (as_of + timedelta(days=6)).isoformat())
    complete = ri.latest_complete_acled_week(max_date)
    ri.store_acled_weeks(bq, recent[recent.week_start <= complete])
    history = ri.load_acled_weeks(bq)
    print(f"ACLED: latest event {max_date}, complete through week {complete}, history {len(history)} rows")
    risk = ri.compute_risk(df, results, calib, history, complete, ri.trade_exposure())
    risk.insert(2, "model_version", MODEL_VERSION)
    risk.insert(3, "generated_at_utc", datetime.now(timezone.utc).replace(microsecond=0))
    replace_week(bq, RISK_TABLE, risk, as_of, "risk_version", ri.RISK_VERSION)
    upload_csv(risk, "risk_", as_of)
    top = risk.head(5)[["country", "risk_score", "risk_band"]].to_dict("records")
    return f"risk index for 20 countries (top: {top})"


def run_site():
    """Rebuild and publish the website bundle from what is in BigQuery now."""
    global _models
    import build_site_data as bs
    try:
        if _models is None:
            _models = lp.load_models(lp.download_release(BUCKET, MODEL_DIR, MODEL_VERSION))
        bundle = bs.build_bundle(bigquery.Client(project=PROJECT), _models)
        url = bs.publish(bs.to_json(bundle), bundle["meta"]["as_of_week"])
        msg = (f"OK: site bundle as of {bundle['meta']['as_of_week']} -> {url}; "
               f"briefs: {bundle['summaries']['provenance'].get('source')}; "
               f"ACLED: {'yes' if bundle['meta']['acled'] else 'no'}; lanes: {'yes' if bundle['lanes'] else 'no'}")
        print(msg)
        return msg, 200
    except Exception:
        print(traceback.format_exc())
        return "SITE BUILD FAILED - see logs", 500


@functions_framework.http
def run_pipeline(request):
    global _models
    if request.args.get("site") == "1":
        return run_site()
    today = datetime.now(timezone.utc).date()
    if not is_run_week(today) and request.args.get("force") != "1":
        msg = f"SKIPPED: {today} is not in a run week (every 2nd week from {REFRESH_ANCHOR_MONDAY})"
        print(msg)
        return msg, 200

    as_of = lp.last_complete_week(today)
    bq = bigquery.Client(project=PROJECT)

    n_rows, billed = refresh_gdelt_table(bq, as_of)
    print(f"Rebuilt {lp.SOURCE_TABLE} through week {as_of}: {n_rows} rows, {billed / 1e9:.0f} GB billed")

    if _models is None:
        _models = lp.load_models(lp.download_release(BUCKET, MODEL_DIR, MODEL_VERSION))
    df = lp.pull_panel(bq, as_of)
    check = lp.check_panel(df, as_of)
    pairs = _models["goldstein"][0].get("external_dyads", [])
    ext = lp.pull_external(bq, as_of, pairs) if pairs else None
    print(f"Pulled {df.week_start.nunique()} complete weeks ending {as_of} (+{len(pairs)} external dyads): "
          f"{json.dumps(check)}")
    results = lp.predict(df, _models, ext)
    replace_week(bq, FORECASTS_TABLE, results, as_of, "model_version", MODEL_VERSION)
    upload_csv(results, "", as_of)
    msg = f"OK: {len(results)} forecasts, model {MODEL_VERSION}, as-of week {as_of}"
    if check["warning"]:
        msg += f" - WARNING: {check['warning']}"

    try:
        msg += "; " + run_risk(bq, df, results, as_of)
    except Exception:
        print(traceback.format_exc())
        msg += "; RISK INDEX FAILED (forecasts were still written) - see logs"
    print(msg)
    return msg, 200
