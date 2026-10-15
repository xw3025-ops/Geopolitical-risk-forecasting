#!/usr/bin/env bash
# Deploy the trade-data refresh as its own small Cloud Function + Scheduler job.
# Run from the repo root after `gcloud auth login` with an account that owns the project:
#
#     bash cloud_function/deploy_trade.sh
#
# What it does (idempotent - re-running updates in place):
#   1. assemble the source: cloud_function/trade_refresh.py (as main.py), src/config.py,
#      src/trade_sources.py, cloud_function/trade_requirements.txt
#   2. deploy Cloud Function "trade-refresh" (gen2, HTTP, private), running as the same
#      service account as the forecasting function, which already writes to BigQuery
#   3. create/update Cloud Scheduler job "trade-refresh-weekly"
#   4. run it once and print the result
# Writes BigQuery tables trade.imf_imts_raw and trade.portwatch_daily (dataset created on first run).
set -euo pipefail

PROJECT=geopolitical-risk-forecasting
REGION=us-central1
FUNCTION=trade-refresh
JOB=trade-refresh-weekly
MAIN_FUNCTION=run-pipeline   # forecasting function: we reuse its service account
# Weekly, Tuesdays 07:00 UTC (an hour after the forecasting run). Weekly rather than every 2nd
# week: the IMF data change about once a month at unpredictable dates, and a run costs nothing
# (no BigQuery scan, just two small table loads).
SCHEDULE="0 7 * * 2"

echo "Account: $(gcloud config get-value account 2>/dev/null)  Project: $PROJECT"

SA=$(gcloud functions describe "$MAIN_FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
     --format='value(serviceConfig.serviceAccountEmail)')
echo "Service account: $SA"

BUILD=$(mktemp -d)
cp cloud_function/trade_refresh.py "$BUILD/main.py"
cp src/config.py src/trade_sources.py "$BUILD/"
cp cloud_function/trade_requirements.txt "$BUILD/requirements.txt"
echo "Function source: $(ls "$BUILD" | tr '\n' ' ')"

gcloud functions deploy "$FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
  --runtime=python312 --source="$BUILD" --entry-point=refresh_trade \
  --trigger-http --no-allow-unauthenticated --service-account="$SA" \
  --memory=1GiB --cpu=1 --timeout=600s

URL=$(gcloud functions describe "$FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
      --format='value(serviceConfig.uri)')
# let the scheduler's service account call the (private) function
gcloud functions add-invoker-policy-binding "$FUNCTION" --project="$PROJECT" --region="$REGION" \
  --member="serviceAccount:$SA" >/dev/null

if gcloud scheduler jobs describe "$JOB" --project="$PROJECT" --location="$REGION" >/dev/null 2>&1; then
  ACTION=update
else
  ACTION=create
fi
gcloud scheduler jobs $ACTION http "$JOB" --project="$PROJECT" --location="$REGION" \
  --schedule="$SCHEDULE" --time-zone=UTC --uri="$URL" --http-method=POST \
  --oidc-service-account-email="$SA" --oidc-token-audience="$URL" \
  --attempt-deadline=600s --max-retry-attempts=1

echo
echo "Deployed $FUNCTION; schedule '$SCHEDULE' (UTC). First run now:"
curl -s -X POST -d '' -H "Authorization: bearer $(gcloud auth print-identity-token)" "$URL"
echo
