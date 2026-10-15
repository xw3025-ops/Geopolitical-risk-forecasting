#!/usr/bin/env bash
# Deploy the release v2.2 live pipeline. Run from the repo root, with the
# Google Cloud SDK installed and `gcloud auth login` done:
#
#     bash cloud_function/deploy.sh
#
# Steps:
#   1. upload the release model files to gs://$BUCKET/models/$VERSION/
#   2. assemble the function source from the repo (single source of truth -
#      nothing is copied into cloud_function/ by hand, so it can't drift)
#   3. deploy the Cloud Function (gen2, HTTP, private)
#   4. create or update the Cloud Scheduler job (weekly trigger; the function
#      itself only works every 2nd week)
#
# Check REGION / FUNCTION / JOB match what is already deployed:
#     gcloud functions list --project=$PROJECT
#     gcloud scheduler jobs list --project=$PROJECT --location=$REGION
set -euo pipefail

# Windows / Git Bash: use the installed Cloud SDK even if it isn't on PATH
SDK_BIN="/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin"
if ! command -v gcloud >/dev/null 2>&1 && [ -x "$SDK_BIN/gcloud.cmd" ]; then
  export PATH="$SDK_BIN:$PATH"
  gcloud() { gcloud.cmd "$@"; }
fi

PROJECT=geopolitical-risk-forecasting
REGION=us-central1
FUNCTION=run-pipeline        # the existing v1.0 function - deploying replaces it
JOB=weekly-pipeline-run      # the existing scheduler job (was Mondays 10:00 UTC) - updated in place
BUCKET=geopolitical-risk-forecasting-models
VERSION=v2.2                 # release folder in GCS: goldstein v2.2 + conflict-surge v2.2
GOLDSTEIN=v2.2
SCHEDULE="0 6 * * 2"   # Tuesdays 06:00 UTC: >1 day after the as-of week ends, so GDELT has caught up

# ---- 0. the release must still reproduce its own test predictions
PY=.venv/bin/python                                   # macOS / Linux venv
[ -x "$PY" ] || PY=.venv/Scripts/python.exe           # Windows venv
"$PY" verify_live_pipeline.py 2>/dev/null | tail -1 | grep -q PASS \
  || { echo "verify_live_pipeline.py did not PASS - not deploying"; exit 1; }

# ---- 1. model files
gcloud storage cp \
  model_release_${GOLDSTEIN}_goldstein_s*.weights.h5 \
  model_release_v2.2_surge_s*.weights.h5 \
  artifacts/goldstein_${GOLDSTEIN}_preprocessing.joblib \
  artifacts/surge_v2.2_preprocessing.joblib \
  artifacts/risk_v1_calibration.json \
  "gs://${BUCKET}/models/${VERSION}/" --project="$PROJECT"

# ---- ACLED credentials for the risk index: attached only if they exist in Secret Manager
# (without them the function still forecasts; its risk step reports "skipped").
if gcloud secrets describe acled-email --project="$PROJECT" >/dev/null 2>&1 \
   && gcloud secrets describe acled-password --project="$PROJECT" >/dev/null 2>&1; then
  SECRETS_FLAG="--set-secrets=ACLED_EMAIL=acled-email:latest,ACLED_PASSWORD=acled-password:latest"
  echo "ACLED secrets found - risk index enabled"
else
  SECRETS_FLAG="--clear-secrets"
  echo "ACLED secrets not in Secret Manager - deploying without them (risk index will be skipped)"
fi

# ---- 2. function source
BUILD=$(mktemp -d)
cp cloud_function/main.py cloud_function/gdelt_refresh.py cloud_function/requirements.txt \
   live_pipeline.py honest_common.py model_architectures.py risk_index.py target_variants.py "$BUILD/"
echo "Function source: $(ls "$BUILD" | tr '\n' ' ')"

# ---- 3. deploy
gcloud functions deploy "$FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
  --runtime=python312 --source="$BUILD" --entry-point=run_pipeline \
  --trigger-http --no-allow-unauthenticated \
  --memory=4GiB --cpu=2 --timeout=3600s \
  --set-env-vars=MODEL_VERSION="$VERSION" \
  $SECRETS_FLAG

URL=$(gcloud functions describe "$FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
      --format='value(serviceConfig.uri)')
SA=$(gcloud functions describe "$FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
     --format='value(serviceConfig.serviceAccountEmail)')

# ---- 4. scheduler (OIDC-authenticated call; no retries, so a slow run is never started twice)
if gcloud scheduler jobs describe "$JOB" --project="$PROJECT" --location="$REGION" >/dev/null 2>&1; then
  ACTION=update
else
  ACTION=create
fi
gcloud scheduler jobs $ACTION http "$JOB" --project="$PROJECT" --location="$REGION" \
  --schedule="$SCHEDULE" --time-zone=UTC --uri="$URL" --http-method=POST \
  --oidc-service-account-email="$SA" --oidc-token-audience="$URL" \
  --attempt-deadline=1800s --max-retry-attempts=0

echo
echo "Deployed $FUNCTION ($VERSION). Trigger a run now (ignores the every-2-weeks rule):"
echo "  curl -X POST -d '' -H \"Authorization: bearer \$(gcloud auth print-identity-token)\" \"$URL?force=1\""
