#!/usr/bin/env bash
# One-time setup of the public bucket the website reads its data from. Run from the repo root
# after `gcloud auth login` (re-running is harmless):
#
#     bash cloud_function/setup_site.sh
#
#   1. create gs://$SITE_BUCKET (us-central1: inside Cloud Storage's always-free 5 GB)
#   2. make its objects publicly readable - the bundle holds only what the site shows anyway
#   3. allow browsers on any origin to fetch it (CORS), so Vercel previews work too
#   4. let the forecasting function's service account write to it
# The function publishes site/latest.json (+ site/archive/<week>.json) on every ?site=1 run.
set -euo pipefail

SDK_BIN="/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin"
if ! command -v gcloud >/dev/null 2>&1 && [ -x "$SDK_BIN/gcloud.cmd" ]; then
  export PATH="$SDK_BIN:$PATH"
  gcloud() { gcloud.cmd "$@"; }
fi

PROJECT=geopolitical-risk-forecasting
REGION=us-central1
FUNCTION=run-pipeline
SITE_BUCKET=geopolitical-risk-forecasting-site

if ! gcloud storage buckets describe "gs://$SITE_BUCKET" --project="$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$SITE_BUCKET" --project="$PROJECT" --location="$REGION" \
    --uniform-bucket-level-access --default-storage-class=STANDARD
fi
gcloud storage buckets add-iam-policy-binding "gs://$SITE_BUCKET" \
  --member=allUsers --role=roles/storage.objectViewer >/dev/null

CORS=$(mktemp)
cat > "$CORS" <<'EOF'
[{"origin": ["*"], "method": ["GET", "HEAD"], "responseHeader": ["Content-Type"], "maxAgeSeconds": 3600}]
EOF
gcloud storage buckets update "gs://$SITE_BUCKET" --cors-file="$CORS"

SA=$(gcloud functions describe "$FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
     --format='value(serviceConfig.serviceAccountEmail)')
gcloud storage buckets add-iam-policy-binding "gs://$SITE_BUCKET" \
  --member="serviceAccount:$SA" --role=roles/storage.objectAdmin >/dev/null

echo "gs://$SITE_BUCKET is ready; the website reads"
echo "  https://storage.googleapis.com/$SITE_BUCKET/site/latest.json"
