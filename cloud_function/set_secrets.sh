#!/usr/bin/env bash
# Copy the API keys from the local .env into Google Secret Manager, so the Cloud Function can read
# them. Run from the repo root after `gcloud auth login`:
#
#     bash cloud_function/set_secrets.sh
#
# Values are never printed. A secret only gets a new version when its value changed, and older
# versions are destroyed, so the project keeps <= 5 active versions (Secret Manager's free tier
# is 6 active versions and 10,000 accesses a month). Keys missing from .env are skipped.
# Then run cloud_function/deploy.sh, which attaches whichever secrets exist.
set -euo pipefail

SDK_BIN="/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin"
if ! command -v gcloud >/dev/null 2>&1 && [ -x "$SDK_BIN/gcloud.cmd" ]; then
  export PATH="$SDK_BIN:$PATH"
  gcloud() { gcloud.cmd "$@"; }
fi

PROJECT=geopolitical-risk-forecasting
REGION=us-central1
FUNCTION=run-pipeline

[ -f .env ] || { echo "no .env in $(pwd) - run from the repo root"; exit 1; }
gcloud services enable secretmanager.googleapis.com --project="$PROJECT"

# the function's runtime service account (default compute account if not deployed yet)
SA=$(gcloud functions describe "$FUNCTION" --gen2 --project="$PROJECT" --region="$REGION" \
     --format='value(serviceConfig.serviceAccountEmail)' 2>/dev/null || true)
if [ -z "$SA" ]; then
  NUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
  SA="${NUM}-compute@developer.gserviceaccount.com"
fi
echo "Function service account: $SA"

env_value() {  # KEY -> value from .env (quotes and Windows line endings stripped)
  grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d '\r' | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

for pair in ACLED_EMAIL=acled-email ACLED_PASSWORD=acled-password DEEPSEEK_API_KEY=deepseek-api-key \
            GUARDIAN_API_KEY=guardian-api-key NYT_API_KEY=nyt-api-key; do
  key=${pair%%=*}; name=${pair#*=}
  val=$(env_value "$key")
  if [ -z "$val" ]; then
    echo "$key: not in .env - skipped"
    continue
  fi
  if ! gcloud secrets describe "$name" --project="$PROJECT" >/dev/null 2>&1; then
    gcloud secrets create "$name" --project="$PROJECT" --replication-policy=automatic >/dev/null
  fi
  cur=$(gcloud secrets versions access latest --secret="$name" --project="$PROJECT" 2>/dev/null || true)
  if [ "$cur" = "$val" ]; then
    echo "$key: unchanged"
  else
    printf '%s' "$val" | gcloud secrets versions add "$name" --project="$PROJECT" --data-file=- >/dev/null
    latest=$(gcloud secrets versions list "$name" --project="$PROJECT" --filter='state=ENABLED' \
             --sort-by='~createTime' --limit=1 --format='value(name.basename())')
    for v in $(gcloud secrets versions list "$name" --project="$PROJECT" --filter='state!=DESTROYED' \
               --format='value(name.basename())'); do
      [ "$v" = "$latest" ] || gcloud secrets versions destroy "$v" --secret="$name" --project="$PROJECT" --quiet >/dev/null
    done
    echo "$key: stored in secret $name"
  fi
  gcloud secrets add-iam-policy-binding "$name" --project="$PROJECT" \
    --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor >/dev/null
done
echo "Done. Now run: bash cloud_function/deploy.sh"
