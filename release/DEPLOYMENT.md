# Deployment

{{STATE}}

## 1. Components

| Component | What it is | Runs on (production) | Portable to |
|---|---|---|---|
| Website | static HTML / CSS / JavaScript (`site/`), no build step | Vercel | any static host |
| Data bundle | one JSON file (~1.3 MB, served gzip at ~0.2 MB) that holds every number the site shows | Google Cloud Storage, public bucket with CORS | any public object storage or CDN |
| Forecasting run | `pipeline/cloud_function/main.py`: rebuilds the GDELT panel in BigQuery, runs the v2.2 models | Cloud Functions gen2 (Python 3.12, 4 GiB, 2 CPU, 60 min) | any container or scheduled job with Python 3.12 |
| Trade refresh | `pipeline/cloud_function/trade_refresh.py`: IMF trade and PortWatch into BigQuery `trade.*` | Cloud Functions gen2 | as above |
| Site refresh | same function with `?site=1`: scores, ACLED, news, DeepSeek briefs, publishes the bundle | Cloud Functions gen2 | as above |
| Schedules | Tue 06:00 UTC (forecasts, every 2nd week), 07:00 (trade), 08:00 (site) | Cloud Scheduler | cron, GitHub Actions, any scheduler |
| Secrets | `ACLED_EMAIL`, `ACLED_PASSWORD`, `DEEPSEEK_API_KEY`, `GUARDIAN_API_KEY`, `NYT_API_KEY` | Secret Manager | any secret store, as environment variables |

The pipeline is Python 3.12 with pinned dependencies (`pipeline/requirements.txt` for local runs,
`pipeline/cloud_function/requirements.txt` for the function). The models load with Keras 3 on TensorFlow or PyTorch.

## 2. Redeploying on Google Cloud (current setup)

Run from `pipeline/` with the Google Cloud SDK installed and `gcloud auth login` done. All scripts are idempotent.

1. `cp .env.example .env` and fill in the keys; `bash cloud_function/set_secrets.sh` copies them to Secret Manager.
2. `bash cloud_function/deploy.sh`: checks the release against its test predictions, uploads the model files,
   deploys the function and creates the forecasting and site-refresh schedules.
3. `bash cloud_function/setup_site.sh`: creates the public data bucket (CORS, public read).
4. `bash cloud_function/deploy_trade.sh`: deploys the weekly trade refresh.
5. Build the first bundle: call the function with `?site=1` (command printed by `deploy.sh`).
6. Deploy the website: from `site/`, `npx vercel --prod`, or upload `site/` to any static host.

Set the project, region and bucket names at the top of each script if deploying to a different Google Cloud project.

## 3. Moving to another platform

**Website.** Upload `site/` to any static host (Netlify, Cloudflare Pages, GitHub Pages, AWS S3 + CloudFront,
Alibaba Cloud OSS + CDN). Set `dataUrl` in `site/config.js` to wherever the bundle is published; `fallbackUrl`
keeps the bundled copy as a backup. `vercel.json` only sets cache headers; set the equivalent on the new host
(short cache for `index.html` and the bundle).

**Data bundle.** Any public object storage with CORS enabled for GET. Publishing is one function,
`publish()` in `pipeline/build_site_data.py` (about 15 lines, Google Cloud Storage client); replace it with the
new platform's SDK and keep the gzip encoding.

**Pipeline.** The function code runs anywhere Python 3.12 runs: as a container (the entry point uses
`functions-framework`), or as a scheduled job calling `python build_site_data.py --gcs` after the forecasting
step. One dependency is not cloud-neutral: GDELT is queried through its public BigQuery dataset
(`gdelt-bq.gdeltv2.events`). Either keep a Google Cloud project for those queries, or rewrite
`cloud_function/gdelt_refresh.py` to aggregate GDELT's raw event files (data.gdeltproject.org).

**Mainland China.** `vercel.app`, `storage.googleapis.com`, Google Fonts and jsDelivr are blocked or unreliable.
Host the site and the bundle in China (mainland hosting requires an ICP filing), and copy the three JavaScript
libraries and the fonts into `site/vendor/` instead of loading them from CDNs (`site/index.html`).

## 4. Verifying a deployment

- `python verify_live_pipeline.py` (from `pipeline/`) confirms that the live code reproduces the release's test
  predictions exactly. It needs the local GDELT panel (`data/*.parquet`), rebuilt from BigQuery by the
  repository's panel scripts.
- After a site refresh, the site header shows the new date; the bundle's `meta.generated_utc` gives the build time.
