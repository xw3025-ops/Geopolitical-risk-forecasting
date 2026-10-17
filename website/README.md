# MENA Risk Monitor (website)

A static site: `index.html` + `app.js` / `map3d.js` / `styles.css`, the data bundle
`data/latest.json` and the world map `vendor/countries-50m.json`. No build step.

## Run locally

```bash
python -m http.server 8765 --directory website
# open http://localhost:8765
```

Preview helpers: `?2d` forces the 2D map, `?focus=IRN` shows a hover state, `#/about` is the
Data & model page.

## Where the data comes from

`config.js`: the site loads `dataUrl`, the bundle the Cloud Function publishes
(`https://storage.googleapis.com/geopolitical-risk-forecasting-site/site/latest.json`), and falls
back to the committed `data/latest.json` if that cannot be reached. On localhost (or with `?local`)
it reads the committed copy directly.

## Refresh the data

In production nothing is manual: Cloud Scheduler calls `run-pipeline?site=1` every Tuesday 08:00
UTC (after the forecasts and the trade refresh), which rebuilds the bundle and publishes it; the
next page load shows it. Set up once with `cloud_function/set_secrets.sh`,
`cloud_function/deploy.sh` and `cloud_function/setup_site.sh`.

Locally:

```bash
python build_site_data.py          # from the repo root; needs gcloud auth + the keys in .env
python build_site_data.py --gcs    # ... and publish it to the live bucket
```

Rebuilds `website/data/latest.json` from BigQuery (release v2.2 forecasts), ACLED, IMF trade and
PortWatch (local CSVs if present, else BigQuery `trade.*`), Guardian / NYT news and DeepSeek (AI
briefs). See `.env.example` for the keys.

## Deploy on Vercel

New project -> import this GitHub repository -> **Root Directory: `website`**, Framework
Preset: **Other**, no build command, no output directory. Every push then gets its own preview
URL; pushes to `main` update production. `vercel.json` sets cache headers. Data refreshes do not
need a redeploy.
