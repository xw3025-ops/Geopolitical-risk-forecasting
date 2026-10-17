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

## Refresh the data

```bash
python build_site_data.py          # from the repo root; needs gcloud auth + the keys in .env
```

Rebuilds `website/data/latest.json` from BigQuery (release v2.2 forecasts), ACLED, IMF trade
(`src/fetch_imf_imts.py`), IMF PortWatch (`src/fetch_portwatch.py`), Guardian / NYT news and
DeepSeek (AI briefs). See `.env.example` for the keys.

## Deploy on Vercel

New project -> import this GitHub repository -> **Root Directory: `website`**, Framework
Preset: **Other**, no build command, no output directory. Every push then gets its own preview
URL; pushes to `main` update production. `vercel.json` sets cache headers.

To serve live data instead of the committed file, point `dataUrl` in `config.js` at the bundle
the Cloud Function publishes after each refresh.
