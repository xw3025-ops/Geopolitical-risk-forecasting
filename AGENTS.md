# Handover note for AI agents (and developers)

Last updated 2026-11-12. Read this first in a new session. Owners: Lixing XIE (lx2339@columbia.edu), Xingyi WANG
(xw3025@columbia.edu), Risk-A-Lab, Columbia. GitHub: `xw3025-ops/Geopolitical-risk-forecasting` (private).

## 1. What this is and where it stands

The **MENA Risk Monitor**: a transformer forecasts all bilateral relations among 20 Middle East / North Africa
states (GDELT events, 13 weeks ahead, plus a "worst month" conflict-surge flag); a site builder combines the
forecasts with ACLED violence, IMF trade and PortWatch shipping into a country risk score and DeepSeek-written
bilingual briefs; a static website shows it. Built for a competition (finished 2nd).

**State since 2026-11-12: frozen demonstration.**
- Website https://mena-risk-monitor.vercel.app is live but frozen on the refresh of 2026-10-27 (forecasts as of
  week 2026-10-12). `website/config.js`: `dataUrl: "data/latest.json"`, `frozen: true` (shows a "Refresh paused"
  pill + contact cards). The site does not read the cloud bucket.
- Cloud Scheduler jobs **paused** (`weekly-pipeline-run`, `trade-refresh-weekly`, `weekly-site-refresh`); Cloud
  Functions `run-pipeline`, `trade-refresh` deployed but idle. Resume steps: `release/STATE_FROZEN.md`.
- ACLED moved to cursor paging (Nov 2026); fixed in `risk_index.acled_read` (used by `build_site_data.py` too),
  tested locally, **not deployed**. Redeploy (`cloud_function/deploy.sh`) before resuming.

## 2. Repository map (what matters)

| Path | Role |
|---|---|
| `release_v2.2_goldstein.py`, `release_v2.2_surge.py` | train the production models (v2.2); weights `model_release_v2.2_*_s{42,1,2,3,4}.weights.h5`, preprocessing in `artifacts/` |
| `live_pipeline.py` | inference used in production (same code verified by `verify_live_pipeline.py`) |
| `honest_common.py`, `model_architectures.py`, `target_variants.py`, `risk_index.py` | shared data, model, targets, ACLED / risk index |
| `build_site_data.py` | builds the site bundle (`build_bundle`, `publish`); `--gcs` publishes |
| `cloud_function/` | `main.py` (forecast run; `?site=1` = site refresh), `gdelt_refresh.py`, `trade_refresh.py`, deploy and setup scripts |
| `src/` | trade module (Xingyi): IMF, PortWatch, Comtrade, regressions, trade baseline |
| `website/` | static site: `index.html`, `app.js` (UI, routes, EN/ZH text), `map3d.js` (Three.js map), `styles.css`, `config.js`, `data/latest.json` (frozen snapshot), `vendor/`, `img/` |
| `paper/` | `build_paper.py` (merged EN paper -> `risk_paper.docx`), `build_paper_zh.py` (Chinese, <5,000 Word count), `build_docx.py` (standalone transformer section), stats/figure scripts, PDFs |
| `release/` | `build_release.py` -> `dist/mena-risk-monitor-<ver>-<date>.zip` (curated production package with checksums) + its doc templates |
| `legacy/`, `train_*.py`, `experiment_*.py`, `benchmark_*.py` | research history; not production |

## 3. Local data that is NOT in git (do not delete; back up)

`logs/` (experiment results the paper scripts read, e.g. `ladder_results.json`, `goldstein_experiments.jsonl`),
`artifacts/experiments/` (experiment predictions), `data/*.parquet` (GDELT dyad-week panel, needed by
`verify_live_pipeline.py` and training), `data/news_cache/`, `data/imf_imts/`, `data/portwatch/`, `data/processed/`,
`models/v2.0/`, `.env` (API keys), `website/.vercel/` (Vercel project link). Most can be rebuilt from APIs/BigQuery,
but `logs/` and `artifacts/experiments/` are results of past runs and cannot be recreated cheaply.

## 4. Environment (Windows 11)

- Shells: PowerShell and Git Bash. Python venv `.venv` (Python 3.12): `.venv/Scripts/python.exe`. GPU training
  with `KERAS_BACKEND=torch` (CUDA torch); the Cloud Function uses TensorFlow.
- gcloud: `C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin` (Bash: call `gcloud.cmd`); project
  `geopolitical-risk-forecasting`, region `us-central1`. Auth expires: the user must run `gcloud auth login`.
- Node.js: `C:\Program Files\nodejs` (not on Git Bash PATH). Deploy the site from `website/`:
  `npx vercel deploy --prod --yes` (project `mena-risk-monitor`, team `geo-risk-forecasting`). GitHub pushes do
  not deploy the site.
- Headless Chrome for screenshots: `chrome.exe --headless=new --lang=zh-CN|en-US --user-data-dir=<abs path>`.
  Site preview helpers: `?2d`, `?focus=IRN`, `?alert` (opens the paused notice), `?local`.
- Microsoft Word via COM (PowerShell) exports PDFs and renders the Chinese paper's tables; documents open in Word
  are locked and cannot be rebuilt.

## 5. Cloud resources

Buckets: `geopolitical-risk-forecasting-models` (models under `models/v2.2/`), `geopolitical-risk-forecasting-site`
(public, CORS; `site/latest.json`, `site/archive/<as_of>.json`). BigQuery: table `1.1` (GDELT dyad-week panel),
`predictions.forecasts_v22`, `predictions.risk_country_v1`, `risk.acled_country_week`, `trade.imf_imts_raw`,
`trade.portwatch_daily`. Secret Manager: `acled-email`, `acled-password`, `deepseek-api-key`, `guardian-api-key`,
`nyt-api-key` (`cloud_function/set_secrets.sh` copies them from `.env`). Stay within the Google free tier.

## 6. Working rules from the owners

- Commit and push only when asked; push to `main` directly (no branches). End commit messages with the
  Co-Authored-By line the session specifies.
- Never print secret values; check only whether keys are set. Ask before changing cloud resources.
- One heavy local job at a time; GPU only when safe. Google free tier only.
- Paper tone: a confident methodological contribution, neither hyped nor self-deprecating.
- Chinese paper: under 5,000 by Word's count, mainland academic style, 2-character first-line indent, tables as
  images. Both paper builders fail on unfilled f-string placeholders, by design.

## 7. Pitfalls met before

- Git Bash heredocs mangle backslashes and `\n` inside Python: write patch scripts to a file and run them.
- python-docx: heading styles use theme fonts (strip `w:asciiTheme` etc. to force fonts); `List Number` continues
  numbering across a document (use the manual `numbered()` helpers); merged cells repeat in `row.cells`.
- ACLED: OAuth token + cursor paging (`next_cursor` -> `cursor`). LinkedIn blocks anonymous downloads.
- Site data archive in the bucket keeps one file per forecast week (later refreshes overwrite it).
