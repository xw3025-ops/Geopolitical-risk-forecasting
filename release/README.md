# MENA Risk Monitor — release {{VERSION}}

| | |
|---|---|
| Production site | {{SITE_URL}} |
| Status | {{STATUS}} |
| Data | {{DATA_LINE}} |
| Release | {{VERSION}}, packaged {{DATE}} from commit `{{COMMIT}}` |
| Data in this package | forecasts as of the week of {{AS_OF}}, built {{GENERATED}} |

The MENA Risk Monitor is an automated early-warning system for organisations with trade at stake in the Middle
East and North Africa. A spatio-temporal transformer forecasts every bilateral relationship in the region jointly,
13 weeks ahead, and flags relationships at risk of a conflict surge. The forecasts are combined with recent ACLED
violence, IMF trade data and IMF PortWatch shipping data into a transparent country risk score, published weekly
on a bilingual (English / Chinese) website with machine-written China-exposure briefs. The method and its
evaluation are documented in `docs/risk_paper.pdf` (English, full) and `docs/risk_paper_zh.pdf` (Chinese, summary).

## Contents

```
README.md          this file
RELEASE_NOTES.md   what is in this release, version facts, known limitations
DEPLOYMENT.md      how the production system runs, how to redeploy it and how to move it to another platform
THIRD_PARTY.md     data sources and open-source software, with their terms
MANIFEST.sha256    SHA-256 checksum of every file in the package
site/              the website exactly as deployed (static files, no build step)
pipeline/          the production code and model files, in the repository layout (scripts run from this folder)
  model_release_v2.2_*.weights.h5   release v2.2 weights (5 seeds x 2 models)
  artifacts/                        their preprocessing, calibration and test predictions
data_snapshot/     the outputs at packaging time: country scores, 400 relationship forecasts, shipping lanes,
                   ACLED events (CSV, UTF-8) and the complete data bundle (JSON)
docs/              the papers (PDF) and screenshots of the production site
```

## For reviewers

1. Open the production site; the *Data & model* page explains the method in plain language.
2. Read `docs/risk_paper.pdf`: Section 2 defines risk, Section 3 the indicator system, Sections 5–6 the evidence.
3. Browse `data_snapshot/` (opens in Excel), or preview the site offline:
   `python -m http.server 8765 --directory site`, then open http://localhost:8765.

## System overview

```
GDELT 2.0 (BigQuery) ──► forecasting run (Cloud Function, every 2nd Tuesday) ──► BigQuery forecasts
IMF trade, PortWatch ──► trade refresh (weekly)                              ──► BigQuery trade.*
                         site refresh (weekly): scores, ACLED, news, briefs  ──► public data bundle (JSON)
                                                                                   │
                         static website (any host) ◄── reads on every visit ──────┘
```

| Release facts | |
|---|---|
| Forecasting model | v2.2: temporal + cross-system attention over 1,959 directed dyads; 176,537 parameters per model, 5-seed ensemble |
| Accuracy (held-out 2024–2026) | R² 0.658 for 13-week mean Goldstein (trailing-mean baseline 0.600) |
| Surge flag | 44% of weekly top-10 alerts followed by a surge (base rate 15%) |
| Refresh | {{REFRESH}} |

## Integrity and security

- Verify the files: `sha256sum -c MANIFEST.sha256` (Linux/macOS/Git Bash). In PowerShell, compare
  `Get-FileHash <file> -Algorithm SHA256` against the manifest.
- The package contains no credentials. `pipeline/.env.example` lists the keys the system needs (names only);
  in production they are held in Google Secret Manager.
