# Release notes — {{VERSION}}

| | |
|---|---|
| Packaged | {{DATE}} |
| Source | GitHub `xw3025-ops/Geopolitical-risk-forecasting`, commit `{{COMMIT}}` |
| Uncommitted changes in packaged files | {{DIRTY}} |
| Production site | {{SITE_URL}} |
| Status | {{STATUS}} |
| Data snapshot | forecasts as of the week of {{AS_OF}}, built {{GENERATED}} |

{{DIRTY_LIST}}
## What is in production

- **Relationship forecast (Goldstein v2.2).** Mean Goldstein cooperation–conflict score of each of 400 directed
  MENA dyads over the next 13 weeks. Temporal attention over each dyad's 13-week history, then cross-system
  attention across 1,959 dyads (MENA and MENA–world), residual on the 52-week mean; trained with 5,122 additional
  world dyads; 5-seed ensemble. Held-out test R² {{R2}} (March 2024 – June 2026).
- **Conflict-surge flag (v2.2).** Probability that a dyad's worst four-week conflict intensity in the next 13
  weeks exceeds its typical quarterly peak by more than the largest tenth of historical gaps; the ten highest
  each week are alerts. Replaces the v2.0 four-week direction model.
- **Country risk score.** √(conflict severity × China trade at stake), with severity = 0.40 forecast hostility +
  0.30 forecast surge risk + 0.30 recent ACLED violence. Shipping-lane status and Hormuz exposure reported beside it.
- **Website.** Bilingual (English / Chinese) monitor: 3D regional map, country pages, shipping lanes, Data & model
  page, CSV / JSON export, DeepSeek-written China-exposure briefs citing Guardian and NYT coverage.
- **Automation.** Forecasts every second Tuesday; trade, shipping, scores and briefs weekly; the site reads the
  latest data on every visit.

## Known limitations

- GDELT 2.0 begins in 2015: few region-wide crises to learn from; results are a lower bound for the method.
- Surge probabilities are calibrated on 2023 and run high in 2024–2026; the ranking and alerts are unaffected.
- The trade effect of a Hormuz closure is identified from the 2026 episode only.
- IMF trade data lag three to four months; forecasts are current to the last complete week.
- Briefs are machine-written from the listed inputs and cited articles.
- Not reachable from mainland China in its current hosting (see DEPLOYMENT.md, section 3).
- ACLED switched its API from page numbers to cursor paging in November 2026; the deployed code (commit `90637c6`,
  which produced the frozen snapshot of 27 October) sends `page` and is now rejected, so the refreshes of 3 and 10
  November ran without ACLED. This package contains the fixed pipeline (`pipeline/risk_index.py`, `acled_read`):
  identical queries, fields and outputs, cursor paging only; tested on the snapshot's window. It is not yet
  deployed to the paused Cloud Function: redeploy before resuming (DEPLOYMENT.md, section 0).
