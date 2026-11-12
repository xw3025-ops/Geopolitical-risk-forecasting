## 0. Current state: frozen demonstration

The development stage is complete. Since 12 November 2026:

- **Website**: live at the production URL, frozen on the refresh of 27 October 2026 (forecasts as of the week of
  12 October). `site/config.js` has `dataUrl: "data/latest.json"` and `frozen: true`, so the site reads only the
  snapshot shipped with it and shows a "Refresh paused" notice with the developers' contacts.
- **Backend**: the three Cloud Scheduler jobs (`weekly-pipeline-run`, `trade-refresh-weekly`,
  `weekly-site-refresh`) are **paused**. The Cloud Functions (`run-pipeline`, `trade-refresh`) remain deployed and
  idle; secrets, model files and BigQuery tables are kept. No API calls are made.

**To resume the live system**

1. `gcloud auth login`, then from `pipeline/`: `bash cloud_function/deploy.sh` (deploys the current code, including
   the ACLED cursor-paging fix, and re-creates the schedules).
2. `gcloud scheduler jobs resume <job> --location=us-central1` for each of the three jobs (if still paused).
3. Trigger a site refresh (`?site=1`, command printed by `deploy.sh`) and check the bundle has ACLED events.
4. In `site/config.js` set `dataUrl` to the live bundle, `fallbackUrl: "data/latest.json"`, `frozen: false`;
   redeploy the website.
