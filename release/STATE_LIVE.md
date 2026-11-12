## 0. Current state: live

The website reads the live data bundle on every visit; the three Cloud Scheduler jobs refresh it automatically
(section 1). To freeze the site on a snapshot, copy the bundle to `site/data/latest.json`, set `dataUrl` to that file
and `frozen: true` in `site/config.js`, redeploy the website, and pause the scheduler jobs.
