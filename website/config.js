// Where the site reads its data. Locally: the file build_site_data.py writes.
// In production: the bundle the Cloud Function publishes after each refresh
// (set at publish time, e.g. "https://storage.googleapis.com/<public-bucket>/site/latest.json").
window.SITE_CONFIG = {
  dataUrl: "data/latest.json",
  mapUrl: "vendor/countries-50m.json",
};
