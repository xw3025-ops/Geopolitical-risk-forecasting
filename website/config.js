// Where the site reads its data. dataUrl: the bundle the Cloud Function publishes after each
// refresh (cloud_function/main.py, ?site=1). fallbackUrl: the copy committed with the site
// (build_site_data.py), used when dataUrl cannot be reached - and when previewing locally.
window.SITE_CONFIG = {
  dataUrl: "https://storage.googleapis.com/geopolitical-risk-forecasting-site/site/latest.json",
  fallbackUrl: "data/latest.json",
  mapUrl: "vendor/countries-50m.json",
};
