// Where the site reads its data.
// The refresh is paused: the site shows the frozen snapshot shipped with it (data/latest.json, the refresh of
// 27 October 2026). To go live again, set dataUrl back to the bundle the Cloud Function publishes,
//   "https://storage.googleapis.com/geopolitical-risk-forecasting-site/site/latest.json",
// set fallbackUrl to "data/latest.json" and frozen to false.
window.SITE_CONFIG = {
  dataUrl: "data/latest.json",
  fallbackUrl: null,
  mapUrl: "vendor/countries-50m.json",
  frozen: true,  // shows the "refresh paused" alert in the top bar
  contacts: [
    { name: "Xingyi WANG", email: "xw3025@columbia.edu", linkedin: "https://www.linkedin.com/in/xingyi-wang-329a77194/",
      photo: "img/xingyi-wang.jpg" },
    { name: "Lixing XIE", email: "lx2339@columbia.edu", linkedin: "https://www.linkedin.com/in/lixing-xie-025907287/",
      photo: "img/lixing-xie.jpg" },
  ],
};
