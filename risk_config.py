"""
Shared configuration for the v2 risk pipeline (build_risk_panel.py,
build_gdelt_panel.py, build_ucdp_country_week.py, risk_features.py,
train_risk_gbm.py, predict_risk.py).

Country codes are GDELT/CAMEO actor country codes, which match ISO 3166
alpha-3 for every country listed here.
"""

MENA = [
    "ARE", "BHR", "CYP", "DJI", "EGY", "IRN", "IRQ", "ISR", "JOR", "KWT",
    "LBN", "LBY", "OMN", "PSE", "QAT", "SAU", "SDN", "SYR", "TUR", "YEM",
]

CHINA = "CHN"

# Host countries with large Chinese enterprise exposure (Belt and Road
# projects, mining, energy, construction) outside MENA. Edit freely - every
# script reads this list. More countries = more dyads (N^2): 50 countries is
# 2,500 dyads x ~600 weeks, still fine for LightGBM.
EXTRA_HOSTS = [
    # Central / South Asia
    "PAK", "AFG", "KAZ", "UZB", "KGZ", "TJK", "BGD", "LKA", "NPL",
    # Southeast Asia
    "IDN", "MYS", "MMR", "LAO", "KHM", "THA", "PHL",
    # Africa
    "ETH", "KEN", "NGA", "AGO", "ZMB", "COD", "TZA", "ZAF", "MOZ", "DZA",
    "MAR", "MLI", "NER", "SOM", "SSD",
    # Europe / Latin America
    "SRB", "RUS", "BRA", "PER", "VEN", "ECU",
]

# Countries pulled by build_gdelt_panel.py (all ordered pairs among them).
GDELT_COUNTRIES = MENA + [CHINA, "USA"] + EXTRA_HOSTS

# ISO3 -> UCDP GED `country` name. UCDP records events in Gaza and the West
# Bank under "Israel"; build_ucdp_country_week.py moves them to PSE using
# adm_1. A country with no UCDP events simply has zero recorded deaths.
UCDP_COUNTRY_NAMES = {
    "ARE": "United Arab Emirates", "BHR": "Bahrain", "CYP": "Cyprus",
    "DJI": "Djibouti", "EGY": "Egypt", "IRN": "Iran", "IRQ": "Iraq",
    "ISR": "Israel", "JOR": "Jordan", "KWT": "Kuwait", "LBN": "Lebanon",
    "LBY": "Libya", "OMN": "Oman", "QAT": "Qatar", "SAU": "Saudi Arabia",
    "SDN": "Sudan", "SYR": "Syria", "TUR": "Turkey", "YEM": "Yemen (North Yemen)",
    "CHN": "China", "USA": "United States of America",
    "PAK": "Pakistan", "AFG": "Afghanistan", "KAZ": "Kazakhstan",
    "UZB": "Uzbekistan", "KGZ": "Kyrgyzstan", "TJK": "Tajikistan",
    "BGD": "Bangladesh", "LKA": "Sri Lanka", "NPL": "Nepal",
    "IDN": "Indonesia", "MYS": "Malaysia", "MMR": "Myanmar (Burma)",
    "LAO": "Laos", "KHM": "Cambodia (Kampuchea)", "THA": "Thailand",
    "PHL": "Philippines", "ETH": "Ethiopia", "KEN": "Kenya", "NGA": "Nigeria",
    "AGO": "Angola", "ZMB": "Zambia", "COD": "DR Congo (Zaire)",
    "TZA": "Tanzania", "ZAF": "South Africa", "MOZ": "Mozambique",
    "DZA": "Algeria", "MAR": "Morocco", "MLI": "Mali", "NER": "Niger",
    "SOM": "Somalia", "SSD": "South Sudan", "SRB": "Serbia (Yugoslavia)",
    "RUS": "Russia (Soviet Union)", "BRA": "Brazil", "PER": "Peru",
    "VEN": "Venezuela", "ECU": "Ecuador",
}
PSE_ADM1_IN_ISRAEL = {"Gaza Strip", "West Bank"}

# UCDP candidate data is published monthly with a lag, so a forecast made in
# week t can only see UCDP events up to about week t-4. Features use this lag
# at training time too, so the model never relies on data that would not be
# available live.
UCDP_REPORTING_LAG_WEEKS = 4

PANEL_START = "2014-12-29"  # first Monday in the original GDELT exports

# File locations
RISK_PANEL_PATH = "raw data/risk_panel.csv"
GDELT_PANEL_PATH = "raw data/gdelt_panel.csv"
UCDP_DIR = "raw data/ucdp"
UCDP_COUNTRY_WEEK_PATH = "raw data/ucdp_country_week.csv"
GOLDSTEIN_WEIGHTS_PATH = "raw data/goldstein_root_weights.json"
MODEL_DIR = "models/risk_gbm"
OUTPUT_DIR = "outputs"
