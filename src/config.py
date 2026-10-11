"""Shared constants and paths. Run all scripts from the repo root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_IMF = ROOT / "data" / "imf_imts"  # raw IMF downloads only
RAW_COMTRADE = ROOT / "data" / "comtrade"  # raw Comtrade downloads only
PARAMS = ROOT / "data" / "hormuz"  # hand-built parameter tables
PROCESSED = ROOT / "data" / "processed"  # everything derived

# 20 MENA countries + USA
PARTNERS = [
    "ARE",
    "BHR",
    "CYP",
    "DJI",
    "EGY",
    "IRN",
    "IRQ",
    "ISR",
    "JOR",
    "KWT",
    "LBN",
    "LBY",
    "OMN",
    "PSE",
    "QAT",
    "SAU",
    "SDN",
    "SYR",
    "TUR",
    "YEM",
    "USA",
]
CHINA = "CHN"
# Where IMF uses a different code from the ISO3 used elsewhere in this project
IMF_CODE = {"PSE": "WBG"}  # West Bank and Gaza
IMF_TO_ISO3 = {v: k for k, v in IMF_CODE.items()}
WORLD = "G001"  # IMF code for "World" counterpart

# Baseline window for predetermined shares (12 months). 2024 kept for robustness.
BASE_START, BASE_END = "2025-02", "2026-01"
BASE_YEAR = "2024"

# UN Comtrade numeric (M49) codes for the 21 partners
M49 = {
    "ARE": 784,
    "BHR": 48,
    "CYP": 196,
    "DJI": 262,
    "EGY": 818,
    "IRN": 364,
    "IRQ": 368,
    "ISR": 376,
    "JOR": 400,
    "KWT": 414,
    "LBN": 422,
    "LBY": 434,
    "OMN": 512,
    "PSE": 275,
    "QAT": 634,
    "SAU": 682,
    "SDN": 729,
    "SYR": 760,
    "TUR": 792,
    "YEM": 887,
    "USA": 842,
}
