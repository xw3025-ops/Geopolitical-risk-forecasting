"""Pull monthly bilateral goods trade from the IMF IMTS SDMX 3.0 API.

Output: data/imf_imts/imts_raw.csv  (long format)
  reporter, indicator, counterpart, period (YYYY-MM), value (as published), side
  side = 'china'   -> China is the reporter (中国申报)
         'partner' -> the partner reports its trade with China (对方申报, mirror)

Indicators: XG_FOB_USD (exports, FOB) and MG_CIF_USD (imports, CIF).
Run:  python src/fetch_imf_imts.py
"""

import io
import re
import sys

import pandas as pd
import requests

from config import CHINA, IMF_CODE, IMF_TO_ISO3, PARTNERS, RAW_IMF, WORLD

BASE = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/IMTS/1.0.0"
INDICATORS = "XG_FOB_USD+MG_CIF_USD"
START = "2017-01"


def get(key: str) -> pd.DataFrame:
    # key order: COUNTRY.INDICATOR.COUNTERPART_COUNTRY.FREQUENCY ('+' joins values)
    url = f"{BASE}/{key}"
    r = requests.get(
        url,
        params={"c[TIME_PERIOD]": f"ge:{START}", "attributes": "all"},
        headers={"Accept": "text/csv"},
        timeout=180,
    )
    if r.status_code != 200:
        sys.exit(f"IMF API {r.status_code} for {key}\n{r.text[:500]}")
    return pd.read_csv(io.StringIO(r.text))


def code(series: pd.Series) -> pd.Series:
    """Keep the leading code if the column is 'CODE: label' style."""
    return series.astype(str).str.extract(r"^\s*([A-Za-z0-9_]+)", expand=False)


def find(df: pd.DataFrame, name: str) -> str:
    for c in df.columns:
        if (
            c.upper() == name
            or c.upper().startswith(name + ":")
            or c.upper().startswith(name + " ")
        ):
            return c
    sys.exit(f"Column {name} not found. Columns returned: {list(df.columns)}")


def tidy(df: pd.DataFrame, side: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "reporter": code(df[find(df, "COUNTRY")]).replace(IMF_TO_ISO3),
            "indicator": code(df[find(df, "INDICATOR")]),
            "counterpart": code(df[find(df, "COUNTERPART_COUNTRY")]).replace(
                IMF_TO_ISO3
            ),
            "period": df[find(df, "TIME_PERIOD")]
            .astype(str)
            .str.replace("-M", "-", regex=False),
            "value": pd.to_numeric(df[find(df, "OBS_VALUE")], errors="coerce"),
            "side": side,
        }
    )
    # keep any status/scale attributes for later inspection
    for c in df.columns:
        if re.match(r"(?i)^(STATUS|SCALE|UNIT|DERIVATION)", c):
            out[c] = df[c]
    return out


def main():
    RAW_IMF.mkdir(parents=True, exist_ok=True)
    imf_partners = [IMF_CODE.get(p, p) for p in PARTNERS]  # e.g. PSE -> WBG
    partners = "+".join(imf_partners + [WORLD])
    china_side = tidy(get(f"{CHINA}.{INDICATORS}.{partners}.M"), "china")
    partner_side = tidy(
        get(f"{'+'.join(imf_partners)}.{INDICATORS}.{CHINA}.M"), "partner"
    )
    df = pd.concat([china_side, partner_side], ignore_index=True)
    df = df.drop_duplicates(
        ["reporter", "indicator", "counterpart", "period", "side"], keep="last"
    )
    df.to_csv(RAW_IMF / "imts_raw.csv", index=False)
    print(f"{len(df):,} rows -> {RAW_IMF / 'imts_raw.csv'}")
    print(df.groupby("side").period.agg(["min", "max"]))
    missing = sorted(set(PARTNERS) - set(df[df.side == "china"].counterpart))
    print("Partners with no China-reported data:", missing or "none")
    print(
        "Spot-check units: compare one China-reported month vs China customs (values assumed full USD)."
    )


if __name__ == "__main__":
    main()
