"""Download functions for the trade module's external sources. Used both locally
(fetch_imf_imts.py, fetch_portwatch.py) and by the Cloud Function
(cloud_function/trade_refresh.py), so the two can never drift.

  imf_imts()   IMF IMTS monthly bilateral goods trade, China-reported and partner-reported
  portwatch()  IMF PortWatch daily chokepoint transits (public ArcGIS service, no key needed;
               daily data from 2019-01-01)
"""
import io
import re
import sys

import pandas as pd
import requests

from config import CHINA, IMF_CODE, IMF_TO_ISO3, PARTNERS, WORLD

IMF_BASE = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/IMTS/1.0.0"
IMF_INDICATORS = "XG_FOB_USD+MG_CIF_USD"
IMF_START = "2017-01"

PORTWATCH_URL = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
                 "Daily_Chokepoints_Data/FeatureServer/0/query")
# Chokepoints on China's sea routes to the Gulf, Red Sea and Mediterranean
CHOKEPOINTS = {"chokepoint6": "hormuz", "chokepoint4": "bab_el_mandeb", "chokepoint1": "suez",
               "chokepoint7": "cape_of_good_hope", "chokepoint5": "malacca"}
PORTWATCH_FIELDS = ["date", "portid", "portname", "n_tanker", "n_cargo", "n_total",
                    "capacity_tanker", "capacity_cargo", "capacity"]


# ------------------------------------------------------------ IMF IMTS

def _imf_get(key):
    # key order: COUNTRY.INDICATOR.COUNTERPART_COUNTRY.FREQUENCY ('+' joins values)
    r = requests.get(f"{IMF_BASE}/{key}", params={"c[TIME_PERIOD]": f"ge:{IMF_START}", "attributes": "all"},
                     headers={"Accept": "text/csv"}, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"IMF API {r.status_code} for {key}\n{r.text[:500]}")
    return pd.read_csv(io.StringIO(r.text))


def _code(series):
    """Keep the leading code if the column is 'CODE: label' style."""
    return series.astype(str).str.extract(r"^\s*([A-Za-z0-9_]+)", expand=False)


def _find(df, name):
    for c in df.columns:
        if c.upper() == name or c.upper().startswith(name + ":") or c.upper().startswith(name + " "):
            return c
    raise KeyError(f"Column {name} not found. Columns returned: {list(df.columns)}")


def _tidy(df, side):
    out = pd.DataFrame({
        "reporter": _code(df[_find(df, "COUNTRY")]).replace(IMF_TO_ISO3),
        "indicator": _code(df[_find(df, "INDICATOR")]),
        "counterpart": _code(df[_find(df, "COUNTERPART_COUNTRY")]).replace(IMF_TO_ISO3),
        "period": df[_find(df, "TIME_PERIOD")].astype(str).str.replace("-M", "-", regex=False),
        "value": pd.to_numeric(df[_find(df, "OBS_VALUE")], errors="coerce"),
        "side": side,
    })
    for c in df.columns:  # keep status/scale attributes for later inspection
        if re.match(r"(?i)^(STATUS|SCALE|UNIT|DERIVATION)", c):
            out[c] = df[c]
    return out


def imf_imts():
    """Long table: reporter, indicator, counterpart, period (YYYY-MM), value (full USD), side,
    attribute columns. side = 'china' (China reports) or 'partner' (partner reports, mirror)."""
    imf_partners = [IMF_CODE.get(p, p) for p in PARTNERS]  # e.g. PSE -> WBG
    china = _tidy(_imf_get(f"{CHINA}.{IMF_INDICATORS}.{'+'.join(imf_partners + [WORLD])}.M"), "china")
    partner = _tidy(_imf_get(f"{'+'.join(imf_partners)}.{IMF_INDICATORS}.{CHINA}.M"), "partner")
    df = pd.concat([china, partner], ignore_index=True)
    return df.drop_duplicates(["reporter", "indicator", "counterpart", "period", "side"], keep="last")


# ------------------------------------------------------------ PortWatch

def portwatch(chokepoints=CHOKEPOINTS, start="2019-01-01", page=1000):
    """Daily transit calls and capacity (metric tons) per chokepoint, paged (service max 1000 rows)."""
    frames = []
    for pid in chokepoints:
        offset = 0
        while True:
            r = requests.get(PORTWATCH_URL, timeout=120, params={
                "where": f"portid='{pid}' AND date >= DATE '{start}'", "outFields": ",".join(PORTWATCH_FIELDS),
                "orderByFields": "date", "resultOffset": offset, "resultRecordCount": page,
                "returnGeometry": "false", "f": "json"})
            r.raise_for_status()
            js = r.json()
            if "error" in js:
                raise RuntimeError(f"PortWatch error for {pid}: {js['error']}")
            rows = [f["attributes"] for f in js.get("features", [])]
            frames.append(pd.DataFrame(rows))
            if len(rows) < page and not js.get("exceededTransferLimit"):
                break
            offset += len(rows)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df.date).dt.date
    df["chokepoint"] = df.portid.map(chokepoints)
    return df.sort_values(["chokepoint", "date"]).drop_duplicates(["portid", "date"]).reset_index(drop=True)


def portwatch_weekly(daily):
    """Monday weeks (same calendar as the GDELT panel); only complete 7-day weeks."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d.date)
    d["week_start"] = d.date - pd.to_timedelta(d.date.dt.weekday, unit="D")
    num = [c for c in PORTWATCH_FIELDS if c.startswith(("n_", "capacity"))]
    w = d.groupby(["chokepoint", "week_start"]).agg(days=("date", "nunique"), **{c: (c, "sum") for c in num})
    return w[w.days == 7].drop(columns="days").reset_index()


if __name__ == "__main__":
    sys.exit("import this module; run fetch_imf_imts.py or fetch_portwatch.py")
