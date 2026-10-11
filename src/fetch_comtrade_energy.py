"""China's 2024 imports of crude oil (HS 2709) and petroleum gases (HS 2711) by partner.

This is the ONLY Comtrade series the pipeline still needs: IMF IMTS has no
commodity detail, and Comtrade has no China monthly data after 2024-12.

Needs COMTRADE_KEY in .env.  Output: data/comtrade/china_energy_2024.csv
  iso3, crude_bn, gas_bn   (USD billions, primaryValue)
Run:  python src/fetch_comtrade_energy.py
"""
import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

from config import M49, RAW_COMTRADE, ROOT

load_dotenv(ROOT / ".env")
KEY = os.getenv("COMTRADE_KEY") or sys.exit("COMTRADE_KEY missing in .env")
URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
INV = {v: k for k, v in M49.items()}


def main():
    RAW_COMTRADE.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, headers={"Ocp-Apim-Subscription-Key": KEY}, timeout=120, params={
        "reporterCode": 156, "period": 2024, "flowCode": "M",
        "partnerCode": ",".join(map(str, M49.values())),
        "cmdCode": "2709,2711", "includeDesc": "true"})
    r.raise_for_status()
    d = pd.DataFrame(r.json().get("data", []))
    if d.empty:
        sys.exit("Comtrade returned no rows")
    d["iso3"] = d.partnerCode.map(INV)
    d = d[d.iso3.notna()]
    p = d.pivot_table(index="iso3", columns="cmdCode", values="primaryValue", aggfunc="sum").fillna(0) / 1e9
    p.columns = ["crude_bn" if str(c) == "2709" else "gas_bn" for c in p.columns]
    p = p.reindex(columns=["crude_bn", "gas_bn"], fill_value=0).reset_index()
    p.to_csv(RAW_COMTRADE / "china_energy_2024.csv", index=False)
    print(p.sort_values("crude_bn", ascending=False).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
