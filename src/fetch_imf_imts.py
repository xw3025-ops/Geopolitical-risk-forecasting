"""Pull monthly bilateral goods trade from the IMF IMTS SDMX 3.0 API.

Output: data/imf_imts/imts_raw.csv  (long format)
  reporter, indicator, counterpart, period (YYYY-MM), value (full USD), side
  side = 'china'   -> China is the reporter (中国申报)
         'partner' -> the partner reports its trade with China (对方申报, mirror)

Indicators: XG_FOB_USD (exports, FOB) and MG_CIF_USD (imports, CIF).
The download code lives in trade_sources.py (shared with the Cloud Function).
Run:  python src/fetch_imf_imts.py
"""
from config import PARTNERS, RAW_IMF
from trade_sources import imf_imts


def main():
    RAW_IMF.mkdir(parents=True, exist_ok=True)
    df = imf_imts()
    df.to_csv(RAW_IMF / "imts_raw.csv", index=False)
    print(f"{len(df):,} rows -> {RAW_IMF / 'imts_raw.csv'}")
    print(df.groupby("side").period.agg(["min", "max"]))
    missing = sorted(set(PARTNERS) - set(df[df.side == "china"].counterpart))
    print("Partners with no China-reported data:", missing or "none")
    print("Units: values are full USD (checked against GACS 2024 totals in check_trade_data.py).")


if __name__ == "__main__":
    main()
