"""Merge trade, GDELT conflict and Hormuz exposure into one country-month panel.

Inputs
  data/imf_imts/imts_raw.csv              China-reported monthly exports/imports by partner
  data/gdelt/gdelt_country_month.csv      conflict index by country-month
  data/processed/exposure_table.csv       predetermined (baseline) exposure by country
Output
  data/processed/panel_country_month.csv  one row per iso3 x month
Run:  python src/build_panel.py
"""
import numpy as np
import pandas as pd

from config import CHINA, PARTNERS, PROCESSED, RAW_IMF

IMP, EXP = "MG_CIF_USD", "XG_FOB_USD"
ROOT_DIR = RAW_IMF.parent


def main():
    # --- trade (China-reported), USD bn, partner level
    raw = pd.read_csv(RAW_IMF / "imts_raw.csv")
    t = raw[(raw.side == "china") & (raw.reporter == CHINA) & raw.counterpart.isin(PARTNERS)]
    t = t.pivot_table(index=["counterpart", "period"], columns="indicator", values="value").reset_index()
    t = t.rename(columns={"counterpart": "iso3", IMP: "imp_usd", EXP: "exp_usd"})
    t["period"] = pd.PeriodIndex(t.period, freq="M")
    t = t.sort_values(["iso3", "period"])
    for c in ("imp", "exp"):
        t[f"{c}_bn"] = t[f"{c}_usd"] / 1e9
        g = t.groupby("iso3")[f"{c}_usd"]
        t[f"{c}_yoy"] = t[f"{c}_usd"] / g.shift(12) - 1                       # year-on-year growth
        t[f"{c}_dlog12"] = np.log(t[f"{c}_usd"]) - np.log(g.shift(12))        # log diff, safer for regressions
    t = t.drop(columns=["imp_usd", "exp_usd"])

    # --- GDELT conflict: drop the incomplete current month, add coverage-robust ratios
    gd = pd.read_csv(ROOT_DIR / "gdelt" / "gdelt_country_month.csv")
    gd["period"] = pd.PeriodIndex(gd.ym.astype(str), freq="M")
    gd = gd[gd.period < pd.Period(pd.Timestamp.today(), "M")]
    gd["t2t4_share"] = gd.n_t2_t4 / gd.n_events
    gd["ln_t2t4"] = np.log1p(gd.n_t2_t4)
    gd = gd.sort_values(["iso3", "period"])
    for c in ("conflict_share", "t2t4_share", "ln_t2t4"):
        for k in (1, 2, 3):
            gd[f"{c}_l{k}"] = gd.groupby("iso3")[c].shift(k)
    gd = gd.drop(columns=["ym"])

    # --- exposure (time-invariant, predetermined)
    ex = pd.read_csv(PROCESSED / "exposure_table.csv")
    keep = ["iso3", "imp_share_base", "exp_share_base", "imp_share_2024", "exp_share_2024",
            "energy_bn", "hormuz_exposed_bn_low", "hormuz_exposed_bn_high",
            "hormuz_exposed_share_imports_low", "hormuz_exposed_share_imports_high",
            "hormuz_exposed_share_partner_imports_low", "hormuz_exposed_share_partner_imports_high"]
    ex = ex[keep]

    panel = t.merge(gd, on=["iso3", "period"], how="inner").merge(ex, on="iso3", how="left")
    panel["period"] = panel.period.astype(str)
    out = PROCESSED / "panel_country_month.csv"
    panel.to_csv(out, index=False)
    print(f"{len(panel):,} rows, {panel.iso3.nunique()} countries, {panel.period.min()} ~ {panel.period.max()} -> {out}")
    print("\nNon-missing by column (key variables):")
    print(panel[["imp_yoy", "exp_yoy", "conflict_share", "t2t4_share", "hormuz_exposed_share_imports_low"]].notna().sum())
    missing = set(PARTNERS) - set(panel.iso3)
    if missing:
        print("WARNING: no panel rows for", sorted(missing))


if __name__ == "__main__":
    main()
