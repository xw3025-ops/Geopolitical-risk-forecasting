"""Build the country-level exposure table.

Inputs
  data/imf_imts/imts_raw.csv              (fetch_imf_imts.py)
  data/comtrade/china_energy_2024.csv     (fetch_comtrade_energy.py)
  data/hormuz/hormuz_dependence.csv       (hand-built parameters, see 'basis'/'sources' columns)
Outputs (data/processed/)
  china_world_totals.csv    China's totals with the World, by indicator and year
  baseline_dependence.csv   predetermined partner shares (baseline window and 2024)
  exposure_table.csv        shares + Hormuz-exposed energy imports (low/high scenarios)
Run:  python src/build_exposure.py
"""
import pandas as pd

from config import (BASE_END, BASE_START, BASE_YEAR, CHINA, PARTNERS, PROCESSED,
                    PARAMS, RAW_COMTRADE, RAW_IMF, WORLD)

IMP, EXP = "MG_CIF_USD", "XG_FOB_USD"


def window_sum(df, lo, hi, name):
    w = df[(df.period >= lo) & (df.period <= hi)]
    # require a complete window so partial months do not bias shares
    n = w.groupby(["indicator", "counterpart"]).period.nunique()
    s = w.groupby(["indicator", "counterpart"]).value.sum()
    s[n < (pd.Period(hi) - pd.Period(lo)).n + 1] = float("nan")
    return s.rename(name)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_IMF / "imts_raw.csv")
    ch = raw[(raw.side == "china") & (raw.reporter == CHINA)].copy()

    # totals with the World
    wld = ch[ch.counterpart == WORLD].copy()
    wld["year"] = wld.period.str[:4]
    tot = wld.groupby(["indicator", "year"]).value.sum().unstack(0)
    tot.to_csv(PROCESSED / "china_world_totals.csv")

    # baseline shares: partner / world, over the baseline window and over 2024
    b = pd.concat([window_sum(ch, BASE_START, BASE_END, "v_base"),
                   window_sum(ch, f"{BASE_YEAR}-01", f"{BASE_YEAR}-12", "v_2024")], axis=1)
    base = pd.DataFrame(index=PARTNERS)
    for ind, tag in ((IMP, "imp"), (EXP, "exp")):
        for col, suf in (("v_base", "base"), ("v_2024", "2024")):
            world = b.loc[(ind, WORLD), col]
            base[f"{tag}_bn_{suf}"] = [b[col].get((ind, p), float("nan")) / 1e9 for p in PARTNERS]
            base[f"{tag}_share_{suf}"] = [b[col].get((ind, p), float("nan")) / world for p in PARTNERS]
    base.index.name = "iso3"
    base.reset_index().to_csv(PROCESSED / "baseline_dependence.csv", index=False)

    # Hormuz-exposed energy imports (Comtrade 2024 commodity values x dependence scenarios)
    en = pd.read_csv(RAW_COMTRADE / "china_energy_2024.csv").set_index("iso3")
    dep = pd.read_csv(PARAMS / "hormuz_dependence.csv").set_index("iso3")
    x = base.join(en[["crude_bn", "gas_bn"]]).fillna({"crude_bn": 0, "gas_bn": 0})
    x["energy_bn"] = x.crude_bn + x.gas_bn
    tot24 = tot.loc[BASE_YEAR, IMP] / 1e9
    for s in ("low", "high"):
        x[f"dep_crude_{s}"] = dep[f"dep_crude_{s}"].reindex(x.index).fillna(0)
        x[f"dep_gas_{s}"] = dep[f"dep_gas_{s}"].reindex(x.index).fillna(0)
        x[f"hormuz_exposed_bn_{s}"] = x.crude_bn * x[f"dep_crude_{s}"] + x.gas_bn * x[f"dep_gas_{s}"]
        x[f"hormuz_exposed_share_imports_{s}"] = x[f"hormuz_exposed_bn_{s}"] / tot24
        x[f"hormuz_exposed_share_partner_imports_{s}"] = x[f"hormuz_exposed_bn_{s}"] / x["imp_bn_2024"]
    x.reset_index().to_csv(PROCESSED / "exposure_table.csv", index=False)

    cols = ["energy_bn", "hormuz_exposed_bn_low", "hormuz_exposed_bn_high"]
    print(x[cols].sort_values("energy_bn", ascending=False).round(2).head(10).to_string())
    lo, hi = x.hormuz_exposed_bn_low.sum(), x.hormuz_exposed_bn_high.sum()
    print(f"\nTotal Hormuz-exposed energy imports: {lo:.1f} ~ {hi:.1f} bn USD "
          f"= {lo/tot24:.2%} ~ {hi/tot24:.2%} of China's {BASE_YEAR} imports ({tot24:.0f} bn)")


if __name__ == "__main__":
    main()
