"""Checks on the IMF trade data that the handoff left unverified.

1. Units: China's annual totals vs the customs (GACS) headline figures.
2. Mirror data: is the partner-reported side an independent source?
   Compares partner-reported imports from China with China-reported exports to
   the partner (and the reverse), and the share of STATUS 'e' (estimated) rows.
   If the IMF filled partner data from China's own figures, the ratio would be
   flat at ~1 (or a fixed CIF/FOB factor) and dominated by 'e' rows.
3. Why imp_yoy has fewer non-missing rows than exp_yoy.

Output: printed report + data/processed/mirror_check.csv
Run:  python src/check_trade_data.py
"""
import numpy as np
import pandas as pd

from config import CHINA, PARTNERS, PROCESSED, RAW_IMF, WORLD

IMP, EXP = "MG_CIF_USD", "XG_FOB_USD"
# GACS annual headline totals, USD bn (as published in each January press release).
# Typed from memory of the releases: re-check against the GACS release before citing.
GACS_TOTALS = {"2023": (3380.0, 2556.0), "2024": (3577.0, 2587.0), "2025": (None, None)}


def main():
    raw = pd.read_csv(RAW_IMF / "imts_raw.csv")

    print("== 1. Units ==")
    print("SCALE values:", raw.SCALE.unique().tolist(), "(values are stored as full USD; SCALE=6 = published in millions)")
    w = raw[(raw.side == "china") & (raw.counterpart == WORLD)].assign(year=lambda d: d.period.str[:4])
    tot = w.groupby(["year", "indicator"]).value.sum().unstack() / 1e9
    for y, (x, m) in GACS_TOTALS.items():
        if x is None or y not in tot.index:
            continue
        print(f"{y}: IMF exports {tot.loc[y, EXP]:,.0f} vs GACS {x:,.0f} bn ({tot.loc[y, EXP] / x - 1:+.1%}); "
              f"IMF imports {tot.loc[y, IMP]:,.0f} vs GACS {m:,.0f} bn ({tot.loc[y, IMP] / m - 1:+.1%})")

    print("\n== 2. Mirror data (partner-reported vs China-reported) ==")
    ch = raw[(raw.side == "china") & (raw.reporter == CHINA)]
    pa = raw[(raw.side == "partner") & (raw.counterpart == CHINA)]
    # partner imports from China (CIF) mirror China exports to partner (FOB); and vice versa
    pairs = [("partner_imp_vs_china_exp", pa[pa.indicator == IMP], ch[ch.indicator == EXP]),
             ("partner_exp_vs_china_imp", pa[pa.indicator == EXP], ch[ch.indicator == IMP])]
    rows = []
    for name, p, c in pairs:
        m = p.rename(columns={"reporter": "iso3", "value": "partner", "STATUS": "status"})[
            ["iso3", "period", "partner", "status"]].merge(
            c.rename(columns={"counterpart": "iso3", "value": "china"})[["iso3", "period", "china"]],
            on=["iso3", "period"])
        m = m.sort_values(["iso3", "period"])
        m["ratio"] = m.partner / m.china
        for c_ in ("partner", "china"):
            m[f"{c_}_dlog12"] = np.log(m[c_].where(m[c_] > 0)) - np.log(m.groupby("iso3")[c_].shift(12).where(lambda s: s > 0))
        for iso, g in m.groupby("iso3"):
            ok = g[["partner_dlog12", "china_dlog12"]].dropna()
            rows.append({
                "check": name, "iso3": iso, "months": len(g),
                "share_status_e": (g.status == "e").mean(), "share_status_r": (g.status == "r").mean(),
                "median_ratio": g.ratio.median(), "ratio_iqr": g.ratio.quantile(.75) - g.ratio.quantile(.25),
                "share_ratio_exactly_1": (g.ratio.sub(1).abs() < 1e-6).mean(),
                "corr_dlog12": ok.partner_dlog12.corr(ok.china_dlog12) if len(ok) > 12 else np.nan,
                "last_partner_month": g.period.max(),
            })
    mc = pd.DataFrame(rows)
    mc.to_csv(PROCESSED / "mirror_check.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(mc.round(3).to_string(index=False))
    e = pa.STATUS.eq("e")
    print(f"\nPartner-side rows: {len(pa):,}; STATUS e = {e.mean():.1%}, r = {pa.STATUS.eq('r').mean():.1%}")
    est_after = pa[e].groupby("reporter").period.min().sort_values()
    print("First 'e' month by partner (estimated rows cluster at the end of each partner's series?):")
    print(est_after.to_string())
    flat = mc[(mc.ratio_iqr < 0.02) | (mc.share_ratio_exactly_1 > 0.5)]
    print("\nPartners whose mirror series looks derived from China data (flat ratio):",
          sorted(set(flat.iso3)) or "none")

    print("\n== 3. imp_yoy vs exp_yoy missing rows ==")
    pnl = pd.read_csv(PROCESSED / "panel_country_month.csv")
    miss = pnl.assign(imp_na=pnl.imp_yoy.isna(), exp_na=pnl.exp_yoy.isna()).groupby("iso3")[["imp_na", "exp_na"]].sum()
    extra = miss[miss.imp_na != miss.exp_na]
    print("Countries where imp_yoy is missing more often than exp_yoy (first 12 months excluded = 12 each):")
    print(extra.to_string())
    raw_imp = ch[ch.indicator == IMP]
    for iso in extra.index:
        months = sorted(set(pnl[pnl.iso3 == iso].period) - set(raw_imp[raw_imp.counterpart == iso].period))
        print(f"  {iso}: no China-reported import value in {len(months)} months: {months[:15]}{' ...' if len(months) > 15 else ''}")
    tiny = pnl.groupby("iso3").imp_bn.median().sort_values()
    print("Median monthly imports (USD m) of the smallest flows:",
          (tiny.head(8) * 1e3).round(2).to_dict())
    missing = sorted(set(PARTNERS) - set(pnl.iso3))
    if missing:
        print("WARNING: no panel rows for", missing)

    print("\n== 4. Revealed Hormuz dependence: the 2026 shock vs the assumed parameters ==")
    print("Imports Apr-Jun 2026 vs Apr-Jun 2025 (and Mar-Jun month by month). USD values, not volumes:")
    print("higher oil prices make value drops UNDERSTATE volume drops.")
    ex = pd.read_csv(PROCESSED / "exposure_table.csv").set_index("iso3")
    imp = ch[ch.indicator == IMP].pivot(index="period", columns="counterpart", values="value")
    q26, q25 = imp.loc["2026-04":"2026-06"].sum(), imp.loc["2025-04":"2025-06"].sum()
    rev = pd.DataFrame({
        "imp_bn_q2_2025": q25 / 1e9, "imp_bn_q2_2026": q26 / 1e9, "change": q26 / q25 - 1,
        **{f"m{m}": imp.loc[f"2026-{m:02d}"] / imp.loc[f"2025-{m:02d}"] - 1 for m in (3, 4, 5, 6)},
        "energy_share": ex.energy_bn / ex.imp_bn_2024,
        "dep_crude_low": ex.dep_crude_low, "dep_crude_high": ex.dep_crude_high,
    }).reindex(["IRQ", "KWT", "QAT", "ARE", "SAU", "OMN", "IRN", "BHR", "USA", "TUR", "ISR", "EGY", WORLD])
    # implied dependence: drop in energy imports needed to explain the total drop (other goods unchanged)
    rev["implied_dep_energy"] = (-rev.change / rev.energy_share).clip(-1, 1.5).where(rev.energy_share >= 0.5)
    with pd.option_context("display.width", 200):
        print(rev.round(2).to_string())
    rev.to_csv(PROCESSED / "revealed_dependence_2026.csv")


if __name__ == "__main__":
    main()
