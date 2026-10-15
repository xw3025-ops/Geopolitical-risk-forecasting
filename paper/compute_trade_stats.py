"""
Numbers for the trade-module sections of the paper (paper/trade_section_draft.md and
paper/pipeline_summary.md). Reads the outputs of the src/ pipeline; run that first:

    python src/build_exposure.py && python src/build_panel.py && python src/check_trade_data.py
    python src/run_regression.py && python src/fetch_portwatch.py && python src/trade_baseline.py
    python paper/compute_trade_stats.py

Writes paper/trade_stats.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import CHINA, PARTNERS, WORLD  # noqa: E402

P = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "paper", "trade_stats.json")
GULF8 = ["IRN", "IRQ", "KWT", "SAU", "ARE", "BHR", "QAT", "OMN"]


def rnd(x, n=3):
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main():
    st = {}

    # ---------------------------------------------------------------- data
    raw = pd.read_csv(os.path.join(ROOT, "data", "imf_imts", "imts_raw.csv"))
    ch = raw[(raw.side == "china") & (raw.reporter == CHINA)]
    tot = ch[ch.counterpart == WORLD].assign(y=lambda d: d.period.str[:4]).groupby(["y", "indicator"]).value.sum() / 1e9
    imp = ch[ch.indicator == "MG_CIF_USD"].pivot(index="period", columns="counterpart", values="value")
    full = pd.period_range(ch.period.min(), ch.period.max(), freq="M").astype(str)
    missing = imp.reindex(full)[PARTNERS].isna().sum()
    mirror = pd.read_csv(os.path.join(P, "mirror_check.csv"))
    derived = sorted(set(mirror[(mirror.ratio_iqr < 0.02) | (mirror.share_ratio_exactly_1 > 0.5)].iso3))
    gm = pd.read_csv(os.path.join(ROOT, "data", "gdelt", "gdelt_country_month.csv"))
    gw = pd.read_csv(os.path.join(ROOT, "data", "risk_panel.csv"), usecols=["week_start"])
    st["data"] = {
        "imf_rows": len(raw), "imf_start": ch.period.min(), "imf_end": ch.period.max(), "n_partners": len(PARTNERS),
        "china_totals_bn_2024": {"exports": tot[("2024", "XG_FOB_USD")], "imports": tot[("2024", "MG_CIF_USD")]},
        "missing_import_months": missing[missing > 0].to_dict(),
        "mirror_derived_partners": derived,
        "mirror_share_status_e": float((raw[raw.side == "partner"].STATUS == "e").mean()),
        "gdelt_month_start": str(gm.ym.min()), "gdelt_month_end": str(gm.ym.max()), "gdelt_month_rows": len(gm),
        "gdelt_week_start": gw.week_start.min(), "gdelt_week_end": gw.week_start.max(),
    }
    pw_path = os.path.join(ROOT, "data", "portwatch", "chokepoints_daily.csv")
    if os.path.exists(pw_path):
        pw = pd.read_csv(pw_path, parse_dates=["date"])
        h = pw[pw.chokepoint == "hormuz"].set_index("date").n_tanker
        st["data"]["portwatch"] = {
            "start": str(h.index.min().date()), "end": str(h.index.max().date()), "days": int(h.size),
            "hormuz_tankers_per_day_2019_2025": h[:"2025-12-31"].mean(),
            "hormuz_tankers_per_day_jan_feb_2026": h["2026-01-01":"2026-02-28"].mean(),
            "hormuz_tankers_per_day_since_mar_2026": h["2026-03-02":].mean(),
            "first_day_below_5": str(h["2026-01-01":][h["2026-01-01":] < 5].index.min().date()),
            "hormuz_tankers_per_day_last_28d": h.tail(28).mean(),
        }

    # ---------------------------------------------------------------- the 2026 shock
    w = pd.read_csv(os.path.join(ROOT, "data", "risk_panel.csv"), parse_dates=["week_start"])
    w["violent"] = w[["count_18", "count_19", "count_20"]].sum(axis=1)
    wk = w.groupby("week_start").violent.sum()
    jan = wk["2026-01-05":"2026-02-16"].mean()
    peak_week = wk["2026-02-01":"2026-04-30"].idxmax()
    rev = pd.read_csv(os.path.join(P, "revealed_dependence_2026.csv"), index_col=0)
    st["shock"] = {
        "violent_events_per_week_jan_feb_2026": jan, "peak_week": str(peak_week.date()),
        "peak_violent_events": int(wk[peak_week]), "peak_ratio": wk[peak_week] / jan,
        "import_change_q2_2026_vs_q2_2025": rev.change.dropna().to_dict(),
        "implied_dependence": rev.implied_dep_energy.dropna().to_dict(),
        "monthly_change_2026": rev[["m3", "m4", "m5", "m6"]].dropna().to_dict(orient="index"),
    }

    # ---------------------------------------------------------------- exposure
    ex = pd.read_csv(os.path.join(P, "exposure_table.csv")).set_index("iso3")
    imp24 = tot[("2024", "MG_CIF_USD")]
    dep = pd.read_csv(os.path.join(ROOT, "data", "hormuz", "hormuz_dependence.csv")).set_index("iso3")
    st["exposure"] = {
        "total_bn_low": ex.hormuz_exposed_bn_low.sum(), "total_bn_high": ex.hormuz_exposed_bn_high.sum(),
        "share_of_china_imports_low": ex.hormuz_exposed_bn_low.sum() / imp24,
        "share_of_china_imports_high": ex.hormuz_exposed_bn_high.sum() / imp24,
        "china_imports_2024_bn": imp24,
        "by_country": ex.loc[ex.energy_bn > 0.5, ["energy_bn", "hormuz_exposed_bn_low", "hormuz_exposed_bn_high",
                                                  "hormuz_exposed_share_partner_imports_low",
                                                  "hormuz_exposed_share_partner_imports_high"]]
        .sort_values("energy_bn", ascending=False).to_dict(orient="index"),
        "parameters": dep[["dep_crude_low", "dep_crude_high", "dep_gas_low", "dep_gas_high", "confidence"]]
        .to_dict(orient="index"),
        "n_exposed_partners": int((ex.hormuz_exposed_share_partner_imports_high > 0).sum()),
    }

    # ---------------------------------------------------------------- regression
    rg = pd.read_csv(os.path.join(P, "regression_results.csv"))
    rg["lag"] = rg["lag"].astype(str)
    keep = ["spec", "outcome", "shock", "exposure", "lag", "sample", "n", "n_country", "n_month", "beta",
            "se_2way", "p_2way", "p_month", "p_dk", "p_perm", "note"]
    cum = rg[rg.lag == "0-1-2-3"][keep]
    st["regression"] = {
        "cumulative": rnd(cum.to_dict(orient="records"), 4),
        "by_lag_main": rnd(rg[(rg.spec == "B") & (rg["sample"] == "all") & (rg.shock == "gulf")
                              & (rg.lag != "0-1-2-3")][keep].to_dict(orient="records"), 4),
        "own_conflict_by_lag": rnd(rg[(rg.spec == "A") & (rg.lag != "0-1-2-3")][keep].to_dict(orient="records"), 4),
    }

    # ---------------------------------------------------------------- baseline forecast
    meta = json.load(open(os.path.join(ROOT, "models", "trade_baseline", "meta.json")))
    bt = {name: pd.DataFrame(rows) for name, rows in meta["backtest"].items()}
    st["baseline"] = {
        "k": meta["k"], "pub_lag_days": meta["pub_lag_days"], "origin": meta["origin"],
        "trade_data_end": meta["trade_data_end"], "interval_80_log_quantiles": meta["interval_80_log_quantiles"],
        "hormuz": meta["hormuz"], "walk_forward_k": meta["walk_forward_k"],
        "backtest_all": {name: df[df.year == "all"].set_index("flow")[
            ["n", "skill_vs_zero", "r2", "mae_model", "mae_zero", "direction_hit", "coverage_80"]].to_dict(orient="index")
            for name, df in bt.items()},
        "backtest_by_year_imf": bt["imf"][bt["imf"].year != "all"][
            ["year", "flow", "k", "skill_vs_zero", "r2", "coverage_80"]].to_dict(orient="records"),
    }
    fc = pd.read_csv(os.path.join(ROOT, "outputs", f"trade_forecast_{meta['origin']}.csv")).set_index("iso3")
    st["baseline"]["forecast_selected"] = fc.loc[["IRQ", "KWT", "QAT", "ARE", "SAU", "OMN", "TUR", "EGY"],
                                                 ["imp_forecast_pct", "imp_lo80_pct", "imp_hi80_pct", "imp_latest_3m_pct",
                                                  "exp_forecast_pct", "exp_lo80_pct", "exp_hi80_pct",
                                                  "exp_latest_3m_pct"]].to_dict(orient="index")

    # ---------------------------------------------------------------- transformer (research track)
    mt_path = os.path.join(P, "trade_model_metrics.csv")
    tm_path = os.path.join(ROOT, "models", "trade_transformer", "meta.json")
    if os.path.exists(mt_path) and os.path.exists(tm_path):
        mt = pd.read_csv(mt_path)
        tm = json.load(open(tm_path))
        st["transformer"] = {
            "params": tm["params"], "d_model": tm["d_model"], "heads": tm["heads"], "ff_dim": tm["ff_dim"],
            "temporal_blocks": tm["temporal_blocks"], "context_blocks": tm["context_blocks"],
            "seeds": len(tm["seeds"]), "val_mse": tm.get("val_mse"),
            "test": {p_: mt[mt.period == p_].pivot(index="model", columns="flow", values="skill_vs_zero").to_dict(orient="index")
                     for p_ in ("test_all", "test_2025", "test_2026_shock")},
            "note": "28-day publication lag; trained before the Hormuz parameter revision and the missing-month fix",
        }

    json.dump(rnd(st), open(OUT, "w"), indent=1, default=str)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
