"""Operational trade forecast: shrunk persistence (the transformer stays in
train_trade_transformer.py for future attempts).

Forecast, made at the end of each week for every MENA partner and both flows (China's imports
from / exports to the partner):
    next-13-week year-on-year log change  =  k_flow x (latest published 3-month year-on-year change)
k_flow (one number per flow) is the least-squares slope through the origin. The target is
the same as the transformer's (make_training_samples.py): trade over the 91 days after the
origin vs the same days a year earlier; % change = exp(y) - 1.

Publication lag. A month counts as available only PUB_LAG_DAYS after it ends:
  imf      100 days  the automated pipeline's source (on 2026-10-14 the IMF had data through
                     June 2026, not July: the real lag is between ~76 and ~106 days)
  customs   28 days  China customs (GACS) country tables, if you feed them in by hand
Both are backtested so the value of faster data is visible; the forecast uses --lag (imf).

Evaluation: walk-forward by calendar year 2020..2026. For test year Y, k is fitted on origins
whose target window ended before Y began, then scored on Y's origins. Grading as in the
samples: only flows with >= $100m in the last published 12 months.
Uncertainty: 80% interval = forecast + 10th/90th percentile of out-of-sample errors from
earlier years (per flow); coverage is checked in the backtest.
Hormuz flag: PortWatch tanker transits through Hormuz over the last 4 complete weeks vs the
2019-2025 weekly average, shown for partners with Hormuz-exposed energy exports to China. It is
a warning for the analyst; it does not change the number.

Outputs
  models/trade_baseline/meta.json                 k, interval quantiles, backtest metrics
  data/processed/trade_baseline_backtest.csv      every out-of-sample forecast
  outputs/trade_forecast_<origin>.csv             latest forecast
Run:  python src/trade_baseline.py [--lag imf|customs] [--as-of YYYY-MM-DD]
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import MENA20, PROCESSED, ROOT, WORLD
from make_training_samples import MIN_FLOW_USD, monthly_trade, targets, trade_features

LAGS = {"imf": 100, "customs": 28}
FLOWS = ("imp", "exp")
FIRST_TEST_YEAR = 2020
Q_LO, Q_HI = 0.10, 0.90
HORMUZ_DISRUPTED = 0.5  # flag when transits fall below half of normal
MODEL_DIR = ROOT / "models" / "trade_baseline"


def build(weeks, pub_lag_days):
    trade = monthly_trade(MENA20)
    tf, pub, staleness, _ = trade_features(weeks, trade, MENA20 + [WORLD], pub_lag_days=pub_lag_days)
    n = len(MENA20)
    base = np.stack([tf["imp_yoy_3m"][:, :n], tf["exp_yoy_3m"][:, :n]], -1)
    graded = np.stack([tf["imp_last12m"][:, :n] >= MIN_FLOW_USD, tf["exp_last12m"][:, :n] >= MIN_FLOW_USD], -1)
    y = targets(weeks, trade, MENA20)
    last_month = trade["imp"].index[np.clip(pub, 0, None)].astype(str).to_numpy()
    return dict(base=base, graded=graded, y=y, staleness=staleness, last_month=last_month,
                trade_end=trade["imp"].index[-1])


def fit_k(base, y):
    k = []
    for j in range(2):
        b, t = base[..., j].ravel(), y[..., j].ravel()
        ok = ~np.isnan(b) & ~np.isnan(t)
        k.append(float(np.sum(b[ok] * t[ok]) / np.sum(b[ok] ** 2)))
    return np.array(k)


def backtest(weeks, d):
    """Walk-forward by year; returns long table of out-of-sample forecasts."""
    origin = weeks + pd.Timedelta(days=7)
    window_end = origin + pd.Timedelta(days=91)
    y = np.where(d["graded"], d["y"], np.nan)
    rows, ks = [], {}
    for year in range(FIRST_TEST_YEAR, origin.year.max() + 1):
        start = pd.Timestamp(f"{year}-01-01")
        tr = np.where(window_end <= start)[0]
        te = np.where((origin >= start) & (origin < pd.Timestamp(f"{year + 1}-01-01"))
                      & ~np.isnan(y).all(axis=(1, 2)))[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        k = fit_k(d["base"][tr], y[tr])
        ks[year] = k.tolist()
        for i in te:
            for c, iso in enumerate(MENA20):
                for j, flow in enumerate(FLOWS):
                    if np.isnan(y[i, c, j]) or np.isnan(d["base"][i, c, j]):
                        continue
                    rows.append({"year": year, "week_start": weeks[i].date(), "iso3": iso, "flow": flow,
                                 "y": y[i, c, j], "latest_yoy_3m": d["base"][i, c, j], "k": k[j],
                                 "forecast": k[j] * d["base"][i, c, j]})
    bt = pd.DataFrame(rows)
    bt["error"] = bt.y - bt.forecast
    # intervals from errors of EARLIER years only, then coverage
    bt["lo"] = bt["hi"] = np.nan
    for year in sorted(bt.year.unique())[1:]:
        for flow in FLOWS:
            past = bt[(bt.year < year) & (bt.flow == flow)].error
            sel = (bt.year == year) & (bt.flow == flow)
            bt.loc[sel, "lo"] = bt.loc[sel, "forecast"] + past.quantile(Q_LO)
            bt.loc[sel, "hi"] = bt.loc[sel, "forecast"] + past.quantile(Q_HI)
    return bt, ks


def summarise(bt):
    out = []
    for (year, flow), g in list(bt.groupby(["year", "flow"])) + [(("all", f), bt[bt.flow == f]) for f in FLOWS]:
        mse0, mse = np.mean(g.y ** 2), np.mean(g.error ** 2)
        cov = g.dropna(subset=["lo"])
        out.append({"year": year, "flow": flow, "n": len(g), "k": g.k.mean(),
                    "skill_vs_zero": 1 - mse / mse0,
                    # R2 against the evaluated rows' own mean (a mean nobody knows in advance)
                    "r2": 1 - mse / np.mean((g.y - g.y.mean()) ** 2), "mae_model": g.error.abs().mean(), "mae_zero": g.y.abs().mean(),
                    "direction_hit": np.mean(np.sign(g.forecast) == np.sign(g.y)),
                    "coverage_80": ((cov.y >= cov.lo) & (cov.y <= cov.hi)).mean() if len(cov) else np.nan})
    return pd.DataFrame(out)


def hormuz_status(origin):
    path = ROOT / "data" / "portwatch" / "chokepoints_weekly.csv"
    if not path.exists():
        return None, "PortWatch not downloaded (python src/fetch_portwatch.py)"
    w = pd.read_csv(path, parse_dates=["week_start"])
    h = w[w.chokepoint == "hormuz"].set_index("week_start").n_tanker
    normal = h[(h.index >= "2019-01-01") & (h.index < "2026-01-01")].mean()
    recent = h[h.index + pd.Timedelta(days=7) <= origin].tail(4)
    ratio = recent.mean() / normal
    return ratio, (f"Hormuz tanker transits, 4 weeks to {recent.index[-1].date()}: {recent.mean():.0f}/week "
                   f"= {ratio:.0%} of the 2019-2025 average ({normal:.0f}/week)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", choices=list(LAGS), default="imf")
    ap.add_argument("--as-of", help="Monday of the last complete week (default: last complete week)")
    args = ap.parse_args()
    today = pd.Timestamp.today().normalize()
    as_of = pd.Timestamp(args.as_of) if args.as_of else today - pd.Timedelta(days=today.weekday() + 7)
    weeks = pd.date_range("2017-01-02", as_of, freq="W-MON")

    results = {}
    for name, lag in LAGS.items():
        d = build(weeks, lag)
        bt, ks = backtest(weeks, d)
        results[name] = (d, bt, ks, summarise(bt))
    print("Walk-forward backtest (skill = 1 - MSE / MSE of 'no change'):")
    for name, (_, _, _, sm) in results.items():
        a = sm[sm.year == "all"].set_index("flow")
        print(f"  {name:8s} lag {LAGS[name]:3d}d  skill imp {a.loc['imp', 'skill_vs_zero']:+.3f} "
              f"exp {a.loc['exp', 'skill_vs_zero']:+.3f}  | R2 imp {a.loc['imp', 'r2']:+.3f} "
              f"exp {a.loc['exp', 'r2']:+.3f}  | 80% coverage imp {a.loc['imp', 'coverage_80']:.0%} "
              f"exp {a.loc['exp', 'coverage_80']:.0%}")
    d, bt, ks, sm = results[args.lag]
    print(f"\nBy year ({args.lag} lag):")
    with pd.option_context("display.width", 200):
        print(sm.pivot(index="year", columns="flow", values=["k", "skill_vs_zero", "r2", "mae_model", "mae_zero",
                                                                 "direction_hit", "coverage_80"]).round(3).to_string())

    # final fit on every labelled origin; intervals from all out-of-sample errors
    y = np.where(d["graded"], d["y"], np.nan)
    k = fit_k(d["base"], y)
    q = {f: bt[bt.flow == f].error.quantile([Q_LO, Q_HI]).tolist() for f in FLOWS}

    i = len(weeks) - 1
    origin = weeks[i] + pd.Timedelta(days=7)
    ratio, hormuz_text = hormuz_status(origin)
    ex = pd.read_csv(PROCESSED / "exposure_table.csv").set_index("iso3")
    exposed = ex.hormuz_exposed_share_partner_imports_high.reindex(MENA20).fillna(0) > 0
    fc = pd.DataFrame({"iso3": MENA20})
    for j, flow in enumerate(FLOWS):
        b = d["base"][i, :, j]
        f = k[j] * b
        fc[f"{flow}_forecast_pct"] = np.expm1(f) * 100
        fc[f"{flow}_lo80_pct"] = np.expm1(f + q[flow][0]) * 100
        fc[f"{flow}_hi80_pct"] = np.expm1(f + q[flow][1]) * 100
        fc[f"{flow}_latest_3m_pct"] = np.expm1(b) * 100
        fc[f"{flow}_graded"] = d["graded"][i, :, j]
        fc.loc[~fc[f"{flow}_graded"], [f"{flow}_forecast_pct", f"{flow}_lo80_pct", f"{flow}_hi80_pct"]] = np.nan
    disrupted = ratio is not None and ratio < HORMUZ_DISRUPTED
    fc["hormuz_exposed"] = exposed.to_numpy()
    fc["note"] = ""
    fc.loc[fc.hormuz_exposed & disrupted, "note"] = ("Hormuz disrupted: forecast assumes part of the recent "
                                                     "change persists; full persistence if the strait stays closed")
    fc.loc[~fc.imp_graded & fc.exp_graded & (fc.note == ""), "note"] = "imports < $100m/yr: no import forecast"
    num = [c for c in fc if c.endswith("_pct")]
    fc[num] = fc[num].round(1)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"trade_forecast_{origin.date()}.csv"
    fc.to_csv(path, index=False)
    bt.to_csv(PROCESSED / "trade_baseline_backtest.csv", index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"model": "shrunk persistence of the latest published 3-month yoy change",
            "lag": args.lag, "pub_lag_days": LAGS[args.lag], "k": dict(zip(FLOWS, k.tolist())),
            "interval_80_log_quantiles": q, "fit_origins": f"{weeks[0].date()} .. last origin with a complete target",
            "trade_data_end": str(d["trade_end"]), "as_of": str(as_of.date()), "origin": str(origin.date()),
            "walk_forward_k": ks, "backtest": {n: r[3].assign(year=r[3].year.astype(str)).to_dict("records")
                                               for n, r in results.items()},
            "hormuz": hormuz_text}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1, default=float))

    print(f"\nFinal k: imports {k[0]:.3f}, exports {k[1]:.3f}  (lag {args.lag}, {LAGS[args.lag]} days)")
    print(f"Forecast origin {origin.date()} (window {origin.date()} .. {(origin + pd.Timedelta(days=90)).date()}), "
          f"latest trade month used: {d['last_month'][i]}")
    print(hormuz_text)
    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        print(fc.drop(columns=["imp_graded", "exp_graded"]).to_string(index=False))
    print(f"-> {path}\n-> {MODEL_DIR / 'meta.json'}, {PROCESSED / 'trade_baseline_backtest.csv'}")


if __name__ == "__main__":
    main()
