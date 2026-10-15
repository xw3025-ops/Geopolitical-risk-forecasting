"""Country x week training samples for the trade transformer (train_trade_transformer.py).

Forecast origin. Row (country c, week w) is a forecast made at the end of week w: origin
O = week_start + 7 days. Inputs use only what is known at O:
  - weekly GDELT (dyad-week panel among the 20 MENA countries), through week w
  - UCDP fatalities, lagged UCDP_LAG_WEEKS (candidate data arrive about a month late)
  - monthly location-based GDELT (fetch_gdelt.py): last complete month before O
  - IMF trade: month m only once month_end(m) + PUB_LAG_DAYS <= O (China customs publishes
    the partner breakdown about 3 weeks after month end)

Targets (one per flow: China's imports from c, China's exports to c):
  y = log(trade over days [O, O+91)) - log(trade over the same 91 days one year earlier)
  i.e. year-on-year growth of the next 13 weeks (exp(y) - 1 = % change). Monthly values are
  spread evenly over the days of the month. Year-on-year removes seasonality (Chinese New
  Year moves between January and February). Clipped to +-Y_CLIP. A flow is graded (target not
  NaN) only if its last published 12 months total >= MIN_FLOW_USD; smaller flows (e.g. Syria's
  ~$1m/yr of exports to China) have meaningless growth rates.

Splits (by origin, each separated by a 13-week embargo so no target window crosses a split):
  train  origins until TRAIN_END        (target windows end by 2023-09)
  val    VAL_START .. VAL_END            (early stopping, model choice)
  test   TEST_START .. last labelled     (2025-01 .. 2026-03: includes the 2026 Hormuz shock)
  live   the latest origin whose trade inputs are as fresh as in training (no target yet)
Standardisation statistics are computed on the train split only.

Outputs (data/processed/):
  trade_weekly_samples.npz      X (weeks, 20, F), static (20, S), y / baseline (weeks, 20, 2), splits
  trade_weekly_samples.csv      the same rows in long form (inspection, gradient-boosting benchmark)
Run:  python src/make_training_samples.py
"""
import json

import numpy as np
import pandas as pd

from config import (CHINA, GDELT_MONTH, MENA20, PROCESSED, RAW_IMF, UCDP_WEEK, WEEKLY_GDELT, WORLD)

HORIZON_DAYS = 91  # 13 weeks
LOOKBACK = 13  # weeks of input history per token
PUB_LAG_DAYS = 28
UCDP_LAG_WEEKS = 4
MIN_FLOW_USD = 1e8  # per 12 months
Y_CLIP = 1.5  # log points (-78% .. +348%)
EPS_MONTH = 1e6  # USD per month added inside logs of small flows
TRAIN_END, VAL_START, VAL_END, TEST_START = "2023-06-26", "2023-10-02", "2024-09-30", "2025-01-06"
FLOWS = {"imp": "MG_CIF_USD", "exp": "XG_FOB_USD"}

DYNAMIC = [  # per country-week inputs
    "dom_attention", "dom_violent_share", "dom_matconf_share", "dom_goldstein",
    "rel_attention", "rel_violent_share", "inbound_violent_share", "rel_goldstein",
    "violent_surge_52", "ucdp_log_deaths",
    "geo_t2t4_share", "geo_conflict_share",
    "imp_yoy_1m", "imp_yoy_3m", "imp_yoy_12m", "imp_log_level",
    "exp_yoy_1m", "exp_yoy_3m", "exp_yoy_12m", "exp_log_level",
    "world_imp_yoy_3m", "world_exp_yoy_3m", "trade_staleness_weeks",
    "season_sin", "season_cos",
]
STATIC = ["hormuz_share_low", "hormuz_share_high", "energy_share_imports", "log_imp_share_2024",
          "log_exp_share_2024"]


# ------------------------------------------------------------ GDELT weekly

def gdelt_weekly(countries):
    p = pd.read_csv(WEEKLY_GDELT, parse_dates=["week_start"])
    p["violent"] = p[["count_18", "count_19", "count_20"]].sum(axis=1)
    p["matconf"] = p[[f"count_{i}" for i in range(15, 21)]].sum(axis=1)
    weeks = pd.DatetimeIndex(sorted(p.week_start.unique()))
    total = p.groupby("week_start").n_events.sum().reindex(weeks)
    cols = ["n_events", "violent", "matconf", "goldstein_sum"]
    self_ = p[p.Actor1CountryCode == p.Actor2CountryCode]
    other = p[p.Actor1CountryCode != p.Actor2CountryCode]
    grid = lambda df, key: df.groupby(["week_start", key])[cols].sum().unstack(key).reindex(weeks).fillna(0)
    dom = grid(self_, "Actor1CountryCode")
    out_ = grid(other, "Actor1CountryCode")
    in_ = grid(other, "Actor2CountryCode")
    rel = out_ + in_

    def per(df, col):
        return df[col].reindex(columns=countries, fill_value=0)

    share = lambda num, den: (num / den.where(den > 0)).fillna(0)
    f = {}
    per_10k = (1e4 / total).to_numpy()[:, None]
    f["dom_attention"] = np.log1p(per(dom, "n_events") * per_10k)
    f["dom_violent_share"] = share(per(dom, "violent"), per(dom, "n_events"))
    f["dom_matconf_share"] = share(per(dom, "matconf"), per(dom, "n_events"))
    f["dom_goldstein"] = share(per(dom, "goldstein_sum"), per(dom, "n_events"))
    f["rel_attention"] = np.log1p(per(rel, "n_events") * per_10k)
    f["rel_violent_share"] = share(per(rel, "violent"), per(rel, "n_events"))
    f["inbound_violent_share"] = share(per(in_, "violent"), per(in_, "n_events"))
    f["rel_goldstein"] = share(per(rel, "goldstein_sum"), per(rel, "n_events"))
    # violent events relative to the past year, as a share of all panel events (removes volume drift)
    v = (per(dom, "violent") + per(rel, "violent")) * per_10k
    f["violent_surge_52"] = np.log((v + 1) / (v.rolling(52, min_periods=13).mean().shift(1) + 1)).fillna(0)
    return weeks, {k: pd.DataFrame(np.asarray(v_, float), index=weeks, columns=countries) for k, v_ in f.items()}


def ucdp_weekly(weeks, countries):
    u = pd.read_csv(UCDP_WEEK, parse_dates=["week_start"])
    d = u.pivot_table(index="week_start", columns="iso3", values="ucdp_deaths", aggfunc="sum")
    d = d.reindex(index=weeks, columns=countries).fillna(0).shift(UCDP_LAG_WEEKS).fillna(0)
    return np.log1p(d)


def geo_monthly(weeks, countries):
    """Monthly location-based GDELT: value of the last complete calendar month before origin."""
    g = pd.read_csv(GDELT_MONTH)
    g["month"] = pd.PeriodIndex(g.ym.astype(str), freq="M")
    g["t2t4_share"] = g.n_t2_t4 / g.n_events
    origin = weeks + pd.Timedelta(days=7)
    last_month = (origin - pd.Timedelta(days=1)).to_period("M") - 1  # last month fully before origin
    out = {}
    for col, name in (("t2t4_share", "geo_t2t4_share"), ("conflict_share", "geo_conflict_share")):
        m = g.pivot_table(index="month", columns="iso3", values=col).reindex(columns=countries)
        out[name] = pd.DataFrame(m.reindex(last_month).to_numpy(), index=weeks, columns=countries).fillna(0)
    return out


# ------------------------------------------------------------ trade

def monthly_trade(countries):
    raw = pd.read_csv(RAW_IMF / "imts_raw.csv")
    ch = raw[(raw.side == "china") & (raw.reporter == CHINA)].copy()
    ch["month"] = pd.PeriodIndex(ch.period, freq="M")
    out = {}
    for flow, ind in FLOWS.items():
        t = ch[ch.indicator == ind].pivot_table(index="month", columns="counterpart", values="value")
        months = pd.period_range(t.index.min(), t.index.max(), freq="M")
        out[flow] = t.reindex(index=months, columns=countries + [WORLD])
    return out


def trade_features(weeks, trade, countries, pub_lag_days=PUB_LAG_DAYS):
    """Features from the months published by each origin (month m counts as published
    pub_lag_days after it ends). A window containing a month the source does not report
    is NaN (unknown), never zero trade."""
    origin = weeks + pd.Timedelta(days=7)
    months = trade["imp"].index
    month_end = months.to_timestamp(how="end").normalize()
    # last month published by each origin (index into `months`, -1 if none)
    pub = np.searchsorted(month_end + pd.Timedelta(days=pub_lag_days), origin, side="right") - 1
    f = {}
    for flow, m in trade.items():
        v = m.to_numpy(float)
        zeros = np.zeros((1, v.shape[1]))
        csum = np.vstack([zeros, np.cumsum(np.nan_to_num(v), axis=0)])
        cmiss = np.vstack([zeros, np.cumsum(np.isnan(v), axis=0)])

        def window(end_idx, n):  # sum of months end_idx-n+1 .. end_idx (NaN if out of range or a gap)
            lo = end_idx - n + 1
            ok = (lo >= 0) & (end_idx >= 0)
            s = np.full((len(end_idx), v.shape[1]), np.nan)
            s[ok] = csum[end_idx[ok] + 1] - csum[lo[ok]]
            gap = np.zeros_like(s, dtype=bool)
            gap[ok] = (cmiss[end_idx[ok] + 1] - cmiss[lo[ok]]) > 0
            s[gap] = np.nan
            return s

        for n, tag in ((1, "1m"), (3, "3m"), (12, "12m")):
            now, before = window(pub, n), window(pub - 12, n)
            f[f"{flow}_yoy_{tag}"] = np.clip(np.log((now + EPS_MONTH * n) / (before + EPS_MONTH * n)), -2, 2)
        f[f"{flow}_log_level"] = np.log10(window(pub, 12) + 12 * EPS_MONTH)
        f[f"{flow}_last12m"] = window(pub, 12)
    world = countries.index(WORLD) if WORLD in countries else -1
    staleness = (origin - month_end[np.clip(pub, 0, None)]).days.to_numpy() / 7.0
    return f, pub, staleness, world


def daily_csum(m):
    """Cumulative sum of each column after spreading monthly values evenly over days."""
    days = pd.date_range(m.index[0].start_time, m.index[-1].end_time.normalize(), freq="D")
    per_day = m.div(m.index.days_in_month, axis=0)
    d = per_day.reindex(days.to_period("M")).to_numpy()  # NaN stays NaN (missing month)
    return days, np.vstack([np.zeros((1, d.shape[1])), np.nancumsum(d, axis=0)]), np.vstack(
        [np.zeros((1, d.shape[1])), np.cumsum(np.isnan(d), axis=0)])


def targets(weeks, trade, countries):
    origin = weeks + pd.Timedelta(days=7)
    y = np.full((len(weeks), len(countries), 2), np.nan)
    for j, (flow, m) in enumerate(trade.items()):
        days, cs, cnan = daily_csum(m[countries])
        start = (origin - days[0]).days.to_numpy()

        def total(s):
            e = s + HORIZON_DAYS
            ok = (s >= 0) & (e <= len(days))
            out = np.full((len(s), len(countries)), np.nan)
            out[ok] = cs[e[ok]] - cs[s[ok]]
            missing = np.zeros_like(out, dtype=bool)
            missing[ok] = (cnan[e[ok]] - cnan[s[ok]]) > 0
            out[missing] = np.nan
            return out

        now, last_year = total(start), total(start - 364)
        eps = EPS_MONTH * 3
        y[:, :, j] = np.clip(np.log((now + eps) / (last_year + eps)), -Y_CLIP, Y_CLIP)
    return y


# ------------------------------------------------------------ main

def main():
    countries = MENA20
    weeks, gd = gdelt_weekly(countries)
    feats = dict(gd)
    feats["ucdp_log_deaths"] = ucdp_weekly(weeks, countries)
    feats.update(geo_monthly(weeks, countries))
    trade = monthly_trade(countries)
    tf, pub, staleness, _ = trade_features(weeks, trade, countries + [WORLD])
    nC = len(countries)
    for k in ("imp_yoy_1m", "imp_yoy_3m", "imp_yoy_12m", "imp_log_level",
              "exp_yoy_1m", "exp_yoy_3m", "exp_yoy_12m", "exp_log_level"):
        feats[k] = pd.DataFrame(tf[k][:, :nC], index=weeks, columns=countries)
    for flow in ("imp", "exp"):
        feats[f"world_{flow}_yoy_3m"] = pd.DataFrame(np.repeat(tf[f"{flow}_yoy_3m"][:, [nC]], nC, 1),
                                                     index=weeks, columns=countries)
    feats["trade_staleness_weeks"] = pd.DataFrame(np.repeat(staleness[:, None], nC, 1), index=weeks, columns=countries)
    mid = (weeks + pd.Timedelta(days=7 + HORIZON_DAYS // 2)).dayofyear.to_numpy()
    for k, fn in (("season_sin", np.sin), ("season_cos", np.cos)):
        feats[k] = pd.DataFrame(np.repeat(fn(2 * np.pi * mid / 365.25)[:, None], nC, 1), index=weeks, columns=countries)
    X = np.stack([feats[k].to_numpy(float) for k in DYNAMIC], axis=-1)  # (W, C, F)
    # a partner's unreported months leave its own trade features unknown: enter them as 0 (no
    # change) in the network inputs; the persistence baseline below keeps NaN (no forecast)
    country_trade = [DYNAMIC.index(k) for k in DYNAMIC if k.startswith(("imp_", "exp_"))]
    enough_history = pub >= 23  # 12-month yoy needs 24 published months
    sub = X[:, :, country_trade]
    sub[np.isnan(sub) & enough_history[:, None, None]] = 0.0
    X[:, :, country_trade] = sub

    y = targets(weeks, trade, countries)
    graded = np.stack([tf["imp_last12m"][:, :nC] >= MIN_FLOW_USD, tf["exp_last12m"][:, :nC] >= MIN_FLOW_USD], -1)
    y_raw = y.copy()
    y[~graded] = np.nan
    base = np.stack([feats["imp_yoy_3m"].to_numpy(), feats["exp_yoy_3m"].to_numpy()], -1)  # persistence

    # static country attributes (2024 trade structure; see the handoff's caveats)
    ex = pd.read_csv(PROCESSED / "exposure_table.csv").set_index("iso3").reindex(countries)
    static = np.stack([ex.hormuz_exposed_share_partner_imports_low.fillna(0),
                       ex.hormuz_exposed_share_partner_imports_high.fillna(0),
                       (ex.energy_bn / ex.imp_bn_2024).replace(np.inf, np.nan).fillna(0).clip(0, 1),
                       np.log10(ex.imp_share_2024.clip(lower=1e-7)),
                       np.log10(ex.exp_share_2024.clip(lower=1e-7))], -1).astype(float)

    # usable origins: full lookback with trade features, and trade inputs as fresh as in training
    feat_ok = ~np.isnan(X).any(axis=(1, 2))
    window_ok = pd.Series(feat_ok).rolling(LOOKBACK).min().fillna(0).astype(bool).to_numpy()
    first = np.argmax(window_ok)  # every week of the lookback has all inputs (12m yoy needs 24 months)
    last_month_idx = len(trade["imp"].index) - 1
    fresh = pub <= last_month_idx  # beyond this the next month would already be published
    next_pub = trade["imp"].index[-1].to_timestamp(how="end").normalize() + pd.Timedelta(days=PUB_LAG_DAYS + 31)
    fresh &= (weeks + pd.Timedelta(days=7)) < next_pub
    idx = np.arange(len(weeks))
    usable = (idx >= first) & fresh
    labelled = usable & ~np.isnan(y_raw).all(axis=(1, 2))
    ws = pd.Series(weeks.strftime("%Y-%m-%d"), index=idx)
    split = np.full(len(weeks), "", dtype=object)
    split[labelled & (ws <= TRAIN_END).to_numpy()] = "train"
    split[labelled & ((ws >= VAL_START) & (ws <= VAL_END)).to_numpy()] = "val"
    split[labelled & (ws >= TEST_START).to_numpy()] = "test"
    live = idx[usable][-1]
    split[live] = "live" if split[live] == "" else split[live]

    # standardisation from training origins (every week a training token can see)
    tr = np.where(split == "train")[0]
    seen = np.unique(np.concatenate([np.arange(w - LOOKBACK + 1, w + 1) for w in tr]))
    mu, sd = X[seen].reshape(-1, X.shape[-1]).mean(0), X[seen].reshape(-1, X.shape[-1]).std(0) + 1e-6
    s_mu, s_sd = static.mean(0), static.std(0) + 1e-6

    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED / "trade_weekly_samples.npz",
        X=X.astype("float32"), static=static.astype("float32"), y=y.astype("float32"),
        y_ungraded=y_raw.astype("float32"), baseline=base.astype("float32"), graded=graded,
        weeks=np.asarray(weeks.strftime("%Y-%m-%d"), dtype=str), split=split.astype(str), countries=np.array(countries),
        features=np.array(DYNAMIC), static_features=np.array(STATIC), flows=np.array(list(FLOWS)),
        mu=mu.astype("float32"), sd=sd.astype("float32"), s_mu=s_mu.astype("float32"), s_sd=s_sd.astype("float32"),
        meta=json.dumps(dict(horizon_days=HORIZON_DAYS, lookback=LOOKBACK, pub_lag_days=PUB_LAG_DAYS,
                             ucdp_lag_weeks=UCDP_LAG_WEEKS, min_flow_usd=MIN_FLOW_USD, y_clip=Y_CLIP,
                             splits=dict(train_end=TRAIN_END, val=[VAL_START, VAL_END], test_start=TEST_START),
                             last_trade_month=str(trade["imp"].index[-1]))))

    keep = np.where(split != "")[0]
    long = pd.DataFrame({
        "week_start": np.repeat(weeks[keep].strftime("%Y-%m-%d"), nC),
        "iso3": np.tile(countries, len(keep)),
        "split": np.repeat(split[keep], nC),
    })
    for j, k in enumerate(DYNAMIC):
        long[k] = X[keep, :, j].reshape(-1)
    for j, flow in enumerate(FLOWS):
        long[f"y_{flow}"] = y[keep, :, j].reshape(-1)
        long[f"baseline_{flow}"] = base[keep, :, j].reshape(-1)
    long.to_csv(PROCESSED / "trade_weekly_samples.csv", index=False)

    print(f"weeks {weeks[0].date()} .. {weeks[-1].date()}, {nC} countries, {len(DYNAMIC)} dynamic + "
          f"{len(STATIC)} static features")
    for s in ("train", "val", "test", "live"):
        w = np.where(split == s)[0]
        if len(w):
            n_lab = (~np.isnan(y[w])).sum(axis=(0, 1))
            print(f"  {s:5s} {len(w):4d} origins  {weeks[w[0]].date()} .. {weeks[w[-1]].date()}"
                  f"  graded imp/exp labels {n_lab[0]:,}/{n_lab[1]:,}")
    print("graded share by country (imports, exports):")
    g = pd.DataFrame(graded[split == "train"].mean(0), index=countries, columns=list(FLOWS)).round(2)
    print(g.T.to_string())
    print(f"-> {PROCESSED / 'trade_weekly_samples.npz'} and .csv")


if __name__ == "__main__":
    main()
