"""
Country risk index (v1) for the 20 MENA countries, built on the release v2.0
forecasts. One row per country per as-of week.

    RISK (1-25) = LIKELIHOOD (1-5)  x  IMPACT (1-5)
    IMPACT      = sqrt( SEVERITY x EXPOSURE )          (geometric mean, as in the INFORM index)
    SEVERITY    = mean( HOSTILITY level, ACLED level )  (two sources, one construct: averaged)

LIKELIHOOD - will this country's relations materially deteriorate?
  Probability that, over the next 13 weeks, the country's relations
  (activity-weighted mean goldstein across all its dyads, incl. domestic)
  fall >= 0.5 below their own trailing 52-week normal. Logistic calibration
  of two signals, fit on held-out weeks (build_risk_calibration.py):
    forecast_change - goldstein v2.0 forecast vs normal (model)
    recent_change   - last 4 weeks vs normal (already under way)
  Levels: P < 10% -> 1, 10-20% -> 2, 20-35% -> 3, 35-50% -> 4, >= 50% -> 5.
  (The direction model's "escalation probability" is NOT used: at the
  country level it ranks worse than random - it mostly flags countries that
  just had an unusually good week. See build_risk_calibration.py output.)

SEVERITY - how hostile is the environment?
  HOSTILITY: the goldstein forecast for the next 13 weeks, activity-weighted
    across the country's dyads and negated (higher = more hostile). The full
    signed forecast, not just its negative part: clipping at 0 made ~20% of
    country-weeks tie at exactly 0 and hid slides from strong to weak
    cooperation.
  ACLED: political-violence, protest and strategic-development events in
    the country over the last 4 complete weeks (what just happened).
  Each -> 1-5 by fixed historical quintiles (not re-scaled every week, so a
  calm week can score low everywhere).

EXPOSURE - what is at stake?
  Total trade (exports + imports of goods and services, current USD, World
  Bank, latest year available), ranked across the 20 countries into
  quintiles (4 countries per level).

Risk bands: < 5 Low, 5-10 Moderate, 10-15 High, >= 15 Severe.
"""

import json
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from honest_common import MENA, panel_from_df, trailing_mean

RISK_VERSION = "risk-v1"
CALIBRATION_FILE = "risk_v1_calibration.json"
LIKELIHOOD_BANDS = [0.10, 0.20, 0.35, 0.50]
RISK_BANDS = [(15, "Severe"), (10, "High"), (5, "Moderate"), (0, "Low")]
ACLED_WEEKS = 4
ACLED_REFERENCE = ("2018-01-01", "2025-12-31")  # fixed window the ACLED quintiles come from

ACLED_COUNTRY = {
    "ARE": "United Arab Emirates", "EGY": "Egypt", "IRN": "Iran", "IRQ": "Iraq", "ISR": "Israel",
    "JOR": "Jordan", "LBY": "Libya", "PSE": "Palestine", "SAU": "Saudi Arabia", "SDN": "Sudan",
    "SYR": "Syria", "TUR": "Turkey", "YEM": "Yemen", "BHR": "Bahrain", "CYP": "Cyprus",
    "KWT": "Kuwait", "LBN": "Lebanon", "OMN": "Oman", "QAT": "Qatar", "DJI": "Djibouti",
}
ACLED_EVENT_TYPES = ["Battles", "Explosions/Remote violence", "Violence against civilians",
                     "Protests", "Riots", "Strategic developments"]


# ---------------------------------------------------------------- country aggregation

def country_matrix(a1, a2):
    """(C, D): 1 where dyad d involves country c (as actor 1 or 2; self-dyad once)."""
    return np.array([(a1 == c) | (a2 == c) for c in MENA], dtype="float64")


def aggregate(values, weights, M):
    """values, weights: (..., D) -> (..., C) weighted mean over each country's dyads."""
    return ((values * weights) @ M.T) / (weights @ M.T)


def dyad_context(p):
    """Per dyad-week quantities the index needs, from a panel (honest_common.panel_from_df)."""
    return {
        "normal": trailing_mean(p["gold_zero"], 52),
        "recent": trailing_mean(p["gold_zero"], 4),
        "weight": np.log1p(trailing_mean(p["counts"].sum(axis=2), 52)),
    }


def country_signals(forecast, normal, recent, weight, M):
    """All (..., D) -> dict of (..., C): the model-based country signals."""
    return {
        "forecast_change": aggregate(forecast - normal, weight, M),
        "recent_change": aggregate(recent - normal, weight, M),
        "hostility": aggregate(-forecast, weight, M),
    }


# ---------------------------------------------------------------- scaling

def likelihood_probability(signals, calib):
    c = calib["likelihood"]
    z = c["intercept"] + c["coef_forecast_change"] * signals["forecast_change"] \
        + c["coef_recent_change"] * signals["recent_change"]
    return 1.0 / (1.0 + np.exp(-z))


def level_from_thresholds(x, thresholds):
    """1 + number of thresholds exceeded -> 1..len(thresholds)+1"""
    return np.digitize(x, thresholds) + 1


def risk_band(score):
    return next(label for lo, label in RISK_BANDS if score >= lo)


# ---------------------------------------------------------------- trade exposure (World Bank API)

WB = "https://api.worldbank.org/v2/country/{}/indicator/{}?format=json&date=2010:2030&per_page=1000"


def _wb_latest(indicator):
    r = requests.get(WB.format(";".join(MENA), indicator), timeout=60)
    r.raise_for_status()
    latest = {}
    for row in r.json()[1] or []:
        c, v = row["countryiso3code"], row["value"]
        if v is not None and (c not in latest or int(row["date"]) > latest[c][0]):
            latest[c] = (int(row["date"]), float(v))
    return latest


def trade_exposure():
    """Total trade (goods & services, current USD), latest year per country;
    falls back to merchandise trade where the national-accounts series is
    missing. -> DataFrame country, trade_usd, trade_year, trade_source."""
    sources = [("goods & services", "NE.EXP.GNFS.CD", "NE.IMP.GNFS.CD"),
               ("merchandise", "TX.VAL.MRCH.CD.WT", "TM.VAL.MRCH.CD.WT")]
    rows = {}
    for label, exp_ind, imp_ind in sources:
        exp, imp = _wb_latest(exp_ind), _wb_latest(imp_ind)
        for c in MENA:
            if c in rows or c not in exp or c not in imp:
                continue
            year = min(exp[c][0], imp[c][0])  # same year for both sides where possible
            rows[c] = dict(country=c, trade_usd=exp[c][1] + imp[c][1], trade_year=year, trade_source=label)
    missing = sorted(set(MENA) - set(rows))
    if missing:
        raise ValueError(f"no World Bank trade data for {missing}")
    df = pd.DataFrame(rows.values())
    # quintiles across the 20 countries: 4 per level
    df["exposure_level"] = pd.qcut(df.trade_usd.rank(method="first"), 5, labels=False).astype(int) + 1
    return df


# ---------------------------------------------------------------- ACLED API

ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL = "https://acleddata.com/api/acled/read"


def acled_token(email, password):
    r = requests.post(ACLED_TOKEN_URL, data={"username": email, "password": password, "grant_type": "password",
                                             "client_id": "acled", "scope": "authenticated"}, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def acled_events(token, country_name, start, end, page_size=5000):
    """All ACLED events in one country between start and end (dates, inclusive)."""
    frames, page = [], 1
    while True:
        r = requests.get(ACLED_READ_URL, headers={"Authorization": f"Bearer {token}"}, timeout=300, params={
            "_format": "json", "country": country_name,
            "event_date": f"{start}|{end}", "event_date_where": "BETWEEN",
            "fields": "event_id_cnty|event_date|event_type|fatalities",
            "limit": page_size, "page": page})
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        frames.append(pd.DataFrame(data))
        if len(data) < page_size:
            break
        if page > 1 and frames[-1].event_id_cnty.iloc[0] == frames[-2].event_id_cnty.iloc[0]:
            raise RuntimeError("ACLED pagination returned the same page twice - check the 'page' parameter")
        page += 1
        time.sleep(0.5)
    if not frames:
        return pd.DataFrame(columns=["event_id_cnty", "event_date", "event_type", "fatalities"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("event_id_cnty")


def acled_country_weeks(email, password, start, end):
    """Weekly (Monday-start) ACLED event counts per country and event type,
    for all 20 countries. -> (DataFrame country, week_start, events,
    fatalities, n_<event type>..., latest event date seen in the pull)"""
    token = acled_token(email, password)
    out, max_date = [], None
    for iso3, name in ACLED_COUNTRY.items():
        ev = acled_events(token, name, start, end)
        if ev.empty:
            continue
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        max_date = max(filter(None, [max_date, ev.event_date.max()]))
        ev["week_start"] = ev.event_date - pd.to_timedelta(ev.event_date.dt.weekday, unit="D")
        ev["fatalities"] = pd.to_numeric(ev["fatalities"], errors="coerce").fillna(0)
        g = ev.groupby("week_start")
        wk = pd.DataFrame({"events": g.size(), "fatalities": g.fatalities.sum()})
        for t in ACLED_EVENT_TYPES:
            wk[_type_col(t)] = ev[ev.event_type == t].groupby("week_start").size()
        wk = wk.fillna(0).astype(int).reset_index()
        wk.insert(0, "country", iso3)
        out.append(wk)
    df = pd.concat(out, ignore_index=True)
    # dense: every country x every week in range (a week with no events is a real 0)
    weeks = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=pd.Timestamp(start).weekday()),
                          pd.Timestamp(end), freq="W-MON")
    grid = pd.MultiIndex.from_product([list(ACLED_COUNTRY), weeks], names=["country", "week_start"])
    df = df.set_index(["country", "week_start"]).reindex(grid, fill_value=0).reset_index()
    df["week_start"] = df.week_start.dt.date
    return df, (max_date.date() if max_date is not None else None)


ACLED_TABLE = "geopolitical-risk-forecasting.risk.acled_country_week"


def latest_complete_acled_week(max_event_date):
    """Monday of the latest week ACLED has published in full: the week of the
    latest event date if that date is a Sunday, else the week before (ACLED
    data for that week may still be arriving)."""
    last = pd.Timestamp(max_event_date)
    monday = last - pd.Timedelta(days=last.weekday())
    return (monday if last.weekday() == 6 else monday - pd.Timedelta(weeks=1)).date()


def store_acled_weeks(bq, weekly, table=ACLED_TABLE):
    """Replace the weeks in `weekly` in the BigQuery history table (creates it on first use)."""
    from google.cloud import bigquery
    first = min(weekly.week_start)
    try:
        bq.query(f"DELETE FROM `{table}` WHERE week_start >= @w",
                 job_config=bigquery.QueryJobConfig(query_parameters=[
                     bigquery.ScalarQueryParameter("w", "DATE", first)])).result()
    except Exception as e:
        if "Not found" not in str(e):
            raise
        dataset = table.rsplit(".", 1)[0]
        bq.create_dataset(dataset, exists_ok=True)
    bq.load_table_from_dataframe(weekly, table, job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND")).result()


def load_acled_weeks(bq, table=ACLED_TABLE):
    return bq.query(f"SELECT * FROM `{table}` ORDER BY country, week_start").to_dataframe()


def _type_col(t):
    return "n_" + t.lower().replace("/", "_").replace(" ", "_")


def acled_window(history, as_of_week, latest_complete):
    """Sum of the last ACLED_WEEKS complete weeks ending at as_of_week, or
    earlier if ACLED hasn't published that far yet."""
    end = min(pd.Timestamp(as_of_week), pd.Timestamp(latest_complete))
    start = end - pd.Timedelta(weeks=ACLED_WEEKS - 1)
    h = history.copy()
    h["week_start"] = pd.to_datetime(h.week_start)
    win = h[(h.week_start >= start) & (h.week_start <= end)]
    return win.groupby("country").events.sum(), end.date()


def acled_reference_thresholds(history):
    """Quintile thresholds of log(1 + 4-week event totals), pooled over all
    countries and the fixed ACLED_REFERENCE window."""
    h = history.copy()
    h["week_start"] = pd.to_datetime(h.week_start)
    h = h.sort_values(["country", "week_start"])
    roll = h.groupby("country").events.transform(lambda s: s.rolling(ACLED_WEEKS).sum())
    ref = roll[(h.week_start >= ACLED_REFERENCE[0]) & (h.week_start <= ACLED_REFERENCE[1])].dropna()
    if len(ref) < 1000:
        raise ValueError(f"only {len(ref)} reference country-windows in ACLED history - run backfill_acled.py")
    return list(np.quantile(np.log1p(ref), [0.2, 0.4, 0.6, 0.8]))


# ---------------------------------------------------------------- the index

def compute_risk(panel_df, forecasts, calib, acled_history, acled_latest_complete, exposure):
    """
    panel_df      the live pull (live_pipeline.pull_panel), last week = as-of week
    forecasts     live_pipeline.predict output (400 dyads)
    calib         dict from risk_v1_calibration.json
    acled_history weekly ACLED counts (acled_country_weeks), incl. the reference window
    acled_latest_complete  Monday of the latest week ACLED has fully published
    exposure      trade_exposure() output
    -> one row per country
    """
    p = panel_from_df(panel_df, MENA)
    ctx = dyad_context(p)
    f = forecasts.set_index(["actor1", "actor2"]).goldstein_forecast_next13w
    forecast = f.reindex(pd.MultiIndex.from_arrays([p["a1"], p["a2"]])).to_numpy()
    M = country_matrix(p["a1"], p["a2"])
    sig = country_signals(forecast, ctx["normal"][:, -1], ctx["recent"][:, -1], ctx["weight"][:, -1], M)

    prob = likelihood_probability(sig, calib)
    out = pd.DataFrame({
        "country": MENA,
        "likelihood_probability": prob,
        "likelihood_level": level_from_thresholds(prob, LIKELIHOOD_BANDS),
        "forecast_change_vs_normal": sig["forecast_change"],
        "recent_change_vs_normal": sig["recent_change"],
        "hostility_forecast": sig["hostility"],
        "hostility_level": level_from_thresholds(sig["hostility"], calib["hostility_thresholds"]),
    })

    events, window_end = acled_window(acled_history, p["weeks"][-1], acled_latest_complete)
    out["acled_events_4w"] = out.country.map(events).fillna(0).astype(int)
    out["acled_window_end"] = window_end
    out["acled_level"] = level_from_thresholds(np.log1p(out.acled_events_4w),
                                               acled_reference_thresholds(acled_history))
    out = out.merge(exposure, on="country", how="left")

    out["severity"] = (out.hostility_level + out.acled_level) / 2
    out["impact"] = np.sqrt(out.severity * out.exposure_level)
    out["risk_score"] = out.likelihood_level * out.impact
    out["risk_band"] = out.risk_score.map(risk_band)
    out.insert(0, "as_of_week", p["weeks"][-1].date())
    out.insert(1, "risk_version", RISK_VERSION)
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


def load_calibration(model_dir):
    for path in (os.path.join(model_dir, CALIBRATION_FILE), os.path.join(model_dir, "artifacts", CALIBRATION_FILE)):
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    raise FileNotFoundError(f"{CALIBRATION_FILE} not found in {model_dir}")
