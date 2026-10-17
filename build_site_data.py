"""
Build the data bundle the website reads (website/data/latest.json).

One JSON file holds everything the site shows, so the site itself is static
and every number refreshes when this script runs - locally now, and after
each Cloud Function run once wired in (upload with --gcs).

Sources
  forecasts   release v2.2 (live_pipeline.py): goldstein forecast and the
              conflict-surge flag for the 400 MENA dyads, as of the latest
              complete week in BigQuery table 1.1; the same run 2 weeks
              earlier gives the change since the previous refresh
  history     weekly dyad activity / goldstein from the same pull
  ACLED       political-violence events of the last 28 days (event level,
              for the map) and the 4-week fatality count per country
  trade       IMF IMTS: China's exports + imports with each country over the
              latest 12 published months (src/fetch_imf_imts.py output, or
              BigQuery trade.imf_imts_raw once trade-refresh is deployed);
              Hormuz route dependence from data/hormuz/hormuz_dependence.csv
  lanes       IMF PortWatch daily vessel transits through Hormuz, Bab el-Mandeb,
              Suez, the Cape of Good Hope and Malacca (src/fetch_portwatch.py
              output, or BigQuery trade.portwatch_daily once trade-refresh runs):
              last 4 complete weeks vs the pre-crisis normal (2019 - Oct 2023),
              with the risk of the states on each lane's shores
  summaries   one regional and one per-country brief, in English and Chinese,
              written by DeepSeek (deepseek-chat) from the numbers in this
              bundle (DEEPSEEK_API_KEY in .env); without a key, clearly
              labelled template text

Country scores (0-100; documented on the site's Method panel)
  SEVERITY  = 100 x weighted mean of three parts, each scaled to 0..1:
     0.40  predicted hostility: activity-weighted goldstein forecast of all the
           country's dyads (incl. domestic), +1.5 -> 0, -4.0 -> 1
     0.30  conflict-surge risk: activity-weighted surge probability, 0 -> 0, 0.6 -> 1
     0.30  recent violence: ACLED fatalities, last 4 weeks, log scale, 2,000+ -> 1
     (without ACLED the first two are re-weighted to 4:3)
  EXPOSURE  = 100 x log(1 + China trade, bn USD / 12 mo) / log(1 + the largest)
  RISK      = sqrt(SEVERITY x EXPOSURE)   (geometric mean: neither part can dominate alone)
  REGION    = mean of the 20 country risks

    python build_site_data.py [--as-of 2026-10-05] [--no-acled] [--no-ai] [--gcs]
"""

import argparse
import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import live_pipeline as lp
from honest_common import MENA, panel_from_df, trailing_mean
from risk_index import ACLED_COUNTRY, ACLED_READ_URL, acled_token

OUT = "website/data/latest.json"
IMF_RAW = "data/imf_imts/imts_raw.csv"
HORMUZ = "data/hormuz/hormuz_dependence.csv"
BUCKET = "geopolitical-risk-forecasting-models"
SITE_OBJECT = "site/latest.json"
REFRESH_WEEKS = 2
CHART_WEEKS = 52
ACLED_DAYS = 28
ACTIVE_EVENTS = 5.0  # dyads with >= 5 events/week on average count as "active" (worst-dyad metric)
AI_MODEL = "deepseek-chat"                         # DeepSeek's general chat model
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
PORTWATCH = "data/portwatch/chokepoints_weekly.csv"
LANES = {  # PortWatch chokepoint -> where it is and which states line it / ship through it
    "hormuz": dict(name="Strait of Hormuz", name_zh="霍尔木兹海峡", lon=56.35, lat=26.55, on_map=True,
                   littoral=["IRN", "OMN", "ARE"], dependent=["IRQ", "KWT", "QAT", "BHR", "IRN", "SAU", "ARE"]),
    "bab_el_mandeb": dict(name="Bab el-Mandeb", name_zh="曼德海峡", lon=43.35, lat=12.6, on_map=True,
                          littoral=["YEM", "DJI"], dependent=[]),
    "suez": dict(name="Suez Canal", name_zh="苏伊士运河", lon=32.35, lat=30.4, on_map=True,
                 littoral=["EGY"], dependent=[]),
    "cape_of_good_hope": dict(name="Cape of Good Hope", name_zh="好望角", lon=18.47, lat=-34.36, on_map=False,
                              littoral=[], dependent=[]),
    "malacca": dict(name="Strait of Malacca", name_zh="马六甲海峡", lon=101.0, lat=2.5, on_map=False,
                    littoral=[], dependent=[]),
}

NAMES = {
    "ARE": "United Arab Emirates", "BHR": "Bahrain", "CYP": "Cyprus", "DJI": "Djibouti", "EGY": "Egypt",
    "IRN": "Iran", "IRQ": "Iraq", "ISR": "Israel", "JOR": "Jordan", "KWT": "Kuwait", "LBN": "Lebanon",
    "LBY": "Libya", "OMN": "Oman", "PSE": "Palestine", "QAT": "Qatar", "SAU": "Saudi Arabia", "SDN": "Sudan",
    "SYR": "Syria", "TUR": "Turkey", "YEM": "Yemen",
}
ISO_NUM = {"ARE": "784", "BHR": "048", "CYP": "196", "DJI": "262", "EGY": "818", "IRN": "364", "IRQ": "368",
           "ISR": "376", "JOR": "400", "KWT": "414", "LBN": "422", "LBY": "434", "OMN": "512", "PSE": "275",
           "QAT": "634", "SAU": "682", "SDN": "729", "SYR": "760", "TUR": "792", "YEM": "887"}

clip01 = lambda x: float(min(1.0, max(0.0, x)))


# ---------------------------------------------------------------- forecasts

def forecasts(bq, models, as_of):
    """-> (now, prev) predict() outputs, the dyad panel p over the pull, activity weights."""
    n = lp.HISTORY_WEEKS + REFRESH_WEEKS + CHART_WEEKS
    df = lp.pull_panel(bq, as_of, n_weeks=n)
    pairs = models["goldstein"][0].get("external_dyads", [])
    ext = lp.pull_external(bq, as_of, pairs, n_weeks=n)
    weeks = sorted(df.week_start.unique())

    def run(end):
        w = [x for x in weeks if x <= end][-lp.HISTORY_WEEKS:]
        sel = lambda d: d[d.week_start.isin(w)]
        return lp.predict(sel(df), models, sel(ext))

    now = run(pd.Timestamp(as_of))
    prev = run(pd.Timestamp(as_of) - pd.Timedelta(weeks=REFRESH_WEEKS))
    p = panel_from_df(df, MENA)
    return now, prev, p


def country_view(results, p):
    """Activity-weighted country signals from one predict() output."""
    r = results.set_index(["actor1", "actor2"]).reindex(pd.MultiIndex.from_arrays([p["a1"], p["a2"]]))
    events = p["counts"].sum(2)
    vol = trailing_mean(events, 52)[:, -1]  # events / week, last 52 weeks
    out = {}
    for c in MENA:
        m = (p["a1"] == c) | (p["a2"] == c)
        w = vol[m] + 1e-6
        g = r.goldstein_forecast_next13w.to_numpy()[m]
        nrm = r.goldstein_trailing52w.to_numpy()[m]
        s = r.conflict_surge_probability.to_numpy()[m]
        active = m & (vol >= ACTIVE_EVENTS)
        worst = r.goldstein_forecast_next13w.to_numpy()[active].min() if active.any() else float(g.min())
        out[c] = dict(forecast=float(np.average(g, weights=w)), normal=float(np.average(nrm, weights=w)),
                      surge=float(np.average(s, weights=w)), worst_active_dyad=float(worst),
                      alerts=int(r.surge_alert.to_numpy()[m].sum()))
        out[c]["change"] = out[c]["forecast"] - out[c]["normal"]
    return out, vol


def severity(cv, fatalities):
    h = clip01((1.5 - cv["forecast"]) / 5.5)
    s = clip01(cv["surge"] / 0.6)
    if fatalities is None:
        return 100 * (4 * h + 3 * s) / 7, dict(hostility=h, surge=s, violence=None)
    v = clip01(math.log1p(fatalities) / math.log1p(2000))
    return 100 * (0.4 * h + 0.3 * s + 0.3 * v), dict(hostility=h, surge=s, violence=v)


# ---------------------------------------------------------------- trade

def trade_exposure():
    raw = pd.read_csv(IMF_RAW)
    ch = raw[(raw.side == "china") & (raw.reporter == "CHN") & raw.counterpart.isin(MENA)]
    last = ch.groupby("period").value.count()
    last = last[last >= len(MENA)].index.max()  # latest month published for every partner
    months = pd.period_range(end=pd.Period(last, "M"), periods=12, freq="M").astype(str)
    prior = pd.period_range(end=pd.Period(last, "M") - 12, periods=12, freq="M").astype(str)
    hz = pd.read_csv(HORMUZ).set_index("iso3")
    out = {}
    for c in MENA:
        d = ch[ch.counterpart == c]
        cur = d[d.period.isin(months)].groupby("indicator").value.sum() / 1e9
        old = d[d.period.isin(prior)].groupby("indicator").value.sum() / 1e9
        exp_bn, imp_bn = float(cur.get("XG_FOB_USD", 0)), float(cur.get("MG_CIF_USD", 0))
        total, total_old = exp_bn + imp_bn, float(old.sum())
        monthly = d[d.period.isin(months)].groupby("period").value.sum().reindex(months).fillna(0) / 1e9
        out[c] = dict(exports_bn=exp_bn, imports_bn=imp_bn, total_bn=total,
                      yoy=(total / total_old - 1) if total_old > 0 else None,
                      monthly_bn=[round(float(x), 3) for x in monthly],
                      hormuz_low=float(hz.dep_crude_low.get(c)) if c in hz.index else None,
                      hormuz_high=float(hz.dep_crude_high.get(c)) if c in hz.index else None,
                      hormuz_confidence=str(hz.confidence.get(c)) if c in hz.index else None)
    top = max(v["total_bn"] for v in out.values())
    for v in out.values():
        v["exposure"] = 100 * math.log1p(v["total_bn"]) / math.log1p(top)
    return out, dict(window=f"{months[0]}..{months[-1]}", source="IMF IMTS, China-reported")


# ---------------------------------------------------------------- ACLED

ACLED_FIELDS = "event_id_cnty|event_date|event_type|sub_event_type|actor1|location|latitude|longitude|fatalities|notes"


def acled_recent(end):
    """Event-level ACLED for the last ACLED_DAYS days up to `end`, per country."""
    token = acled_token(os.environ["ACLED_EMAIL"], os.environ["ACLED_PASSWORD"])
    start = end - timedelta(days=ACLED_DAYS - 1)
    out, latest = {}, None
    for iso, name in ACLED_COUNTRY.items():
        rows, page = [], 1
        while True:
            r = requests.get(ACLED_READ_URL, headers={"Authorization": f"Bearer {token}"}, timeout=300, params={
                "_format": "json", "country": name, "event_date": f"{start}|{end}",
                "event_date_where": "BETWEEN", "fields": ACLED_FIELDS, "limit": 5000, "page": page})
            r.raise_for_status()
            data = r.json().get("data", [])
            rows += data
            if len(data) < 5000:
                break
            page += 1
            time.sleep(0.5)
        ev = []
        for e in rows:
            try:
                ev.append(dict(d=e["event_date"], t=e["event_type"], s=e.get("sub_event_type", ""),
                               a=(e.get("actor1") or "")[:60], l=(e.get("location") or "")[:40],
                               lat=round(float(e["latitude"]), 3), lon=round(float(e["longitude"]), 3),
                               f=int(float(e.get("fatalities") or 0)), n=(e.get("notes") or "")[:160]))
            except (KeyError, ValueError):
                continue
        ev.sort(key=lambda e: (-e["f"], e["d"]))
        dates = [e["d"] for e in ev]
        if dates:
            latest = max(filter(None, [latest, max(dates)]))
        last4 = [e for e in ev if e["d"] >= str(end - timedelta(days=27))]
        by_type = pd.Series([e["t"] for e in ev]).value_counts().to_dict() if ev else {}
        out[iso] = dict(events=ev[:400], n_events=len(ev), fatalities_4w=int(sum(e["f"] for e in last4)),
                        by_type={k: int(v) for k, v in by_type.items()})
        time.sleep(0.3)
    return out, dict(window=f"{start}..{end}", latest_event=latest, source="ACLED (acleddata.com)")


# ---------------------------------------------------------------- AI summaries

SUMMARY_SYSTEM = """You write the short briefs on a geopolitical risk dashboard. Readers are managers at companies
trading with the Middle East and North Africa, not statisticians. Each brief says, in plain words, what is expected
over the next 13 weeks and what it means for doing business there.

Rules:
- Use only the data provided. Do not add events, names, causes or figures that are not in it.
- Plain language. Never write "Goldstein", "normal of", "exposure", "severity score" or raw decimals. Translate
  numbers into meaning: e.g. "relations are expected to be somewhat more hostile than usual".
- Round numbers: risk scores as whole numbers out of 100, deaths with thousands separators, shipping as ships per
  week, changes as percentages. At most two numbers per sentence.
- Keep the past and the future apart: ACLED deaths are what happened in the last four weeks; surge alerts,
  forecasts and risk describe the next 13 weeks.
- Say when signals disagree (e.g. high violence but low trade at stake).
- Never state a cause, a link between two facts or a ranking unless the data states it. Rank countries on a field
  only with the rank fields given (rank for risk, trade_rank for trade at stake). A country is on a shipping lane
  only if the lane lists it in littoral; even then say it borders the lane, not that it caused the disruption.
- Region brief: 3 sentences. Country briefs: 2 sentences each.
- Also write every brief in Simplified Chinese (region_zh, countries_zh), naturally for Chinese business readers,
  not word for word.

Field meanings:
- risk (0-100): overall business risk = conflict severity combined with China's trade at stake. rank: 1 = highest of 20.
- risk_change: change in risk since the previous refresh two weeks ago.
- severity (0-100): how bad the conflict environment is expected to be.
- exposure (0-100): China trade at stake with the country (call it "trade at stake"); trade_rank: 1 = most of 20.
- forecast / normal: average tone of the country's relations, next 13 weeks / past year, on a scale from -10
  (hostile) to +10 (cooperative). change = forecast - normal; below -0.2 means worse than usual, above +0.2 better.
- surge: average chance that a relationship sees an unusual surge of conflict; alerts: how many of the country's
  relationships are flagged for such a surge in the next 13 weeks.
- acled_fatalities_4w: deaths from political violence recorded in the last four weeks.
- top_partners: the country's most active relationships (ISO3 codes; use country names).
- shipping_lanes: ships per week through each chokepoint in the last four weeks (transits_week) vs the
  2019-2023 normal (normal_week); change is the fraction above or below normal; status disrupted / reduced / normal.
- regional_risk: average of the 20 country risks now (risk) and at the previous refresh (risk_prev).
"""


def ai_summaries(bundle):
    """-> (dict region/countries/region_zh/countries_zh, provenance). DeepSeek chat API in JSON mode
    (DEEPSEEK_API_KEY in .env, or a Secret Manager secret in the cloud); template text if unavailable."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    compact = {"as_of_week": bundle["meta"]["as_of_week"], "regional_risk": bundle["region"],
               "countries": {c["iso3"]: {k: c[k] for k in ("name", "risk", "severity", "exposure", "rank",
                                                           "risk_change", "forecast", "normal", "change",
                                                           "surge", "alerts", "acled_fatalities_4w",
                                                           "top_partners")}
                             for c in bundle["countries"]},
               "shipping_lanes": [dict({k: l[k] for k in ("name", "status", "change", "transits_week", "normal_week")},
                                       littoral=[x["iso3"] for x in l["littoral"]])
                                  for l in (bundle.get("lanes") or []) if l.get("on_map")]}
    by_trade = sorted(bundle["countries"], key=lambda c: -c["exposure"])
    for i, c in enumerate(by_trade, 1):
        compact["countries"][c["iso3"]]["trade_rank"] = i
    shape = ('Reply with a single JSON object with exactly these keys: "region" (string), "countries" (object '
             f'mapping each of these ISO3 codes to a string: {", ".join(MENA)}), "region_zh" (string) and '
             '"countries_zh" (the same ISO3 codes, Simplified Chinese strings).')
    r = requests.post(DEEPSEEK_URL, timeout=240, headers={"Authorization": f"Bearer {key}"}, json={
        "model": AI_MODEL, "temperature": 0.3, "max_tokens": 8000, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SUMMARY_SYSTEM + " " + shape},
                     {"role": "user", "content": "Dashboard data (JSON):\n" + json.dumps(compact, sort_keys=True)}]})
    r.raise_for_status()
    body = r.json()
    if body["choices"][0].get("finish_reason") == "length":
        raise RuntimeError("DeepSeek response was cut off (max_tokens)")
    out = json.loads(body["choices"][0]["message"]["content"])
    missing = [k for k in ("region", "region_zh") if not isinstance(out.get(k), str)] + \
              [f"{grp}.{c}" for grp in ("countries", "countries_zh") for c in MENA if not (out.get(grp) or {}).get(c)]
    if missing:
        raise RuntimeError(f"DeepSeek response is missing {missing[:5]}")
    return out, dict(source="ai", model=body.get("model", AI_MODEL))


# ---------------------------------------------------------------- news (online context for the China briefs)

NEWS_DAYS = 90
NEWS_CACHE = "data/news_cache"
SEARCH_NAME = {"ARE": "United Arab Emirates", "PSE": "Palestinian", "SAU": "Saudi Arabia", "TUR": "Turkey"}


def _news_guardian(name, key, since):
    r = requests.get("https://content.guardianapis.com/search", timeout=60, params={
        "q": f'China AND "{name}"', "from-date": since, "order-by": "relevance", "page-size": 8,
        "show-fields": "trailText", "api-key": key})
    r.raise_for_status()
    return [dict(source="The Guardian", title=x["webTitle"], date=x["webPublicationDate"][:10], url=x["webUrl"],
                 snippet=(x.get("fields") or {}).get("trailText", "")[:300]) for x in r.json()["response"]["results"]]


def _news_nyt(name, key, since):
    r = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json", timeout=60, params={
        "q": f"China {name}", "begin_date": since.replace("-", ""), "sort": "relevance", "api-key": key})
    if r.status_code == 429:
        return []
    r.raise_for_status()
    docs = (r.json().get("response") or {}).get("docs") or []
    return [dict(source="The New York Times", title=d["headline"]["main"], date=d["pub_date"][:10], url=d["web_url"],
                 snippet=(d.get("abstract") or "")[:300]) for d in docs[:8]]


def news_context(as_of):
    """Recent Guardian / NYT coverage mentioning both China and each country (last NEWS_DAYS days).
    Cached per as-of week, so re-runs do not re-query the APIs."""
    os.makedirs(NEWS_CACHE, exist_ok=True)
    cache = os.path.join(NEWS_CACHE, f"news_{as_of}.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    g, n = os.environ.get("GUARDIAN_API_KEY"), os.environ.get("NYT_API_KEY")
    since = str(as_of - timedelta(days=NEWS_DAYS))
    out = {}
    for c in MENA:
        name = SEARCH_NAME.get(c, NAMES[c])
        items = []
        try:
            items += _news_guardian(name, g, since) if g else []
        except Exception as e:
            print(f"Guardian {c}: {e}")
        try:
            items += _news_nyt(name, n, since) if n else []
            time.sleep(6.5)  # NYT allows ~10 requests a minute
        except Exception as e:
            print(f"NYT {c}: {e}")
        # keep items that actually concern both China and the country
        key = name.split()[0].lower()
        rel = [x for x in items if "chin" in (x["title"] + x["snippet"]).lower() and key[:5] in (x["title"] + x["snippet"]).lower()]
        seen, uniq = set(), []
        for x in sorted(rel or items, key=lambda x: x["date"], reverse=True):
            if x["title"] not in seen:
                seen.add(x["title"]); uniq.append(x)
        out[c] = uniq[:6]
    json.dump(out, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


CHINA_SYSTEM = """You write the country page of a risk dashboard for Chinese companies and organisations active in the
Middle East and North Africa (exporters, importers and energy buyers, contractors and investors, shippers, staff on
the ground). For ONE country you receive: the dashboard numbers, China's trade with the country (IMF, last 12
months), its dependence on the Strait of Hormuz, the shipping lanes it borders, and recent news articles (numbered).

Write, in plain business language:
1. outlook: 2 sentences on the risk outlook for the next 13 weeks.
2. china: 2-3 sentences on the trade relationship with China - its size, direction (what China mainly exports there
   vs imports from there, as far as the numbers show) and recent trend - and anything the news adds about
   Chinese involvement (projects, investment, diplomacy, energy).
3. at_risk: 2-4 short bullet points on what could be at risk for Chinese entities if the outlook materialises,
   each naming WHO (e.g. Chinese exporters, crude buyers, contractors, shipping lines, nationals) and HOW.
4. sources: the numbers of the news articles you used (empty list if none).
Then the same three texts in Simplified Chinese (outlook_zh, china_zh, at_risk_zh), written naturally for Chinese
business readers, not word for word.

Rules - accuracy matters more than detail:
- Every figure must come from the data given. Round numbers; at most two per sentence; no raw decimals.
- Facts about specific projects, companies, deals or events must come from the numbered articles and be cited in
  sources. Without an article, stay at the level of sectors and general exposure (e.g. "Chinese crude buyers
  relying on Gulf supply") - never invent names, projects or amounts.
- Do not claim causes or links the data and articles do not state. Do not repeat a claim from an article as
  certain if it is reported as a possibility.
- Separate the past (deaths in the last four weeks, trade over the last 12 months) from the forecast (next 13 weeks).
- Field meanings: risk 0-100 combines conflict severity with China's trade at stake (rank 1 = highest of 20);
  forecast/normal = average tone of the country's relations next 13 weeks / past year on a -10 (hostile) to +10
  (cooperative) scale; alerts = relationships flagged for an unusual conflict surge in the next 13 weeks;
  exports_bn / imports_bn = China's exports to / imports from the country, USD bn, last 12 months;
  hormuz_dependence = share of the country's oil exports that cannot bypass the Strait of Hormuz (low-high case).
Reply with one JSON object with keys: outlook, china, at_risk (list of strings), sources (list of integers),
outlook_zh, china_zh, at_risk_zh (list of strings)."""


def _deepseek(system, user, max_tokens=4000):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    r = requests.post(DEEPSEEK_URL, timeout=240, headers={"Authorization": f"Bearer {key}"}, json={
        "model": AI_MODEL, "temperature": 0.2, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    body = r.json()
    if body["choices"][0].get("finish_reason") == "length":
        raise RuntimeError("DeepSeek response was cut off")
    return json.loads(body["choices"][0]["message"]["content"]), body.get("model", AI_MODEL)


def china_brief(c, news, lanes):
    data = {k: c[k] for k in ("name", "risk", "rank", "risk_change", "severity", "forecast", "normal", "change",
                              "alerts", "acled_fatalities_4w")}
    tr = c["trade"]
    data["china_trade"] = dict(exports_bn=round(tr["exports_bn"], 2), imports_bn=round(tr["imports_bn"], 2),
                               total_bn=round(tr["total_bn"], 2), yoy=None if tr["yoy"] is None else round(tr["yoy"], 3),
                               monthly_total_bn_last_12=tr["monthly_bn"],
                               hormuz_dependence=None if tr["hormuz_high"] is None else [tr["hormuz_low"], tr["hormuz_high"]])
    data["borders_lanes"] = [dict(name=l["name"], status=l["status"], change_vs_normal=round(l["change"], 2))
                             for l in lanes if c["iso3"] in [x["iso3"] for x in l["littoral"]]]
    arts = "\n".join(f"[{i + 1}] {a['source']}, {a['date']}: {a['title']} - {a['snippet']}" for i, a in enumerate(news)) or "(none)"
    out, model = _deepseek(CHINA_SYSTEM, f"Country data (JSON):\n{json.dumps(data)}\n\nNews articles:\n{arts}")
    for k in ("outlook", "china", "outlook_zh", "china_zh"):
        if not isinstance(out.get(k), str) or not out[k].strip():
            raise RuntimeError(f"{c['iso3']}: missing {k}")
    for k in ("at_risk", "at_risk_zh"):
        if not isinstance(out.get(k), list) or not out[k]:
            raise RuntimeError(f"{c['iso3']}: missing {k}")
    used = [news[i - 1] for i in out.get("sources", []) if isinstance(i, int) and 1 <= i <= len(news)]
    return dict(outlook=out["outlook"], outlook_zh=out["outlook_zh"], china=out["china"], china_zh=out["china_zh"],
                at_risk=out["at_risk"][:4], at_risk_zh=out["at_risk_zh"][:4], sources=used), model


def china_briefs(bundle, news):
    from concurrent.futures import ThreadPoolExecutor
    lanes = bundle.get("lanes") or []
    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {c["iso3"]: ex.submit(china_brief, c, news.get(c["iso3"], []), lanes) for c in bundle["countries"]}
        for iso, fu in futs.items():
            try:
                res[iso], _ = fu.result()
            except Exception as e:
                print(f"China brief {iso} failed: {str(e)[:120]}")
    return res


def template_summaries(bundle):
    zh_names = ZH_NAMES

    def line(c):
        tone = "deteriorate" if c["change"] < -0.2 else "improve" if c["change"] > 0.2 else "stay close to normal"
        parts = [f"{c['name']} ranks #{c['rank']} of 20 on risk ({c['risk']:.0f}/100). Relations are forecast to "
                 f"{tone} over the next 13 weeks (average Goldstein {c['forecast']:+.2f} vs a normal of "
                 f"{c['normal']:+.2f})."]
        if c["alerts"]:
            parts.append(f"{c['alerts']} of its relationships carry a conflict-surge alert.")
        if c.get("acled_fatalities_4w"):
            parts.append(f"ACLED records {c['acled_fatalities_4w']:,} fatalities in the last four weeks.")
        return " ".join(parts)

    def line_zh(c):
        tone = "恶化" if c["change"] < -0.2 else "改善" if c["change"] > 0.2 else "大致保持常态"
        parts = [f"{zh_names[c['iso3']]}风险在20国中排第{c['rank']}位（{c['risk']:.0f}/100）。预测未来13周其对外关系将{tone}"
                 f"（平均戈德斯坦值{c['forecast']:+.2f}，常态为{c['normal']:+.2f}）。"]
        if c["alerts"]:
            parts.append(f"其中{c['alerts']}组双边关系触发冲突激增预警。")
        if c.get("acled_fatalities_4w"):
            parts.append(f"ACLED记录过去四周死亡{c['acled_fatalities_4w']:,}人。")
        return "".join(parts)

    top = bundle["countries"][:3]
    n_alerts = sum(d["al"] for d in bundle["dyads"])
    region = (f"Highest risk this refresh: {', '.join(c['name'] for c in top)}. "
              f"{n_alerts} relationships carry a conflict-surge alert for the next 13 weeks.")
    region_zh = (f"本期风险最高：{'、'.join(zh_names[c['iso3']] for c in top)}。"
                 f"未来13周共有{n_alerts}组双边关系触发冲突激增预警。")
    return dict(region=region, countries={c["iso3"]: line(c) for c in bundle["countries"]},
                region_zh=region_zh, countries_zh={c["iso3"]: line_zh(c) for c in bundle["countries"]}), \
        dict(source="template")


ZH_NAMES = {
    "ARE": "阿联酋", "BHR": "巴林", "CYP": "塞浦路斯", "DJI": "吉布提", "EGY": "埃及", "IRN": "伊朗", "IRQ": "伊拉克",
    "ISR": "以色列", "JOR": "约旦", "KWT": "科威特", "LBN": "黎巴嫩", "LBY": "利比亚", "OMN": "阿曼", "PSE": "巴勒斯坦",
    "QAT": "卡塔尔", "SAU": "沙特阿拉伯", "SDN": "苏丹", "SYR": "叙利亚", "TUR": "土耳其", "YEM": "也门",
}


# ---------------------------------------------------------------- shipping lanes

def shipping_lanes(countries, trade):
    """Transits through each chokepoint, last 4 complete weeks vs the pre-crisis normal (2019 - Oct 2023,
    before the Red Sea attacks), and the risk of the states on its shores."""
    try:
        w = pd.read_csv(PORTWATCH, parse_dates=["week_start"])
    except FileNotFoundError:
        return None, None
    risk = {c["iso3"]: c for c in countries}
    hz = pd.read_csv(HORMUZ).set_index("iso3")
    out = []
    for key, L in LANES.items():
        d = w[w.chokepoint == key].sort_values("week_start")
        if d.empty:
            continue
        last = d.tail(4)
        normal = d[(d.week_start >= "2019-01-01") & (d.week_start < "2023-10-01")]
        r = dict(id=key, name=L["name"], name_zh=L["name_zh"], lon=L["lon"], lat=L["lat"], on_map=L["on_map"],
                 weeks=f"{last.week_start.min().date()}..{(last.week_start.max() + pd.Timedelta(days=6)).date()}",
                 transits_week=round(float(last.n_total.mean()), 1), normal_week=round(float(normal.n_total.mean()), 1),
                 tankers_week=round(float(last.n_tanker.mean()), 1), normal_tankers=round(float(normal.n_tanker.mean()), 1),
                 history=[int(x) for x in d.n_total.tail(52)])
        r["change"] = r["transits_week"] / r["normal_week"] - 1 if r["normal_week"] else None
        lit = [risk[c] for c in L["littoral"] if c in risk]
        r["littoral"] = [dict(iso3=c["iso3"], risk=c["risk"], alerts=c["alerts"]) for c in lit]
        r["littoral_risk"] = max((c["risk"] for c in lit), default=None)
        if L["dependent"]:  # China trade with partners whose oil exports depend on the lane (high case)
            r["china_trade_at_stake_bn"] = round(float(sum(
                trade[c]["total_bn"] * float(hz.dep_crude_high.get(c, 0) or 0) for c in L["dependent"] if c in trade)), 1)
        r["status"] = ("disrupted" if r["change"] is not None and r["change"] <= -0.5 else
                       "reduced" if r["change"] is not None and r["change"] <= -0.2 else "normal")
        out.append(r)
    meta = dict(source="IMF PortWatch (daily vessel transits)", normal="2019-01 to 2023-09 average",
                latest=str(w.week_start.max().date()))
    return out, meta


# ---------------------------------------------------------------- model card

def model_performance():
    """Held-out test results of the release models (paper/v22_stats.json, written by
    paper/compute_v22_stats.py) and the paper's benchmarks, for the site's Data & model page."""
    try:
        v = json.load(open("paper/v22_stats.json"))
        lin = json.load(open("paper/linear_baselines.json"))
    except FileNotFoundError:
        return None
    g, fl = v["v2.2"], v["flag"]
    surge, direction = fl["v2.2 conflict-surge model"], fl["v2.0 direction model (4-week escalation probability)"]
    return dict(
        test_period="March 2024 - June 2026 (never used in training)",
        goldstein=dict(r2=g["r2"], r2_trailing_mean=0.600, r2_linear_handbuilt=lin["OLS 44 history features"]["r2"],
                       r2_linear_raw=lin["OLS current week"]["r2"], direction_accuracy=g["dir_acc"]),
        surge=dict(auc=surge["conflict_surge"]["auc"], top10_hit=surge["conflict_surge"]["top10"],
                   base_rate=v["surge_rate"], old_direction_top10=direction["conflict_surge"]["top10"]),
        training=dict(weeks="May 2015 - Oct 2022", mena_dyads=400, external_dyads=1559, world_dyads=v["world_dyads"]))


# ---------------------------------------------------------------- bundle

def build(args):
    from google.cloud import bigquery
    bq = bigquery.Client(project="geopolitical-risk-forecasting")
    as_of = date.fromisoformat(args.as_of) if args.as_of else \
        list(bq.query("SELECT MAX(week_start) w FROM `geopolitical-risk-forecasting.1.1`").result())[0].w
    print(f"as-of week {as_of}", flush=True)
    models = lp.load_models(".")
    now, prev, p = forecasts(bq, models, as_of)
    cv_now, vol = country_view(now, p)
    cv_prev, _ = country_view(prev, p)
    trade, trade_meta = trade_exposure()
    acled, acled_meta = (None, None)
    if not args.no_acled and os.environ.get("ACLED_EMAIL"):
        try:
            acled, acled_meta = acled_recent(as_of + timedelta(days=6))
        except Exception as e:  # the site still builds without the overlay
            print(f"ACLED skipped: {e}")

    # dyads
    r = now.set_index(["actor1", "actor2"])
    idx = {(a, b): i for i, (a, b) in enumerate(zip(p["a1"], p["a2"]))}
    dyads = []
    for (a, b), row in r.iterrows():
        dyads.append(dict(a=a, b=b, f=round(row.goldstein_forecast_next13w, 3), n=round(row.goldstein_trailing52w, 3),
                          s=round(row.conflict_surge_probability, 3), al=bool(row.surge_alert),
                          v=round(float(vol[idx[(a, b)]]), 2)))
    # undirected threads between distinct countries
    threads = []
    for i, a in enumerate(MENA):
        for b in MENA[i + 1:]:
            ab, ba = r.loc[(a, b)], r.loc[(b, a)]
            va, vb = vol[idx[(a, b)]], vol[idx[(b, a)]]
            w = va + vb + 1e-6
            threads.append(dict(a=a, b=b, volume=round(float(va + vb), 2),
                                tone=round(float((ab.goldstein_forecast_next13w * va + ba.goldstein_forecast_next13w * vb) / w), 3),
                                change=round(float(((ab.goldstein_forecast_next13w - ab.goldstein_trailing52w) * va +
                                                    (ba.goldstein_forecast_next13w - ba.goldstein_trailing52w) * vb) / w), 3),
                                surge=round(float(max(ab.conflict_surge_probability, ba.conflict_surge_probability)), 3),
                                alert=bool(ab.surge_alert or ba.surge_alert)))

    # weekly history (last CHART_WEEKS) per country, activity-weighted with today's weights
    T = p["gold_zero"].shape[1]
    hist_weeks = [str(w.date()) for w in p["weeks"][T - CHART_WEEKS:]]
    ev_total = p["counts"].sum(2)
    countries = []
    for c in MENA:
        m = (p["a1"] == c) | (p["a2"] == c)
        w = vol[m] + 1e-6
        g = (p["gold_zero"][m][:, T - CHART_WEEKS:] * w[:, None]).sum(0) / w.sum()
        events = ev_total[m][:, T - CHART_WEEKS:].sum(0)
        neg = p["df"]["negative_goldstein_intensity"].to_numpy("float64").reshape(p["gold_zero"].shape)[m][:, T - CHART_WEEKS:].sum(0)
        fat = acled[c]["fatalities_4w"] if acled and c in acled else None
        sev, parts = severity(cv_now[c], fat)
        sev_prev, _ = severity(cv_prev[c], fat)
        ex = trade[c]["exposure"]
        partners = sorted([d for d in dyads if c in (d["a"], d["b"]) and d["a"] != d["b"]], key=lambda d: -d["v"])
        dom = next(d for d in dyads if d["a"] == c and d["b"] == c)
        countries.append(dict(
            iso3=c, num=ISO_NUM[c], name=NAMES[c],
            risk=round(math.sqrt(sev * ex), 2), risk_prev=round(math.sqrt(sev_prev * ex), 2),
            severity=round(sev, 2), severity_parts={k: (round(v, 3) if v is not None else None) for k, v in parts.items()},
            exposure=round(ex, 2),
            forecast=round(cv_now[c]["forecast"], 3), normal=round(cv_now[c]["normal"], 3),
            change=round(cv_now[c]["change"], 3), change_prev=round(cv_prev[c]["change"], 3),
            surge=round(cv_now[c]["surge"], 3), alerts=cv_now[c]["alerts"],
            worst_active_dyad=round(cv_now[c]["worst_active_dyad"], 3),
            domestic=dict(f=dom["f"], n=dom["n"], s=dom["s"], al=dom["al"]),
            acled_fatalities_4w=fat,
            top_partners=[f"{d['b'] if d['a'] == c else d['a']}" for d in partners[:5]],
            history=dict(goldstein=[round(float(x), 3) for x in g], events=[int(x) for x in events],
                         conflict=[round(float(np.log1p(x)), 3) for x in neg]),
            trade=trade[c], acled=(acled or {}).get(c)))
    countries.sort(key=lambda c: -c["risk"])
    for i, c in enumerate(countries, 1):
        c["rank"] = i
        c["risk_change"] = round(c["risk"] - c["risk_prev"], 2)

    bundle = dict(
        meta=dict(as_of_week=str(as_of), data_through=str(as_of + timedelta(days=6)),
                  previous_as_of=str(as_of - timedelta(weeks=REFRESH_WEEKS)),
                  generated_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                  model=lp.MODEL_VERSION, refresh="forecasts every 2 weeks; trade weekly",
                  trade=trade_meta, acled=acled_meta, chart_weeks=hist_weeks,
                  method=dict(severity="0.40 predicted hostility + 0.30 conflict-surge risk + 0.30 recent ACLED "
                                       "fatalities (re-weighted 4:3 without ACLED); each scaled 0-1",
                              exposure="log China trade (exports + imports, last 12 months) relative to the largest",
                              risk="geometric mean of severity and exposure, sqrt(severity x exposure)", thread_color="forecast Goldstein of the pair "
                              "(activity-weighted over both directions)", thread_width="events per week between "
                              "the pair (both directions, 52-week average)")),
        region=dict(risk=round(float(np.mean([c["risk"] for c in countries])), 2),
                    risk_prev=round(float(np.mean([c["risk_prev"] for c in countries])), 2),
                    severity=round(float(np.mean([c["severity"] for c in countries])), 2),
                    alerts=int(sum(d["al"] for d in dyads))),
        countries=countries, threads=threads, dyads=dyads)

    bundle["meta"]["performance"] = model_performance()
    bundle["lanes"], bundle["meta"]["lanes"] = shipping_lanes(countries, trade)
    for c in bundle["countries"]:
        c["name_zh"] = ZH_NAMES[c["iso3"]]

    summaries, prov = None, None
    if not args.no_ai:
        try:
            summaries, prov = ai_summaries(bundle)
        except Exception as e:
            print(f"AI summaries unavailable ({type(e).__name__}: {str(e)[:120]}); using template text")
    if summaries is None:
        summaries, prov = template_summaries(bundle)
    if prov.get("source") == "ai":  # per-country China briefs with online news context
        news = news_context(as_of)
        detail = china_briefs(bundle, news)
        for iso, d in detail.items():  # the country outlook replaces the dashboard-only summary
            summaries["countries"][iso] = d["outlook"]
            summaries.setdefault("countries_zh", {})[iso] = d["outlook_zh"]
        summaries["detail"] = detail
        print(f"China briefs: {len(detail)}/20; news items: {sum(len(v) for v in news.values())}")
    bundle["summaries"] = dict(summaries, **{"provenance": prov})
    return bundle


def _clean(x):
    """NaN/inf -> None, numpy scalars -> Python, recursively (strict JSON for the browser)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of")
    ap.add_argument("--no-acled", action="store_true")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--gcs", action="store_true", help=f"also upload to gs://{BUCKET}/{SITE_OBJECT}")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    bundle = build(args)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    text = json.dumps(_clean(bundle), separators=(",", ":"), allow_nan=False)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text) / 1e6:.2f} MB); summaries: {bundle['summaries']['provenance']}")
    for c in bundle["countries"][:5]:
        print(f"  #{c['rank']} {c['name']:<22} risk {c['risk']:5.1f}  severity {c['severity']:5.1f}  exposure {c['exposure']:5.1f}")
    if args.gcs:
        from google.cloud import storage
        blob = storage.Client().bucket(BUCKET).blob(SITE_OBJECT)
        blob.cache_control = "public, max-age=300"
        blob.upload_from_string(text, content_type="application/json")
        print(f"uploaded gs://{BUCKET}/{SITE_OBJECT}")


if __name__ == "__main__":
    main()
