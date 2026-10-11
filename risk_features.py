"""
Feature and target construction for the v2 risk models. Used by BOTH
train_risk_gbm.py and predict_risk.py, so training and live scoring always
build inputs the same way (the v1 transformers refit their preprocessing
inside every script, which let the two silently diverge).

Every feature for week t uses data up to and including week t only; UCDP
features additionally lag by UCDP_REPORTING_LAG_WEEKS (see risk_config.py).

Two row types:

  dyad rows    (week, Actor1 -> Actor2), including self-dyads (A -> A =
               domestic events) and, when present in the panel, CHN <-> host
               dyads. Two targets:
               y_worsen_{H} (diplomatic TONE): the next-H-week EVENT-WEIGHTED
               goldstein ends up >= DETERIORATION_DROP points below the
               dyad's own trailing 26-week level. Rows without enough events
               to measure either side get NaN (not graded).
               y_violence_{H} (INTENSITY): violent events (CAMEO 18-20) in
               the next H weeks reach >= SURGE_FACTOR x the dyad's trailing
               26-week rate and >= MIN_VIOLENT_FUTURE[H]. Rates are relative
               to all OTHER events in the panel, which removes GDELT's volume
               drift. Needed because average tone can IMPROVE during a war
               (ISR -> PSE after Oct 7 2023 went from -3.8 to -2.4: ceasefire
               and aid events dilute the average), so y_worsen misses it.

  country rows (week, country). Target y_escalate_{H}: UCDP organised-violence
               deaths in the next H weeks reach max(ESCALATION_MIN_DEATHS[H],
               2 x deaths in the last H weeks), i.e. a new conflict or a
               doubling of an existing one. NaN beyond UCDP coverage.
"""

import numpy as np
import pandas as pd

from risk_config import CHINA, UCDP_REPORTING_LAG_WEEKS

HORIZONS = (4, 13)
REF_WEEKS = 26
DETERIORATION_DROP = 1.0
MIN_EVENTS_REF = 20
MIN_EVENTS_FUTURE = {4: 10, 13: 20}
SURGE_FACTOR = 2.0
MIN_VIOLENT_FUTURE = {4: 20, 13: 50}
ESCALATION_MIN_DEATHS = {4: 10, 13: 25}
DISPLAY_MIN_EVENTS = 100  # watchlist hides dyads with fewer events in 26 weeks (too sparse to trust)

COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
# CAMEO root-code groups (quad classes plus two sub-groups of interest).
GROUPS = {
    "verb_coop": range(1, 6), "mat_coop": range(6, 10), "verb_conf": range(10, 15),
    "mat_conf": range(15, 21), "violent": range(18, 21), "protest": [14],
}

FEATURE_DESCRIPTIONS = {
    "gw": "event-weighted Goldstein (cooperation level)",
    "g_trend_4_26": "Goldstein last 4w vs last 26w (recent change)",
    "g_trend_13_52": "Goldstein last 13w vs last 52w",
    "g_week_std_13": "week-to-week volatility of Goldstein",
    "g_week_min_13": "worst single week in last 13w",
    "log_events": "event volume (log of weekly events)",
    "event_surge_4_52": "event volume last 4w vs 52w (media attention surge)",
    "attention_share_4": "share of all regional events (volume-normalised)",
    "share_mat_conf": "share of material-conflict events (CAMEO 15-20)",
    "share_violent": "share of violent events (CAMEO 18-20)",
    "share_verb_conf": "share of verbal-conflict events (CAMEO 10-14)",
    "share_protest": "share of protest events (CAMEO 14)",
    "share_mat_coop": "share of material-cooperation events (CAMEO 06-09)",
    "conf_trend_4_52": "material-conflict share last 4w vs 52w",
    "violent_surge_4_26": "violent events last 4w vs 26w (surge)",
    "imputed_share_13": "share of Goldstein values imputed from counts",
    "weeks_active_13": "share of last 13 weeks with any events",
    "rev_": "reverse direction (B -> A): ",
    "a1_": "actor 1 country: ", "a2_": "actor 2 country: ",
    "dom_": "domestic (self-dyad): ", "ctry_": "all relations of the country: ",
    "inbound_share_mat_conf_4": "material conflict directed AT the country by others",
    "region_": "whole panel: ",
    "ucdp_log_deaths": "UCDP deaths (log of total, lagged)",
    "ucdp_death_trend": "UCDP deaths last 13w vs 52w average (lagged)",
    "ucdp_weeks_since_fatal": "weeks since last UCDP fatal event (lagged)",
    "ucdp_share_onesided_52": "share of deaths from one-sided violence against civilians",
    "is_domestic": "domestic dyad (A -> A)", "involves_china": "dyad involves China",
    "actor1": "identity of actor 1", "actor2": "identity of actor 2", "country": "country identity",
}


def describe(feature):
    """Plain-language label for a feature name, for alert explanations."""
    prefix = ""
    for p in ("rev_", "a1_", "a2_", "dom_", "ctry_", "region_"):
        if feature.startswith(p) and feature not in FEATURE_DESCRIPTIONS:
            prefix += FEATURE_DESCRIPTIONS[p]
            feature = feature[len(p):]
            break
    if feature in FEATURE_DESCRIPTIONS:
        return prefix + FEATURE_DESCRIPTIONS[feature]
    for key, text in FEATURE_DESCRIPTIONS.items():
        if feature.startswith(key):
            return prefix + text + feature[len(key):].replace("_", " ") + "w"
    return prefix + feature


# ---------------------------------------------------------------- helpers

def load_panel(path):
    panel = pd.read_csv(path, parse_dates=["week_start"])
    panel["dyad"] = panel.Actor1CountryCode + "_" + panel.Actor2CountryCode
    return panel


def load_ucdp(path):
    return pd.read_csv(path, parse_dates=["week_start"])


def _roll(W, w):
    return W.rolling(w, min_periods=1).sum()


def _fwd(W, h):
    """Sum over weeks t+1..t+h; NaN when any of those weeks is missing."""
    return W.iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1].shift(-1)


def _ratio(num, den):
    return num / den.where(den > 0)


def _log_rate(W, w):
    return np.log1p(_roll(W, w) / w)


def _weeks_since_positive(W, cap=104):
    idx = np.arange(len(W), dtype=float)[:, None] * np.ones((1, W.shape[1]))
    last = pd.DataFrame(np.where(W.to_numpy() > 0, idx, np.nan), index=W.index, columns=W.columns).ffill()
    since = idx - last.to_numpy()
    since[np.isnan(W.to_numpy())] = np.nan
    return pd.DataFrame(np.minimum(np.nan_to_num(since, nan=cap), cap), index=W.index, columns=W.columns
                        ).where(W.notna())


class PanelMatrices:
    """Wide (weeks x dyads) matrices of the raw panel, plus country aggregates."""

    def __init__(self, panel):
        self.weeks = pd.DatetimeIndex(np.sort(panel.week_start.unique()))
        self.dyads = np.sort(panel.dyad.unique())
        parts = pd.Series(self.dyads).str.split("_", expand=True)
        self.a1, self.a2 = parts[0].to_numpy(), parts[1].to_numpy()
        self.countries = np.array(sorted(set(self.a1) | set(self.a2)))

        def wide(values):
            return (panel.assign(_v=values).pivot(index="week_start", columns="dyad", values="_v")
                    .reindex(index=self.weeks, columns=self.dyads).astype("float32"))

        self.n = wide(panel.n_events)
        self.gsum = wide(panel.goldstein_sum)
        self.g = wide(panel.goldstein)
        self.imputed = wide(panel.goldstein_imputed)
        self.groups = {name: wide(panel[[f"count_{i:02d}" for i in codes]].sum(axis=1))
                       for name, codes in GROUPS.items()}

        # Incidence matrices (dyads x countries) for country-level aggregates.
        c_idx = {c: i for i, c in enumerate(self.countries)}
        self.M_any = np.zeros((len(self.dyads), len(self.countries)), dtype="float32")
        self.M_in = np.zeros_like(self.M_any)
        for d, (x, y) in enumerate(zip(self.a1, self.a2)):
            self.M_any[d, c_idx[x]] = 1
            self.M_any[d, c_idx[y]] = 1
            if x != y:
                self.M_in[d, c_idx[y]] = 1  # others acting on y
        self.self_dyad = {c: f"{c}_{c}" for c in self.countries if f"{c}_{c}" in set(self.dyads)}

    def to_country(self, W, M):
        return pd.DataFrame(W.to_numpy() @ M, index=self.weeks, columns=self.countries)


def _relation_features(n, gsum, g, groups, imputed, total_n):
    """Feature matrices for any set of columns (dyads or countries)."""
    f = {}
    for w in (4, 13, 26, 52):
        f[f"gw_{w}"] = _ratio(_roll(gsum, w), _roll(n, w))
    f["g_trend_4_26"] = f["gw_4"] - f["gw_26"]
    f["g_trend_13_52"] = f["gw_13"] - f["gw_52"]
    if g is not None:
        f["g_week_std_13"] = g.rolling(13, min_periods=3).std()
        f["g_week_min_13"] = g.rolling(13, min_periods=1).min()
    for w in (4, 13, 52):
        f[f"log_events_{w}"] = _log_rate(n, w)
    f["event_surge_4_52"] = f["log_events_4"] - f["log_events_52"]
    f["attention_share_4"] = np.log(_ratio(_roll(n, 4), _roll(total_n, 4)).clip(lower=1e-7))
    for grp in ("mat_conf", "violent", "verb_conf", "protest", "mat_coop"):
        for w in ((4, 13) if grp in ("mat_conf", "violent") else (4,)):
            f[f"share_{grp}_{w}"] = _ratio(_roll(groups[grp], w), _roll(n, w))
    f["conf_trend_4_52"] = f["share_mat_conf_4"] - _ratio(_roll(groups["mat_conf"], 52), _roll(n, 52))
    f["violent_surge_4_26"] = _log_rate(groups["violent"], 4) - _log_rate(groups["violent"], 26)
    if imputed is not None:
        f["imputed_share_13"] = imputed.rolling(13, min_periods=1).mean()
        f["weeks_active_13"] = (n > 0).astype("float32").rolling(13, min_periods=1).mean()
    return f


def _ucdp_features(ucdp, weeks, countries):
    deaths = (ucdp.pivot(index="week_start", columns="iso3", values="ucdp_deaths")
              .reindex(index=weeks, columns=countries))
    onesided = (ucdp.pivot(index="week_start", columns="iso3", values="ucdp_deaths_onesided")
                .reindex(index=weeks, columns=countries))
    lag = UCDP_REPORTING_LAG_WEEKS
    d_lag, o_lag = deaths.shift(lag), onesided.shift(lag)
    f = {f"ucdp_log_deaths_{w}": np.log1p(d_lag.rolling(w, min_periods=1).sum()) for w in (4, 13, 52)}
    f["ucdp_death_trend"] = f["ucdp_log_deaths_13"] - np.log1p(d_lag.rolling(52, min_periods=1).sum() / 4)
    f["ucdp_weeks_since_fatal"] = _weeks_since_positive(d_lag)
    f["ucdp_share_onesided_52"] = _ratio(o_lag.rolling(52, min_periods=1).sum(), d_lag.rolling(52, min_periods=1).sum())
    return f, deaths


def _to_long(mats, weeks, cols, key_name):
    out = pd.DataFrame({
        "week_start": np.repeat(weeks.to_numpy(), len(cols)),
        key_name: np.tile(cols, len(weeks)),
    })
    for name, mat in mats.items():
        out[name] = np.asarray(mat, dtype="float32").ravel()
    return out


# ---------------------------------------------------------------- frames

def build_frames(panel, ucdp):
    """Return (dyad_frame, country_frame, feature lists)."""
    P = PanelMatrices(panel)
    total_n = pd.DataFrame(np.repeat(P.n.sum(axis=1).to_numpy()[:, None], len(P.dyads), axis=1),
                           index=P.weeks, columns=P.dyads)

    # Country-level aggregates (all relations involving the country, and inbound).
    cn = P.to_country(P.n, P.M_any)
    cgs = P.to_country(P.gsum, P.M_any)
    cgroups = {k: P.to_country(v, P.M_any) for k, v in P.groups.items()}
    c_total = pd.DataFrame(np.repeat(cn.sum(axis=1).to_numpy()[:, None], len(P.countries), axis=1),
                           index=P.weeks, columns=P.countries)
    ctry = _relation_features(cn, cgs, None, cgroups, None, c_total)
    inbound = _ratio(_roll(P.to_country(P.groups["mat_conf"], P.M_in), 4), _roll(P.to_country(P.n, P.M_in), 4))
    dom_cols = [P.self_dyad.get(c) for c in P.countries]

    dyad_rel = _relation_features(P.n, P.gsum, P.g, P.groups, P.imputed, total_n)
    ucdp_f, deaths = _ucdp_features(ucdp, P.weeks, P.countries)
    region_conf = _ratio(_roll(P.groups["mat_conf"].sum(axis=1), 4), _roll(P.n.sum(axis=1), 4))
    region_gw = _ratio(_roll(P.gsum.sum(axis=1), 4), _roll(P.n.sum(axis=1), 4))

    # ----- dyad frame
    D = dict(dyad_rel)
    rev = [f"{y}_{x}" for x, y in zip(P.a1, P.a2)]
    for k in ("gw_13", "g_trend_4_26", "share_mat_conf_4"):
        D[f"rev_{k}"] = dyad_rel[k][rev].to_numpy()
    country_ctx = {"ctry_gw_4": ctry["gw_4"], "ctry_share_mat_conf_4": ctry["share_mat_conf_4"],
                   "ctry_conf_trend_4_52": ctry["conf_trend_4_52"], "ctry_event_surge_4_52": ctry["event_surge_4_52"],
                   "ucdp_log_deaths_13": ucdp_f["ucdp_log_deaths_13"], "ucdp_death_trend": ucdp_f["ucdp_death_trend"]}
    for role, actors in (("a1", P.a1), ("a2", P.a2)):
        for k, mat in country_ctx.items():
            D[f"{role}_{k}"] = mat[actors].to_numpy()
    D["region_share_mat_conf_4"] = np.repeat(region_conf.to_numpy()[:, None], len(P.dyads), axis=1)
    D["region_gw_4"] = np.repeat(region_gw.to_numpy()[:, None], len(P.dyads), axis=1)

    ref_n = _roll(P.n, REF_WEEKS)
    for h in HORIZONS:
        fut_n = _fwd(P.n, h)
        gw_next = _ratio(_fwd(P.gsum, h), fut_n)
        gradable = (fut_n >= MIN_EVENTS_FUTURE[h]) & (ref_n >= MIN_EVENTS_REF)
        y = (gw_next <= dyad_rel["gw_26"] - DETERIORATION_DROP).astype("float32").where(gradable)
        D[f"gw_next_{h}"] = gw_next
        D[f"y_worsen_{h}"] = y

    violent = P.groups["violent"]
    others = total_n - P.n
    ref_rate = _ratio(_roll(violent, REF_WEEKS), _roll(others, REF_WEEKS)).fillna(0)
    for h in HORIZONS:
        fut_v, fut_others = _fwd(violent, h), _fwd(others, h)
        surge = (_ratio(fut_v, fut_others) >= SURGE_FACTOR * ref_rate) & (fut_v >= MIN_VIOLENT_FUTURE[h])
        D[f"violent_next_{h}"] = fut_v
        D[f"y_violence_{h}"] = surge.astype("float32").where(fut_others.notna())
    D["ref_events_26"] = ref_n

    dyad_frame = _to_long(D, P.weeks, P.dyads, "dyad")
    dyad_frame["actor1"] = np.tile(P.a1, len(P.weeks))
    dyad_frame["actor2"] = np.tile(P.a2, len(P.weeks))
    dyad_frame["is_domestic"] = (dyad_frame.actor1 == dyad_frame.actor2).astype("float32")
    dyad_frame["involves_china"] = ((dyad_frame.actor1 == CHINA) | (dyad_frame.actor2 == CHINA)).astype("float32")

    # ----- country frame
    C = {}
    for k, mat in dyad_rel.items():
        if k in ("gw_4", "gw_13", "g_trend_4_26", "log_events_4", "event_surge_4_52", "share_mat_conf_4",
                 "share_mat_conf_13", "share_violent_4", "share_protest_4", "g_week_min_13", "violent_surge_4_26"):
            C[f"dom_{k}"] = np.column_stack([mat[c].to_numpy() if c else np.full(len(P.weeks), np.nan)
                                             for c in dom_cols])
    for k in ("gw_4", "gw_13", "g_trend_4_26", "share_mat_conf_4", "share_violent_4", "conf_trend_4_52",
              "log_events_4", "event_surge_4_52", "violent_surge_4_26"):
        C[f"ctry_{k}"] = ctry[k]
    C["inbound_share_mat_conf_4"] = inbound
    C.update(ucdp_f)
    C["region_share_mat_conf_4"] = np.repeat(region_conf.to_numpy()[:, None], len(P.countries), axis=1)
    for h in HORIZONS:
        d_next = _fwd(deaths, h)
        d_prev = deaths.rolling(h, min_periods=h).sum()
        threshold = np.maximum(ESCALATION_MIN_DEATHS[h], 2 * d_prev)
        C[f"deaths_next_{h}"] = d_next
        C[f"y_escalate_{h}"] = (d_next >= threshold).astype("float32").where(d_next.notna() & d_prev.notna())
    country_frame = _to_long(C, P.weeks, P.countries, "country")

    target_like = ("y_", "gw_next_", "deaths_next_", "violent_next_", "ref_events_")
    dyad_features = [c for c in dyad_frame.columns
                     if c not in ("week_start", "dyad") and not c.startswith(target_like)]
    country_features = [c for c in country_frame.columns
                        if c != "week_start" and not c.startswith(target_like)]
    return dyad_frame, country_frame, dyad_features, country_features
