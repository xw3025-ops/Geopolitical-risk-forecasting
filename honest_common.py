"""
Shared data loading, splitting, labels and feature building for the v2.0
release models (release_v2.0_goldstein.py, release_v2.0_direction.py) and
benchmark_honest.py. No TensorFlow here.

Data source: the BigQuery table geopolitical-risk-forecasting.1.1 (the same
table the live pipeline pulls from), via two local pulls of it:
  raw data/bq_panel_22_with_target_4w.parquet   (build_bq_panel_22_4w.py)
      every feature column + the 4-week target, 2015-02-16..2026-08-24
  raw data/bq_panel_22_with_target_13w.parquet  (build_bq_panel_22.py)
      the 13-week target, valid through 2026-06-22
The 13-week target is merged onto the 4-week panel's dyad-week grid (NaN for
the last weeks where 13 future weeks don't exist yet).

Why not the CSV panel release_v1.0_goldstein.py trained on: its goldstein
column came from a separate export whose country list had ERI and SOM in
place of LBY and SDN, so every LBY/SDN dyad had avg_goldstein padded to 0 and
a constant-0 target. The BigQuery table covers LBY and SDN properly, and for
the other 324 dyads its goldstein_mean and event counts are the same numbers
as the CSV (checked: identical on >99.9% of overlapping dyad-weeks; the
remainder are small GDELT revisions between the two pulls).

Conventions (all arrays are dyad-major, D dyads x T weeks):
  gold_zero[d, t] - the week's mean goldstein, 0 if the dyad-week had no
                    events. Both targets are built from this, matching v1.0.
  gold_obs[d, t]  - the same, but NaN for empty weeks ("no data", not
                    "neutral") - used for features.
"""

import warnings

import numpy as np
import pandas as pd

PANEL_PATH = "raw data/bq_panel_22_with_target_4w.parquet"
TARGET13_PATH = "raw data/bq_panel_22_with_target_13w.parquet"
TARGET4_COL = "target_avg_goldstein_next4w"
TARGET13_COL = "target_avg_goldstein_next13w"

MENA = [
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
]
EXTRA = ["USA", "CHN"]  # direction model only: attention context, never graded
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]
LOOKBACK_WEEKS = 13

# Splits: the same validation/test start weeks as the v1.0 release scripts,
# with an embargo equal to each target's horizon (see split_by_dates).
GOLDSTEIN_HORIZON, GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START = 13, "2023-02-06", "2024-03-18"
DIRECTION_HORIZON, DIRECTION_VAL_START, DIRECTION_TEST_START = 4, "2023-04-10", "2024-05-27"

# CAMEO root codes -> QuadClass groups (0-based column slices of count_01..20)
CAMEO_GROUPS = {
    "vcoop": slice(0, 5),      # 01-05 verbal cooperation
    "mcoop": slice(5, 9),      # 06-09 material cooperation
    "vconf": slice(9, 14),     # 10-14 verbal conflict
    "mconf": slice(14, 20),    # 15-20 material conflict
    "violent": slice(17, 20),  # 18-20 assault / fight / mass violence
}


# ---------------------------------------------------------------- loading

def load_panel(countries=MENA):
    """The training panel: the two local BigQuery pulls, merged."""
    key = ["week_start", "Actor1CountryCode", "Actor2CountryCode"]
    df = pd.read_parquet(PANEL_PATH)
    t13 = pd.read_parquet(TARGET13_PATH, columns=key + [TARGET13_COL])
    for frame in (df, t13):
        frame["week_start"] = pd.to_datetime(frame["week_start"])
    return panel_from_df(df.merge(t13, on=key, how="left"), countries)


def panel_from_df(df, countries=MENA):
    """Dense dyad-major arrays over `countries` (all ordered pairs incl. self)
    from a dyad-week frame shaped like table 1.1 (nulls already 0). Used for
    training AND by live_pipeline.py, so live inputs are built by the same
    code. Target columns are optional (absent in a live pull -> NaN).
    `df` in the result is the filtered, sorted frame."""
    df = df[df.Actor1CountryCode.isin(countries) & df.Actor2CountryCode.isin(countries)]
    df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
    T = df.week_start.nunique()
    D = len(countries) ** 2
    per_dyad = df.groupby(["Actor1CountryCode", "Actor2CountryCode"]).size()
    assert len(per_dyad) == D and (per_dyad == T).all(), "expected a fully dense panel"

    def grid(col, dtype="float32"):
        if col not in df.columns:
            return np.full((D, T), np.nan, dtype=dtype)
        return df[col].to_numpy(dtype=dtype).reshape(D, T)

    counts = df[COUNT_COLS].to_numpy(dtype="float32").reshape(D, T, 20)
    has_events = grid("total_count") > 0
    gold_zero = grid("goldstein_mean")
    p = dict(
        df=df,
        a1=df.Actor1CountryCode.to_numpy().reshape(D, T)[:, 0],
        a2=df.Actor2CountryCode.to_numpy().reshape(D, T)[:, 0],
        weeks=pd.DatetimeIndex(sorted(df.week_start.unique())),
        counts=counts,
        has_events=has_events,
        gold_zero=gold_zero,
        gold_obs=np.where(has_events, gold_zero, np.nan).astype("float32"),
        tone_obs=np.where(has_events, grid("tone_mean"), np.nan).astype("float32"),
        target4=grid(TARGET4_COL),
        target13=grid(TARGET13_COL),
    )
    p["escalation"] = escalation_label(p["target4"], gold_zero)
    return p


# ---------------------------------------------------------------- model inputs
# Shared by the release scripts (training) and live_pipeline.py (inference).

def long_memory_inputs(p):
    """(D, T, 3): trailing 26- and 52-week mean goldstein, log 52-week mean
    event count. Needs 52 weeks of history before the first window week to
    match training exactly."""
    total = p["counts"].sum(axis=2)
    return np.stack([trailing_mean(p["gold_zero"], 26),
                     trailing_mean(p["gold_zero"], 52),
                     np.log1p(trailing_mean(total, 52))], axis=2)


def goldstein_numeric_inputs(p, long_features=True, has_events=False):
    """(D, T, F) per-week inputs of the goldstein model: log1p count_01..20,
    goldstein_mean (0 when empty), optional has_events, optional long-memory."""
    parts = [np.log1p(p["counts"]), p["gold_zero"][:, :, None]]
    if has_events:
        parts.append(p["has_events"][:, :, None].astype("float32"))
    if long_features:
        parts.append(long_memory_inputs(p))
    return np.concatenate(parts, axis=2).astype("float32")


DIRECTION_EXCLUDE_COLS = {"week_start", "Actor1CountryCode", "Actor2CountryCode", "dyad_scope",
                          "top_event_root", "action_lat_mean", "action_long_mean",
                          TARGET4_COL, TARGET13_COL}


def direction_feature_columns(df):
    """Training-time split of table-1.1 columns into count columns (integer
    in BigQuery -> log1p) and continuous ones. Saved in the release
    artifacts; live inference uses the saved lists, not this function."""
    cols = [c for c in df.columns if c not in DIRECTION_EXCLUDE_COLS]
    log_cols = [c for c in cols if str(df[c].dtype).startswith("Int")]
    return log_cols, [c for c in cols if c not in log_cols]


def direction_numeric_inputs(p, log_cols, raw_cols):
    """(D, T, F) per-week inputs of the direction model: log1p counts, raw
    continuous columns, long-memory inputs."""
    df = p["df"]
    D, T = p["gold_zero"].shape
    return np.concatenate([
        np.log1p(df[log_cols].to_numpy(dtype="float32")).reshape(D, T, -1),
        df[raw_cols].to_numpy(dtype="float32").reshape(D, T, -1),
        long_memory_inputs(p),
    ], axis=2).astype("float32")


def escalation_label(target4, gold_now):
    """Direction label (v2.0): 1 = ESCALATION if the mean goldstein over the
    next 4 weeks is LOWER than this week's - goldstein falling means
    relations moving toward conflict. 0 = de-escalation or flat.
    (v1.0 had this inverted: 1 meant goldstein ROSE.)"""
    return (target4 < gold_now).astype("int32")


# ---------------------------------------------------------------- splitting

def split_by_dates(weeks, target_row, val_start, test_start, horizon, lookback=LOOKBACK_WEEKS):
    """Week indices for train / val / test, with an EMBARGO of `horizon`
    weeks before val and before test: a row at week t has a target averaging
    weeks t+1..t+horizon, so without the gap the last training rows' targets
    overlap the evaluation period. Prediction weeks start at lookback-1 (the
    first week with 13 weeks of history, as in the release scripts) and end
    at the last week with a valid target."""
    v = weeks.get_loc(pd.Timestamp(val_start))
    s = weeks.get_loc(pd.Timestamp(test_start))
    last = int(np.where(~np.isnan(target_row))[0].max())
    return (np.arange(lookback - 1, v - horizon),
            np.arange(v, s - horizon),
            np.arange(s, last + 1))


# ---------------------------------------------------------------- rolling helpers

def _trailing(x, w, nan_aware):
    """Mean over weeks t-w+1..t along axis 1 (partial windows at the start).
    nan_aware: mean over the non-NaN weeks only (NaN if there are none)."""
    valid = ~np.isnan(x) if nan_aware else np.ones_like(x, dtype=bool)
    vals = np.where(valid, x, 0.0)
    cs = np.cumsum(vals, axis=1, dtype="float64")
    cn = np.cumsum(valid, axis=1, dtype="float64")
    pad = np.zeros((x.shape[0], 1))
    cs, cn = np.concatenate([pad, cs], axis=1), np.concatenate([pad, cn], axis=1)
    T = x.shape[1]
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - w, 0)
    s, n = cs[:, hi] - cs[:, lo], cn[:, hi] - cn[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan).astype("float32")


def trailing_mean(x, w):
    return _trailing(x, w, nan_aware=False)


def trailing_nanmean(x, w):
    return _trailing(x, w, nan_aware=True)


def trailing_count(x, w):
    """How many of weeks t-w+1..t are non-NaN."""
    return trailing_mean((~np.isnan(x)).astype("float32"), w) * np.minimum(np.arange(1, x.shape[1] + 1), w)


def trailing_nanstd(x, w):
    m = trailing_nanmean(x, w)
    return np.sqrt(np.maximum(trailing_nanmean(x * x, w) - m * m, 0)).astype("float32")


def trailing_nanmin(x, w):
    out = np.full(x.shape, np.inf, dtype="float32")
    filled = np.where(np.isnan(x), np.inf, x)
    for k in range(w):
        out[:, k:] = np.minimum(out[:, k:], filled[:, :x.shape[1] - k])
    return np.where(np.isinf(out), np.nan, out).astype("float32")


def _shift_forward(a, fill):
    """b[:, t] = a[:, t+1]"""
    return np.concatenate([a[:, 1:], np.full((a.shape[0], 1), fill, dtype=a.dtype)], axis=1)


def forward_mean(x, h):
    """Mean over weeks t+1..t+h; NaN where fewer than h future weeks exist."""
    m = _shift_forward(trailing_mean(x[:, ::-1], h)[:, ::-1], np.nan)
    m[:, -h:] = np.nan
    return m


def forward_nanmean(x, h):
    """Mean over the non-NaN weeks among t+1..t+h, and how many there were."""
    m = _shift_forward(trailing_nanmean(x[:, ::-1], h)[:, ::-1], np.nan)
    n = _shift_forward(trailing_count(x[:, ::-1], h)[:, ::-1].astype("float32"), 0.0)
    m[:, -h:] = np.nan
    n[:, -h:] = 0
    return m, n


# ---------------------------------------------------------------- features

def build_features(p):
    """Hand-built lag features, (D, T, F) float32, using only weeks <= t.
    No country identity / embeddings - everything is computed from history."""
    g_obs, g_zero, counts = p["gold_obs"], p["gold_zero"], p["counts"]
    a1, a2 = p["a1"], p["a2"]
    D, T = g_zero.shape
    feats, names = [], []

    def add(name, arr):
        names.append(name)
        feats.append(np.asarray(arr, dtype="float32"))

    # the dyad's own goldstein history
    add("gold_now", g_obs)
    add("gold_now_zero", g_zero)
    for w in (4, 13, 26, 52):
        add(f"gold_mean{w}", trailing_nanmean(g_obs, w))
    for w in (4, 13, 26):
        add(f"gold_zmean{w}", trailing_mean(g_zero, w))
    add("gold_std13", trailing_nanstd(g_obs, 13))
    add("gold_min13", trailing_nanmin(g_obs, 13))
    add("gold_trend_4v26", trailing_nanmean(g_obs, 4) - trailing_nanmean(g_obs, 26))

    # news tone
    add("tone_now", p["tone_obs"])
    for w in (4, 13):
        add(f"tone_mean{w}", trailing_nanmean(p["tone_obs"], w))

    # activity level and CAMEO mix
    total = counts.sum(axis=2)
    add("events_now", np.log1p(total))
    for w in (4, 13):
        add(f"events_mean{w}", np.log1p(trailing_mean(total, w)))
    for w in (13, 52):
        add(f"active_frac{w}", trailing_mean((total > 0).astype("float32"), w))
    for label, sl in CAMEO_GROUPS.items():
        grp = counts[:, :, sl].sum(axis=2)
        for w in (1, 4, 13):
            num, den = (grp, total) if w == 1 else (trailing_mean(grp, w), trailing_mean(total, w))
            with np.errstate(invalid="ignore", divide="ignore"):
                add(f"{label}_share{w}", np.where(den > 0, num / den, np.nan))

    # the reverse dyad (B->A)
    idx = {(x, y): i for i, (x, y) in enumerate(zip(a1, a2))}
    rev = np.array([idx[(y, x)] for x, y in zip(a1, a2)])
    for w in (4, 13):
        add(f"rev_gold_mean{w}", trailing_nanmean(g_obs, w)[rev])
    add("rev_gold_zmean13", trailing_mean(g_zero, 13)[rev])

    # country- and region-level context: mean goldstein across every dyad a
    # country takes part in, and across the whole region
    countries = sorted(set(a1))
    ci = {c: i for i, c in enumerate(countries)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN weeks -> NaN
        cmat = np.stack([np.nanmean(g_obs[(a1 == c) | (a2 == c)], axis=0) for c in countries])
        region = np.nanmean(g_obs, axis=0, keepdims=True)
    for w in (4, 13):
        cm = trailing_nanmean(cmat, w)
        add(f"actor1_ctx{w}", cm[[ci[c] for c in a1]])
        add(f"actor2_ctx{w}", cm[[ci[c] for c in a2]])
        add(f"region_ctx{w}", np.repeat(trailing_nanmean(region, w), D, axis=0))

    return np.stack(feats, axis=2), names


def windows_at(numeric, week_idx, lookback=LOOKBACK_WEEKS):
    """(D, T, F) -> (len(week_idx), D, lookback, F): for each prediction week
    t, every dyad's weeks t-lookback+1..t - the transformer input."""
    return np.stack([numeric[:, t - lookback + 1: t + 1] for t in week_idx]).astype("float32")


def rows(arr, week_idx):
    """(D, T, ...) -> (len(week_idx) * D, ...), week-major (all dyads of the
    first week, then all dyads of the next, ...)."""
    return np.swapaxes(arr[:, week_idx], 0, 1).reshape((-1,) + arr.shape[2:])
