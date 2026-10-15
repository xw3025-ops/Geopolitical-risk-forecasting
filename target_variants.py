"""
Alternative 13-week targets for the goldstein model (experiment_targets.py,
benchmark_targets.py). Every target is built from a weekly series s (D x T)
with the same 13-week horizon as the release, so warnings still leave
organisations a quarter to act:

  mean-gold       mean goldstein over weeks t+1..t+13 (the v2.1 release target)
  mean-conflict   mean conflict intensity over t+1..t+13, where weekly conflict
                  intensity = log(1 + sum of |negative goldstein|). Cooperative
                  events (aid, talks) cannot mask fighting, and volume counts.
  peak-gold       the WORST 4-week mean goldstein within t+1..t+13 - how bad
                  will relations get at their low point this quarter.
  peak-conflict   the HIGHEST 4-week mean conflict intensity within t+1..t+13.

Peaks use 4-week means rather than single weeks: a single week of a quiet
dyad can be one or two events, so its extreme is mostly noise.

`normal` is each target's reference level known at week t (the residual
baseline the model corrects, and the zero point for change metrics):
  mean-*   trailing 52-week mean of s
  peak-*   mean over the last 52 weeks of the realised past-13-week peak,
           i.e. "this dyad's typical quarterly low (or high) lately".
"""

import numpy as np

from honest_common import trailing_mean

HORIZON = 13
PEAK_WINDOW = 4
TARGETS = ["mean-gold", "mean-conflict", "peak-gold", "peak-conflict"]


def _future_windows(r, lo, hi):
    """stack r[:, t+k] for k in lo..hi (NaN past the end)."""
    D, T = r.shape
    out = np.full((hi - lo + 1, D, T), np.nan, dtype="float32")
    for i, k in enumerate(range(lo, hi + 1)):
        out[i, :, :T - k] = r[:, k:]
    return out


def _past_windows(r, n):
    """stack r[:, t-k] for k in 0..n-1 (NaN before the start)."""
    D, T = r.shape
    out = np.full((n, D, T), np.nan, dtype="float32")
    for k in range(n):
        out[k, :, k:] = r[:, :T - k]
    return out


def peak_future(s, worst_is_low):
    """Worst 4-week mean within t+1..t+13 (4-week windows ending t+4..t+13)."""
    r = trailing_mean(s, PEAK_WINDOW)
    w = _future_windows(r, PEAK_WINDOW, HORIZON)
    out = w.min(0) if worst_is_low else w.max(0)
    out[:, s.shape[1] - HORIZON:] = np.nan
    return out


def peak_past(s, worst_is_low):
    """Worst 4-week mean within t-12..t (known at week t)."""
    r = trailing_mean(s, PEAK_WINDOW)
    w = _past_windows(r, HORIZON - PEAK_WINDOW + 1)
    return np.nanmin(w, 0) if worst_is_low else np.nanmax(w, 0)


def forward_mean13(s):
    from honest_common import forward_mean
    return forward_mean(s, HORIZON)


def surge_numeric_inputs(src):
    """(D, T, F) inputs of the conflict-surge model (release v2.2), shared by
    training and the live pipeline. src: dict with counts (D, T, 20),
    gold_zero (D, T) and neg = weekly negative goldstein intensity (D, T).
    = the goldstein model's inputs + weekly conflict intensity and the signed
    log of SUMMED goldstein (volume x tone), each with its 52-week mean."""
    from honest_common import goldstein_numeric_inputs
    c = np.log1p(np.asarray(src["neg"], dtype="float64")).astype("float32")
    ssum = src["gold_zero"] * src["counts"].sum(2)
    ssum = (np.sign(ssum) * np.log1p(np.abs(ssum))).astype("float32")
    extra = np.stack([c, trailing_mean(c, 52), ssum, trailing_mean(ssum, 52)], 2)
    return np.concatenate([goldstein_numeric_inputs(src, long_features=True), extra], 2).astype("float32")


def build(target, gold_zero, neg_intensity):
    """-> dict(y, normal, series, worse_up) for one source of dyads.
    worse_up: True when a HIGHER value is worse (conflict targets)."""
    conflict = target.endswith("conflict")
    s = np.log1p(np.asarray(neg_intensity, dtype="float64")).astype("float32") if conflict \
        else np.asarray(gold_zero, dtype="float32")
    worst_is_low = not conflict
    if target.startswith("mean"):
        y, normal = forward_mean13(s), trailing_mean(s, 52)
    else:
        y = peak_future(s, worst_is_low)
        normal = trailing_mean(peak_past(s, worst_is_low), 52)
    return dict(y=y.astype("float32"), normal=normal.astype("float32"), series=s, worse_up=conflict)
