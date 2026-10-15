"""
Benchmarks for the alternative targets in target_variants.py, and the shared
evaluation used by experiment_targets.py.

For each target: naive forecasts, linear regression and gradient boosting on
the 44 hand-built features (+ 3 conflict-history features for conflict
targets), fit on training weeks; gradient boosting's tree count is chosen on
validation. Same splits and 13-week embargo as the release.

Evaluation (test weeks x 400 MENA dyads), `evaluate()`:
  own-target   r2, mae, change_corr, dir_acc, esc_auc (forecast_metrics; the
               escalation threshold is the worst 10% of training-period changes)
  amplitude    sd(forecast - normal) / sd(actual - normal): 1 = moves as much as
               reality, near 0 = hugs the normal
  crisis AUCs  SHARED labels, the same for every target so that targets can be
               compared: how well each forecast ranks dyad-weeks before a quarter
               whose worst month is unusually bad FOR THAT DYAD:
                 gold-crisis      worst 4-week mean goldstein in the next 13 weeks
                                  is below the dyad's typical quarterly low (52-week
                                  average of its past-quarter lows) by more than the
                                  worst 10% of training-period gaps
                 conflict-surge   the same for the highest 4-week conflict intensity
               (= the peak targets' own escalation labels). Each forecast's
               score = how much worse than its own normal it says things will be.
  ISR-PSE      change_corr and amplitude on the two Israel-Palestine dyads.

    python benchmark_targets.py           -> logs/targets_benchmarks.json
"""

import json
import warnings
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score

import target_variants as tv
from benchmark_honest import fit_gbm
from benchmark_ladder import forecast_metrics
from honest_common import (GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA, build_features,
                           load_panel, rows, split_by_dates, trailing_mean)

OUT = "logs/targets_benchmarks.json"
FOCUS = [("ISR", "PSE"), ("PSE", "ISR")]


def neg_intensity(p):
    D, T = p["gold_zero"].shape
    return p["df"]["negative_goldstein_intensity"].to_numpy(dtype="float64").reshape(D, T)


class Context:
    """Everything evaluate() needs that does not depend on the forecast."""

    def __init__(self, p, tr, va, te):
        self.p, self.tr, self.va, self.te = p, tr, va, te
        neg = neg_intensity(p)
        self.T = {t: tv.build(t, p["gold_zero"], neg) for t in tv.TARGETS}
        self.focus = [int(np.where((p["a1"] == a) & (p["a2"] == b))[0][0]) for a, b in FOCUS]
        self.esc = {}
        for t, d in self.T.items():  # worst 10% of training-period changes, in the worse direction
            ch = (d["y"] - d["normal"])[:, tr]
            ch = ch[np.isfinite(ch)]
            self.esc[t] = float(np.quantile(ch, 0.9) if d["worse_up"] else -np.quantile(ch, 0.1))
        g, c = self.T["peak-gold"], self.T["peak-conflict"]
        with np.errstate(invalid="ignore"):
            self.labels = {"gold-crisis": g["y"] - g["normal"] <= -self.esc["peak-gold"],
                           "conflict-surge": c["y"] - c["normal"] >= self.esc["peak-conflict"]}

    def evaluate(self, target, f, weeks=None):
        """f: (weeks, 400) forecast of `target` on the test (or given) weeks."""
        w = self.te if weeks is None else weeks
        d = self.T[target]
        y, normal = d["y"][:, w].T, d["normal"][:, w].T
        out = forecast_metrics(y, f, normal, self.esc[target], d["worse_up"])
        out["amplitude"] = float(np.std(f - normal) / np.std(y - normal))
        score = (f - normal) if d["worse_up"] else (normal - f)
        for name, lab in self.labels.items():
            L = lab[:, w].T
            ok = np.isfinite(score)
            out[f"auc_{name}"] = float(roc_auc_score(L[ok], score[ok]))
            out[f"rate_{name}"] = float(L.mean())
        for (a, b), j in zip(FOCUS, self.focus):
            dy, df = y[:, j] - normal[:, j], f[:, j] - normal[:, j]
            out[f"{a}-{b}_change_corr"] = float(np.corrcoef(dy, df)[0, 1])
            out[f"{a}-{b}_amplitude"] = float(np.std(df) / np.std(dy))
        return out


def summary_line(name, m):
    return (f"  {name:<34} R² {m['r2']:.3f} chg-corr {m['change_corr']:.3f} esc {m['esc_auc']:.3f} "
            f"amp {m['amplitude']:.2f} | crisis-AUC gold {m['auc_gold-crisis']:.3f} "
            f"conflict {m['auc_conflict-surge']:.3f} | ISR-PSE corr {m['ISR-PSE_change_corr']:.2f} "
            f"amp {m['ISR-PSE_amplitude']:.2f}, PSE-ISR {m['PSE-ISR_change_corr']:.2f} "
            f"amp {m['PSE-ISR_amplitude']:.2f}")


def make_context():
    p = load_panel(MENA)
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START,
                                GOLDSTEIN_HORIZON)
    ctx = Context(p, tr, va, te)
    # the forward-built targets need 13 future weeks inside the weekly panel (ends 2026-08-24), so the
    # last test weeks of the stored release target drop out; every model is scored on the same weeks
    ok = np.all([np.isfinite(d["y"][:, te]).all(0) for d in ctx.T.values()], axis=0)
    ctx.te = te[ok]
    print(f"test weeks with every target defined: {ok.sum()} of {len(te)} "
          f"({p['weeks'][ctx.te[0]].date()}..{p['weeks'][ctx.te[-1]].date()})", flush=True)
    return ctx


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ctx = make_context()
    p, tr, va, te = ctx.p, ctx.tr, ctx.va, ctx.te
    D = len(p["a1"])
    F44, _ = build_features(p)
    c = ctx.T["mean-conflict"]["series"]
    Fc = np.concatenate([F44, np.stack([c, trailing_mean(c, 13), trailing_mean(c, 52)], 2)], 2)
    results = {}
    for t in tv.TARGETS:
        d = ctx.T[t]
        print(f"{t}: escalation threshold {ctx.esc[t]:.2f}", flush=True)
        res = results[t] = {}
        res["naive: normal"] = ctx.evaluate(t, d["normal"][:, te].T)
        if t.startswith("peak"):
            past = tv.peak_past(d["series"], not d["worse_up"])
            res["naive: last 13 weeks' peak"] = ctx.evaluate(t, past[:, te].T)
        F = Fc if d["worse_up"] else F44
        ok = lambda w: np.isfinite(rows(d["y"], w))
        Xtr, ytr = rows(F, tr)[ok(tr)], rows(d["y"], tr)[ok(tr)]
        Xva, yva = rows(F, va)[ok(va)], rows(d["y"], va)[ok(va)]
        Xte = rows(F, te)
        lin = LinearRegression().fit(np.nan_to_num(Xtr), ytr)  # NaN = no events yet -> 0 for OLS only
        res["linear regression, hand-built*"] = ctx.evaluate(t, lin.predict(np.nan_to_num(Xte)).reshape(len(te), D))
        gbm, n = fit_gbm(HistGradientBoostingRegressor, Xtr, ytr, Xva, yva, mean_squared_error)
        res["gradient boosting, hand-built*"] = ctx.evaluate(t, gbm.predict(Xte).reshape(len(te), D))
        res["_gbm_trees"] = n
        for name, m in res.items():
            if not name.startswith("_"):
                print(summary_line(name, m), flush=True)
    os.makedirs("logs", exist_ok=True)
    json.dump(dict(esc_thresholds=ctx.esc, results=results), open(OUT, "w"), indent=1)
    print(f"crisis base rates (test): gold {results['mean-gold']['naive: normal']['rate_gold-crisis']:.1%}, "
          f"conflict {results['mean-gold']['naive: normal']['rate_conflict-surge']:.1%}; saved {OUT}")


if __name__ == "__main__":
    main()
