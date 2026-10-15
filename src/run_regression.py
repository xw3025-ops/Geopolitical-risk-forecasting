"""Fixed-effects regressions of China's trade growth on conflict (country x month panel).

  A. own conflict:      y_it = b * c_{i,t-k} + FE_i + FE_t
  B. Gulf shock x exposure:
                        y_it = b * gulf_{t-k} * exposure_i + FE_i + FE_t
     y = imp_dlog12 / exp_dlog12; c = z-score of t2t4_share; gulf_t = z-score (over months) of
     the mean t2t4_share of the 8 Hormuz littoral states; exposure = Hormuz-exposed share of
     China's 2024 imports from the partner (0-1, low/high scenario).
  Each lag k is run on its own; lag "0-1-2-3" puts all four lags in one model and reports
  their sum (the cumulative effect).

Inference. Shocks are common to all countries in a month, so there are only ~100 independent
months, and exposure > 0 for only 5 partners. Reported:
  se_2way  Cameron-Gelbach-Miller two-way cluster (country, month). When V is not positive
           semi-definite its eigenvalues are clipped at 0 (CGM 2011); flagged in `note`.
  se_month cluster by month;  se_dk  Driscoll-Kraay (Bartlett kernel, 3 lags).
  p_perm   (B only) randomisation inference: exposure re-assigned across countries --perm times.
           With 5 exposed partners this is the most honest p-value of the four.

Output: data/processed/regression_results.csv
Run:  python src/run_regression.py [--start 2018-01] [--lags 0 1 2 3] [--perm 999]
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED

GULF8 = ["IRN", "IRQ", "KWT", "SAU", "ARE", "BHR", "QAT", "OMN"]
WAR = ["SYR", "SDN", "YEM", "LBY"]
DK_LAGS = 3


# ------------------------------------------------------------ estimation

class TwoWayFE:
    """Country + month fixed effects on one (possibly unbalanced) sample. resid(A) partials the
    dummies out of every column of A (Frisch-Waugh-Lovell), so b and its sandwich covariance
    equal those of the full dummy regression."""

    def __init__(self, d):
        D = np.hstack([pd.get_dummies(d.iso3).to_numpy(float),
                       pd.get_dummies(d.period).to_numpy(float)[:, 1:]])
        self.D, self.Dpinv = D, np.linalg.pinv(D)
        self.k_fe = D.shape[1]

    def resid(self, A):
        return A - self.D @ (self.Dpinv @ A)


def fit(d, y, xs):
    fe = TwoWayFE(d)
    X, Y = fe.resid(d[xs].to_numpy(float)), fe.resid(d[y].to_numpy(float))
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    sc = X * (Y - X @ b)[:, None]  # per-observation scores
    n, k = len(d), len(xs) + fe.k_fe
    sandwich = lambda meat: XtX_inv @ meat @ XtX_inv

    def clustered(groups):
        s = pd.DataFrame(sc).groupby(np.asarray(groups)).sum().to_numpy()
        g = len(s)
        return g / (g - 1) * (n - 1) / (n - k) * sandwich(s.T @ s), g

    Vc, gc = clustered(d.iso3)
    Vt, gt = clustered(d.period)
    V2 = Vc + Vt - n / (n - k) * sandwich(sc.T @ sc)  # intersection = single observations
    note = ""
    w, Q = np.linalg.eigh(V2)
    if (w < 0).any():
        clipped = Q @ np.diag(np.clip(w, 0, None)) @ Q.T
        if (np.diag(clipped) > 0).all():
            V2, note = clipped, "2way V not PSD: eigenvalues clipped"
        else:  # clipping leaves a zero variance: use the larger one-way covariance (conservative)
            V2 = Vc if np.trace(Vc) >= np.trace(Vt) else Vt
            note = "2way V not PSD: larger one-way V used"
    ht = pd.DataFrame(sc).groupby(d.period.to_numpy()).sum().sort_index().to_numpy()
    S = ht.T @ ht
    for lag in range(1, DK_LAGS + 1):
        G = ht[lag:].T @ ht[:-lag]
        S += (1 - lag / (DK_LAGS + 1)) * (G + G.T)
    Vdk = n / (n - k) * sandwich(S)
    return dict(b=b, V={"2way": V2, "month": Vt, "country": Vc, "dk": Vdk},
                dof={"2way": min(gc, gt) - 1, "month": gt - 1, "country": gc - 1, "dk": gt - 1},
                n=n, n_country=gc, n_month=gt, note=note, fe=fe, Y=Y)


def summarise(res, contrast):
    b = float(contrast @ res["b"])
    out = {"beta": b, "n": res["n"], "n_country": res["n_country"], "n_month": res["n_month"],
           "note": res["note"]}
    for tag, V in res["V"].items():
        se = float(np.sqrt(max(contrast @ V @ contrast, 0)))
        out[f"se_{tag}"] = se
        out[f"p_{tag}"] = 2 * stats.t.sf(abs(b) / se, res["dof"][tag]) if se > 0 else np.nan
    return out


def permutation_p(d, res, shock_cols, contrast, n_perm, seed=0):
    """Randomisation inference: shuffle the exposure values across countries, keep everything
    else, and count how often |sum of interaction coefficients| is at least the observed one."""
    rng = np.random.default_rng(seed)
    countries = d.iso3.unique()
    expo = d.groupby("iso3").exposure.first().reindex(countries).to_numpy()
    idx = pd.Index(countries).get_indexer(d.iso3)
    shocks = d[shock_cols].to_numpy(float)
    observed = abs(contrast @ res["b"])
    hits = 0
    for _ in range(n_perm):
        X = res["fe"].resid(shocks * rng.permutation(expo)[idx][:, None])
        b = np.linalg.lstsq(X, res["Y"], rcond=None)[0]
        hits += abs(contrast @ b) >= observed - 1e-12
    return (hits + 1) / (n_perm + 1)


# ------------------------------------------------------------ data

def load(max_lag):
    p = pd.read_csv(PROCESSED / "panel_country_month.csv")
    ex = pd.read_csv(PROCESSED / "exposure_table.csv").set_index("iso3")
    p = p.sort_values(["iso3", "period"]).reset_index(drop=True)
    z = lambda s: (s - s.mean()) / s.std()
    p["c_z"] = z(p.t2t4_share)
    for shock, members, col in (("gulf", GULF8, "t2t4_share"), ("gulf_conf", GULF8, "conflict_share"),
                                ("gulf_wide", GULF8 + ["ISR", "YEM"], "t2t4_share")):
        p[shock] = p.period.map(z(p[p.iso3.isin(members)].groupby("period")[col].mean()))
    for k in range(max_lag + 1):
        for c in ("c_z", "gulf", "gulf_conf", "gulf_wide"):
            p[f"{c}_l{k}"] = p.groupby("iso3")[c].shift(k)

    def exposure(scen="high", overrides=None, no_dep=False):
        dc, dg = ex[f"dep_crude_{scen}"].copy(), ex[f"dep_gas_{scen}"].copy()
        for iso, v in (overrides or {}).items():
            dc[iso] = dg[iso] = v
        num = ex.energy_bn if no_dep else ex.crude_bn * dc + ex.gas_bn * dg
        return (num / ex.imp_bn_2024).replace([np.inf, -np.inf], np.nan).fillna(0)

    variants = {"high": exposure("high"), "low": exposure("low"),
                "oman_1": exposure(overrides={"OMN": 1.0}), "saudi_0.8": exposure(overrides={"SAU": 0.8}),
                "energy_share_no_dep": exposure(no_dep=True)}
    return p, variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--perm", type=int, default=999)
    args = ap.parse_args()
    lags = args.lags
    p, variants = load(max(lags))
    rows = []

    def run(spec, outcome, shock, exposure, lag_set, start=args.start, end=None, drop=(), only=None, label="all"):
        d = p[(p.period >= start) & (p.period <= (end or p.period.max())) & ~p.iso3.isin(drop)]
        if only is not None:
            d = d[d.iso3.isin(only)]
        d = d.copy()
        shock_cols = [f"{shock}_l{k}" for k in lag_set]
        if spec == "A":
            xs = shock_cols
        else:
            d["exposure"] = d.iso3.map(variants[exposure])
            xs = [f"x_l{k}" for k in lag_set]
            for s, x in zip(shock_cols, xs):
                d[x] = d[s] * d.exposure
        d = d.dropna(subset=[outcome] + xs).reset_index(drop=True)
        res = fit(d, outcome, xs)
        contrast = np.ones(len(xs))
        out = {"spec": spec, "outcome": outcome, "shock": shock, "exposure": exposure if spec == "B" else "",
               "lag": "-".join(map(str, lag_set)), "start": start, "end": end or p.period.max(), "sample": label}
        out.update(summarise(res, contrast))
        if spec == "B" and args.perm:
            out["p_perm"] = permutation_p(d, res, shock_cols, contrast, args.perm)
        rows.append(out)

    lag_sets = [[k] for k in lags] + ([lags] if len(lags) > 1 else [])
    for y in ("imp_dlog12", "exp_dlog12"):
        for ls in lag_sets:
            run("A", y, "c_z", None, ls)
        for e in ("high", "low"):
            for ls in lag_sets:
                run("B", y, "gulf", e, ls)

    material = sorted(p[p.period.str.startswith("2024")].groupby("iso3").imp_bn.sum().loc[lambda s: s >= 0.1].index)
    robustness = [
        dict(start="2023-01", label="start 2023"),
        dict(end="2025-12", label="pre-2026 (no Hormuz shock)"),
        dict(drop=["USA"], label="excl USA"),
        dict(drop=WAR, label="excl SYR SDN YEM LBY"),
        dict(drop=["IRN", "ARE"], label="excl IRN ARE (sanctions / re-export)"),
        dict(only=material, label="import flows >= $100m in 2024"),
        dict(shock="gulf_conf", label="shock = conflict_share"),
        dict(shock="gulf_wide", label="shock incl ISR YEM"),
        dict(exposure="oman_1", label="Oman dependence = 1"),
        dict(exposure="saudi_0.8", label="Saudi dependence = 0.8"),
        dict(exposure="energy_share_no_dep", label="energy share, no route parameters"),
    ]
    for r in robustness:
        r = dict(r)
        shock, exposure = r.pop("shock", "gulf"), r.pop("exposure", "high")
        for ls in lag_sets:
            run("B", "imp_dlog12", shock, exposure, ls, **r)

    res = pd.DataFrame(rows)
    out = PROCESSED / "regression_results.csv"
    res.to_csv(out, index=False)
    show = ["spec", "outcome", "shock", "exposure", "lag", "sample", "n", "beta", "se_2way", "p_2way",
            "p_month", "p_dk", "p_perm", "note"]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 38):
        print(res[show].round(4).to_string(index=False))
    print(f"\n{len(res)} specifications -> {out}")


if __name__ == "__main__":
    main()
