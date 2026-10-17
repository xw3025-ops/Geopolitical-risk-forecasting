"""
Experiment: forecast DEATHS (UCDP organised violence) instead of Goldstein.

Unit: country-week (UCDP is recorded per country). Each week is one sequence
of 59 COUNTRY tokens (every country in data/ucdp_country_week.csv: the 20 MENA
states plus China, the USA and Belt-and-Road hosts). The release architecture
is reused unchanged: a temporal transformer over each country's last 13 weeks,
country embeddings, and two cross-country attention blocks (the "system"),
predicting a residual on the country's history average.

Target  log(1 + UCDP deaths over the next 13 weeks); weeks whose 13-week
        window runs past UCDP coverage (2026-08-24) are not graded.
Normal  log(1 + 13 x the country's trailing 52-week mean weekly deaths).
Inputs per country-week (all known at week t):
  UCDP:  log1p deaths, log1p events, covered flag
  GDELT, summed over every dyad the country is part of (MENA-MENA, MENA-world
         and world dyads in our panels): log1p counts of the 20 CAMEO roots,
         log1p total events, conflict intensity log1p(sum |negative goldstein|);
         plus the domestic dyad's mean goldstein
  history: trailing 13/26/52-week means of log1p deaths, 52-week mean of
         conflict intensity
Escalation label (as in Xingyi's country_escalate_h13, train_risk_gbm.py):
  deaths over the next 13 weeks >= max(25, 2 x deaths over the last 13).

Benchmarks on the same rows: last quarter's deaths, the 52-week normal, and
linear regression / gradient boosting on hand-built history features.
Splits: the release dates (val 2023-02-06, test 2024-03-18) with a 13-week
embargo. Graded on all 59 countries in training; reported for the 20 MENA
countries and for all 59.

    KERAS_BACKEND=torch python experiment_deaths.py [--seeds 42 1 2 3 4]
-> logs/deaths.json
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from benchmark_honest import fit_gbm
from benchmark_targets import neg_intensity
from experiment_scaling import external_pool
from experiment_targets import load_world
from honest_common import (COUNT_COLS, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA, load_panel, rows,
                           split_by_dates, trailing_mean, windows_at)

warnings.filterwarnings("ignore", category=RuntimeWarning)
H, LOOKBACK = 13, 13
UCDP = "data/ucdp_country_week.csv"
LOG = "logs/deaths.json"


def country_arrays():
    """-> weeks, countries, X (C, T, F) inputs, deaths (C, T), covered (C, T)."""
    p = load_panel(MENA)
    weeks = p["weeks"]
    p["neg"] = neg_intensity(p).astype("float32")
    ext = external_pool(weeks, 0.0)
    world = load_world(weeks)
    u = pd.read_csv(UCDP, parse_dates=["week_start"])
    countries = sorted(u.iso3.unique())
    ci = {c: i for i, c in enumerate(countries)}
    C, T = len(countries), len(weeks)

    counts = np.zeros((C, T, 20), "float64")
    neg = np.zeros((C, T), "float64")
    dom_gold = np.zeros((C, T), "float32")
    for src in (p, ext, world):
        for d, (a, b) in enumerate(zip(src["a1"], src["a2"])):
            for c in {a, b}:  # a domestic dyad counts once
                if c in ci:
                    counts[ci[c]] += src["counts"][d]
                    neg[ci[c]] += src["neg"][d]
            if a == b and a in ci:  # domestic dyad (MENA panel or world panel)
                dom_gold[ci[a]] = src["gold_zero"][d]

    u = u.set_index(["iso3", "week_start"])
    grid = pd.MultiIndex.from_product([countries, weeks])
    g = u.reindex(grid)
    deaths = g["ucdp_deaths"].fillna(0).to_numpy("float64").reshape(C, T)
    events = g["ucdp_events"].fillna(0).to_numpy("float64").reshape(C, T)
    covered = g["ucdp_covered"].fillna(0).to_numpy("float64").reshape(C, T) > 0

    ld = np.log1p(deaths).astype("float32")
    ci_ = np.log1p(neg).astype("float32")
    X = np.concatenate([
        ld[..., None], np.log1p(events)[..., None], covered[..., None],
        np.log1p(counts), np.log1p(counts.sum(2))[..., None], ci_[..., None], dom_gold[..., None],
        np.stack([trailing_mean(ld, 13), trailing_mean(ld, 26), trailing_mean(ld, 52), trailing_mean(ci_, 52)], 2),
    ], 2).astype("float32")
    return weeks, countries, X, deaths, covered


def targets(deaths, covered):
    C, T = deaths.shape
    nxt = np.full((C, T), np.nan)
    prv = np.full((C, T), np.nan)
    cs = np.concatenate([np.zeros((C, 1)), np.cumsum(deaths, 1)], 1)
    ok = covered.all(0)  # weeks covered for every country
    last_cov = int(np.where(ok)[0].max())
    for t in range(T):
        if t + H <= last_cov:
            nxt[:, t] = cs[:, t + H + 1] - cs[:, t + 1]
        if t >= H - 1:
            prv[:, t] = cs[:, t + 1] - cs[:, t + 1 - H]
    normal = np.log1p(H * trailing_mean(deaths.astype("float32"), 52))
    y = np.log1p(nxt).astype("float32")
    esc = np.where(np.isfinite(nxt), (nxt >= np.maximum(25, 2 * prv)).astype(float), np.nan)
    return y, normal.astype("float32"), np.log1p(prv).astype("float32"), esc


def hand_features(X):
    """Hand-built history features for the classical benchmarks (C, T, F)."""
    ld, ev, tot, ci_ = X[..., 0], X[..., 1], X[..., 23], X[..., 24]  # see country_arrays() column order
    f = [ld, ev, tot, ci_, X[..., 25]]
    for s in (ld, ev, ci_, tot):
        for w in (4, 13, 26, 52):
            f.append(trailing_mean(s, w))
        f.append(trailing_mean(s, 4) - trailing_mean(s, 52))
    return np.stack(f, 2).astype("float32")


def metrics(y, f, prev, esc, sel):
    y, f, prev, esc = (a[:, sel] for a in (y, f, prev, esc))
    ok = np.isfinite(y) & np.isfinite(f)
    out = dict(r2=float(r2_score(y[ok], f[ok])), mae=float(mean_absolute_error(y[ok], f[ok])))
    e_ok = ok & np.isfinite(esc) & np.isfinite(prev)
    score = f - prev  # how much worse than last quarter the forecast is
    if 0 < esc[e_ok].mean() < 1:
        out["esc_auc"] = float(roc_auc_score(esc[e_ok], score[e_ok]))
        out["esc_rate"] = float(esc[e_ok].mean())
    # weekly top-5 alerts among the selected countries
    hits = []
    for t in range(y.shape[0]):
        m = e_ok[t]
        if m.sum() >= 5:
            top = np.argsort(-np.where(m, score[t], -np.inf))[:5]
            hits.append(esc[t, top].mean())
    out["top5_hit"] = float(np.mean(hits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    weeks, countries, X, deaths, covered = country_arrays()
    y, normal, prev, esc = targets(deaths, covered)
    C = len(countries)
    last_valid = np.where(np.isfinite(y).all(0))[0].max()
    tr, va, te = split_by_dates(weeks, np.where(np.arange(len(weeks)) <= last_valid, 0.0, np.nan),
                                GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START, H, lookback=LOOKBACK)
    mena = np.array([c in MENA for c in countries])
    print(f"{C} countries x {len(weeks)} weeks; train {len(tr)}, val {len(va)}, test {len(te)} weeks "
          f"({weeks[te[0]].date()}..{weeks[te[-1]].date()}); data {(time.time() - t0) / 60:.1f} min", flush=True)

    Y = {k: y[:, w].T for k, w in (("tr", tr), ("va", va), ("te", te))}
    NORM = {k: normal[:, w].T for k, w in (("tr", tr), ("va", va), ("te", te))}
    PREV = {k: prev[:, w].T for k, w in (("va", va), ("te", te))}
    ESC = {k: esc[:, w].T for k, w in (("va", va), ("te", te))}
    results = {}

    def report(name, f_te, f_va=None):
        r = {"test_mena": metrics(Y["te"], f_te, PREV["te"], ESC["te"], mena),
             "test_all": metrics(Y["te"], f_te, PREV["te"], ESC["te"], np.ones(C, bool))}
        if f_va is not None:
            r["val_mse_all"] = float(np.nanmean((f_va - Y["va"]) ** 2))
        results[name] = r
        m, a = r["test_mena"], r["test_all"]
        print(f"  {name:<44} MENA: R² {m['r2']:.3f} MAE {m['mae']:.3f} esc-AUC {m.get('esc_auc', float('nan')):.3f} "
              f"top5 {m['top5_hit']:.0%} | all 59: R² {a['r2']:.3f} esc-AUC {a.get('esc_auc', float('nan')):.3f} "
              f"top5 {a['top5_hit']:.0%}", flush=True)

    print("Naive and classical benchmarks")
    report("naive: last quarter's deaths", PREV["te"], prev[:, va].T)
    report("naive: 52-week history average", NORM["te"], NORM["va"])
    F = hand_features(X)
    ok_tr, ok_va = np.isfinite(rows(y, tr)), np.isfinite(rows(y, va))
    Xtr, Xva, Xte = rows(F, tr), rows(F, va), rows(F, te)
    lin = LinearRegression().fit(np.nan_to_num(Xtr[ok_tr]), rows(y, tr)[ok_tr])
    report("linear regression, hand-built features",
           lin.predict(np.nan_to_num(Xte)).reshape(len(te), C), lin.predict(np.nan_to_num(Xva)).reshape(len(va), C))
    from sklearn.metrics import mean_squared_error
    gbm, n = fit_gbm(HistGradientBoostingRegressor, Xtr[ok_tr], rows(y, tr)[ok_tr], Xva[ok_va], rows(y, va)[ok_va],
                     mean_squared_error)
    report("gradient boosting, hand-built features", gbm.predict(Xte).reshape(len(te), C),
           gbm.predict(Xva).reshape(len(va), C))

    # ---------------------------------------------------------------- transformers
    import keras
    from keras import ops
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from model_architectures import build_goldstein_model

    n_num = X.shape[-1]
    scaler = StandardScaler().fit(windows_at(X, tr, LOOKBACK).reshape(-1, n_num))
    sc = lambda A: np.nan_to_num(scaler.transform(A.reshape(-1, n_num)).reshape(A.shape)).astype("float32")
    idx = LabelEncoder().fit(countries).transform(countries).astype("int32")

    def inputs(w):
        return {"numeric_seq": sc(windows_at(X, w, LOOKBACK)), "actor1_idx": np.tile(idx, (len(w), 1)),
                "actor2_idx": np.tile(idx, (len(w), 1)), "baseline": np.nan_to_num(normal[:, w].T)}

    def masked_mse(y_true, y_pred):
        m = ops.isfinite(y_true)
        e = ops.where(m, ops.square(y_pred - ops.where(m, y_true, 0.0)), 0.0)
        return ops.sum(e) / ops.maximum(ops.sum(ops.cast(m, "float32")), 1.0)

    x_tr, x_va, x_te = inputs(tr), inputs(va), inputs(te)
    variants = [("transformer: cross-country attention + embeddings", 2, True),
                ("  ablation: no cross-country attention", 0, True),
                ("  ablation: no country embeddings", 2, False)]
    print(f"Transformers ({len(args.seeds)} seeds each)")
    for name, blocks, emb in variants:
        seeds = args.seeds if blocks == 2 and emb else args.seeds[:3]
        pv, pt = [], []
        for seed in seeds:
            keras.utils.set_random_seed(seed)
            m = build_goldstein_model(n_dyads=C, lookback_weeks=LOOKBACK, n_numeric=n_num, n_countries=C,
                                      dropout=0.2, residual_baseline=True, crossdyad_blocks=blocks,
                                      country_embedding=emb)
            m.compile(optimizer=keras.optimizers.Adam(3e-4), loss=masked_mse)
            m.fit(x_tr, Y["tr"], validation_data=(x_va, Y["va"]), epochs=args.epochs, batch_size=8, shuffle=True,
                  verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                                        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)])
            pv.append(m.predict(x_va, verbose=0))
            pt.append(m.predict(x_te, verbose=0))
        report(f"{name} ({len(seeds)} seeds)", np.mean(pt, 0), np.mean(pv, 0))
        if blocks == 2 and emb:
            np.savez("artifacts/experiments/deaths_transformer.npz", te_week_idx=te, countries=countries,
                     pred=np.mean(pt, 0), y=Y["te"], prev=PREV["te"], normal=NORM["te"])

    os.makedirs("logs", exist_ok=True)
    json.dump(dict(countries=countries, test_weeks=[str(weeks[te[0]].date()), str(weeks[te[-1]].date())],
                   results=results), open(LOG, "w"), indent=1)
    print(f"saved {LOG} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
