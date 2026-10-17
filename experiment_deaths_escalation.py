"""
Experiment: train the country-token transformer DIRECTLY on the escalation
label (experiment_deaths.py trained on log deaths and ranked escalations
poorly).

Label (Xingyi's country_escalate_h13): UCDP deaths over the next 13 weeks
>= max(25, 2 x deaths over the last 13 weeks). Classification with a masked
binary cross-entropy on logits.

Same data, tokens (59 countries), inputs, splits and embargo as
experiment_deaths.py. The network predicts a correction to a HISTORY PRIOR:
the logit of a logistic regression of the label on (52-week normal - last
quarter) in log deaths, fitted on training weeks only - the classification
analogue of the residual on the history average.

Benchmarks on the same rows: the history prior alone; logistic regression and
gradient boosting (classifier) on the hand-built features of
experiment_deaths.py. Metrics: AUC, average precision, weekly top-5 hit rate,
for the 20 MENA countries and all 59.

    KERAS_BACKEND=torch python experiment_deaths_escalation.py [--seeds 42 1 2 3 4]
-> logs/deaths_escalation.json
"""

import argparse
import json
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from benchmark_honest import fit_gbm
from experiment_deaths import H, LOOKBACK, country_arrays, hand_features, targets
from honest_common import GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA, rows, split_by_dates, windows_at

warnings.filterwarnings("ignore")
LOG = "logs/deaths_escalation.json"


def metrics(lab, score, sel):
    L, S = lab[:, sel], score[:, sel]
    ok = np.isfinite(L) & np.isfinite(S)
    hits = []
    for t in range(L.shape[0]):
        m = ok[t]
        if m.sum() >= 5:
            top = np.argsort(-np.where(m, S[t], -np.inf))[:5]
            hits.append(L[t, top].mean())
    return dict(auc=float(roc_auc_score(L[ok], S[ok])), ap=float(average_precision_score(L[ok], S[ok])),
                top5_hit=float(np.mean(hits)), rate=float(L[ok].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    args = ap.parse_args()
    t0 = time.time()
    weeks, countries, X, deaths, covered = country_arrays()
    y, normal, prev, esc = targets(deaths, covered)
    C = len(countries)
    last_valid = np.where(np.isfinite(esc).all(0))[0].max()
    tr, va, te = split_by_dates(weeks, np.where(np.arange(len(weeks)) <= last_valid, 0.0, np.nan),
                                GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START, H, lookback=LOOKBACK)
    mena = np.array([c in MENA for c in countries])
    E = {k: esc[:, w].T for k, w in (("tr", tr), ("va", va), ("te", te))}
    print(f"{C} countries; train {len(tr)}, val {len(va)}, test {len(te)} weeks; escalation rate train "
          f"{np.nanmean(E['tr']):.3f}, test {np.nanmean(E['te']):.3f}", flush=True)
    results = {}

    def report(name, s_te, s_va=None):
        r = {"test_mena": metrics(E["te"], s_te, mena), "test_all": metrics(E["te"], s_te, np.ones(C, bool))}
        if s_va is not None:
            r["val_auc_all"] = metrics(E["va"], s_va, np.ones(C, bool))["auc"]
        results[name] = r
        m, a = r["test_mena"], r["test_all"]
        print(f"  {name:<46} MENA: AUC {m['auc']:.3f} AP {m['ap']:.3f} top5 {m['top5_hit']:.0%} | "
              f"all 59: AUC {a['auc']:.3f} AP {a['ap']:.3f} top5 {a['top5_hit']:.0%}"
              + (f" | val AUC {r['val_auc_all']:.3f}" if s_va is not None else ""), flush=True)

    # history prior: logistic on (normal - last quarter), training weeks only
    gap = (normal - prev)
    ok_tr = np.isfinite(rows(esc, tr)) & np.isfinite(rows(gap, tr))
    prior = LogisticRegression().fit(rows(gap, tr)[ok_tr, None], rows(esc, tr)[ok_tr])
    prior_logit = (prior.coef_[0, 0] * np.nan_to_num(gap) + prior.intercept_[0]).astype("float32")
    print("Benchmarks")
    report("history prior: 52-week normal vs last quarter", prior_logit[:, te].T, prior_logit[:, va].T)

    F = np.concatenate([hand_features(X), gap[..., None]], 2)
    ok_tr = np.isfinite(rows(esc, tr))
    ok_va = np.isfinite(rows(esc, va))
    sc = StandardScaler().fit(np.nan_to_num(rows(F, tr)[ok_tr]))
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(np.nan_to_num(rows(F, tr)[ok_tr])), rows(esc, tr)[ok_tr])
    p = lambda w: lr.decision_function(sc.transform(np.nan_to_num(rows(F, w)))).reshape(len(w), C)
    report("logistic regression, hand-built features", p(te), p(va))
    gbm, n = fit_gbm(HistGradientBoostingClassifier, rows(F, tr)[ok_tr], rows(esc, tr)[ok_tr].astype(int),
                     rows(F, va)[ok_va], rows(esc, va)[ok_va].astype(int), log_loss)
    g = lambda w: gbm.predict_proba(rows(F, w))[:, 1].reshape(len(w), C)
    report(f"gradient boosting classifier, hand-built ({n} trees)", g(te), g(va))

    # ---------------------------------------------------------------- transformers
    import keras
    from keras import ops
    from sklearn.preprocessing import LabelEncoder
    from model_architectures import build_goldstein_model

    n_num = X.shape[-1]
    xs = StandardScaler().fit(windows_at(X, tr, LOOKBACK).reshape(-1, n_num))
    scl = lambda A: np.nan_to_num(xs.transform(A.reshape(-1, n_num)).reshape(A.shape)).astype("float32")
    idx = LabelEncoder().fit(countries).transform(countries).astype("int32")

    def inputs(w, use_prior):
        return {"numeric_seq": scl(windows_at(X, w, LOOKBACK)), "actor1_idx": np.tile(idx, (len(w), 1)),
                "actor2_idx": np.tile(idx, (len(w), 1)),
                "baseline": prior_logit[:, w].T if use_prior else np.zeros((len(w), C), "float32")}

    def masked_bce(y_true, logit):
        m = ops.isfinite(y_true)
        yt = ops.where(m, y_true, 0.0)
        # numerically stable BCE with logits
        loss = ops.maximum(logit, 0.0) - logit * yt + ops.log1p(ops.exp(-ops.abs(logit)))
        return ops.sum(ops.where(m, loss, 0.0)) / ops.maximum(ops.sum(ops.cast(m, "float32")), 1.0)

    variants = [("transformer: attention + embeddings, history prior", 2, True, True),
                ("transformer: attention + embeddings, no prior", 2, True, False),
                ("  ablation: no cross-country attention (prior)", 0, True, True),
                ("  ablation: no country embeddings (prior)", 2, False, True)]
    print("Transformers")
    for name, blocks, emb, use_prior in variants:
        seeds = args.seeds if (blocks, emb, use_prior) == (2, True, True) else args.seeds[:3]
        x_tr, x_va, x_te = inputs(tr, use_prior), inputs(va, use_prior), inputs(te, use_prior)
        pv, pt = [], []
        for seed in seeds:
            keras.utils.set_random_seed(seed)
            m = build_goldstein_model(n_dyads=C, lookback_weeks=LOOKBACK, n_numeric=n_num, n_countries=C,
                                      dropout=0.2, residual_baseline=True, crossdyad_blocks=blocks,
                                      country_embedding=emb)
            m.compile(optimizer=keras.optimizers.Adam(3e-4), loss=masked_bce)
            m.fit(x_tr, E["tr"].astype("float32"), validation_data=(x_va, E["va"].astype("float32")), epochs=200,
                  batch_size=8, shuffle=True, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)])
            pv.append(m.predict(x_va, verbose=0))
            pt.append(m.predict(x_te, verbose=0))
        report(f"{name} ({len(seeds)} seeds)", np.mean(pt, 0), np.mean(pv, 0))

    json.dump(dict(countries=countries, results=results), open(LOG, "w"), indent=1)
    print(f"saved {LOG} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
