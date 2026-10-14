"""
Numbers for the v2.1 version of the paper section: benchmark rows, variance
decomposition, example dyads, attention analysis of the cross-system model,
and the table of data/model experiments. Writes paper/v21_stats.json.

    KERAS_BACKEND=torch python paper/compute_v21_stats.py
"""

import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_ladder import forecast_metrics  # noqa: E402
from honest_common import (GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA,  # noqa: E402
                           goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean, windows_at)

OUT = "paper/v21_stats.json"


def main():
    st = {}
    p = load_panel()
    D = 400
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START, GOLDSTEIN_HORIZON)
    n52 = trailing_mean(p["gold_zero"], 52)[:, te].T
    y = p["target13"][:, te].T

    # ---- benchmark rows (same rows for all)
    v21 = np.load("artifacts/goldstein_v2.1_test_predictions.npz", allow_pickle=True)
    v20 = np.load("artifacts/goldstein_v2.0_test_predictions.npz", allow_pickle=True)
    assert np.array_equal(v21["week_idx"], te) and np.array_equal(v20["week_idx"], te)
    st["benchmark"] = {"v2.1": forecast_metrics(y, v21["pred"], n52), "v2.0": forecast_metrics(y, v20["pred"], n52)}
    st["v21_seed_r2"] = [float(r2_score(y.ravel(), s.ravel())) for s in v21["pred_per_seed"]]
    va21 = np.load("artifacts/goldstein_v2.1_val_predictions.npz")
    st["v21_val_mse"] = float(((va21["pred"] - va21["y_true"]) ** 2).mean())

    # ---- variance decomposition and change skill for v2.1
    f = v21["pred"]
    tot = ((y - y.mean()) ** 2).sum()
    within = ((y - y.mean(0)) ** 2).sum()
    per = 1 - ((y - f) ** 2).mean(0) / ((y - n52) ** 2).mean(0)
    idx = {(a, b): i for i, (a, b) in enumerate(zip(p["a1"], p["a2"]))}
    st["variance"] = dict(between_share=float(1 - within / tot),
                          mse_reduction=float(1 - ((y - f) ** 2).mean() / ((y - n52) ** 2).mean()),
                          change_corr=float(np.corrcoef((y - n52).ravel(), (f - n52).ravel())[0, 1]),
                          share_dyads_better=float((per > 0).mean()),
                          example_reduction={k: float(per[idx[tuple(k.split("-"))]])
                                             for k in ["ISR-PSE", "ISR-IRN", "SAU-YEM", "TUR-SYR"]})
    st["examples"] = {k: dict(weeks=[str(p["weeks"][t].date()) for t in te], actual=y[:, idx[tuple(k.split('-'))]].tolist(),
                              forecast=f[:, idx[tuple(k.split('-'))]].tolist(),
                              trailing52=n52[:, idx[tuple(k.split('-'))]].tolist())
                      for k in ["ISR-PSE", "ISR-IRN", "SAU-YEM", "TUR-SYR"]}

    # ---- attention of the cross-system model (seed 42): where do MENA forecasts look?
    import keras
    from experiment_scaling import external_pool
    from model_architectures import build_goldstein_model
    art = joblib.load("artifacts/goldstein_v2.1_preprocessing.joblib")
    ext = external_pool(p["weeks"], art["config"]["min_active_external"])
    assert list(zip(ext["a1"], ext["a2"])) == [tuple(x) for x in art["external_dyads"]]
    N = len(ext["a1"])
    X = np.concatenate([goldstein_numeric_inputs(p, long_features=True), goldstein_numeric_inputs(ext, long_features=True)])
    base = np.concatenate([trailing_mean(p["gold_zero"], 52), trailing_mean(ext["gold_zero"], 52)])
    enc = art["country_encoder"]
    a1 = enc.transform(np.concatenate([p["a1"], ext["a1"]])).astype("int32")
    a2 = enc.transform(np.concatenate([p["a2"], ext["a2"]])).astype("int32")
    names = [f"{a}-{b}" for a, b in zip(np.concatenate([p["a1"], ext["a1"]]), np.concatenate([p["a2"], ext["a2"]]))]
    m = build_goldstein_model(n_dyads=400 + N, lookback_weeks=13, n_numeric=X.shape[-1], n_countries=len(enc.classes_),
                              dropout=0.2, residual_baseline=True, crossdyad_blocks=2)
    m.load_weights("model_release_v2.1_goldstein_s42.weights.h5")
    st["params"] = int(m.count_params())
    sc = art["scaler"]
    n_num = X.shape[-1]
    received = {}
    ext_share = {}
    for blk, src in [("crossdyad_block0", "dyad_token_projection"), ("crossdyad_block1", "crossdyad_block0_ln2")]:
        sub = keras.Model(m.inputs, m.get_layer(src).output)
        acc = None
        for s in range(0, len(te), 4):
            w = te[s:s + 4]
            xin = {"numeric_seq": sc.transform(windows_at(X, w, 13).reshape(-1, n_num)).reshape(len(w), 400 + N, 13, n_num).astype("float32"),
                   "actor1_idx": np.tile(a1, (len(w), 1)), "actor2_idx": np.tile(a2, (len(w), 1)),
                   "baseline": base[:, w].T.astype("float32")}
            tok = sub.predict(xin, verbose=0)
            _, sco = m.get_layer(f"{blk}_mha")(tok, tok, return_attention_scores=True, training=False)
            sco = keras.ops.convert_to_numpy(sco)[:, :, :400, :]  # queries = MENA dyads only
            part = sco.sum(axis=(0, 2))  # (heads, keys)
            acc = part if acc is None else acc + part
        n_q = len(te) * 400
        rec = acc / n_q * (400 + N)  # x uniform
        for h in range(rec.shape[0]):
            key = f"{blk}_h{h}"
            order = np.argsort(-rec[h])[:12]
            received[key] = [dict(dyad=names[j], lift=float(rec[h, j]), external=bool(j >= 400)) for j in order]
            ext_share[key] = float((acc[h, 400:].sum()) / n_q)
    st["attention"] = dict(received=received, external_share=ext_share, uniform_external_share=N / (400 + N))

    # ---- table of experiments (single seed 42 unless noted; test = Mar 2024 - Jun 2026)
    rel42 = np.load("artifacts/experiments/lb13_res52_long_do0.2_lr0.0003_plat_s42.npz")
    rows = [("Release v2.0 recipe (reference, seed 42)", rel42["te_pred"], "lb13_res52_long_do0.2_lr0.0003_plat_s42")]
    for name, tag in [("+ all 69 GDELT fields (tone, media volume, shares)", "lb13_res52_long_rich_do0.2_lr0.0003_plat_s42"),
                      ("+ ACLED country-level events (2015-)", "lb13_res52_long_acled_do0.2_lr0.0003_plat_s42"),
                      ("+ Guardian news text (pretrained MiniLM)", "lb13_res52_long_news_do0.2_lr0.0003_plat_s42")]:
        rows.append((name, np.load(f"artifacts/experiments/{tag}.npz")["te_pred"], tag))
    exps = {e["tag"]: e for e in (json.loads(l) for l in open("logs/goldstein_experiments.jsonl"))}
    table = []
    for name, pred, tag in rows:
        mtr = forecast_metrics(y, pred, n52)
        table.append(dict(name=name, val_mse=exps[tag]["va_mse"], **{k: mtr[k] for k in ("r2", "dir_acc", "change_corr", "esc_auc")}))
    for name, path in [("+ GDELT 1.0 history 2008-2015 (3 seeds)", "logs/history_2008-01-07_cd2.json"),
                       ("+ 1,200 MENA-world dyads, no attention (3 seeds)", "logs/moredata_min0.5_d64.json"),
                       ("+ same, wider model d=128 (3 seeds)", "logs/moredata_min0.5_d128.json")]:
        if os.path.exists(path):
            j = json.load(open(path))
            vs = [v["val_mse"] for k, v in j.items() if k.startswith("seed")]
            e = j["ensemble"]
            table.append(dict(name=name, val_mse=float(np.mean(vs)), r2=e["r2"], dir_acc=e["dir_acc"],
                              change_corr=e["change_corr"], esc_auc=e["esc_auc"]))
    c = exps.get("lb13_res52_long_Tconflict_do0.2_lr0.0003_plat_s42")
    st["conflict_target"] = c["test_metrics"] if c else None
    st["experiments"] = table
    json.dump(st, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in st.items() if k not in ("examples", "attention")}, indent=1))
    print("external attention share (MENA queries):", {k: round(v, 3) for k, v in ext_share.items()},
          "uniform", round(N / (400 + N), 3))
    for k in ext_share:
        print(k, [f"{d['dyad']}{'*' if d['external'] else ''} {d['lift']:.1f}" for d in received[k][:8]])


if __name__ == "__main__":
    main()
