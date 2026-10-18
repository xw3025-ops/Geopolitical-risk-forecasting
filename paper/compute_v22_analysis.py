"""
Analysis of the release v2.2 Goldstein model for the paper (the v2.1 analyses of
compute_v21_stats.py, redone on the current release): variance decomposition and change
skill, the four example dyads, and the attention of the cross-system blocks.
Writes paper/v22_analysis.json. Inference only (no training).

    KERAS_BACKEND=torch python paper/compute_v22_analysis.py
"""

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from honest_common import (GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START,  # noqa: E402
                           goldstein_numeric_inputs, load_panel, split_by_dates, trailing_mean, windows_at)

OUT = "paper/v22_analysis.json"
EXAMPLES = ["ISR-PSE", "ISR-IRN", "SAU-YEM", "TUR-SYR"]


def main():
    st = {}
    p = load_panel()
    tr, va, te = split_by_dates(p["weeks"], p["target13"][0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START, GOLDSTEIN_HORIZON)
    n52 = trailing_mean(p["gold_zero"], 52)[:, te].T
    y = p["target13"][:, te].T
    v22 = np.load("artifacts/goldstein_v2.2_test_predictions.npz", allow_pickle=True)
    assert np.array_equal(v22["week_idx"], te)

    # ---- variance decomposition and change skill
    f = v22["pred"]
    tot = ((y - y.mean()) ** 2).sum()
    within = ((y - y.mean(0)) ** 2).sum()
    per = 1 - ((y - f) ** 2).mean(0) / ((y - n52) ** 2).mean(0)
    idx = {(a, b): i for i, (a, b) in enumerate(zip(p["a1"], p["a2"]))}
    col = lambda k: idx[tuple(k.split("-"))]
    st["variance"] = dict(between_share=float(1 - within / tot),
                          mse_reduction=float(1 - ((y - f) ** 2).mean() / ((y - n52) ** 2).mean()),
                          change_corr=float(np.corrcoef((y - n52).ravel(), (f - n52).ravel())[0, 1]),
                          share_dyads_better=float((per > 0).mean()),
                          example_reduction={k: float(per[col(k)]) for k in EXAMPLES})
    st["examples"] = {k: dict(weeks=[str(p["weeks"][t].date()) for t in te], actual=y[:, col(k)].tolist(),
                              forecast=f[:, col(k)].tolist(), trailing52=n52[:, col(k)].tolist()) for k in EXAMPLES}

    # ---- attention of the cross-system blocks (seed 42): where do MENA forecasts look?
    import keras
    from experiment_scaling import external_pool
    from model_architectures import build_goldstein_model
    art = joblib.load("artifacts/goldstein_v2.2_preprocessing.joblib")
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
    m.load_weights("model_release_v2.2_goldstein_s42.weights.h5")
    st["params"] = int(m.count_params())
    sc = art["scaler"]
    n_num = X.shape[-1]
    received, ext_share, max_lift = {}, {}, {}
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
            max_lift[key] = float(rec[h].max())
    selective = sorted(max_lift, key=lambda k: -max_lift[k])[:2]  # the two most selective heads
    st["attention"] = dict(received=received, external_share=ext_share, uniform_external_share=N / (400 + N),
                           max_lift=max_lift, selective_heads=selective)
    json.dump(st, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in st.items() if k not in ("examples", "attention")}, indent=1))
    print("external attention share:", {k: round(v, 3) for k, v in ext_share.items()}, "uniform", round(N / (400 + N), 3))
    for k in sorted(max_lift, key=lambda k: -max_lift[k]):
        print(k, f"max {max_lift[k]:.1f}", [f"{d['dyad']}{'*' if d['external'] else ''} {d['lift']:.1f}" for d in received[k][:10]])


if __name__ == "__main__":
    main()
