"""
Numbers behind the paper section (paper/transformer_section.docx). Reads the
release v2.0 models, their saved held-out predictions and the local panel;
writes paper/stats.json. No training.

    python paper/compute_paper_stats.py
"""

import importlib.util
import json
import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiment_goldstein as eg  # noqa: E402
import live_pipeline as lp  # noqa: E402
from honest_common import MENA, load_panel, trailing_mean  # noqa: E402
from model_architectures import StructuralBiasAttention, RELATION_NAMES  # noqa: E402

RELEASE = dict(lookback=13, residual=True, baseline_weeks=52, long_features=True, has_events=False,
               dropout=0.2, lr=3e-4, plateau=True)
OUT = "paper/stats.json"


def relation_matrix(a1, a2):
    spec = importlib.util.spec_from_file_location("rd", "release_v2.0_direction.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    return rd.build_relation_matrix(a1, a2)


def main():
    stats = {}
    p = load_panel()
    D, T = p["gold_zero"].shape
    mena_counts = p["counts"].sum(axis=2)
    stats["data"] = dict(dyads=D, weeks=T, first_week=str(p["weeks"][0].date()), last_week=str(p["weeks"][-1].date()),
                         events=int(mena_counts.sum()), zero_event_share=float((mena_counts == 0).mean()))

    # ---- goldstein test-set metrics for the simple baselines and the release
    z = np.load("artifacts/goldstein_v2.0_test_predictions.npz", allow_pickle=True)
    te, y, f = z["week_idx"], z["y_true"], z["pred"]
    rows = {
        "train mean": None, "per-dyad train mean": None,
        "persistence (this week)": p["gold_zero"][:, te].T,
        "trailing 13-week mean": trailing_mean(p["gold_zero"], 13)[:, te].T,
        "trailing 26-week mean": trailing_mean(p["gold_zero"], 26)[:, te].T,
        "trailing 52-week mean": trailing_mean(p["gold_zero"], 52)[:, te].T,
        "transformer v2.0 (5-seed ensemble)": f,
    }
    bench = {}
    for k, v in rows.items():
        if v is None:
            continue
        bench[k] = dict(r2=float(r2_score(y.ravel(), v.ravel())), mae=float(mean_absolute_error(y.ravel(), v.ravel())))
    stats["goldstein_benchmark"] = bench
    stats["goldstein_per_seed_test_r2"] = [float(r2_score(y.ravel(), s.ravel())) for s in z["pred_per_seed"]]
    stats["test_weeks"] = [str(p["weeks"][te[0]].date()), str(p["weeks"][te[-1]].date()), int(len(te))]

    # ---- example dyads: forecast vs actual vs trailing-52 baseline over the test period
    idx = {(a, b): i for i, (a, b) in enumerate(zip(p["a1"], p["a2"]))}
    base52 = trailing_mean(p["gold_zero"], 52)
    ex = {}
    for a, b in [("ISR", "PSE"), ("ISR", "IRN"), ("SAU", "YEM"), ("TUR", "SYR")]:
        d = idx[(a, b)]
        ex[f"{a}-{b}"] = dict(weeks=[str(p["weeks"][t].date()) for t in te], actual=y[:, d].tolist(),
                              forecast=f[:, d].tolist(), trailing52=base52[d, te].tolist(),
                              r2=float(r2_score(y[:, d], f[:, d])))
    stats["examples"] = ex

    # ---- where does cross-dyad attention go? (release seed 42, first cross-dyad block)
    args = eg.config(**RELEASE, seed=42)
    data = eg.prepare_data(args, p)
    model = eg.build(args, data)
    model.load_weights("model_release_v2.0_goldstein_s42.weights.h5")
    stats["goldstein_params"] = int(model.count_params())
    from tensorflow import keras
    sub = keras.Model(model.inputs, model.get_layer("dyad_token_projection").output)
    mha = model.get_layer("crossdyad_block0_mha")
    rel = relation_matrix(p["a1"], p["a2"])
    inp = data["inputs"]("te")
    mass = np.zeros(len(RELATION_NAMES))
    count = np.array([(rel == r).sum() for r in range(len(RELATION_NAMES))], dtype=float)
    top_example = None
    ex_week = int(np.where(te == p["weeks"].get_loc(np.datetime64("2026-02-16")))[0][0])
    for s in range(0, len(te), 8):
        batch = {k: v[s:s + 8] for k, v in inp.items()}
        tok = sub.predict(batch, verbose=0)
        _, scores = mha(tok, tok, return_attention_scores=True, training=False)
        att = np.asarray(scores).mean(axis=1)  # (b, 400, 400) averaged over heads; rows sum to 1
        for r in range(len(RELATION_NAMES)):
            mass[r] += att[:, rel == r].sum()
        if s <= ex_week < s + 8:
            row = att[ex_week - s, idx[("ISR", "IRN")]]
            order = np.argsort(-row)[:10]
            top_example = [dict(dyad=f"{p['a1'][j]}-{p['a2'][j]}", weight=float(row[j]),
                                relation=RELATION_NAMES[rel[idx[("ISR", "IRN")], j]]) for j in order]
    n_rows = len(te) * D
    share = mass / n_rows                      # average attention mass per query on each relation type
    uniform = count / (D * D)                  # what uniform attention would give
    stats["attention_by_relation"] = {RELATION_NAMES[r]: dict(share=float(share[r]), uniform=float(uniform[r]),
                                                              lift=float(share[r] / uniform[r]))
                                      for r in range(len(RELATION_NAMES))}
    stats["attention_example"] = dict(query="ISR-IRN", week="2026-02-16", top=top_example)

    # ---- direction model: learned structural-relation biases (mean over seeds, heads)
    models = lp.load_models(".")
    biases = []
    for m in models["direction"][1]:
        for layer in m.layers:
            if isinstance(layer, StructuralBiasAttention):
                biases.append(layer.relation_bias.numpy().mean(axis=1))
    biases = np.array(biases).reshape(len(models["direction"][1]), -1, len(RELATION_NAMES))
    stats["direction_relation_bias"] = {RELATION_NAMES[r]: [float(biases[:, b, r].mean()) for b in range(biases.shape[1])]
                                        for r in range(len(RELATION_NAMES))}

    # ---- GDELT limitation example: ISR->PSE around 7 Oct 2023
    d = idx[("ISR", "PSE")]
    weeks = [t for t in range(T) if "2023-06-01" <= str(p["weeks"][t].date()) <= "2024-06-30"]
    tot = mena_counts[d, weeks]
    coop = p["counts"][d][:, :9].sum(axis=1)[weeks]
    viol = p["counts"][d][:, 17:20].sum(axis=1)[weeks]
    stats["gdelt_example"] = dict(dyad="ISR-PSE", weeks=[str(p["weeks"][t].date()) for t in weeks],
                                  goldstein=p["gold_zero"][d, weeks].tolist(), events=tot.tolist(),
                                  coop_share=(coop / np.maximum(tot, 1)).tolist(),
                                  violent_share=(viol / np.maximum(tot, 1)).tolist())

    with open(OUT, "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("examples", "gdelt_example")}, indent=1))


if __name__ == "__main__":
    main()
