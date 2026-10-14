"""
Ladder benchmark: rudimentary networks vs the transformer variants, on the
13-week goldstein target, with level AND change metrics.

Rudimentary models (trained here, 3 seeds each, fast): a plain feed-forward
network on the CURRENT week only - no history window, no transformer, no
residual baseline:
  mlp_counts      log CAMEO counts (20)
  mlp_gold        + this week's mean goldstein
  mlp_gold_emb    + country embeddings (actor 1, actor 2)
Transformer variants are read from their saved held-out predictions
(experiment_goldstein.py runs).

Metrics (test weeks, 400 dyads), all from shared `forecast_metrics`:
  r2, mae                level accuracy
  mse_vs_52              squared error relative to the trailing 52-week mean (negative = better)
  dir_acc                direction accuracy: sign of (forecast - 52wk normal) == sign of (actual - normal)
  change_corr            corr(forecast - normal, actual - normal)
  esc_auc                ROC-AUC for flagging dyad-weeks whose next 13 weeks end >= 1.0 below normal,
                         scored by (normal - forecast)

    python benchmark_ladder.py
"""

import glob
import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from honest_common import (GOLDSTEIN_HORIZON, GOLDSTEIN_TEST_START, GOLDSTEIN_VAL_START, MENA, load_panel,
                           rows, split_by_dates, trailing_mean)

ESC_DROP = 1.0
OUT = "logs/ladder_results.json"


def forecast_metrics(y, f, normal, esc_drop=ESC_DROP, higher_is_worse=False):
    """y, f, normal: (weeks, dyads) on the same target scale. Escalation =
    the outcome ends >= esc_drop worse than normal (lower goldstein, or higher
    conflict intensity when higher_is_worse)."""
    y, f, normal = (np.asarray(a, dtype="float64").ravel() for a in (y, f, normal))
    if higher_is_worse:  # flip so that "down" always means "worse"
        y, f, normal = -y, -f, -normal
    dy, df = y - normal, f - normal
    moved = np.abs(dy) > 1e-9
    esc = (dy <= -esc_drop).astype(int)
    return dict(
        r2=float(r2_score(y, f)), mae=float(mean_absolute_error(y, f)),
        mse_vs_52=float(((y - f) ** 2).mean() / ((y - normal) ** 2).mean() - 1),
        dir_acc=float((np.sign(df[moved]) == np.sign(dy[moved])).mean()),
        change_corr=float(np.corrcoef(dy, df)[0, 1]),
        esc_auc=float(roc_auc_score(esc, -df)) if 0 < esc.mean() < 1 else float("nan"),
        esc_rate=float(esc.mean()),
    )


def train_mlp(Xtr, ytr, Xva, yva, Xte, a_tr, a_va, a_te, use_emb, seed):
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    num = keras.Input((Xtr.shape[1],), name="num")
    inputs, h = [num], num
    if use_emb:
        a1 = keras.Input((1,), dtype="int32", name="a1")
        a2 = keras.Input((1,), dtype="int32", name="a2")
        emb = keras.layers.Embedding(len(MENA), 8)
        h = keras.layers.Concatenate()([num, keras.layers.Flatten()(emb(a1)), keras.layers.Flatten()(emb(a2))])
        inputs += [a1, a2]
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Dense(64, activation="relu")(h)
    m = keras.Model(inputs, keras.layers.Dense(1)(h))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    def inp(X, a):
        return {"num": X, "a1": a[:, :1], "a2": a[:, 1:]} if use_emb else {"num": X}

    m.fit(inp(Xtr, a_tr), ytr, validation_data=(inp(Xva, a_va), yva), epochs=100, batch_size=1024, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return m.predict(inp(Xte, a_te), verbose=0, batch_size=4096).ravel()


def main():
    p = load_panel()
    D = len(p["a1"])
    y_all = p["target13"]
    tr, va, te = split_by_dates(p["weeks"], y_all[0], GOLDSTEIN_VAL_START, GOLDSTEIN_TEST_START, GOLDSTEIN_HORIZON)
    normal = trailing_mean(p["gold_zero"], 52)
    y_te, n_te = y_all[:, te].T, normal[:, te].T
    results = {}

    def add(name, f):
        results[name] = forecast_metrics(y_te, f, n_te)
        r = results[name]
        print(f"  {name:<44} R² {r['r2']:.3f}  vs52 {r['mse_vs_52']:+.1%}  dir {r['dir_acc']:.1%}  "
              f"chg-corr {r['change_corr']:.3f}  esc-AUC {r['esc_auc']:.3f}", flush=True)

    print("Baselines")
    for w in (13, 26, 52):
        add(f"trailing {w}-week mean", trailing_mean(p["gold_zero"], w)[:, te].T)

    print("Rudimentary networks (current week only; 3-seed averages)")
    counts = np.log1p(p["counts"])
    feats = {"mlp_counts": counts, "mlp_gold": np.concatenate([counts, p["gold_zero"][:, :, None]], 2)}
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    enc = LabelEncoder().fit(MENA)
    actors = np.stack([enc.transform(p["a1"]), enc.transform(p["a2"])], 1)  # (D, 2)
    act = lambda w: np.tile(actors, (len(w), 1)).astype("int32")
    for name, F, emb in [("rudimentary: CAMEO counts", feats["mlp_counts"], False),
                         ("rudimentary: counts + goldstein", feats["mlp_gold"], False),
                         ("rudimentary: counts + goldstein + embeddings", feats["mlp_gold"], True)]:
        sc = StandardScaler().fit(rows(F, tr))
        Xtr, Xva, Xte = (sc.transform(rows(F, w)).astype("float32") for w in (tr, va, te))
        preds = [train_mlp(Xtr, rows(y_all, tr), Xva, rows(y_all, va), Xte, act(tr), act(va), act(te), emb, s)
                 for s in (42, 1, 2)]
        add(name, np.mean(preds, 0).reshape(len(te), D))

    print("Transformer variants (saved held-out predictions; 3-seed averages unless noted)")
    variants = [("temporal transformer + residual + long memory, no emb, no cross-dyad",
                 "lb13_res52_long_cd0_noemb_do0.2_lr0.0003_plat_s{}", (42, 1, 2)),
                ("+ country embeddings (no cross-dyad)", "lb13_res52_long_cd0_do0.2_lr0.0003_plat_s{}", (42, 1, 2)),
                ("+ cross-dyad, no embeddings", "lb13_res52_long_noemb_do0.2_lr0.0003_plat_s{}", (42, 1, 2)),
                ("full model (release), 3 seeds", "lb13_res52_long_do0.2_lr0.0003_plat_s{}", (42, 1, 2)),
                ("full model (release), 5 seeds", "lb13_res52_long_do0.2_lr0.0003_plat_s{}", (42, 1, 2, 3, 4))]
    for name, pat, seeds in variants:
        files = [f"artifacts/experiments/{pat.format(s)}.npz" for s in seeds]
        if all(os.path.exists(f) for f in files):
            z = [np.load(f) for f in files]
            assert np.array_equal(z[0]["te_week_idx"], te)
            add(name, np.mean([x["te_pred"] for x in z], 0))
    os.makedirs("logs", exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nescalation base rate (next 13 weeks >= {ESC_DROP} below normal): "
          f"{next(iter(results.values()))['esc_rate']:.1%};  saved {OUT}")


if __name__ == "__main__":
    main()
