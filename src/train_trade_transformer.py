"""Trade transformer: forecast the next 13 weeks of China's trade with each MENA country.

Unit: country x week (20 MENA countries, Monday weeks). Target (make_training_samples.py):
year-on-year log change of China's imports from / exports to the country over the 13 weeks
after the forecast origin; two outputs per country. exp(y) - 1 is the % change.

Architecture (one sample = one origin week = 20 country tokens):
  1. temporal transformer  each country attends over its own last 13 weeks of inputs
                           (GDELT conflict, UCDP deaths, published trade growth, calendar)
  2. country token         temporal summary (last week) + country embedding + static exposure
  3. contextual transformer every country attends to the other 19 countries in the same week
  4. head                  Dense(2) per country: imports, exports
  Both transformers: multi-head attention + SwiGLU feed-forward, residual + LayerNorm.

Variants (same recipe): full | no_context (0 contextual blocks) | trade_only (conflict inputs
zeroed, so the value of the conflict data is measured) | residual (full, but the output is the
shrunk-persistence forecast + the network's correction, as in release v2.0).
The variant with the lowest mean validation loss is refit for the live forecast.
Benchmarks on the same graded rows: zero change, shrunk persistence, country mean, LightGBM.
Each neural variant is a SEEDS-seed ensemble; early stopping on the validation split.
Finally the selected variant is refit on every labelled origin (median best epoch count) and
scores the live origin.

Outputs
  models/trade_transformer/{<variant>_s<seed>.weights.h5, final_s<seed>.weights.h5, meta.json}
  data/processed/trade_model_predictions.csv  (test rows, every model)
  data/processed/trade_model_metrics.csv
  outputs/trade_forecast_<origin>.csv          (live forecast)
Run:  python src/train_trade_transformer.py [--seeds 5] [--variants full no_context trade_only residual]
"""
import argparse
import json
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import lightgbm as lgb
import numpy as np
import pandas as pd
from keras import ops
from sklearn.metrics import roc_auc_score

from config import PROCESSED, ROOT

warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")

D_MODEL = 32
NUM_HEADS = 4
FF_DIM = 64  # SwiGLU hidden width
COUNTRY_EMBED_DIM = 8
TEMPORAL_BLOCKS = 2
CONTEXT_BLOCKS = 2
DROPOUT = 0.2
LR = 1e-3
BATCH = 16
MAX_EPOCHS = 200
PATIENCE = 20
SHOCK_START = "2026-01-01"  # test origins from here have target windows in the 2026 Hormuz shock
DROP = np.log(0.75)  # "large drop" = next 13 weeks at least 25% below a year earlier
CONFLICT_FEATURES = ["dom_attention", "dom_violent_share", "dom_matconf_share", "dom_goldstein",
                     "rel_attention", "rel_violent_share", "inbound_violent_share", "rel_goldstein",
                     "violent_surge_52", "ucdp_log_deaths", "geo_t2t4_share", "geo_conflict_share"]
MODEL_DIR = ROOT / "models" / "trade_transformer"
# variant -> (contextual blocks, conflict inputs zeroed, residual on shrunk persistence)
VARIANTS = {"full": (CONTEXT_BLOCKS, False, False), "no_context": (0, False, False),
            "trade_only": (CONTEXT_BLOCKS, True, False), "residual": (CONTEXT_BLOCKS, False, True)}


# ------------------------------------------------------------ model

def swiglu(x, name):
    gate = keras.layers.Dense(FF_DIM, activation="swish", name=f"{name}_gate")(x)
    value = keras.layers.Dense(FF_DIM, name=f"{name}_value")(x)
    out = keras.layers.Dense(D_MODEL, name=f"{name}_out")(keras.layers.Multiply()([gate, value]))
    return keras.layers.Dropout(DROPOUT)(out)


def block(x, name):
    att = keras.layers.MultiHeadAttention(NUM_HEADS, D_MODEL // NUM_HEADS, dropout=DROPOUT, name=f"{name}_mha")(x, x)
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + keras.layers.Dropout(DROPOUT)(att))
    return keras.layers.LayerNormalization(name=f"{name}_ln2")(x + swiglu(x, f"{name}_ffn"))


def build_model(n_countries, lookback, n_features, n_static, context_blocks=CONTEXT_BLOCKS, residual=False):
    seq = keras.layers.Input((n_countries, lookback, n_features), name="numeric_seq")
    cidx = keras.layers.Input((n_countries,), dtype="int32", name="country_idx")
    static = keras.layers.Input((n_countries, n_static), name="static")

    x = keras.layers.Dense(D_MODEL, name="token_projection")(seq)
    positions = keras.layers.Lambda(lambda t: ops.arange(lookback), output_shape=(lookback,))(seq)
    pos = keras.layers.Embedding(lookback, D_MODEL, name="week_position")(positions)  # (lookback, d)
    x = keras.layers.Lambda(lambda t: t[0] + t[1], output_shape=(n_countries, lookback, D_MODEL),
                            name="add_week_position")([x, pos])
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Lambda(lambda t: ops.reshape(t, (-1, lookback, D_MODEL)), output_shape=(lookback, D_MODEL),
                            name="merge_batch_country")(x)
    for i in range(TEMPORAL_BLOCKS):  # attention over the same country's last 13 weeks
        x = block(x, f"temporal{i}")
    x = keras.layers.Lambda(lambda t: ops.reshape(t[:, -1, :], (-1, n_countries, D_MODEL)),
                            output_shape=(n_countries, D_MODEL), name="last_week_per_country")(x)

    emb = keras.layers.Embedding(n_countries, COUNTRY_EMBED_DIM, name="country_embedding")(cidx)
    st = keras.layers.Dense(COUNTRY_EMBED_DIM, name="static_projection")(static)
    x = keras.layers.Dense(D_MODEL, activation="relu", name="country_token")(
        keras.layers.Concatenate()([x, emb, st]))
    for i in range(context_blocks):  # attention across the 20 countries in the same week
        x = block(x, f"context{i}")
    out = keras.layers.Dense(2, name="imp_exp")(x)
    if not residual:
        return keras.Model([seq, cidx, static], out)
    baseline = keras.layers.Input((n_countries, 2), name="baseline")  # forecast known at the origin
    return keras.Model([seq, cidx, static, baseline], keras.layers.Add(name="add_baseline")([baseline, out]))


def masked_mse(y_true, y_pred):
    mask = ops.logical_not(ops.isnan(y_true))
    diff = ops.where(mask, y_pred - ops.where(mask, y_true, 0.0), 0.0)
    return ops.sum(diff ** 2) / ops.maximum(ops.sum(ops.cast(mask, "float32")), 1.0)


# ------------------------------------------------------------ data

class Data:
    def __init__(self):
        d = np.load(PROCESSED / "trade_weekly_samples.npz", allow_pickle=False)
        self.meta = json.loads(str(d["meta"]))
        self.L = self.meta["lookback"]
        self.X = (d["X"] - d["mu"]) / d["sd"]
        self.Xraw = d["X"]
        self.static = ((d["static"] - d["s_mu"]) / d["s_sd"]).astype("float32")
        self.y, self.base = d["y"], d["baseline"]
        self.weeks, self.split = d["weeks"], d["split"]
        self.countries, self.features = list(d["countries"]), list(d["features"])
        self.idx = {s: np.where(self.split == s)[0] for s in ("train", "val", "test", "live")}
        # shrunk persistence: last published 3-month yoy x a per-flow slope fitted on train
        tr = self.idx["train"]
        b, y = np.nan_to_num(self.base[tr]), self.y[tr]
        self.shrink = np.array([np.nansum(b[..., j] * y[..., j]) / np.sum(b[..., j][~np.isnan(y[..., j])] ** 2)
                                for j in range(2)], dtype="float32")
        self.base_shrunk = (np.nan_to_num(self.base) * self.shrink).astype("float32")

    def inputs(self, w, zero=(), residual=False):
        X = np.stack([self.X[i - self.L + 1:i + 1].transpose(1, 0, 2) for i in w]).astype("float32")
        for f in zero:  # ablation: input replaced by its training mean (0 after standardising)
            X[..., self.features.index(f)] = 0.0
        C = len(self.countries)
        out = {"numeric_seq": X, "country_idx": np.tile(np.arange(C, dtype="int32"), (len(w), 1)),
               "static": np.repeat(self.static[None], len(w), 0)}
        if residual:
            out["baseline"] = self.base_shrunk[w]
        return out


# ------------------------------------------------------------ training

def fit_one(data, seed, variant, train_w, val_w=None, epochs=None):
    keras.utils.set_random_seed(seed)
    blocks, no_conflict, residual = VARIANTS[variant]
    zero = CONFLICT_FEATURES if no_conflict else ()
    m = build_model(len(data.countries), data.L, len(data.features), data.static.shape[-1],
                    context_blocks=blocks, residual=residual)
    m.compile(optimizer=keras.optimizers.AdamW(LR, weight_decay=1e-4), loss=masked_mse)
    kw = dict(batch_size=BATCH, shuffle=True, verbose=0)
    if val_w is not None:
        h = m.fit(data.inputs(train_w, zero, residual), data.y[train_w], epochs=MAX_EPOCHS,
                  validation_data=(data.inputs(val_w, zero, residual), data.y[val_w]), **kw,
                  callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)])
        best = int(np.argmin(h.history["val_loss"])) + 1
        return m, best, float(np.min(h.history["val_loss"]))
    m.fit(data.inputs(train_w, zero, residual), data.y[train_w], epochs=epochs, **kw)
    return m, epochs, np.nan


def predict(m, data, w, variant):
    _, no_conflict, residual = VARIANTS[variant]
    zero = CONFLICT_FEATURES if no_conflict else ()
    return m.predict(data.inputs(w, zero, residual), verbose=0, batch_size=64)


# ------------------------------------------------------------ benchmarks

def tabular(data, w):
    """One row per (week, country): last-week inputs, 13-week means of the conflict inputs, static."""
    rows = []
    C = len(data.countries)
    ci = [data.features.index(f) for f in CONFLICT_FEATURES]
    for i in w:
        win = data.Xraw[i - data.L + 1:i + 1]
        rows.append(np.concatenate([data.Xraw[i], win[:, :, ci].mean(0), data.static, np.arange(C)[:, None]], 1))
    cols = data.features + [f"{f}_mean13" for f in CONFLICT_FEATURES] + \
        [f"static{j}" for j in range(data.static.shape[-1])] + ["country"]
    df = pd.DataFrame(np.concatenate(rows), columns=cols)
    df["country"] = df.country.astype(int).astype("category")
    return df


def lightgbm_preds(data, train_w, val_w, pred_w, seeds=(0, 1, 2)):
    Xtr, Xva, Xp = tabular(data, train_w), tabular(data, val_w), tabular(data, pred_w)
    out = np.zeros((len(pred_w), len(data.countries), 2))
    for j in range(2):
        ytr, yva = data.y[train_w][..., j].reshape(-1), data.y[val_w][..., j].reshape(-1)
        a, b = ~np.isnan(ytr), ~np.isnan(yva)
        for s in seeds:
            g = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=s, verbose=-1)
            g.fit(Xtr[a], ytr[a], eval_set=[(Xva[b], yva[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
            out[..., j] += g.predict(Xp).reshape(len(pred_w), -1) / len(seeds)
    return out


def simple_benchmarks(data, train_w, pred_w):
    ytr = data.y[train_w]
    out = {"zero": np.zeros((len(pred_w), len(data.countries), 2))}
    b_tr, b_p = data.base[train_w], data.base[pred_w]
    k = []
    for j in range(2):
        ok = ~np.isnan(ytr[..., j])
        k.append(np.sum(b_tr[..., j][ok] * ytr[..., j][ok]) / np.sum(b_tr[..., j][ok] ** 2))
    out["persistence_shrunk"] = b_p * np.array(k)  # same slope as Data.shrink
    out["persistence_3m"] = b_p
    out["country_mean"] = np.repeat(np.nan_to_num(np.nanmean(ytr, 0))[None], len(pred_w), 0)
    return out


# ------------------------------------------------------------ evaluation

def metrics(y, p, label):
    rows = []
    for j, flow in enumerate(("imp", "exp")):
        yy, pp = y[..., j].reshape(-1), p[..., j].reshape(-1)
        ok = ~np.isnan(yy)
        yy, pp = yy[ok], pp[ok]
        mse0 = np.mean(yy ** 2)
        drop = yy < DROP
        rows.append({"period": label, "flow": flow, "n": len(yy), "mse": np.mean((pp - yy) ** 2),
                     "mae": np.mean(np.abs(pp - yy)), "skill_vs_zero": 1 - np.mean((pp - yy) ** 2) / mse0,
                     "corr": np.corrcoef(pp, yy)[0, 1] if pp.std() > 0 else np.nan,
                     "drop_rate": drop.mean(),
                     "drop_auc": roc_auc_score(drop, -pp) if 0 < drop.sum() < len(drop) and pp.std() > 0 else np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--no-final", action="store_true", help="skip the refit on all labelled origins")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    data = Data()
    tr, va, te = data.idx["train"], data.idx["val"], data.idx["test"]
    print(f"origins: train {len(tr)}, val {len(va)}, test {len(te)}; {len(data.countries)} countries, "
          f"{len(data.features)} inputs x {data.L} weeks")
    m0 = build_model(len(data.countries), data.L, len(data.features), data.static.shape[-1])
    print(f"full model: {m0.count_params():,} parameters")

    preds = simple_benchmarks(data, tr, te)
    preds["lightgbm"] = lightgbm_preds(data, tr, va, te)
    best_epochs, val_mse, log = {}, {}, []
    for v in args.variants:
        ps, eps = [], []
        for s in seeds:
            m, ep, vl = fit_one(data, s, v, tr, va)
            p = predict(m, data, te, v)
            ps.append(p)
            eps.append(ep)
            log.append({"variant": v, "seed": s, "best_epoch": ep, "val_mse": vl})
            print(f"  {v:10s} seed {s}: best epoch {ep:3d}, val MSE {vl:.4f}", flush=True)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            m.save_weights(MODEL_DIR / f"{v}_s{s}.weights.h5")
        preds[f"transformer_{v}"] = np.mean(ps, 0)
        best_epochs[v] = eps
        val_mse[v] = float(np.mean([r["val_mse"] for r in log if r["variant"] == v]))
    print("mean validation MSE by variant:", {k: round(v, 4) for k, v in val_mse.items()})

    # metrics: whole test, calm 2025, 2026 shock
    y = data.y[te]
    shock = data.weeks[te] >= SHOCK_START
    rows = []
    for name, p in preds.items():
        for label, sel in (("test_all", slice(None)), ("test_2025", ~shock), ("test_2026_shock", shock)):
            for r in metrics(y[sel], p[sel], label):
                rows.append({"model": name, **r})
    met = pd.DataFrame(rows)
    met.to_csv(PROCESSED / "trade_model_metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for label in ("test_all", "test_2025", "test_2026_shock"):
            print(f"\n== {label} ==")
            print(met[met.period == label].drop(columns="period").pivot(index="model", columns="flow")
                  [["mse", "skill_vs_zero", "corr", "drop_auc"]].round(3).to_string())

    # long table of test predictions
    C = len(data.countries)
    long = pd.DataFrame({"week_start": np.repeat(data.weeks[te], C), "iso3": np.tile(data.countries, len(te))})
    for j, flow in enumerate(("imp", "exp")):
        long[f"y_{flow}"] = y[..., j].reshape(-1)
        for name, p in preds.items():
            long[f"{name}_{flow}"] = p[..., j].reshape(-1)
    long.to_csv(PROCESSED / "trade_model_predictions.csv", index=False)

    # what the models said just before the trade collapse (exposed importers)
    show = ["IRQ", "ARE", "QAT", "KWT", "SAU", "OMN", "TUR", "ISR"]
    for wk in ("2026-02-16", "2026-03-02", "2026-03-23"):
        sub = long[(long.week_start == wk) & long.iso3.isin(show)].set_index("iso3").reindex(show)
        cols = ["y_imp", "zero_imp", "lightgbm_imp"] + [c for c in sub if c.startswith("transformer") and c.endswith("_imp")]
        print(f"\nImports, origin week {wk}: % change next 13 weeks vs a year earlier")
        print((np.expm1(sub[cols]) * 100).round(0).to_string())

    meta = {"countries": data.countries, "features": data.features, "conflict_features": CONFLICT_FEATURES,
            "lookback": data.L, "d_model": D_MODEL, "heads": NUM_HEADS, "ff_dim": FF_DIM,
            "temporal_blocks": TEMPORAL_BLOCKS, "context_blocks": CONTEXT_BLOCKS, "dropout": DROPOUT,
            "params": int(m0.count_params()), "seeds": seeds, "best_epochs": best_epochs, "val_mse": val_mse,
            "persistence_shrink": data.shrink.tolist(), "runs": log,
            "samples_meta": data.meta, "scaler": "mu/sd stored in data/processed/trade_weekly_samples.npz"}

    if not args.no_final and val_mse:
        chosen = min(val_mse, key=val_mse.get)  # selected on validation, never on test
        epochs = int(np.median(best_epochs[chosen]))
        all_w = np.concatenate([tr, va, te])
        live = data.idx["live"]
        ps = []
        for s in seeds:
            m, _, _ = fit_one(data, s, chosen, all_w, epochs=epochs)
            m.save_weights(MODEL_DIR / f"final_s{s}.weights.h5")
            ps.append(predict(m, data, live, chosen))
        p = np.mean(ps, 0)[0]
        origin = (pd.Timestamp(str(data.weeks[live[0]])) + pd.Timedelta(days=7)).date()
        fc = pd.DataFrame({"iso3": data.countries,
                           "imports_pct_change": np.expm1(p[:, 0]) * 100,
                           "exports_pct_change": np.expm1(p[:, 1]) * 100,
                           "imports_latest_yoy_3m_pct": np.expm1(data.base[live[0], :, 0]) * 100,
                           "exports_latest_yoy_3m_pct": np.expm1(data.base[live[0], :, 1]) * 100})
        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"trade_forecast_{origin}.csv"
        fc.round(1).to_csv(path, index=False)
        print(f"\nLive forecast, origin {origin} (window {origin} .. {origin + pd.Timedelta(days=90)}), "
              f"variant '{chosen}' refit on {len(all_w)} origins for {epochs} epochs -> {path}")
        print(fc.round(1).to_string(index=False))
        meta["final"] = {"variant": chosen, "epochs": epochs, "origins": int(len(all_w)), "live_origin": str(origin)}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"\nmetrics -> {PROCESSED / 'trade_model_metrics.csv'}; weights + meta -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
