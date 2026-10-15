"""
Target and loss experiments on the v2.1 cross-system recipe (same data, same
1,959 tokens, same splits/embargo, same architecture and training settings):

  --target   mean-gold | mean-conflict | peak-gold | peak-conflict
             (target_variants.py; all 13-week horizon)
  --weight   none | vol
             vol: each dyad's squared error is weighted by its training-period
             volatility, sd(target - normal), normalised to mean 1 and clipped
             to [0.2, 5] - so the many stable dyads do not teach the model to
             stay flat.

  --target A+B   multi-target: one network, two outputs per dyad; each target's
                 squared error is divided by its training-period variance of
                 (target - normal) so neither dominates the loss.
  --inputs   base | extended
             extended adds, for every target, the weekly conflict intensity
             (log of summed |negative goldstein|) and the signed log of SUMMED
             goldstein (volume x tone), each with its 52-week mean.

  --world-chunks C   MORE TRAINING DATA: add C extra training "systems" per week,
             each a sequence of the same 1,959-token shape filled with dyads
             from outside the region (fetch_world_dyads.py: 5,122 non-MENA
             dyads active in >= 50% of weeks; shuffled once, the last chunk
             padded with ungraded repeats). Validation and test are unchanged
             (the MENA system, 400 graded dyads), so results compare directly.

The model forecasts a residual on the target's own `normal`. Conflict targets
add the weekly conflict series and its 26/52-week means to the inputs (as in
experiment_goldstein.py --target conflict). Evaluation: benchmark_targets.py's
`evaluate()`, on validation (for selection) and test.

    KERAS_BACKEND=torch python experiment_targets.py --target peak-gold --weight vol --seeds 42 1 2
-> logs/targets.jsonl, artifacts/experiments/targets_<target>_<weight>.npz
"""

import argparse
import json
import os
import time

import keras
import numpy as np
from keras import ops
from sklearn.preprocessing import LabelEncoder, StandardScaler

import target_variants as tv
from benchmark_targets import make_context, summary_line
from experiment_scaling import external_pool
from honest_common import goldstein_numeric_inputs, trailing_mean, windows_at
from model_architectures import build_goldstein_model

LOOKBACK = 13
LOG = "logs/targets.jsonl"
WORLD = "raw data/world_dyads_panel.parquet"


def load_world(weeks):
    """Non-MENA dyads on the MENA panel's week grid, shaped like external_pool()."""
    import pandas as pd
    from honest_common import COUNT_COLS
    df = pd.read_parquet(WORLD)
    df["week_start"] = pd.to_datetime(df.week_start)
    keys = sorted(df.groupby(["a1", "a2"]).groups)
    full = pd.MultiIndex.from_tuples([(a, b, w) for a, b in keys for w in weeks], names=["a1", "a2", "week_start"])
    df = df.set_index(["a1", "a2", "week_start"]).reindex(full)
    D, T = len(keys), len(weeks)
    g = lambda c: np.nan_to_num(df[c].to_numpy(dtype="float32")).reshape(D, T)
    return dict(a1=np.array([a for a, _ in keys]), a2=np.array([b for _, b in keys]),
                counts=np.nan_to_num(df[COUNT_COLS].to_numpy(dtype="float32")).reshape(D, T, 20),
                gold_zero=g("goldstein_mean"), neg=g("negative_goldstein_intensity"))


def weighted_masked_mse(n_tok):
    """y_true = [targets (n_tok) | weights (n_tok)]; NaN target = not graded."""
    def loss(y_true, y_pred):
        y, w = y_true[:, :n_tok], y_true[:, n_tok:]
        mask = ops.isfinite(y)
        err = ops.where(mask, w * ops.square(y_pred - ops.where(mask, y, 0.0)), 0.0)
        return ops.sum(err) / ops.maximum(ops.sum(ops.where(mask, w, 0.0)), 1e-6)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="one of target_variants.TARGETS, or two joined by +")
    ap.add_argument("--inputs", choices=["base", "extended"], default="base")
    ap.add_argument("--weight", choices=["none", "vol"], default="none")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--world-chunks", type=int, default=0)
    args = ap.parse_args()

    ctx = make_context()
    p, tr, va, te = ctx.p, ctx.tr, ctx.va, ctx.te
    ext = external_pool(p["weeks"], 0.50)
    N = len(ext["a1"])
    n_tok = 400 + N
    names = args.target.split("+")
    assert all(n in tv.TARGETS for n in names), names
    K = len(names)
    mena_ts = [ctx.T[n] for n in names]
    ext_ts = [tv.build(n, ext["gold_zero"], ext["neg"]) for n in names]
    # (n_tok, T, K) -> flattened dyad-major to (n_tok * K, T), matching the model's output order
    stackK = lambda key: np.stack([np.concatenate([m[key], e[key]], 0) for m, e in zip(mena_ts, ext_ts)], 2)
    flat = lambda a: a.transpose(0, 2, 1).reshape(-1, a.shape[1])
    y_all, base = flat(stackK("y")), flat(stackK("normal"))
    ext["neg_series"] = np.log1p(ext["neg"])
    p["neg_series"] = ctx.T["mean-conflict"]["series"]

    def numeric(src):
        x = goldstein_numeric_inputs(src, long_features=True)
        c = src["neg_series"]
        if args.inputs == "extended":
            ssum = src["gold_zero"] * src["counts"].sum(2)
            ssum = np.sign(ssum) * np.log1p(np.abs(ssum))
            extra = [c, trailing_mean(c, 52), ssum, trailing_mean(ssum, 52)]
        elif any(n.endswith("conflict") for n in names):
            extra = [c, trailing_mean(c, 26), trailing_mean(c, 52)]
        else:
            extra = []
        if extra:
            x = np.concatenate([x, np.stack(extra, 2)], 2)
        return x.astype("float32")

    X = np.concatenate([numeric(p), numeric(ext)], 0)
    n_num = X.shape[-1]
    world = load_world(p["weeks"]) if args.world_chunks else None
    enc = LabelEncoder().fit(np.concatenate([p["a1"], p["a2"], ext["a1"], ext["a2"]] +
                                            ([world["a1"], world["a2"]] if world else [])))
    a1 = enc.transform(np.concatenate([p["a1"], ext["a1"]])).astype("int32")
    a2 = enc.transform(np.concatenate([p["a2"], ext["a2"]])).astype("int32")

    ch = (y_all - base)[:, tr].reshape(n_tok, K, -1)
    task_var = np.nanvar(ch.transpose(1, 0, 2).reshape(K, -1), 1) if K > 1 else np.ones(K)

    def token_weights(ch):
        if args.weight == "vol":
            sd = np.sqrt(np.nanmean((ch - np.nanmean(ch, 2, keepdims=True)) ** 2, 2))  # (tokens, K)
            w = np.nan_to_num(np.clip(sd / np.nanmean(sd, 0), 0.2, 5.0), nan=1.0)
        else:
            w = np.ones(ch.shape[:2])
        return (w / task_var).reshape(-1).astype("float32")  # task_var puts multi-targets on one scale

    w_d = token_weights(ch)

    Xtr = windows_at(X, tr, LOOKBACK)
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_num))
    sc = lambda A: np.nan_to_num(scaler.transform(A.reshape(-1, n_num)).reshape(A.shape)).astype("float32")

    def inputs(w, Xw=None):
        return {"numeric_seq": sc(windows_at(X, w, LOOKBACK) if Xw is None else Xw),
                "actor1_idx": np.tile(a1, (len(w), 1)), "actor2_idx": np.tile(a2, (len(w), 1)),
                "baseline": np.nan_to_num(base[:, w].T).astype("float32")}

    def targets(w, grade_ext):
        y = y_all[:, w].T.astype("float32").copy()
        if not grade_ext:
            y[:, 400 * K:] = np.nan
        return np.concatenate([y, np.tile(w_d, (len(w), 1))], 1)

    x_tr, x_va, x_te = inputs(tr, Xtr), inputs(va), inputs(te)
    y_tr, y_va = targets(tr, True), targets(va, False)
    del Xtr

    if world:  # extra training systems of non-MENA dyads, same token count
        Dw = len(world["a1"])
        order = np.resize(np.random.default_rng(0).permutation(Dw), args.world_chunks * n_tok)
        pad = np.arange(len(order)) >= Dw  # repeats beyond the first pass: inputs only, never graded
        world["neg_series"] = np.log1p(world["neg"])
        Xw_all = numeric(world)
        w_ts = [tv.build(n, world["gold_zero"], world["neg"]) for n in names]
        yw_all = np.stack([t["y"] for t in w_ts], 2)
        bw_all = np.stack([t["normal"] for t in w_ts], 2)
        aw1, aw2 = enc.transform(world["a1"]).astype("int32"), enc.transform(world["a2"]).astype("int32")
        xs, ys = [x_tr], [y_tr]
        for c in range(args.world_chunks):
            sel = order[c * n_tok:(c + 1) * n_tok]
            yc, bc = flat(yw_all[sel]), flat(bw_all[sel])
            ych = yc[:, tr].T.astype("float32").copy()
            ych[:, np.repeat(pad[c * n_tok:(c + 1) * n_tok], K)] = np.nan
            wc = token_weights((yc - bc)[:, tr].reshape(n_tok, K, -1))
            xs.append({"numeric_seq": sc(windows_at(Xw_all[sel], tr, LOOKBACK)),
                       "actor1_idx": np.tile(aw1[sel], (len(tr), 1)), "actor2_idx": np.tile(aw2[sel], (len(tr), 1)),
                       "baseline": np.nan_to_num(bc[:, tr].T).astype("float32")})
            ys.append(np.concatenate([ych, np.tile(wc, (len(tr), 1))], 1))
        x_tr = {k: np.concatenate([x[k] for x in xs], 0) for k in x_tr}
        y_tr = np.concatenate(ys, 0)
        del xs, ys, Xw_all
        print(f"world: {Dw} non-MENA dyads in {args.world_chunks} extra systems per week "
              f"({int(pad.sum())} padded tokens); training sequences {len(y_tr)}", flush=True)
    tag = f"{args.target}_{args.weight}" + ("_ext" if args.inputs == "extended" else "") + \
        (f"_world{args.world_chunks}" if world else "")
    print(f"{tag}: {n_tok} tokens, {n_num} inputs, train weeks {len(tr)}, "
          f"graded train cells {np.isfinite(y_tr[:, :n_tok * K]).sum()}", flush=True)

    preds, per_seed = {"va": [], "te": []}, {}
    t0 = time.time()
    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        m = build_goldstein_model(n_dyads=n_tok, lookback_weeks=LOOKBACK, n_numeric=n_num,
                                  n_countries=len(enc.classes_), dropout=0.2, residual_baseline=True,
                                  crossdyad_blocks=2, n_outputs=K)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss=weighted_masked_mse(n_tok * K))
        h = m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=100, batch_size=8, shuffle=True, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        mena_k = lambda out: out[:, :400 * K].reshape(len(out), 400, K)  # (weeks, 400, K)
        pv, pt = mena_k(m.predict(x_va, verbose=0)), mena_k(m.predict(x_te, verbose=0))
        preds["va"].append(pv)
        preds["te"].append(pt)
        per_seed[str(seed)] = dict(best_epoch=int(np.argmin(h.history["val_loss"]) + 1),
                                   val_mse=[float(np.nanmean((pv[..., k] - mena_ts[k]["y"][:, va].T) ** 2))
                                            for k in range(K)])
        print(f"  seed {seed}: best epoch {per_seed[str(seed)]['best_epoch']}, val MSE "
              f"{per_seed[str(seed)]['val_mse']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ens = {k: np.mean(v, 0) for k, v in preds.items()}
    val, test = {}, {}
    for k, n in enumerate(names):
        val[n] = ctx.evaluate(n, ens["va"][..., k], va)
        test[n] = ctx.evaluate(n, ens["te"][..., k])
        print("VAL " + summary_line(f"{tag} -> {n}", val[n]))
        print("TEST" + summary_line(f"{tag} -> {n}", test[n]), flush=True)

    os.makedirs("artifacts/experiments", exist_ok=True)
    np.savez(f"artifacts/experiments/targets_{tag}.npz", va_week_idx=va, te_week_idx=te, targets=names,
             va_pred=ens["va"], te_pred=ens["te"], te_pred_per_seed=np.stack(preds["te"]), a1=p["a1"], a2=p["a2"])
    with open(LOG, "a") as f:
        f.write(json.dumps(dict(target=args.target, weight=args.weight, inputs=args.inputs, seeds=args.seeds,
                                world_chunks=args.world_chunks,
                                n_tokens=n_tok,
                                minutes=round((time.time() - t0) / 60, 1), per_seed=per_seed,
                                val=val, test=test)) + "\n")


if __name__ == "__main__":
    main()
