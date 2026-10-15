"""Score the live origin with every saved trade model, side by side (no retraining).

Why side by side: for the Gulf exporters the models disagree in a way that matters. The
networks learned from 2019-2024 that big drops in China's imports (lumpy oil cargoes) revert
within a quarter; no training year had a sustained chokepoint closure. Shrunk persistence
instead carries part of the latest collapse forward. Neither is validated for this regime,
so the table flags rows whose latest 3-month change is below REGIME_FLAG.

Input:  data/processed/trade_weekly_samples.npz, models/trade_transformer/*.weights.h5
Output: outputs/trade_forecast_<origin>_all_models.csv  (% change, next 13 weeks vs a year earlier)
Run:  python src/predict_trade.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from config import ROOT
from train_trade_transformer import MODEL_DIR, VARIANTS, Data, build_model, predict

REGIME_FLAG = np.log(0.5)  # latest 3-month yoy below -50%


def main():
    data = Data()
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    live = data.idx["live"]
    origin = (pd.Timestamp(str(data.weeks[live[0]])) + pd.Timedelta(days=7)).date()
    out = pd.DataFrame({"iso3": data.countries})
    for j, flow in enumerate(("imp", "exp")):
        out[f"{flow}_latest_yoy_3m"] = data.base[live[0], :, j]
        out[f"{flow}_persistence_shrunk"] = data.base_shrunk[live[0], :, j]
    sets = {v: [f"{v}_s{s}" for s in meta["seeds"]] for v in VARIANTS}
    if "final" in meta:
        sets[f"final_{meta['final']['variant']}"] = [f"final_s{s}" for s in meta["seeds"]]
    for name, files in sets.items():
        variant = name.removeprefix("final_")
        files = [f for f in files if (MODEL_DIR / f"{f}.weights.h5").exists()]
        if not files:
            continue
        blocks, _, residual = VARIANTS[variant]
        ps = []
        for f in files:
            m = build_model(len(data.countries), data.L, len(data.features), data.static.shape[-1],
                            context_blocks=blocks, residual=residual)
            m.load_weights(MODEL_DIR / f"{f}.weights.h5")
            ps.append(predict(m, data, live, variant)[0])
        p = np.mean(ps, 0)
        for j, flow in enumerate(("imp", "exp")):
            out[f"{flow}_{name}"] = p[:, j]
    num = out.columns.drop("iso3")
    out[num] = (np.expm1(out[num].astype(float)) * 100).round(1)
    out["regime_flag"] = np.where((data.base[live[0], :, 0] < REGIME_FLAG) | (data.base[live[0], :, 1] < REGIME_FLAG),
                                  "latest change < -50%: outside validated behaviour", "")
    path = ROOT / "outputs" / f"trade_forecast_{origin}_all_models.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print(f"origin {origin}, window {origin} .. {origin + pd.Timedelta(days=90)}; % change vs a year earlier")
    with pd.option_context("display.width", 250):
        print(out[["iso3"] + [c for c in out if c.startswith("imp_")] + ["regime_flag"]].to_string(index=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
