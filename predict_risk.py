"""
Score one week with the saved v2 risk models (train_risk_gbm.py) and write a
decision-support watchlist. Nothing is refitted here: features come from
risk_features.py, and models, category lists and calibrators come from
models/risk_gbm/<model>/.

Outputs in outputs/<week>/:
  host_country_risk.csv   one row per country: probability of a UCDP violence
                          escalation (4w, 13w), of a domestic violent-event
                          surge and of worsening domestic tone, China <->
                          country relations (when the panel includes CHN),
                          the riskiest foreign relationship, and the main
                          drivers
  <model>.csv             every scored row with its probability and drivers

Drivers are the features that pushed this row's risk UP the most (LightGBM
SHAP contributions, averaged over seeds), in plain language with their
current values. They explain the MODEL, not the world - an analyst should
check them against the news before acting.

Run:
    python predict_risk.py                  # latest week in the panel
    python predict_risk.py --week 2023-09-25
"""

import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from risk_config import CHINA, MODEL_DIR, OUTPUT_DIR, RISK_PANEL_PATH, UCDP_COUNTRY_WEEK_PATH
from risk_features import DISPLAY_MIN_EVENTS, build_frames, describe, load_panel, load_ucdp
from train_risk_gbm import CalibratedEnsemble, file_sha256, prepare_X

N_DRIVERS = 3


def load_model(model_dir):
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    boosters = [lgb.Booster(model_file=p) for p in sorted(glob.glob(os.path.join(model_dir, "seed*.txt")))]
    return meta, CalibratedEnsemble(boosters, meta["calibrator"]["a"], meta["calibrator"]["b"])


def drivers(ens, X, features):
    contrib = np.mean([bst.predict(X, pred_contrib=True) for bst in ens.boosters], axis=0)[:, :-1]
    out = []
    for i in range(len(X)):
        top = np.argsort(-contrib[i])[:N_DRIVERS]
        parts = []
        for j in top:
            if contrib[i, j] <= 0:
                break
            value = X.iloc[i, j]
            shown = value if isinstance(value, str) or pd.isna(value) else f"{float(value):.2f}"
            parts.append(f"{describe(features[j])} = {shown}")
        out.append("; ".join(parts))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", help="Monday week_start to score (default: latest week in the panel)")
    parser.add_argument("--panel", default=RISK_PANEL_PATH)
    parser.add_argument("--ucdp", default=UCDP_COUNTRY_WEEK_PATH)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    panel, ucdp = load_panel(args.panel), load_ucdp(args.ucdp)
    dyad_frame, country_frame, _, _ = build_frames(panel, ucdp)
    week = pd.Timestamp(args.week) if args.week else dyad_frame.week_start.max()
    frames = {"dyad": dyad_frame, "country": country_frame}
    out_dir = os.path.join(OUTPUT_DIR, str(week.date()))
    os.makedirs(out_dir, exist_ok=True)
    hashes = {"panel": file_sha256(args.panel), "ucdp": file_sha256(args.ucdp)}

    scored = {}
    for model_dir in sorted(glob.glob(os.path.join(MODEL_DIR, "*"))):
        meta, ens = load_model(model_dir)
        if week <= pd.Timestamp(meta["calibration_weeks"][1]):
            print(f"WARNING: {week.date()} is inside {meta['name']}'s training/calibration window - its scores "
                  f"are IN-SAMPLE and look better than reality. For honest historical scores use "
                  f"outputs/backtest_{meta['name']}.csv")
        if meta["data_sha256"] != hashes:
            print(f"note: data changed since {meta['name']} was trained ({meta['created']}) - "
                  f"expected for live scoring; re-run train_risk_gbm.py at least monthly")
        rows = frames[meta["task"]]
        rows = rows[rows.week_start == week].reset_index(drop=True)
        if rows.empty:
            raise SystemExit(f"No rows for week {week.date()}")
        X = prepare_X(rows, meta["features"], meta["categories"])
        result = rows[["week_start", meta["key"]]].copy()
        result["probability"] = ens.proba(X)
        result["drivers"] = drivers(ens, X, meta["features"])
        if meta["task"] == "dyad":
            result["low_data"] = (rows.ref_events_26 < DISPLAY_MIN_EVENTS).to_numpy()
        result = result.sort_values("probability", ascending=False)
        result.to_csv(os.path.join(out_dir, f"{meta['name']}.csv"), index=False)
        scored[meta["name"]] = result.set_index(meta["key"])

    # ---- host-country risk card
    card = pd.DataFrame(index=scored["country_escalate_h13"].index)
    card["p_violence_escalation_4w"] = scored["country_escalate_h4"].probability
    card["p_violence_escalation_13w"] = scored["country_escalate_h13"].probability
    tone13, viol13 = scored["dyad_worsen_h13"], scored["dyad_violence_h13"]
    card["p_domestic_violent_surge_13w"] = [viol13.probability.get(f"{c}_{c}", np.nan) for c in card.index]
    card["p_domestic_tone_worsen_13w"] = [tone13.probability.get(f"{c}_{c}", np.nan) for c in card.index]
    if CHINA in card.index:
        for name, scores in (("violent_surge", viol13), ("tone_worsen", tone13)):
            card[f"p_china_to_country_{name}_13w"] = [scores.probability.get(f"{CHINA}_{c}", np.nan) for c in card.index]
            card[f"p_country_to_china_{name}_13w"] = [scores.probability.get(f"{c}_{CHINA}", np.nan) for c in card.index]
    is_foreign = np.array([a != b for a, b in (k.split("_") for k in viol13.index)])
    foreign = viol13[is_foreign & ~viol13.low_data.to_numpy(bool)]
    worst = {}
    for key, p in foreign.probability.items():
        for c in key.split("_"):
            if p > worst.get(c, (-1, ""))[0]:
                worst[c] = (p, key)
    card["max_relation_violent_surge_13w"] = [worst.get(c, (np.nan, ""))[0] for c in card.index]
    card["riskiest_relation"] = [worst.get(c, (np.nan, ""))[1] for c in card.index]
    card["violence_drivers_13w"] = scored["country_escalate_h13"].drivers
    card = card.sort_values("p_violence_escalation_13w", ascending=False)
    card.to_csv(os.path.join(out_dir, "host_country_risk.csv"))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)
    print(f"\n=== Host-country risk card, week of {week.date()} (written to {out_dir}/) ===")
    print(card.drop(columns="violence_drivers_13w").head(args.top).to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nWhy (top 5 countries, violence escalation 13w):")
    for c, row in card.head(5).iterrows():
        print(f"  {c}: {row.violence_drivers_13w}")
    for title, scores in (("violent-event surge", viol13), ("worsening tone", tone13)):
        print(f"\n=== Top {args.top} relationships: {title} over 13 weeks "
              f"(dyads with >= {DISPLAY_MIN_EVENTS} events in 26 weeks) ===")
        for key, row in scores[~scores.low_data].head(args.top).iterrows():
            print(f"  {key:<8} P={row.probability:.2f}  {row.drivers}")

if __name__ == "__main__":
    main()
