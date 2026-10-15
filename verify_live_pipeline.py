"""
Check that live_pipeline.py (what the Cloud Function runs) produces exactly
the predictions the release scripts produced on their test set - i.e. that
live inputs are built, scaled and fed identically to training.

Replays one historical as-of week from the local BigQuery panel: keeps only
the HISTORY_WEEKS complete weeks ending that week, drops the target columns
(a live pull has none), runs live_pipeline.predict, and compares with the
saved release test predictions for the same week. Run after any change to
the pipeline or the models:

    python verify_live_pipeline.py [--as-of 2026-06-22]
"""

import argparse

import numpy as np
import pandas as pd

import live_pipeline as lp
from honest_common import load_panel
from run_pipeline import local_external, local_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-05-25", help="a week in both release test periods")
    args = ap.parse_args()
    as_of = pd.Timestamp(args.as_of)

    df = local_frame(as_of)
    print("panel check:", lp.check_panel(df, as_of.date()))
    models = lp.load_models(".")
    pairs = models["goldstein"][0].get("external_dyads", [])
    live = lp.predict(df, models, local_external(as_of.date(), pairs) if pairs else None)
    live = live.set_index(["actor1", "actor2"])

    weeks = load_panel().get("weeks")
    t = weeks.get_loc(as_of)
    ok = True
    for name, path, key, col in [
        ("goldstein", f"artifacts/goldstein_{lp.GOLDSTEIN_VERSION}_test_predictions.npz", "pred",
         "goldstein_forecast_next13w"),
        ("surge", "artifacts/surge_v2.2_test_predictions.npz", "pred", "conflict_peak_forecast_next13w"),
    ]:
        z = np.load(path, allow_pickle=True)
        row = int(np.where(z["week_idx"] == t)[0][0])
        release = pd.Series(z[key][row], index=pd.MultiIndex.from_arrays([z["a1"], z["a2"]]))
        diff = np.abs(live[col].reindex(release.index).to_numpy() - release.to_numpy()).max()
        status = "OK" if diff < 1e-4 else "MISMATCH"
        ok &= diff < 1e-4
        print(f"{name:<10} live vs release test predictions for {as_of.date()}: max |diff| = {diff:.2e}  {status}")
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
