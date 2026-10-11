"""
Step 3 of the v2 pipeline: turn UCDP Georeferenced Event Dataset (GED)
records into a country-week table of organised-violence fatalities, aligned
with the GDELT panel's Monday weeks.

Why UCDP: GDELT measures media coverage of events, not violence. UCDP GED
codes each organised-violence event with a fatality estimate, from
human-validated sources. It is the standard ground truth in conflict
forecasting (e.g. the ViEWS early-warning system) and is licensed CC BY 4.0,
so commercial use is allowed with attribution. (ACLED has richer coverage
of protests and riots, but its terms require a paid licence for commercial
use - check before adding it.)

Input (download into raw data/ucdp/, see README "Runbook"):
  GEDEvent_v26_1.csv            final release, 1989-2025 (unzip ged261-csv.zip)
  GEDEvent_v26_0*.csv / v26_01_26_06.csv   candidate releases for 2026; when the
                                same event id appears in several files, the
                                newest file wins
Output: raw data/ucdp_country_week.csv with, per country (ISO3) and week:
  ucdp_events, ucdp_deaths (best estimate), ucdp_deaths_state,
  ucdp_deaths_nonstate, ucdp_deaths_onesided, ucdp_deaths_civilians,
  ucdp_covered (0 for weeks after the last UCDP date - unknown, not zero)

Events in Gaza / West Bank (filed under "Israel" by UCDP) are assigned to PSE.
Events with an uncertain date are assigned to the week of date_start.

Run:
    python build_ucdp_country_week.py
"""

import glob
import os

import pandas as pd

from risk_config import (PANEL_START, PSE_ADM1_IN_ISRAEL, UCDP_COUNTRY_NAMES,
                         UCDP_COUNTRY_WEEK_PATH, UCDP_DIR)

USE_COLS = ["id", "country", "adm_1", "type_of_violence", "best", "deaths_civilians", "date_start"]
FINAL_RELEASE = "GEDEvent_v26_1.csv"


def candidate_sort_key(path):
    # GEDEvent_v26_01_26_06.csv (Jan-Jun bundle) is older than v26_0_7, v26_0_8, ...
    name = os.path.basename(path)
    if "_01_" in name:
        return 0
    return int(name.rsplit("_", 1)[-1].split(".")[0])


def main():
    final = pd.read_csv(os.path.join(UCDP_DIR, FINAL_RELEASE), usecols=USE_COLS, low_memory=False)
    final["file_rank"] = -1
    candidates = sorted(set(glob.glob(os.path.join(UCDP_DIR, "GEDEvent_v26_0*.csv"))), key=candidate_sort_key)
    frames = [final]
    for rank, path in enumerate(candidates):
        frame = pd.read_csv(path, usecols=USE_COLS, low_memory=False)
        frame["file_rank"] = rank
        frames.append(frame)
        print(f"candidate file {os.path.basename(path)}: {len(frame):,} events")
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values("file_rank").drop_duplicates("id", keep="last")

    events["date_start"] = pd.to_datetime(events.date_start.str[:10])
    last_date = events.date_start.max()
    name_to_iso = {name: iso for iso, name in UCDP_COUNTRY_NAMES.items()}
    events["iso3"] = events.country.map(name_to_iso)
    events.loc[(events.country == "Israel") & events.adm_1.isin(PSE_ADM1_IN_ISRAEL), "iso3"] = "PSE"
    events = events[events.iso3.notna() & (events.date_start >= PANEL_START)]
    events["week_start"] = events.date_start - pd.to_timedelta(events.date_start.dt.weekday, unit="D")

    for col, tov in (("ucdp_deaths_state", 1), ("ucdp_deaths_nonstate", 2), ("ucdp_deaths_onesided", 3)):
        events[col] = events.best.where(events.type_of_violence == tov, 0)
    weekly = events.groupby(["iso3", "week_start"]).agg(
        ucdp_events=("id", "size"), ucdp_deaths=("best", "sum"),
        ucdp_deaths_state=("ucdp_deaths_state", "sum"), ucdp_deaths_nonstate=("ucdp_deaths_nonstate", "sum"),
        ucdp_deaths_onesided=("ucdp_deaths_onesided", "sum"), ucdp_deaths_civilians=("deaths_civilians", "sum"),
    ).reset_index()

    # Last fully covered week: the week containing last_date is partial.
    last_full_week = last_date - pd.Timedelta(days=last_date.weekday() + (0 if last_date.weekday() == 6 else 7))
    weeks = pd.date_range(PANEL_START, last_date + pd.Timedelta(days=7 * 13), freq="W-MON")
    grid = pd.MultiIndex.from_product([sorted(UCDP_COUNTRY_NAMES) + ["PSE"], weeks],
                                      names=["iso3", "week_start"]).to_frame(index=False)
    out = grid.merge(weekly, on=["iso3", "week_start"], how="left")
    value_cols = [c for c in out.columns if c.startswith("ucdp_")]
    out["ucdp_covered"] = (out.week_start <= last_full_week).astype("int8")
    out[value_cols] = out[value_cols].fillna(0).where(out.ucdp_covered == 1)

    out.to_csv(UCDP_COUNTRY_WEEK_PATH, index=False)
    covered = out[out.ucdp_covered == 1]
    print(f"UCDP events through {last_date.date()} (last complete week {last_full_week.date()}); "
          f"{covered.iso3.nunique()} countries x {covered.week_start.nunique()} covered weeks")
    print("Deaths since 2015 by country (top 15):")
    print(covered.groupby("iso3").ucdp_deaths.sum().sort_values(ascending=False).head(15).astype(int).to_string())
    print(f"Wrote {len(out):,} rows to {UCDP_COUNTRY_WEEK_PATH}")


if __name__ == "__main__":
    main()
