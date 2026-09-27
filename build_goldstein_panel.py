"""
Merge the event-count dataset with the Goldstein-score dataset into a single
complete dyad-week panel, and add a forward-looking target: the average
Goldstein score over the NEXT 13 weeks for each dyad.

Country set: the ORIGINAL 20 (includes LBY, SDN). The Goldstein source uses
a different 20 (ERI, SOM instead of LBY, SDN) - any dyad involving ERI/SOM is
dropped, and any dyad involving LBY/SDN gets avg_goldstein padded to 0 for
every week (that source never covered them).

Inputs:
    raw data/bq-results-20260927-170349-1790530001905.csv   (counts)
    raw data/results-20260927-135201 - results-20260927-135201.csv (goldstein)
Output:
    raw data/panel_with_goldstein_target.csv
"""

import numpy as np
import pandas as pd

COUNTS_SRC = "raw data/bq-results-20260927-170349-1790530001905.csv"
GOLDSTEIN_SRC = "raw data/results-20260927-135201 - results-20260927-135201.csv"
OUT = "raw data/panel_with_goldstein_target.csv"

HORIZON_WEEKS = 13
COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]

# The 20-country universe used previously (from the counts dataset / the
# original list the user supplied), kept fixed regardless of what shows up
# in either source file.
COUNTRIES = [
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
]

counts = pd.read_csv(COUNTS_SRC, parse_dates=["week_start"])
goldstein = pd.read_csv(GOLDSTEIN_SRC, parse_dates=["week_start"])

# Keep only rows within our fixed 20-country universe on both sides.
counts = counts[
    counts.Actor1CountryCode.isin(COUNTRIES) & counts.Actor2CountryCode.isin(COUNTRIES)
]
before = len(goldstein)
goldstein = goldstein[
    goldstein.Actor1CountryCode.isin(COUNTRIES) & goldstein.Actor2CountryCode.isin(COUNTRIES)
]
print(f"Dropped {before - len(goldstein)} goldstein rows involving countries "
      f"outside the fixed 20 (e.g. ERI, SOM)")

weeks = sorted(counts.week_start.unique())
print(f"{len(COUNTRIES)} countries, {len(weeks)} weeks -> "
      f"{len(COUNTRIES)**2 * len(weeks)} full dyad-weeks")

full_index = pd.MultiIndex.from_product(
    [weeks, COUNTRIES, COUNTRIES],
    names=["week_start", "Actor1CountryCode", "Actor2CountryCode"],
)
panel = pd.DataFrame(index=full_index).reset_index()

panel = panel.merge(counts[["week_start", "Actor1CountryCode", "Actor2CountryCode"] + COUNT_COLS],
                     on=["week_start", "Actor1CountryCode", "Actor2CountryCode"], how="left")
panel = panel.merge(goldstein, on=["week_start", "Actor1CountryCode", "Actor2CountryCode"], how="left")

n_missing_counts = panel[COUNT_COLS].isna().all(axis=1).sum()
n_missing_goldstein = panel["avg_goldstein"].isna().sum()
print(f"Padding {n_missing_counts} dyad-weeks with count_01..count_20 = 0")
print(f"Padding {n_missing_goldstein} dyad-weeks with avg_goldstein = 0 "
      f"(includes all LBY/SDN dyads, which the goldstein source never covers)")

panel[COUNT_COLS] = panel[COUNT_COLS].fillna(0).astype(int)
panel["avg_goldstein"] = panel["avg_goldstein"].fillna(0.0)

panel = panel.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)

# Forward target: mean of avg_goldstein over the next HORIZON_WEEKS weeks
# (week t+1 .. t+13), NOT including the current week. Requires all 13 future
# weeks to exist (i.e. be within the observed date range) - rows in the last
# 13 weeks of the panel don't have a complete future window and get NaN.
grouped = panel.groupby(["Actor1CountryCode", "Actor2CountryCode"])["avg_goldstein"]
future_shifts = pd.concat([grouped.shift(-k) for k in range(1, HORIZON_WEEKS + 1)], axis=1)
complete = future_shifts.notna().all(axis=1)
target_col = f"target_avg_goldstein_next{HORIZON_WEEKS}w"
panel[target_col] = np.where(complete, future_shifts.mean(axis=1), np.nan)

n_no_target = panel[target_col].isna().sum()
print(f"{n_no_target} rows have no target (last {HORIZON_WEEKS} weeks per dyad, "
      f"incomplete future window)")

panel = panel.sort_values(["week_start", "Actor1CountryCode", "Actor2CountryCode"]).reset_index(drop=True)

panel.to_csv(OUT, index=False)
print(f"Wrote {len(panel)} rows to {OUT}")
