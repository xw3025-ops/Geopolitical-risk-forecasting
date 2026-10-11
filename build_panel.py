"""
Build a complete, regular dyad-week panel from the raw GDELT-style export and
add a forward-looking (13-week-ahead) target column.

Input:  raw data/bq-results-20260927-170349-1790530001905.csv
Output: raw data/panel_with_target.csv
"""

import pandas as pd

SRC = "raw data/bq-results-20260927-170349-1790530001905.csv"
OUT = "raw data/panel_with_target.csv"
HORIZON_WEEKS = 13

COUNT_COLS = [f"count_{i:02d}" for i in range(1, 21)]

df = pd.read_csv(SRC, parse_dates=["week_start"])

countries = sorted(set(df.Actor1CountryCode) | set(df.Actor2CountryCode))
weeks = sorted(df.week_start.unique())
print(f"{len(countries)} countries, {len(weeks)} weeks -> expecting {len(countries)**2 * len(weeks)} full rows")

# Full grid: every week x every ordered (actor1, actor2) pair, including self-pairs
dyads = pd.MultiIndex.from_product([countries, countries], names=["Actor1CountryCode", "Actor2CountryCode"])
full_index = pd.MultiIndex.from_product(
    [weeks, dyads.get_level_values(0).unique(), dyads.get_level_values(1).unique()],
    names=["week_start", "Actor1CountryCode", "Actor2CountryCode"],
)
full = pd.DataFrame(index=full_index).reset_index()

panel = full.merge(df, on=["week_start", "Actor1CountryCode", "Actor2CountryCode"], how="left")

missing = panel[COUNT_COLS + ["total_count"]].isna().all(axis=1)
print(f"Filling {missing.sum()} missing dyad-weeks (had {len(df)} real rows, grid has {len(panel)} total)")

panel[COUNT_COLS + ["total_count"]] = panel[COUNT_COLS + ["total_count"]].fillna(0).astype(int)
panel["top_label"] = panel["top_label"].fillna("00 Null")

panel = panel.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)

target_col = f"target_top_label_{HORIZON_WEEKS}w_ahead"
panel[target_col] = (
    panel.groupby(["Actor1CountryCode", "Actor2CountryCode"])["top_label"].shift(-HORIZON_WEEKS)
)

n_no_target = panel[target_col].isna().sum()
print(f"{n_no_target} rows have no target (last {HORIZON_WEEKS} weeks per dyad, no future label available yet)")

panel = panel.sort_values(["week_start", "Actor1CountryCode", "Actor2CountryCode"]).reset_index(drop=True)

panel.to_csv(OUT, index=False)
print(f"Wrote {len(panel)} rows to {OUT}")
