"""
Verify that the data the models were trained on - and the data the live
pipeline feeds them - really is GDELT, aggregated by the production query in
cloud_function/gdelt_refresh.py, and not some other query built along the way.

Chain being verified:
    gdelt-bq.gdeltv2.events  --(gdelt_refresh.REFRESH_QUERY)-->  table 1.1
    table 1.1  --(build_bq_panel_22*.py)-->  raw data/bq_panel_22_*.parquet  --> training
    table 1.1  --(live_pipeline.pull_panel)-->  live inputs                   --> forecasts

OFFLINE checks (no credentials):
  A. Source tracing: the only tables the pipeline code reads/writes.
  B. Schema: the production query's output columns = the training data's
     feature columns, with the same integer/float typing; same country list;
     same Monday-week calendar starting 2015-02-16.
  C. Fingerprint: arithmetic identities the production query builds into its
     output (total_count = sum of CAMEO counts, shares = count / total, ...)
     hold in every row of the training data.

ONLINE checks (--online; needs `gcloud auth application-default login`):
  D. Provenance: the SQL of the BigQuery job that last created table 1.1,
     from INFORMATION_SCHEMA job history, compared with REFRESH_QUERY.
  E. Values: run REFRESH_QUERY read-only (SELECT instead of CREATE TABLE)
     directly against gdelt-bq.gdeltv2.events for a few historical weeks and
     compare with the training data. A dry run reports the scan size first;
     add --yes to actually run it (one full scan of the GDELT columns used,
     same as one pipeline refresh).

    python verify_gdelt_source.py
    python verify_gdelt_source.py --online [--yes]
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "cloud_function")
from gdelt_refresh import DESTINATION_TABLE, REFRESH_QUERY  # noqa: E402

import live_pipeline as lp  # noqa: E402
from honest_common import EXTRA, MENA, PANEL_PATH, TARGET4_COL  # noqa: E402

PROJECT = "geopolitical-risk-forecasting"
GDELT_TABLE = "gdelt-bq.gdeltv2.events"
CHECK_WEEKS = ["2016-06-06", "2019-11-11", "2023-10-09", "2026-06-22"]

failures = []


def check(ok, msg):
    print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def training_frame():
    df = pd.read_parquet(PANEL_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


# ---------------------------------------------------------------- A
def check_sources():
    print("\nA. Where the pipeline's queries read from / write to")
    tables = lambda path: set(re.findall(r"`([\w\-]+\.[\w\-]+\.[\w\-]+)`", open(path, encoding="utf-8").read()))
    refresh_src = set(re.findall(r"FROM `([\w.\-]+)`", REFRESH_QUERY))
    check(refresh_src == {GDELT_TABLE}, f"refresh query reads only {GDELT_TABLE} (found: {sorted(refresh_src)})")
    check(REFRESH_QUERY.lstrip().startswith(f"CREATE OR REPLACE TABLE `{DESTINATION_TABLE}`"),
          f"refresh query writes {DESTINATION_TABLE}")
    check(lp.SOURCE_TABLE == DESTINATION_TABLE,
          f"live pull reads the table the refresh writes ({lp.SOURCE_TABLE})")
    other = tables("cloud_function/main.py") | tables("live_pipeline.py")
    other = {t for t in other if "{" not in t}
    check(other <= {"geopolitical-risk-forecasting.predictions.forecasts_v2"},
          f"no other hard-coded input tables in main.py / live_pipeline.py (found: {sorted(other) or 'none'})")
    for script in ("build_bq_panel_22.py", "build_bq_panel_22_4w.py"):
        src = open(script, encoding="utf-8").read()
        check(f'TABLE = "{DESTINATION_TABLE}"' in src, f"training pull {script} reads {DESTINATION_TABLE}")


# ---------------------------------------------------------------- B
def query_output_columns():
    """Output column -> 'int' / 'float' / 'other', read off the production SQL."""
    weekly = REFRESH_QUERY.split("weekly_events AS (")[1].split("FROM raw_events")[0]
    types = {}
    for expr, name in re.findall(r"\n\s*(.+?)\s+AS\s+(\w+),?\s*(?=\n)", weekly):
        if name in ("week_start", "Actor1CountryCode", "Actor2CountryCode", "dyad_scope"):
            continue
        is_float = re.match(r"(AVG|STDDEV)\(", expr) or re.match(r"(MIN|MAX)\((GoldsteinScale|AvgTone)\)", expr) \
            or ("GoldsteinScale" in expr and expr.startswith("SUM"))
        types[name] = "float" if is_float else "int"
    derived = REFRESH_QUERY.split("derived AS (")[1]
    for name in re.findall(r"SAFE_DIVIDE\(.+?\)\s+AS\s+(\w+)", derived):
        types[name] = "float"
    types["top_event_root"] = "other"
    return types


def check_schema(df):
    print("\nB. Production query output vs training data columns")
    q = query_output_columns()
    q_num = {c for c, t in q.items() if t != "other"}
    keys = {"week_start", "Actor1CountryCode", "Actor2CountryCode", TARGET4_COL}
    train_num = set(df.columns) - keys
    check(q_num == train_num, f"same {len(train_num)} numeric columns "
          f"(only in query: {sorted(q_num - train_num) or 'none'}; only in training: {sorted(train_num - q_num) or 'none'})")
    mism = [c for c in q_num & train_num
            if (q[c] == "int") != str(df[c].dtype).startswith("Int")]
    check(not mism, f"integer/float typing matches for all columns (mismatches: {mism or 'none'})")
    sql_countries = set(re.findall(r"'([A-Z]{3})'", REFRESH_QUERY.split("middle_east AS")[1].split(")")[0]))
    check(sql_countries == set(MENA), f"query's MENA country list == the models' 20 countries")
    weeks = pd.DatetimeIndex(sorted(df.week_start.unique()))
    check((weeks.dayofweek == 0).all(), "every training week starts on a Monday (query: WEEK(MONDAY))")
    check(str(weeks[0].date()) == "2015-02-16" and "DATE '2015-02-16'" in REFRESH_QUERY,
          f"calendar starts 2015-02-16 in both (training first week {weeks[0].date()})")
    check(len(weeks) == (weeks[-1] - weeks[0]).days // 7 + 1, "no missing weeks in the training calendar")


# ---------------------------------------------------------------- C
def check_fingerprint(df):
    print("\nC. Production-query identities in every training row")
    df = df[~(df.Actor1CountryCode.isin(EXTRA) & df.Actor2CountryCode.isin(EXTRA))]  # zero pads
    c = lambda a, b: df[[f"count_{i:02d}" for i in range(a, b + 1)]].sum(axis=1)
    tot = df.total_count.astype(float)

    def share(num):
        return np.where(tot > 0, num / tot.where(tot > 0, 1), 0.0)

    ident = {
        "total_count = count_01 + ... + count_20": (df.total_count, c(1, 20)),
        "precursor_count = count_13..17": (df.precursor_count, c(13, 17)),
        "violent_count = count_18..20": (df.violent_count, c(18, 20)),
        "conflict_share = (verbal+material conflict) / total":
            (df.conflict_share, share(df.verbal_conflict_count + df.material_conflict_count)),
        "material_conflict_share = material conflict / total":
            (df.material_conflict_share, share(df.material_conflict_count)),
        "cooperation_share = (verbal+material coop) / total":
            (df.cooperation_share, share(df.verbal_cooperation_count + df.material_cooperation_count)),
        "violent_share = violent / total": (df.violent_share, share(df.violent_count)),
        "precursor_share = precursor / total": (df.precursor_share, share(df.precursor_count)),
        "strongly_negative_share = tone<-5 count / total": (df.strongly_negative_share, share(df.strongly_negative_count)),
        "root_event_share = root events / total": (df.root_event_share, share(df.root_event_count)),
        "mentions_per_event = mentions_total / total": (df.mentions_per_event, share(df.mentions_total)),
        "quad classes sum to total": (df.verbal_cooperation_count + df.material_cooperation_count
                                      + df.verbal_conflict_count + df.material_conflict_count, df.total_count),
    }
    for name, (a, b) in ident.items():
        frac = np.isclose(np.asarray(a, float), np.asarray(b, float), atol=1e-4).mean()
        check(frac > 0.999, f"{name}: holds in {frac:.4%} of {len(df)} rows")
    has = tot > 0
    order = ((df.goldstein_min <= df.goldstein_mean + 1e-6) & (df.goldstein_mean <= df.goldstein_max + 1e-6))[has].mean()
    check(order > 0.999, f"goldstein_min <= goldstein_mean <= goldstein_max: {order:.4%} of rows with events")
    check((df.goldstein_mean[~has] == 0).all(), "weeks with no events have goldstein_mean = 0 (null coalesced)")


# ---------------------------------------------------------------- D, E
def check_provenance(bq):
    print(f"\nD. Which SQL actually created {DESTINATION_TABLE}")
    dataset, table = DESTINATION_TABLE.split(".")[1:]
    meta = bq.get_table(DESTINATION_TABLE)
    print(f"  table location {meta.location}, created {meta.created}, last modified {meta.modified}")
    jobs = bq.query(f"""
        SELECT creation_time, user_email, query
        FROM `{PROJECT}`.`region-{meta.location.lower()}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
        WHERE destination_table.dataset_id = @d AND destination_table.table_id = @t
          AND statement_type IN ('CREATE_TABLE_AS_SELECT', 'CREATE_TABLE')
          AND state = 'DONE' AND error_result IS NULL
        ORDER BY creation_time DESC LIMIT 1""", job_config=_params(d=dataset, t=table)).to_dataframe()
    if jobs.empty:
        check(False, "no CREATE job for the table in the last 180 days of job history")
        return
    j = jobs.iloc[0]
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"--.*", "", s)).strip().rstrip(";").lower()
    live_sql = norm(REFRESH_QUERY).replace("@last_week", "?")
    hist_sql = norm(j.query)
    print(f"  last created by job at {j.creation_time} ({j.user_email})")
    check(GDELT_TABLE in j.query, f"that job read from {GDELT_TABLE}")
    same_except_cutoff = re.sub(r"date_trunc\(current_date\(\), week\(monday\)\)", "?", hist_sql)
    same_except_cutoff = re.sub(r"and e\.sqldate <= cast\(format_date\(.+?\) as int64\) ", "", same_except_cutoff)
    live_cmp = re.sub(r"and e\.sqldate <= cast\(format_date\(.+?\) as int64\) ", "", live_sql)
    check(hist_sql == live_sql or same_except_cutoff == live_cmp,
          "its SQL is the production REFRESH_QUERY (ignoring whitespace and the complete-week cutoff)")


def check_values(bq, df, run):
    print(f"\nE. Production query run directly on {GDELT_TABLE} vs training data")
    sql = REFRESH_QUERY.split(" AS\n", 1)[1]
    sql = re.sub(r"SELECT \* FROM derived\s+ORDER BY[^;]*;", "SELECT * FROM derived WHERE week_start IN UNNEST(@weeks)", sql)
    weeks = [pd.Timestamp(w).date() for w in CHECK_WEEKS]
    from google.cloud import bigquery
    params = [bigquery.ScalarQueryParameter("last_week", "DATE", max(weeks)),
              bigquery.ArrayQueryParameter("weeks", "DATE", weeks)]
    dry = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True))
    print(f"  dry run: would scan {dry.total_bytes_processed / 1e9:.0f} GB "
          f"(~${dry.total_bytes_processed / 1e12 * 6.25:.2f} on-demand, $0 within the free 1 TB/month)")
    if not run:
        print("  not run - add --yes to run it")
        return
    fresh = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    fresh["week_start"] = pd.to_datetime(fresh["week_start"])
    key = ["week_start", "Actor1CountryCode", "Actor2CountryCode"]
    train = df[df.week_start.isin(pd.to_datetime(CHECK_WEEKS))]
    train = train[~(train.Actor1CountryCode.isin(EXTRA) & train.Actor2CountryCode.isin(EXTRA))]
    m = train.merge(fresh, on=key, how="left", suffixes=("_train", "_gdelt"), indicator=True)
    check((m._merge == "both").all(), f"all {len(train)} training dyad-weeks found in the fresh GDELT result")
    for w in CHECK_WEEKS:
        mw = m[m.week_start == pd.Timestamp(w)]
        cols = ["total_count", "goldstein_mean", "tone_mean", "count_19"]
        same = {c: np.isclose(mw[f"{c}_train"].astype(float), mw[f"{c}_gdelt"].fillna(0).astype(float),
                              atol=1e-4).mean() for c in cols}
        print(f"  week {w}: identical rows - " + ", ".join(f"{c} {v:.1%}" for c, v in same.items()))
        check(min(same.values()) > 0.95, f"week {w} matches GDELT (small gaps = GDELT's own later revisions)")


def _params(**kw):
    from google.cloud import bigquery
    return bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter(k, "STRING", v)
                                                     for k, v in kw.items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="also run checks D and E against BigQuery")
    ap.add_argument("--yes", action="store_true", help="actually run check E's GDELT scan")
    args = ap.parse_args()

    df = training_frame()
    check_sources()
    check_schema(df)
    check_fingerprint(df)
    if args.online:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)
        check_provenance(bq)
        check_values(bq, df, args.yes)
    else:
        print("\n(D, E skipped - run with --online to check against BigQuery and GDELT directly)")
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
