"""
Rebuilds geopolitical-risk-forecasting.1.1 from GDELT's public BigQuery
dataset (gdelt-bq.gdeltv2.events, updated by GDELT every 15 minutes), using
the same aggregation query that originally built the table the models were
trained on. CREATE OR REPLACE, so each run is a full, clean rebuild (which
also picks up GDELT's late additions and corrections to earlier weeks).

COMPLETE WEEKS ONLY: the table ends at @last_week, the Monday of the last
fully finished Monday..Sunday week (live_pipeline.last_complete_week). Events
dated after that week's Sunday are excluded, so the week in progress never
appears as a half-empty week.

Cost: gdelt-bq.gdeltv2.events is not partitioned, so every rebuild scans the
selected columns of the whole table regardless of the date filter - which is
why the pipeline only refreshes every 2 weeks (stays inside BigQuery's free
1 TB/month query tier).
"""

from google.cloud import bigquery

DESTINATION_TABLE = "geopolitical-risk-forecasting.1.1"

REFRESH_QUERY = f"""
CREATE OR REPLACE TABLE `{DESTINATION_TABLE}` AS

WITH

middle_east AS (
  SELECT country_code
  FROM UNNEST([
    'BHR','CYP','EGY','IRN','IRQ','ISR','JOR','KWT','LBN','OMN',
    'PSE','QAT','SAU','SYR','TUR','ARE','YEM','LBY','SDN','DJI'
  ]) AS country_code
),

raw_events AS (
  SELECT
    DATE_TRUNC(PARSE_DATE('%Y%m%d', CAST(e.SQLDATE AS STRING)), WEEK(MONDAY)) AS week_start,
    e.Actor1CountryCode,
    e.Actor2CountryCode,
    e.EventRootCode,
    e.QuadClass,
    e.GoldsteinScale,
    e.AvgTone,
    e.NumMentions,
    e.NumSources,
    e.NumArticles,
    e.IsRootEvent,
    e.Actor1Type1Code, e.Actor1Type2Code, e.Actor1Type3Code,
    e.Actor2Type1Code, e.Actor2Type2Code, e.Actor2Type3Code,
    e.Actor1KnownGroupCode, e.Actor2KnownGroupCode,
    e.ActionGeo_CountryCode, e.ActionGeo_ADM1Code, e.ActionGeo_Lat, e.ActionGeo_Long,
    CASE
      WHEN e.Actor1CountryCode IN ('BHR','CYP','EGY','IRN','IRQ','ISR','JOR','KWT','LBN','OMN','PSE','QAT','SAU','SYR','TUR','ARE','YEM','LBY','SDN','DJI')
       AND e.Actor2CountryCode IN ('BHR','CYP','EGY','IRN','IRQ','ISR','JOR','KWT','LBN','OMN','PSE','QAT','SAU','SYR','TUR','ARE','YEM','LBY','SDN','DJI')
      THEN 'INTRA_REGION'
      ELSE 'EXTERNAL_LINK'
    END AS dyad_scope
  FROM `gdelt-bq.gdeltv2.events` e
  WHERE e.SQLDATE >= 20150219
    AND e.SQLDATE <= CAST(FORMAT_DATE('%Y%m%d', DATE_ADD(@last_week, INTERVAL 6 DAY)) AS INT64)
    AND e.EventRootCode IS NOT NULL
    AND (
      e.Actor1CountryCode IN ('BHR','CYP','EGY','IRN','IRQ','ISR','JOR','KWT','LBN','OMN','PSE','QAT','SAU','SYR','TUR','ARE','YEM','LBY','SDN','DJI')
      OR e.Actor2CountryCode IN ('BHR','CYP','EGY','IRN','IRQ','ISR','JOR','KWT','LBN','OMN','PSE','QAT','SAU','SYR','TUR','ARE','YEM','LBY','SDN','DJI')
    )
),

observed_dyads AS (
  SELECT DISTINCT Actor1CountryCode, Actor2CountryCode, dyad_scope
  FROM raw_events
  WHERE Actor1CountryCode IS NOT NULL AND Actor2CountryCode IS NOT NULL
),

calendar AS (
  SELECT week_start
  FROM UNNEST(GENERATE_DATE_ARRAY(
    DATE '2015-02-16', @last_week, INTERVAL 7 DAY
  )) AS week_start
),

spine AS (
  SELECT c.week_start, d.Actor1CountryCode, d.Actor2CountryCode, d.dyad_scope
  FROM calendar c
  CROSS JOIN observed_dyads d
),

weekly_events AS (
  SELECT
    week_start, Actor1CountryCode, Actor2CountryCode, dyad_scope,

    COUNTIF(EventRootCode = '01') AS count_01,
    COUNTIF(EventRootCode = '02') AS count_02,
    COUNTIF(EventRootCode = '03') AS count_03,
    COUNTIF(EventRootCode = '04') AS count_04,
    COUNTIF(EventRootCode = '05') AS count_05,
    COUNTIF(EventRootCode = '06') AS count_06,
    COUNTIF(EventRootCode = '07') AS count_07,
    COUNTIF(EventRootCode = '08') AS count_08,
    COUNTIF(EventRootCode = '09') AS count_09,
    COUNTIF(EventRootCode = '10') AS count_10,
    COUNTIF(EventRootCode = '11') AS count_11,
    COUNTIF(EventRootCode = '12') AS count_12,
    COUNTIF(EventRootCode = '13') AS count_13,
    COUNTIF(EventRootCode = '14') AS count_14,
    COUNTIF(EventRootCode = '15') AS count_15,
    COUNTIF(EventRootCode = '16') AS count_16,
    COUNTIF(EventRootCode = '17') AS count_17,
    COUNTIF(EventRootCode = '18') AS count_18,
    COUNTIF(EventRootCode = '19') AS count_19,
    COUNTIF(EventRootCode = '20') AS count_20,
    COUNT(*) AS total_count,

    COUNTIF(QuadClass = 1) AS verbal_cooperation_count,
    COUNTIF(QuadClass = 2) AS material_cooperation_count,
    COUNTIF(QuadClass = 3) AS verbal_conflict_count,
    COUNTIF(QuadClass = 4) AS material_conflict_count,

    COUNTIF(EventRootCode IN ('13','14','15','16','17')) AS precursor_count,
    COUNTIF(EventRootCode IN ('18','19','20')) AS violent_count,

    AVG(GoldsteinScale) AS goldstein_mean,
    MIN(GoldsteinScale) AS goldstein_min,
    MAX(GoldsteinScale) AS goldstein_max,
    STDDEV(GoldsteinScale) AS goldstein_std,
    SUM(CASE WHEN GoldsteinScale < 0 THEN ABS(GoldsteinScale) ELSE 0 END) AS negative_goldstein_intensity,

    AVG(AvgTone) AS tone_mean,
    MIN(AvgTone) AS tone_min,
    MAX(AvgTone) AS tone_max,
    STDDEV(AvgTone) AS tone_std,
    COUNTIF(AvgTone < -5) AS strongly_negative_count,

    SUM(NumMentions) AS mentions_total,
    AVG(NumMentions) AS mentions_mean,
    MAX(NumMentions) AS mentions_max,
    SUM(NumSources) AS sources_total,
    AVG(NumSources) AS sources_mean,
    MAX(NumSources) AS sources_max,
    SUM(NumArticles) AS articles_total,
    AVG(NumArticles) AS articles_mean,
    MAX(NumArticles) AS articles_max,

    COUNTIF(IsRootEvent = 1) AS root_event_count,

    COUNTIF(Actor1Type1Code IS NOT NULL) AS actor1_typed_event_count,
    COUNTIF(Actor2Type1Code IS NOT NULL) AS actor2_typed_event_count,
    COUNT(DISTINCT Actor1Type1Code) AS actor1_type1_diversity,
    COUNT(DISTINCT Actor2Type1Code) AS actor2_type1_diversity,
    COUNT(DISTINCT Actor1KnownGroupCode) AS actor1_group_diversity,
    COUNT(DISTINCT Actor2KnownGroupCode) AS actor2_group_diversity,

    COUNTIF(ActionGeo_CountryCode IS NOT NULL) AS geocoded_event_count,
    COUNT(DISTINCT ActionGeo_CountryCode) AS unique_action_countries,
    COUNT(DISTINCT ActionGeo_ADM1Code) AS unique_action_admin1s,
    AVG(ActionGeo_Lat) AS action_lat_mean,
    AVG(ActionGeo_Long) AS action_long_mean

  FROM raw_events
  WHERE Actor1CountryCode IS NOT NULL AND Actor2CountryCode IS NOT NULL
  GROUP BY week_start, Actor1CountryCode, Actor2CountryCode, dyad_scope
),

panel AS (
  SELECT
    s.week_start, s.Actor1CountryCode, s.Actor2CountryCode, s.dyad_scope,

    COALESCE(e.count_01, 0) AS count_01, COALESCE(e.count_02, 0) AS count_02,
    COALESCE(e.count_03, 0) AS count_03, COALESCE(e.count_04, 0) AS count_04,
    COALESCE(e.count_05, 0) AS count_05, COALESCE(e.count_06, 0) AS count_06,
    COALESCE(e.count_07, 0) AS count_07, COALESCE(e.count_08, 0) AS count_08,
    COALESCE(e.count_09, 0) AS count_09, COALESCE(e.count_10, 0) AS count_10,
    COALESCE(e.count_11, 0) AS count_11, COALESCE(e.count_12, 0) AS count_12,
    COALESCE(e.count_13, 0) AS count_13, COALESCE(e.count_14, 0) AS count_14,
    COALESCE(e.count_15, 0) AS count_15, COALESCE(e.count_16, 0) AS count_16,
    COALESCE(e.count_17, 0) AS count_17, COALESCE(e.count_18, 0) AS count_18,
    COALESCE(e.count_19, 0) AS count_19, COALESCE(e.count_20, 0) AS count_20,
    COALESCE(e.total_count, 0) AS total_count,

    COALESCE(e.verbal_cooperation_count, 0) AS verbal_cooperation_count,
    COALESCE(e.material_cooperation_count, 0) AS material_cooperation_count,
    COALESCE(e.verbal_conflict_count, 0) AS verbal_conflict_count,
    COALESCE(e.material_conflict_count, 0) AS material_conflict_count,

    COALESCE(e.precursor_count, 0) AS precursor_count,
    COALESCE(e.violent_count, 0) AS violent_count,

    e.goldstein_mean, e.goldstein_min, e.goldstein_max, e.goldstein_std,
    COALESCE(e.negative_goldstein_intensity, 0) AS negative_goldstein_intensity,

    e.tone_mean, e.tone_min, e.tone_max, e.tone_std,
    COALESCE(e.strongly_negative_count, 0) AS strongly_negative_count,

    COALESCE(e.mentions_total, 0) AS mentions_total, e.mentions_mean, e.mentions_max,
    COALESCE(e.sources_total, 0) AS sources_total, e.sources_mean, e.sources_max,
    COALESCE(e.articles_total, 0) AS articles_total, e.articles_mean, e.articles_max,

    COALESCE(e.root_event_count, 0) AS root_event_count,

    COALESCE(e.actor1_typed_event_count, 0) AS actor1_typed_event_count,
    COALESCE(e.actor2_typed_event_count, 0) AS actor2_typed_event_count,
    COALESCE(e.actor1_type1_diversity, 0) AS actor1_type1_diversity,
    COALESCE(e.actor2_type1_diversity, 0) AS actor2_type1_diversity,
    COALESCE(e.actor1_group_diversity, 0) AS actor1_group_diversity,
    COALESCE(e.actor2_group_diversity, 0) AS actor2_group_diversity,

    COALESCE(e.geocoded_event_count, 0) AS geocoded_event_count,
    COALESCE(e.unique_action_countries, 0) AS unique_action_countries,
    COALESCE(e.unique_action_admin1s, 0) AS unique_action_admin1s,
    e.action_lat_mean, e.action_long_mean

  FROM spine s
  LEFT JOIN weekly_events e
    ON s.week_start = e.week_start
   AND s.Actor1CountryCode = e.Actor1CountryCode
   AND s.Actor2CountryCode = e.Actor2CountryCode
   AND s.dyad_scope = e.dyad_scope
),

derived AS (
  SELECT
    *,
    SAFE_DIVIDE(verbal_conflict_count + material_conflict_count, total_count) AS conflict_share,
    SAFE_DIVIDE(material_conflict_count, total_count) AS material_conflict_share,
    SAFE_DIVIDE(verbal_cooperation_count + material_cooperation_count, total_count) AS cooperation_share,
    SAFE_DIVIDE(precursor_count, total_count) AS precursor_share,
    SAFE_DIVIDE(violent_count, total_count) AS violent_share,
    SAFE_DIVIDE(strongly_negative_count, total_count) AS strongly_negative_share,
    SAFE_DIVIDE(root_event_count, total_count) AS root_event_share,
    SAFE_DIVIDE(mentions_total, total_count) AS mentions_per_event,
    SAFE_DIVIDE(sources_total, total_count) AS sources_per_event,
    SAFE_DIVIDE(articles_total, total_count) AS articles_per_event,

    CASE
      WHEN total_count = 0 THEN NULL
      ELSE (
        SELECT code FROM UNNEST([
          STRUCT('01 MAKE PUBLIC STATEMENT' AS code, count_01 AS cnt),
          STRUCT('02 APPEAL', count_02),
          STRUCT('03 EXPRESS INTENT TO COOPERATE', count_03),
          STRUCT('04 CONSULT', count_04),
          STRUCT('05 ENGAGE IN DIPLOMATIC COOPERATION', count_05),
          STRUCT('06 ENGAGE IN MATERIAL COOPERATION', count_06),
          STRUCT('07 PROVIDE AID', count_07),
          STRUCT('08 YIELD', count_08),
          STRUCT('09 INVESTIGATE', count_09),
          STRUCT('10 DEMAND', count_10),
          STRUCT('11 DISAPPROVE', count_11),
          STRUCT('12 REJECT', count_12),
          STRUCT('13 THREATEN', count_13),
          STRUCT('14 PROTEST', count_14),
          STRUCT('15 EXHIBIT FORCE POSTURE', count_15),
          STRUCT('16 REDUCE RELATIONS', count_16),
          STRUCT('17 COERCE', count_17),
          STRUCT('18 ASSAULT', count_18),
          STRUCT('19 FIGHT', count_19),
          STRUCT('20 UNCONVENTIONAL MASS VIOLENCE', count_20)
        ])
        ORDER BY cnt DESC, code ASC
        LIMIT 1
      )
    END AS top_event_root

  FROM panel
)

SELECT * FROM derived
ORDER BY week_start, Actor1CountryCode, Actor2CountryCode;
"""


def refresh_gdelt_table(bq_client: bigquery.Client, last_week):
    """Full rebuild through the week starting `last_week` (a Monday, date).
    Takes a few minutes; returns (rows in the table, bytes billed)."""
    job = bq_client.query(REFRESH_QUERY, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("last_week", "DATE", last_week),
    ]))
    job.result()  # block until done
    return bq_client.get_table(DESTINATION_TABLE).num_rows, job.total_bytes_billed
