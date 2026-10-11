-- GDELT: 21国(20 MENA + USA) × 月 冲突指数
-- ActionGeo_CountryCode 是 FIPS 码，下面的映射请对照 GDELT codebook 核一遍
WITH base AS (
  SELECT
    DIV(SQLDATE, 100) AS ym,                      -- 例如 202603
    CASE ActionGeo_CountryCode
      WHEN 'AE' THEN 'ARE' WHEN 'BA' THEN 'BHR' WHEN 'CY' THEN 'CYP'
      WHEN 'DJ' THEN 'DJI' WHEN 'EG' THEN 'EGY' WHEN 'IR' THEN 'IRN'
      WHEN 'IZ' THEN 'IRQ' WHEN 'IS' THEN 'ISR' WHEN 'JO' THEN 'JOR'
      WHEN 'KU' THEN 'KWT' WHEN 'LE' THEN 'LBN' WHEN 'LY' THEN 'LBY'
      WHEN 'MU' THEN 'OMN' WHEN 'WE' THEN 'PSE' WHEN 'GZ' THEN 'PSE'
      WHEN 'QA' THEN 'QAT' WHEN 'SA' THEN 'SAU' WHEN 'SU' THEN 'SDN'
      WHEN 'SY' THEN 'SYR' WHEN 'TU' THEN 'TUR' WHEN 'YM' THEN 'YEM'
      WHEN 'US' THEN 'USA'
    END AS iso3,
    EventRootCode, QuadClass, GoldsteinScale, NumMentions
  FROM `gdelt-bq.gdeltv2.events`
  WHERE SQLDATE >= 20170101
    AND ActionGeo_CountryCode IN ('AE','BA','CY','DJ','EG','IR','IZ','IS','JO','KU','LE','LY',
                                  'MU','WE','GZ','QA','SA','SU','SY','TU','YM','US')
)
SELECT
  iso3, ym,
  COUNT(*)                                        AS n_events,
  COUNTIF(QuadClass = 4)                          AS n_material_conflict,
  COUNTIF(EventRootCode IN ('15','18','19','20')) AS n_t2_t4,       -- 军事姿态/袭击/交战/大规模暴力
  SUM(IF(QuadClass = 4, NumMentions, 0))          AS mentions_conflict,
  ROUND(AVG(GoldsteinScale), 3)                   AS avg_goldstein,
  ROUND(SAFE_DIVIDE(COUNTIF(QuadClass = 4), COUNT(*)), 4) AS conflict_share
FROM base
WHERE iso3 IS NOT NULL
GROUP BY iso3, ym
ORDER BY iso3, ym;