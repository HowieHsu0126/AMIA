CREATE OR REPLACE FUNCTION DATETIME_SUB(datetime_val TIMESTAMP(3), intvl INTERVAL) RETURNS TIMESTAMP(3) AS $$
BEGIN
RETURN datetime_val - intvl;
END; $$
LANGUAGE PLPGSQL;

DROP TABLE IF EXISTS aki_cr;
CREATE TABLE aki_cr AS
-- Extract all creatinine values from labevents within a 90-day window around patient's ICU stay
WITH cr AS (
    SELECT
        ie.hadm_id
        , ie.stay_id
        , le.charttime
        , le.valuenum AS creat
    FROM mimiciv_icu.icustays ie
    LEFT JOIN mimiciv_hosp.labevents le
        ON ie.subject_id = le.subject_id
            AND le.itemid = 50912
            AND le.valuenum IS NOT NULL
            AND le.valuenum <= 150
            AND le.charttime >= DATETIME_SUB(ie.intime, INTERVAL '90' DAY)
            AND le.charttime <= ie.outtime
)

, cr_avg AS (
    -- Calculate the average creatinine value for each patient over the 90-day window
    SELECT
        hadm_id
        , AVG(creat) AS avg_creat
    FROM cr
    GROUP BY hadm_id
)

, cr_baseline AS (
    -- Select the creatinine value closest to the average for each patient as the baseline
    SELECT
        cr.hadm_id
        , cr.stay_id
        , cr.charttime
        , cr.creat
        , ABS(cr.creat - cr_avg.avg_creat) AS diff
    FROM cr
    INNER JOIN cr_avg
        ON cr.hadm_id = cr_avg.hadm_id
)

, ranked_baseline AS (
    -- Rank the creatinine values by their closeness to the average
    SELECT
        hadm_id
        , stay_id
        , charttime AS baseline_time
        , creat AS baseline_creat
        , ROW_NUMBER() OVER(PARTITION BY hadm_id ORDER BY diff ASC) AS rank
    FROM cr_baseline
)

-- 按照肌酐值的降序为每个病人的每个肌酐记录分配一个行号。然后，我们从这些记录中选出行号为1的记录，即每个病人在48小时或7天内肌酐值最高的那条记录及其对应的时间点。
, cr_48hr_max AS (
    -- Determine the highest creatinine value within 48 hours after ICU admission and its timepoint
    SELECT
        hadm_id
        , stay_id
        , creat AS max_creat_48hr
        , charttime AS max_creat_48hr_time
    FROM (
        SELECT
            cr.hadm_id
            , cr.stay_id
            , cr.creat
            , cr.charttime
            , ROW_NUMBER() OVER (PARTITION BY cr.hadm_id ORDER BY cr.creat DESC) AS rn
        FROM cr
        JOIN ranked_baseline rb
            ON cr.hadm_id = rb.hadm_id
            AND cr.charttime BETWEEN rb.baseline_time AND DATETIME_SUB(rb.baseline_time, INTERVAL '-48' HOUR)
    ) sub
    WHERE rn = 1
)

, cr_7day_max AS (
    -- Determine the highest creatinine value within 7 days after ICU admission and its timepoint
    SELECT
        hadm_id
        , stay_id
        , creat AS max_creat_7day
        , charttime AS max_creat_7day_time
    FROM (
        SELECT
            cr.hadm_id
            , cr.stay_id
            , cr.creat
            , cr.charttime
            , ROW_NUMBER() OVER (PARTITION BY cr.hadm_id ORDER BY cr.creat DESC) AS rn
        FROM cr
        JOIN ranked_baseline rb
            ON cr.hadm_id = rb.hadm_id
            AND cr.charttime BETWEEN rb.baseline_time AND DATETIME_SUB(rb.baseline_time, INTERVAL '-7' DAY)
    ) sub
    WHERE rn = 1
)


, aki_status AS (
    -- Determine AKI status and timepoint based on KDIGO criteria
    SELECT
        rb.hadm_id
        , rb.stay_id
        , rb.baseline_time
        , rb.baseline_creat
        , c48.max_creat_48hr
        , c48.max_creat_48hr_time
        , c7.max_creat_7day
        , c7.max_creat_7day_time
        , CASE
            WHEN c48.max_creat_48hr >= rb.baseline_creat + 0.3 THEN 'AKI within 48hr'
            WHEN c48.max_creat_48hr >= 1.5 * rb.baseline_creat THEN 'AKI within 48hr'
            WHEN c7.max_creat_7day >= 1.5 * rb.baseline_creat THEN 'AKI within 7day'
            ELSE 'No AKI'
          END AS aki_status
        , CASE
            WHEN c48.max_creat_48hr >= rb.baseline_creat + 0.3 OR c48.max_creat_48hr >= 1.5 * rb.baseline_creat THEN c48.max_creat_48hr_time
            WHEN c7.max_creat_7day >= 1.5 * rb.baseline_creat THEN c7.max_creat_7day_time
          END AS aki_timepoint -- 通过比较两个时间点max_creat_48hr_time（在48小时内肌酐值达到最高的时间点）和max_creat_7day_time（在7天内肌酐值达到最高的时间点）来确定的。这个字段将反映出根据KDIGO标准判断为AKI的第一个时间点。
    FROM ranked_baseline rb
    LEFT JOIN cr_48hr_max c48
        ON rb.hadm_id = c48.hadm_id
    LEFT JOIN cr_7day_max c7
        ON rb.hadm_id = c7.hadm_id
    WHERE rb.rank = 1
)

SELECT *
FROM aki_status;