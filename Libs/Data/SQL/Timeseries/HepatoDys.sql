COPY (
WITH lab_events AS (
    SELECT 
        le.subject_id, 
        le.hadm_id, 
        le.charttime, 
        le.itemid, 
        le.valuenum,
        ha.admittime,
        aki.earliest_aki_timepoint AS aki_time
    FROM mimiciv_hosp.labevents le
    INNER JOIN mimiciv_hosp.admissions ha ON le.hadm_id = ha.hadm_id
    INNER JOIN icu_aki aki ON ha.hadm_id = aki.hadm_id
    WHERE le.itemid IN (50861, 50878)
      AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN ha.admittime AND aki_time
),
hourly_averages AS (
    SELECT
        subject_id,
        hadm_id,
        FLOOR(EXTRACT(epoch FROM (charttime - admittime))/3600) + 1 AS hour_interval,
        itemid,
        AVG(valuenum) AS average_valuenum
    FROM lab_events
    GROUP BY subject_id, hadm_id, itemid, hour_interval
)
SELECT
    hadm_id AS admission_id,
    hour_interval AS hour,
    AVG(CASE WHEN itemid = 50861 THEN average_valuenum ELSE NULL END) AS AST,
    AVG(CASE WHEN itemid = 50878 THEN average_valuenum ELSE NULL END) AS ALT
    -- MAX(CASE 
    --         WHEN itemid = 50861 AND average_valuenum > 40 THEN 1
    --         WHEN itemid = 50878 AND average_valuenum > 40 THEN 1
    --         ELSE 0 END) AS HepatoDys
FROM hourly_averages
GROUP BY hadm_id, subject_id, hour_interval
ORDER BY hadm_id, subject_id, hour_interval
) TO :'DATA_DIR'/HepatoDys.csv WITH CSV HEADER;
