COPY (
WITH lab_events AS (
    SELECT 
        ce.subject_id, 
        ce.hadm_id, 
        ce.charttime, 
        ce.itemid, 
        ce.valuenum,
        ha.admittime,
        aki.earliest_aki_timepoint AS aki_time
    FROM mimiciv_icu.chartevents ce
    INNER JOIN mimiciv_hosp.admissions ha ON ce.hadm_id = ha.hadm_id
    INNER JOIN icu_aki aki ON ha.hadm_id = aki.hadm_id
    WHERE ce.itemid IN (220277, 223835)
      AND ce.valuenum IS NOT NULL
      AND ce.charttime BETWEEN ha.admittime AND aki_time
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
    AVG(CASE WHEN itemid = 220277 THEN average_valuenum ELSE NULL END) AS SPO2,
    AVG(CASE WHEN itemid = 223835 THEN average_valuenum ELSE NULL END) AS FiO2
    -- MAX(CASE 
    --         WHEN itemid = 220277 AND average_valuenum < 92 THEN 1
    --         WHEN itemid = 223835 AND average_valuenum > 21 THEN 1
    --         ELSE 0 END) AS O2DiffDys
FROM hourly_averages
GROUP BY hadm_id, subject_id, hour_interval
ORDER BY hadm_id, subject_id, hour_interval
) TO :'DATA_DIR'/O2DiffDys.csv WITH CSV HEADER;

