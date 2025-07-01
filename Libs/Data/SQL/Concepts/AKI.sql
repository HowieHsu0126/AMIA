DROP TABLE IF EXISTS ICU_AKI;
CREATE TABLE ICU_AKI AS
SELECT 
    af.stay_id,
    icu.hadm_id,
    af.final_aki_status, 
    af.earliest_aki_timepoint
FROM 
    aki_final af
INNER JOIN mimiciv_icu.icustays icu 
    ON af.stay_id = icu.stay_id
WHERE 
    af.final_aki_status = 'ICU Acquired AKI';
