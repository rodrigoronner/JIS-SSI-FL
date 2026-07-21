-- extract_cohort.sql
--
-- Cohort extraction for "An Identity-First Software Architecture for Secure
-- Federated Learning in Healthcare" (MIMIC-IV v3.1).
--
-- Prerequisites:
--   1. MIMIC-IV v3.1 loaded into a local PostgreSQL instance following the
--      official build scripts: https://github.com/MIT-LCP/mimic-code
--      Schemas expected: mimiciv_hosp, mimiciv_icu.
--   2. The community "concepts" layer from mimic-code, which materializes
--      mimiciv_derived.charlson (Charlson Comorbidity Index) and
--      mimiciv_derived.icustay_detail. Build it with:
--        cd mimic-code/mimic-iv/concepts_postgres && bash postgres_make_concepts.sh
--
-- Inclusion criteria (Sec. 4.1 of the paper):
--   - Adult patients (age >= 18) at admission time.
--   - First admission per unique patient (ensures sample independence).
--   - Non-null hospital_expire_flag (in-hospital mortality outcome).
--
-- Output columns map to the paper's described feature groups:
--   Demographics          : age, gender, ethnicity, insurance
--   Clinical context      : admission_location, first_careunit,
--                            charlson_comorbidity_index
--   Physiological proxies : los_days, sepsis_flag, heart_failure_flag
--   Chronological key     : admittime (required for the 90/10 chronological
--                            split; NOT present in earlier exported CSVs
--                            used in this repository)
--   Target                : hospital_expire_flag
--
-- NOTE ON DIMENSIONALITY: the raw query below returns ~10 named columns.
-- The paper's "25 variables" refers to the dimensionality of the final
-- feature matrix AFTER one-hot encoding of the categorical columns
-- (gender, ethnicity, insurance, admission_location, first_careunit).
-- The exact final width depends on the cardinality of categories observed
-- in the extracted cohort; data_loader.py reports the realized shape after
-- encoding so this can be checked against the paper's 25-column claim.

WITH first_admission AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.admission_location,
        a.insurance,
        a.race,
        a.hospital_expire_flag,
        ROW_NUMBER() OVER (
            PARTITION BY a.subject_id ORDER BY a.admittime ASC
        ) AS admission_rank
    FROM mimiciv_hosp.admissions a
    WHERE a.hospital_expire_flag IS NOT NULL
),
cohort AS (
    SELECT
        fa.subject_id,
        fa.hadm_id,
        fa.admittime,
        fa.dischtime,
        fa.admission_location,
        fa.insurance,
        fa.race,
        fa.hospital_expire_flag,
        p.gender,
        p.anchor_age
            + (EXTRACT(YEAR FROM fa.admittime) - p.anchor_year) AS age
    FROM first_admission fa
    JOIN mimiciv_hosp.patients p
        ON p.subject_id = fa.subject_id
    WHERE fa.admission_rank = 1
),
icu_first AS (
    -- First ICU stay within the admission (paper: "first care unit").
    SELECT DISTINCT ON (icu.hadm_id)
        icu.hadm_id,
        icu.first_careunit,
        icu.los AS icu_los_days
    FROM mimiciv_icu.icustays icu
    ORDER BY icu.hadm_id, icu.intime ASC
),
dx_flags AS (
    -- ICD-10 groupings for sepsis (A40-A41) and heart failure (I50).
    SELECT
        d.hadm_id,
        MAX(CASE WHEN d.icd_version = 10 AND LEFT(d.icd_code, 3) IN ('A40', 'A41')
                 THEN 1 ELSE 0 END) AS sepsis_flag,
        MAX(CASE WHEN d.icd_version = 10 AND LEFT(d.icd_code, 3) = 'I50'
                 THEN 1 ELSE 0 END) AS heart_failure_flag,
        COUNT(DISTINCT d.icd_code) AS num_diagnoses
    FROM mimiciv_hosp.diagnoses_icd d
    GROUP BY d.hadm_id
),
procedures_ct AS (
    SELECT
        pr.hadm_id,
        COUNT(DISTINCT pr.icd_code) AS num_procedures
    FROM mimiciv_hosp.procedures_icd pr
    GROUP BY pr.hadm_id
)
SELECT
    c.hadm_id,
    c.admittime,                                   -- chronological split key
    c.age,
    c.gender,
    c.race                             AS ethnicity,
    c.insurance,
    c.admission_location,
    COALESCE(icu.first_careunit, 'NONE')            AS first_careunit,
    ch.charlson_comorbidity_index,
    COALESCE(
        EXTRACT(EPOCH FROM (c.dischtime - c.admittime)) / 86400.0,
        icu.icu_los_days
    )                                                AS los_days,
    COALESCE(dx.sepsis_flag, 0)                     AS sepsis_flag,
    COALESCE(dx.heart_failure_flag, 0)               AS heart_failure_flag,
    COALESCE(dx.num_diagnoses, 0)                    AS num_diagnoses,
    COALESCE(pc.num_procedures, 0)                   AS num_procedures,
    c.hospital_expire_flag
FROM cohort c
LEFT JOIN icu_first icu           ON icu.hadm_id = c.hadm_id
LEFT JOIN dx_flags dx             ON dx.hadm_id = c.hadm_id
LEFT JOIN procedures_ct pc        ON pc.hadm_id = c.hadm_id
LEFT JOIN mimiciv_derived.charlson ch ON ch.hadm_id = c.hadm_id
WHERE c.age >= 18
ORDER BY c.admittime ASC;

-- Export from psql, e.g.:
--   \copy (⟨query above⟩) TO 'data/mortalidade_features.csv' WITH CSV HEADER
