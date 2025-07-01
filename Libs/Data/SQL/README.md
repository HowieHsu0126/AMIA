# SQL Extraction Framework for AMIA Project

This directory contains all PostgreSQL scripts used to curate **MIMIC-IV** data for the AMIA research project. The queries transform the raw critical-care database into analysis-ready tables and CSV exports that cover ICU-acquired acute kidney injury (ICU-AKI), SOFA scores, laboratory concepts, time-series features, and treatment information.

---

## 1. Directory Layout

| Path           | Purpose                                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| `function.sql` | Utility UDFs that emulate BigQuery/Standard SQL date-time and regex helpers in PostgreSQL. **Run this first**. |
| `run.sql`      | Master script that orchestrates the end-to-end pipeline by `\i`-including the individual modules below.        |
| `Concepts/`    | Concept-level views or tables (e.g., KDIGO AKI criteria, SOFA components, vitals, labs).                       |
| `Timeseries/`  | Hourly aggregation scripts that export per-admission time-series to CSV.                                       |
| `Medication/`  | Drug-specific extraction queries (e.g., vasopressors, antibiotics).                                            |
| `Treatments/`  | Procedures such as renal-replacement therapy (RRT).                                                            |
| `main.sql`     | Convenience wrapper that sequentially includes all `Timeseries` scripts.                                       |

> All paths inside the scripts are absolute on the author's system. **Edit them to match your environment** before execution.

---

## 2. Prerequisites

1. **PostgreSQL ≥ 12** with the [`mimic-iv`](https://mimic.mit.edu/) schemas loaded (`mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`, etc.).
2. Super-user or a role that can create functions, tables, and copy files to the local filesystem.
3. Sufficient disk space to store the derived tables and exported CSV files.
4. `psql` client or any tool capable of executing `\i` meta-commands.

---

## 3. Quick-Start

```bash
# 1) Enter psql connected to your MIMIC-IV database
psql -U postgres -d mimic

-- 2) Load helper functions (run once)
\i 'Libs/Data/SQL/function.sql'

-- 3) Execute the complete pipeline
\i 'Libs/Data/SQL/run.sql'
```

The `run.sql` script will:

- Compute KDIGO-based AKI status (`kdigo_*` scripts) and write intermediary tables.
- Generate SOFA components and 24-hour rolling scores.
- Produce demographic features (weight, height) and lab concepts.
- Export curated time-series to CSV under the `Input/raw/` folder.

Execution time depends on hardware; expect **tens of minutes** on a standard workstation.

---

## 4. Detailed Module Description

### 4.1 KDIGO / AKI Detection (`Concepts/kdigo_*.sql`)

Implements creatinine- and urine-output–based KDIGO criteria, merges them with RRT information, and labels each ICU stay as **ICU-acquired AKI** or not.

### 4.2 SOFA Score (`Concepts/SOFA.sql`)

Calculates per-hour SOFA organ subscores (respiration, coagulation, liver, cardiovascular, CNS, renal) and the rolling 24-hour maximum.

### 4.3 Laboratory & Vital Concepts

- `BG.sql`, `Chemistry.sql`, `CBC.sql`, `Creatinine.sql`, etc. pivot raw `labevents` and `chartevents` into tidy analytic tables.
- `VitalSign.sql` aggregates heart-rate, blood-pressure, respiratory-rate, temperature, etc.

### 4.4 Time-Series Exports (`Timeseries/*.sql`)

Each script focuses on a clinical domain (e.g., _Acidosis_, _Hypoglycemia_, _Coagulation_) and writes an **hourly averaged** CSV to `Input/raw/`. Modify the `COPY ... TO` path to suit your storage.

### 4.5 Medications & Treatments

Vasoactive infusions, antibiotics, dialysis modalities, and other interventions are captured in `Medication/` and `Treatments/`.

---

## 5. Customisation Tips

- **Output Paths** – Change the absolute `/home/.../Input/raw/` destinations or switch to `COPY ... TO PROGRAM 'gzip > file.csv.gz' WITH (FORMAT CSV)` if storage is an issue.
- **Schema Names** – If your MIMIC-IV schemas differ, adjust the `search_path` or fully qualified names inside the scripts.
- **Filtering Cohorts** – Narrow to specific diagnosis codes or study periods by editing the `WHERE` clauses in the concept queries.

---

## 6. Troubleshooting

| Symptom                          | Likely Cause                             | Fix                                                             |
| -------------------------------- | ---------------------------------------- | --------------------------------------------------------------- |
| _function does not exist_        | `function.sql` not loaded                | `\i function.sql` in your session.                              |
| _permission denied for relation_ | Insufficient privileges to create tables | Connect as a role with `CREATE` rights or use a scratch schema. |
| Long runtimes                    | Missing indices on `charttime`/`itemid`  | Consider adding temporary indices for iterative development.    |

---

## 7. Citation

If you use this pipeline in an academic work, please cite my open-source PostgreSQL collections [AwesomeEHR](git@github.com:HowieHsu0126/AwesomeEHR.git) and acknowledge this repository.

---

## 8. License

The SQL in this directory is released under the **MIT License** unless stated otherwise. See the project root `LICENSE` file for details.
