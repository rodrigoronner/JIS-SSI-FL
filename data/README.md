# data/

This directory is intentionally empty in the repository. MIMIC-IV is
credentialed-access data (PhysioNet Data Use Agreement) and must never be
committed to a public repository, so no dataset ships with this codebase.

## Generating `mortalidade_features.csv`

1. Obtain credentialed access to MIMIC-IV v3.1 via
   [PhysioNet](https://physionet.org/content/mimiciv/) and load it into a
   local PostgreSQL instance using the official
   [mimic-code](https://github.com/MIT-LCP/mimic-code) build scripts.
2. Build the community "concepts" layer (`mimic-iv/concepts_postgres`),
   which materializes `mimiciv_derived.charlson` — required by the
   extraction query below for the Charlson Comorbidity Index.
3. Run [`../sql/extract_cohort.sql`](../sql/extract_cohort.sql) against
   that database and export the result here:

   ```bash
   psql -d mimiciv -f ../sql/extract_cohort.sql \
     -c "\copy (SELECT * FROM cohort_export) TO 'data/mortalidade_features.csv' WITH CSV HEADER"
   ```

   (or run the query interactively in `psql`/pgAdmin and use `\copy` to
   export — see the comment at the bottom of `extract_cohort.sql`).

4. Verify the extraction:

   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/mortalidade_features.csv'); print(df.shape)"
   ```

The query enforces the paper's inclusion criteria (adult patients, first
admission per subject, non-null `hospital_expire_flag`) and includes
`admittime`, which `src/data_loader.py` requires for the chronological
90/10 split described in the paper (Sec. 4.1). Any CSV placed in this
folder is covered by `.gitignore` and will never be committed.
