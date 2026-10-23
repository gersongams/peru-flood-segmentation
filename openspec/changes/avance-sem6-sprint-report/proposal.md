## Why

The course's week-6 deliverable is a sprint report: EDA, feature engineering, and an experiment
with a baseline and one or two variants. Weeks 4 (EDA) and 5 (feature engineering) are already
done and executed (`notebooks/eda.ipynb`, `notebooks/feature_engineering.ipynb`). The
baseline-plus-variants experiment also exists: it was trained, scored on `peru-dev` with 3 seeds,
and evaluated once on the test sets (`docs/EXPERIMENT_LOG.md`, `docs/phaseD_dev.md`,
`docs/phaseD_test.md`). But no notebook presents it, and no single document ties the three parts
together. The report starts as Markdown and will later be converted to LaTeX.

## What Changes

- New branch `avance-sem6-sprint` from `main`, with the same layout as `avance-sem4-eda` and
  `avance-sem5-feature-engineering`: executed Spanish notebook plus its `build_*_nb.py`, a
  write-up in `docs/`, METHODS_NOTES and EXPERIMENT_LOG entries, and a README row.
- New notebook `notebooks/baseline_variantes.ipynb` (Spanish, executed), built by
  `notebooks/build_baseline_nb.py`. It presents results that already exist; it trains nothing
  and evaluates no test split again:
  - **One experiment, with Peru labels:** baseline U-Net (ResNet-34) against variant 1
    TerraMind v1-B and variant 2 Prithvi-EO-2.0 300M, all trained on `peru-train`. The MNDWI
    threshold is a non-learned reference row. `peru-dev` per seed, plus the one-time
    `peru-test` and `peru-2017-test` scores already on record.
  - The label-free adaptation results (phase D) are out of scope for this deliverable (authors'
    decision, 2026-10-23).
- New tracked results table `notebooks/results/sprint6_results.csv`, with one row per value and its
  source (log entry, write-up table, commit). The notebook reads it. When `outputs/` is
  present, the notebook also checks the table against the run summaries.
- New sprint report `docs/informe_sprint_sem6.md` (Spanish), written so it converts easily to
  LaTeX: objectives, data, EDA, feature engineering, experiment, discussion, threats to
  validity, next steps.

## Capabilities

### New Capabilities
- `course-progress-reports`: weekly course deliverables (branch, executed notebook, write-up)
  that present project results without changing the frozen thesis comparison.

### Modified Capabilities
None.

## Impact

- New files only: notebook, builder, CSV, report, OpenSpec change. No change to `src/`,
  `configs/`, `configs/frozen.yaml` or the data.
- No training and no new evaluation. The test numbers shown are the ones from the one-time
  pass of 2026-10-16.
- Docs: new entries in `docs/METHODS_NOTES.md` and `docs/EXPERIMENT_LOG.md` (no new runs), and a
  README row under "Course deliverables".
