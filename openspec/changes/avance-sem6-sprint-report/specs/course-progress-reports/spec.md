# Spec Delta

## Purpose

Weekly course deliverables (EDA, feature engineering, sprint reports) that present project
results in Spanish, traceable to their sources, without changing the frozen thesis comparison.

## ADDED Requirements

### Requirement: One branch per weekly deliverable
Each weekly deliverable SHALL live on its own branch named `avance-sem<N>-<topic>`, created from
`main`. It SHALL contain the executed Spanish notebook, the script that builds it, a write-up in
`docs/`, entries in `docs/METHODS_NOTES.md` and `docs/EXPERIMENT_LOG.md`, and a README row under
"Course deliverables".

#### Scenario: Week-6 branch layout
- **WHEN** the week-6 branch `avance-sem6-sprint` is checked out
- **THEN** it contains `notebooks/baseline_variantes.ipynb` with outputs, `notebooks/build_baseline_nb.py`, `docs/informe_sprint_sem6.md`, the two log entries and a README row for week 6

### Requirement: Results notebook reports existing results only
The baseline-and-variants notebook SHALL present results already on record. It SHALL NOT train
a model, and it SHALL NOT read any test-split image or label. Test scores SHALL come only from
the one-time evaluation on record.

#### Scenario: Notebook runs without outputs
- **WHEN** the notebook is executed on a machine without `outputs/`
- **THEN** it builds every table and figure from the tracked results table and reads no file under `data/` that belongs to a test split

#### Scenario: Notebook runs with outputs
- **WHEN** `outputs/phaseD/dev_table_runs.csv` exists
- **THEN** the notebook compares the per-seed dev values of the tracked table with those files and reports any difference larger than 1e-4

### Requirement: Every reported number has a source
Every value in the tracked results table SHALL carry a source: the `docs/EXPERIMENT_LOG.md`
entry or write-up table it comes from, and the commit of the run where the log records one.
The notebook and the sprint report SHALL use no number that is not in that table or in a cited
project document.

#### Scenario: Row without a source
- **WHEN** the notebook loads a results row whose source field is empty
- **THEN** it stops with an error naming the row

### Requirement: Baseline and variants are named explicitly
Each experiment in the notebook SHALL name one baseline and one or two variants. It SHALL report
water IoU on `peru-dev` as mean ± sample std over 3 seeds, with the per-seed values, and SHALL
state the training data of each row (Peru labels or Sen1Floods11 only).

#### Scenario: Experiment table
- **WHEN** a reader opens the results table of an experiment
- **THEN** the baseline row comes first and is labelled as the baseline, each variant is labelled with its number, and each row shows its 3 seeds and mean ± std

### Requirement: Sprint report covers the whole sprint
The sprint report SHALL be a Spanish Markdown document that covers EDA, feature engineering and
the baseline-and-variants experiment. It SHALL include threats to validity and next steps, and
SHALL point to the notebook and source document of each result.

#### Scenario: Report sections
- **WHEN** a reader opens `docs/informe_sprint_sem6.md`
- **THEN** it has sections for objectives, data, EDA, feature engineering, experiment, discussion, threats to validity and next steps, each pointing to its notebook or source document

### Requirement: Report converts cleanly to LaTeX
The sprint report SHALL use only Markdown constructs that map directly to LaTeX: headings,
paragraphs, lists, pipe tables, inline math and fenced code. It SHALL NOT use raw HTML, and each
figure SHALL be an image file path in the repository.

#### Scenario: Pandoc conversion
- **WHEN** the report is converted with `pandoc -f markdown -t latex`
- **THEN** the conversion finishes without errors and every table and figure appears in the output
