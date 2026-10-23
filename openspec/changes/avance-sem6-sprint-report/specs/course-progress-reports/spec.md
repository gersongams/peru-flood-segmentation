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
- **THEN** the notebook compares the per-seed dev IoU of every row of the tracked table that the file contains (regimes `s1f11_src`, `trhm`, `trfda`, `peru`) and reports any difference larger than 1e-4

### Requirement: Every reported number has a source
Every value in the tracked results table SHALL carry a source: the `docs/EXPERIMENT_LOG.md`
entry or write-up table it comes from, and the commit of the run where the log records one.
The notebook and the sprint report SHALL use no number that is not in that table or in a cited
project document.

#### Scenario: Row without a source
- **WHEN** the notebook loads a results row whose source field is empty
- **THEN** it stops with an error naming the row

### Requirement: Baseline and variants are named explicitly
The week-6 experiment SHALL name one baseline and at most two variants, each with a short
experiment ID (for example `UNet_s1f11_trHM`). It SHALL report water IoU on `peru-dev` as mean ±
sample std over 3 seeds, with the per-seed values, and SHALL state each row's model and training
data.

#### Scenario: Experiment table
- **WHEN** a reader opens the results table of the experiment
- **THEN** the baseline row comes first and is labelled as the baseline, each variant is labelled Var1 or Var2 with its experiment ID, and each row shows its 3 seeds and mean ± std

### Requirement: One change per variant
Each variant SHALL differ from the baseline in exactly one setting. The model, config, splits,
seeds and stopping rule SHALL stay the same. The notebook SHALL list the settings shared by all
rows and the one setting each variant changes.

#### Scenario: Data variants
- **WHEN** the variants are variants of the data
- **THEN** every row uses the same U-Net config trained on Sen1Floods11 `train` and selected on `valid`, and each variant differs only in the transform applied to the training images

#### Scenario: Reference rows
- **WHEN** a row differs from the baseline in more than one setting (for example the Peru-trained U-Net)
- **THEN** it is labelled as a reference row, not as a variant, and the difference is stated

### Requirement: Standard A/B results table
The notebook and the report SHALL contain one table with these columns for the baseline, Var1
and Var2: experiment ID, what changed (one line), central metric (`peru-dev` water IoU), secondary
metrics (precision, recall, and `peru-test` IoU where it exists), cost/latency, and log path.

#### Scenario: Value not on record
- **WHEN** a cell's value is not in any tracked document (for example Var2's dev precision)
- **THEN** the cell says it is not on record instead of leaving a blank or estimating it

### Requirement: Adopt-or-discard decision
The report SHALL state which variant is adopted and justify it with the central metric, seed
stability and cost. The decision SHALL rest on `peru-dev` only. Test scores SHALL be shown only
for rows that were evaluated in the one-time pass.

#### Scenario: Discarded variant
- **WHEN** a variant was discarded on dev before the test pass
- **THEN** it has no test score, and the report says that this follows the protocol

### Requirement: Validation without leakage
The report SHALL describe the validation scheme. It SHALL cover the spatial-block split with a
buffer (a grouped split), the fixed seeds, and that the data transforms are fitted only on
unlabelled training-split images. It SHALL also explain why no K-fold cross-validation is used.

#### Scenario: Transform fitted on training data
- **WHEN** a reader checks how a data variant was built
- **THEN** the report states that its statistics come only from `peru-train` images, never from `peru-dev` or test images, and that no Peru label is used

### Requirement: Single chart
The experiment SHALL have one main comparison chart for the baseline and both variants on
`peru-dev`. It SHALL be saved as an image file for the report.

#### Scenario: Chart file
- **WHEN** the notebook is built
- **THEN** `docs/figs/sprint6/` contains the A/B chart as PNG and PDF

### Requirement: Reproducibility record
The report SHALL give:
- the commands that trained the baseline, Var1 and Var2;
- where their logs are;
- a data snapshot: date, the Peru v1 checksum file, the chip count and the data size;
- the versions of the main libraries.

#### Scenario: Library versions
- **WHEN** a reader looks for the software versions
- **THEN** the report lists the Python, PyTorch, Lightning and TerraTorch versions taken from `uv.lock`

### Requirement: Demo script
The deliverable SHALL include a short demo script in Spanish for the week-6 presentation, in
order: context, baseline, variants, result, decision, next step. Each step SHALL point to the
notebook cell or figure to show.

#### Scenario: Presenting the demo
- **WHEN** an author presents the sprint
- **THEN** the demo script gives, for each step, what to say and which figure or table to show

### Requirement: Model comparison as annex
The comparison of models trained on Peru labels (U-Net, TerraMind v1-B, Prithvi-EO-2.0) SHALL
appear only as a short annex in the notebook and the report. It SHALL NOT be presented as the
week-6 variants.

#### Scenario: Annex placement
- **WHEN** a reader opens the report
- **THEN** the model comparison appears after the A/B experiment, in an annex labelled as context

### Requirement: Sprint report covers the whole sprint
The sprint report SHALL be a Spanish Markdown document that covers EDA, feature engineering and
the A/B experiment, following the course's week-6 structure. It SHALL include threats to
validity, risks and next steps, and SHALL point to the notebook and source document of each
result.

#### Scenario: Report sections
- **WHEN** a reader opens `docs/informe_sprint_sem6.md`
- **THEN** it has sections for context and central metric, data, EDA, feature engineering, baseline, A/B variants, results (table and chart), validation, decision, reproducibility, threats to validity, risks and next steps, demo script, and the model-comparison annex, each pointing to its notebook or source document

### Requirement: Report converts cleanly to LaTeX
The sprint report SHALL use only Markdown constructs that map directly to LaTeX: headings,
paragraphs, lists, pipe tables, inline math and fenced code. It SHALL NOT use raw HTML, and each
figure SHALL be an image file path in the repository.

#### Scenario: Pandoc conversion
- **WHEN** the report is converted with `pandoc -f markdown -t latex`
- **THEN** the conversion finishes without errors and every table and figure appears in the output
