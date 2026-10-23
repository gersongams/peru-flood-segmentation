## Why

The course's week-6 deliverable is a sprint report covering EDA, feature engineering, and an A/B
experiment: a baseline and at most two variants. The class slides ("Experimentos A/B, validación
reforzada y reporte comparativo", Proyecto de Investigación 2) set the rules:
- **one change per variant**, with the same splits, seeds and data;
- a standard table with the central metric, secondary metrics, cost/latency, a one-line note
  and the log path;
- one single chart;
- a validation section without leakage ("divide primero, transforma después");
- an adopt/discard decision;
- reproducibility (commands, data snapshot, library versions);
- risks and next steps;
- a demo draft for the presentation.

The authors asked for the variants to be **variants of the data**. Week 5 found that the
bottleneck is the dry-land shift between Sen1Floods11 and Peru, not missing features. So the
natural week-6 test is to change only the training data and see what that recovers. That
experiment already exists in the project: 3 seeds on `peru-dev`, the adopted variant evaluated
once on test (`docs/phaseD_dev.md`, `docs/phaseD_test.md`). No notebook presents it yet. Weeks 4
and 5 are done (`notebooks/eda.ipynb`, `notebooks/feature_engineering.ipynb`). The report starts
as Markdown and will later be converted to LaTeX.

## What Changes

- Branch `avance-sem6-sprint` from `main`, with the layout of `avance-sem4-eda` and
  `avance-sem5-feature-engineering`: an executed Spanish notebook plus its `build_*_nb.py`, a
  write-up in `docs/`, METHODS_NOTES and EXPERIMENT_LOG entries, and a README row. Already
  created.
- **The experiment becomes an A/B test of the data** (replacing the model comparison). The model
  is fixed: a U-Net with a ResNet-34 encoder, trained on Sen1Floods11 `train` with one identical
  config (LR 3e-4, batch 8, Dice, D4, early stopping on `valid`) and 3 seeds. Only the training
  images change:
  - **Baseline `UNet_s1f11_raw`:** raw Sen1Floods11 images.
  - **Var1 `UNet_s1f11_trHM`:** training images histogram-matched, per band, to the unlabelled
    `peru-train` images.
  - **Var2 `UNet_s1f11_trFDA_b0.01`:** the low-frequency Fourier amplitude of each training image
    replaced by that of a random unlabelled `peru-train` image.
  - The decision was taken on `peru-dev`: Var1 adopted, Var2 discarded. Only the adopted variant
    and the baseline have test scores, from the one-time pass.
  - Reference rows, not variants: the Peru-trained U-Net (upper bound, uses Peru labels) and the
    MNDWI thresholds.
- **The model comparison moves to a short annex.** Peru-trained U-Net vs TerraMind v1-B vs
  Prithvi-EO-2.0, as already built.
- The tracked results table `notebooks/results/sprint6_results.csv` gains the data-variant rows
  and the columns the slides ask for: experiment ID, what changed, run directory, config and
  cost/latency.
- The sprint report `docs/informe_sprint_sem6.md` follows the slides' "Entrega sprint semana 6"
  structure:
  - context;
  - baseline;
  - A/B variants;
  - standard table and single chart;
  - validation;
  - decision;
  - reproducibility;
  - risks and next steps.

  It adds a demo script for the week-6 presentation.

## Capabilities

### New Capabilities
- `course-progress-reports`: weekly course deliverables (branch, executed notebook, write-up)
  that present project results, traceable to their sources, without changing the frozen thesis
  comparison. For week 6, an A/B report that follows the course's format.

### Modified Capabilities
None.

## Impact

- New and revised files only: notebook, builder, CSV, figures, report, OpenSpec change. No
  change to `src/`, `configs/`, `configs/frozen.yaml` or the data.
- No training and no new evaluation. The test numbers are those of the one-time pass of
  2026-10-16. Var2 has no test score because it was discarded on dev.
- Docs: updated entries in `docs/METHODS_NOTES.md` and `docs/EXPERIMENT_LOG.md` (no new runs), and
  the week-6 README row reworded.
