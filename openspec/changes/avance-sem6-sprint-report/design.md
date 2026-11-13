# Design

## Context

- The data is rebuilt on this machine and matches Peru v1 (`scripts/freeze_peru.py --check`:
  2,990 files). `outputs/` (checkpoints, logits, run summaries) is not on this machine. A
  notebook that recomputes metrics from logits could not run here.
- The values needed are recorded in tracked documents:
  - **Data A/B experiment** (U-Net trained on Sen1Floods11):
    - `docs/EXPERIMENT_LOG.md` "Phase-D source-only models" (2026-10-15), per-seed dev IoU of
      `s1f11_src` U-Net;
    - "Pod session 2 evaluation" (2026-10-16), per-seed dev IoU of the trHM and trFDA U-Net;
    - "Pod session 2: launch" (2026-10-15/16), the transform cost per chip and per epoch, and the
      pod rate;
    - section 1 of 2026-10-14/15, the U-Net LR sweep and wall clock per run.
  - `docs/METHODS_NOTES.md`:
    - 2026-10-15 "Pod session 2, pre-registered", the transform definitions and the HM-vs-FDA
      rule;
    - 2026-10-16 "Pod session 2 results", the dev mean ± std, precision and recall, and the rule
      outcome.
  - `docs/phaseD_test.md`: `peru-test` and `peru-2017-test` for U-Net trHM and U-Net source-only,
    with CIs and paired differences.
  - **Annex** (Peru-trained models): EXPERIMENT_LOG 2026-10-13/14 and `docs/phaseD_test.md`.
    These are already in the CSV.
- Weeks 4 and 5 use the same pattern: a `notebooks/build_*_nb.py` script builds the notebook with
  `nbformat` and executes it with `nbclient`, and the notebook is committed with its outputs.
- The test sets were evaluated once on 2026-10-16 and must not be evaluated again.
- The class slides on A/B experiments set the format: one change per variant, standard table,
  single chart, validation, decision, reproducibility, risks, demo (see proposal.md).

## Goals / Non-Goals

**Goals:**
- A notebook that runs on any clone, with or without `outputs/`, and shows the data A/B experiment
  in the course's format, with every number sourced.
- A sprint report in Markdown, following the slides' week-6 structure, that converts to LaTeX
  with little rework.

**Non-Goals:**
- New training, new variants, or new evaluation on any split.
- Recomputing metrics from logits. That excludes a PR curve or a threshold curve for the models,
  which needs logits that are not on this machine.
- K-fold cross-validation (see decision 4).
- Re-running the EDA or feature-engineering notebooks.
- The LaTeX version of the report.

## Decisions

1. **The experiment is an A/B test of the training data, with the model fixed.**
   - **Setup.** U-Net, ResNet-34 encoder (ImageNet), 13 L1C bands. Trained on Sen1Floods11 `train`
     (252 chips) with one config: LR 3e-4 (the section-1 sweep winner), batch 8, Dice, D4,
     AdamW + ReduceLROnPlateau, early stopping on `valid` water IoU, seeds 0–2. Scored on
     `peru-dev`.
   - **Baseline `UNet_s1f11_raw`:** raw images.
   - **Var1 `UNet_s1f11_trHM`:** per-band histogram matching of each training (and `valid`) image
     to the unlabelled `peru-train` images, using 1,001 quantiles.
   - **Var2 `UNet_s1f11_trFDA_b0.01`:** FDA. The centred low-frequency amplitude window (β = 0.01)
     of each training image is replaced by that of a random `peru-train` image.
   - **Why data variants.** Week 5 found that the problem is the dry-land shift between datasets,
     so changing only the data tests that finding directly. The setting is also the thesis'
     realistic case: a new flood without Peru labels.
   - **Why these meet "one change per variant".** The three rows share the code commit, config,
     LR, splits, seeds and stopping rule (EXPERIMENT_LOG and METHODS_NOTES 2026-10-15). Only the
     training transform differs. The transform is also applied to `valid`, so checkpoint
     selection sees the same input style. That is part of the same change.
   - **Decision.** The HM-vs-FDA choice was pre-registered on seed-0 dev IoU (HM 0.9148 vs FDA
     0.7251), so Var1 was adopted and frozen, and Var2 was never evaluated on test. The report
     shows this as the adopt/discard step. The 3-seed dev means confirm it.
   - **Reference rows, not variants:**
     - Peru-trained U-Net: different data, LR and selection set; the upper bound with labels.
     - MNDWI thresholds.
   - **Annex.** The Peru-trained U-Net vs TerraMind vs Prithvi comparison, kept brief.
   - Alternatives rejected:
     - Baseline Peru-trained U-Net, Var1 Sen1Floods11 → Peru fine-tuning, Var2 trHM. Var2 would
       change four things at once (training data, transform, LR, selection set).
     - Test-time HM as Var2. It changes inference, not training, and has no U-Net test row.
     - Model variants as the main experiment. The authors asked for data variants.
     - A new variant (U-Net + MNDWI channel). It needs training, and its test score could not be
       reported because the test pass is closed.
2. **Source of numbers: a tracked CSV transcribed from the logs.**
   - File `notebooks/results/sprint6_results.csv`, in long format. Columns:
     `experiment, role, model, method, train_data, split, seed, metric, value, source, commit`,
     plus `exp_id`, `change` (one line), `run_dir` and `config`.
   - Cost and latency are stored as metric rows: `s_per_chip_transform`, `s_per_epoch`,
     `wall_clock_s_per_run`, `usd_per_hour`. Each has its source.
   - Experiment `A` is renamed `annex`; experiment `AB` holds the data rows.
   - Alternative rejected: reading `outputs/` directly. It is absent here.
   - Mitigation: the check cell compares the per-seed dev IoU with
     `outputs/phaseD/dev_table_runs.csv` when that file exists (regimes `s1f11_src`, `trhm`,
     `trfda`, `peru`).
3. **Test values are aggregate only.** `docs/phaseD_test.md` records mean ± std, the CI,
   precision and recall, and paired differences. The CSV stores them as recorded; the notebook
   does not recompute them.
4. **Validation: one grouped holdout, no K-fold.**
   - **Split.** Peru uses spatial blocks with a buffer. That is a grouped split: a location is in
     one split on both dates, and no held-out chip touches a training chip. Sen1Floods11 uses its
     official splits.
   - **Seeds.** Fixed (0–2).
   - **Transform fitting.** HM quantiles and FDA amplitudes are fitted only on `peru-train`
     images, without labels. Dev and test images are never used. This is "divide primero,
     transforma después".
   - **Why no K-fold.** The project rules forbid re-splitting Sen1Floods11. A K-fold over Peru
     blocks would need retraining every row K times on rented GPUs. Seed spread and chip-bootstrap
     CIs are used instead. The report states this as a limitation.
5. **Single chart.** Per-seed dev IoU (points) and mean ± std for the baseline, Var1 and Var2. A
   dashed line marks the Peru-trained upper bound. The slides suggest a PR curve or confusion
   matrix for classification, but both need logits, which are not on this machine. A second panel
   shows dev precision and recall for the rows that have them on record. The annex keeps its own
   figures.
6. **Figures saved as files for the report.** PNG and PDF in `docs/figs/sprint6/`, plus a CSV of
   each displayed table. Figures use the shared thesis style (`scripts/figures/style.py`) when it
   is importable.
7. **Report structure.** It follows the slides' "Entrega sprint semana 6" list, after short EDA and
   FE summaries:
   - context and central metric;
   - baseline;
   - A/B variants;
   - results: standard table and chart;
   - validation;
   - decision;
   - reproducibility;
   - threats to validity;
   - risks and next steps;
   - demo script;
   - annex.
8. **Report language and form.** Spanish. Only pandoc-safe Markdown. Citations use pandoc syntax
   (`[@key]` → `\citep{}` with `--natbib`), and only keys that exist in
   `thesis/3_3_BIBLIOGRAFIA/library.bib` or `docs/literature_thesis.bib`.
9. **Branch base.** `avance-sem6-sprint` from `main` (already created). Pushed only with the
   authors' approval.

## Risks / Trade-offs

- [Transcription error in the CSV] → each row cites its log entry. Every dev mean ± std is
  recomputed from its seeds and compared with the log, and the check cell compares with
  `outputs/` when present.
- [Var2's dev precision/recall and per-run wall clock for trHM/trFDA are not in tracked
  documents] → the table says "not on record"; nothing is estimated.
- [The baseline's high seed spread (std 0.022) could hide the effect] → Var1 is above the
  baseline on every seed (paired by seed: +0.072 / +0.055 / +0.100, METHODS_NOTES 2026-10-16). On
  test the gain is +0.22.
- [Readers may take Var1 as the final answer] → the report states that it stays 0.030 [0.013,
  0.048] below the Peru-trained U-Net on `peru-test`. On `peru-2017-test` the unadapted rows score
  higher, because the 2017 labels only measure clear-water recall.
- [The annex could be read as the week-6 variants] → it is labelled as context and placed after
  the A/B experiment.

## Migration Plan

Not applicable (new and revised files only). To roll back, reset the branch to the commit before
the revision.

## Open Questions

- The exact page limit the course wants. The Markdown can be trimmed during the LaTeX step.
