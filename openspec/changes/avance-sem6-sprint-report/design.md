# Design

## Context

- The data is rebuilt on this machine and matches Peru v1 (`scripts/freeze_peru.py --check`:
  2,990 files). `outputs/` (checkpoints, logits, run summaries) is not on this machine. A
  notebook that recomputes metrics from logits could not run here.
- The per-seed dev values and the one-time test scores are recorded in tracked documents:
  - `docs/EXPERIMENT_LOG.md`: "Peru-only, 3 seeds" (2026-10-13/14);
  - `docs/phaseD_dev.md` (index baselines);
  - `docs/phaseD_test.md` (`peru-test` and `peru-2017-test`, chip-bootstrap 95 % CIs).
- Weeks 4 and 5 use the same pattern: a `notebooks/build_*_nb.py` script builds the notebook with
  `nbformat` and executes it with `nbclient`, and the notebook is committed with its outputs.
- The test sets were evaluated once on 2026-10-16 and must not be evaluated again.

## Goals / Non-Goals

**Goals:**
- A notebook that runs on any clone, with or without `outputs/`, and shows the
  baseline-and-variants results with their sources.
- A sprint report in Markdown that converts to LaTeX for the thesis template with little
  rework.

**Non-Goals:**
- New training, new variants, or new evaluation on any split.
- Recomputing metrics from logits.
- Re-running the EDA or feature-engineering notebooks. Their executed versions are cited as
  they are.
- The LaTeX version of the report (a later step).

## Decisions

1. **Which experiment is "baseline + variants".** U-Net baseline, TerraMind v1-B variant 1,
   Prithvi-EO-2.0 300M variant 2, all trained on `peru-train`. This is the thesis' core question
   (foundation models against a CNN). Every row has 3 seeds on dev and a one-time test score.
   - The MNDWI threshold rows are non-learned reference rows, not variants.
   - Alternatives rejected:
     - A new variant (U-Net + MNDWI channel). It needs 6–12 h of training, and its result
       could not be compared on test, because the test pass is closed.
     - Adding the label-free phase-D experiment (source-only U-Net, AdaBN, training-time
       histogram matching). The authors kept the deliverable to one experiment (2026-10-23).
       The report mentions it only as next work.
2. **Source of numbers: a tracked CSV transcribed from the logs.** File
   `notebooks/results/sprint6_results.csv`, in long format with columns `experiment, role, model,
   method, train_data, split, seed, metric, value, source, commit`. The rows are transcribed
   from the documents listed in Context.
   - Alternative rejected: reading `outputs/` directly. It is absent here, and copying it from
     the other machine would block the deliverable.
   - Mitigation: an optional check cell compares the CSV with
     `outputs/phaseD/dev_table_runs.csv` (regime `peru`) when it
     exists (spec: "Notebook runs with outputs").
3. **Test values are aggregate only.** `docs/phaseD_test.md` records mean ± std, the CI,
   precision and recall per row, but not per-seed test values. The CSV stores them with
   `seed = mean` / `std` / `ci_low` / `ci_high`. The notebook shows them as recorded and does
   not recompute them.
4. **Figures saved as files for the report.** The builder saves each figure as PNG in
   `docs/figs/sprint6/`, plus a CSV of each displayed table. The Markdown report links those
   files, so the LaTeX version can `\includegraphics` them. Figures follow the existing shared
   figure style if it is importable (commit `7dc9620`); otherwise they use matplotlib defaults.
5. **Report language and form.** Spanish, as in the EDA and FE notebooks. Only pandoc-safe
   Markdown, as the spec requires. Citations use pandoc syntax (for example
   `[@bonafilia_sen1floods11_2020]`), which `pandoc --natbib` turns into `\citep{}`. No key is
   invented: a key is used only if it already exists in `thesis/3_3_BIBLIOGRAFIA/library.bib` or
   in `docs/literature_thesis.bib` (the project's thesis bibliography, used when `thesis/` is not
   cloned). Otherwise the text says "(referencia pendiente)".
6. **Branch base.** `avance-sem6-sprint` starts from `main`, which already contains the week-4 and
   week-5 notebooks. The branch is pushed only with the authors' approval.

## Risks / Trade-offs

- [Transcription error in the CSV] → each row cites its log line; the check cell catches dev
  differences when `outputs/` is present. In this session, every dev mean is recomputed from the
  three seeds and compared with the mean ± std in the log.
- [Readers take the dev numbers as final results] → the report labels `peru-dev` as the
  selection split, and the test values as the one-time pass.
- [On `peru-2017-test` the ranking differs from dev (TerraMind 0.892 against U-Net and Prithvi
  0.878)] → the report states it and adds that the 2017 labels only measure clear-water recall
  (`docs/phaseD_test.md`). It does not claim that one model wins both events.
- [The figure style module may not import outside the thesis scripts] → fall back to matplotlib
  defaults; the content does not change.

## Migration Plan

Not applicable (new files only). To roll back, delete the branch.

## Open Questions

- The exact page limit and format the course wants for the sprint report. The Markdown
  structure works for any length; it can be trimmed during the LaTeX step.
