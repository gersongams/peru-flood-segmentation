# Tasks

## 1. Branch

- [x] 1.1 Create branch `avance-sem6-sprint` from an up-to-date `main` (`git pull`). Verify: `git status -sb` shows the new branch and a clean tree, and `notebooks/eda.ipynb` and `notebooks/feature_engineering.ipynb` are present.
- [x] 1.2 Commit this OpenSpec change on the branch. Verify: `openspec validate avance-sem6-sprint-report --strict` passes.

## 2. Tracked results table

- [x] 2.1 Write `notebooks/results/sprint6_results.csv` (columns per design decision 2): per-seed `peru-dev` IoU/mIoU of the Peru-trained U-Net, TerraMind and Prithvi (EXPERIMENT_LOG 2026-10-13/14); MNDWI reference rows (`docs/phaseD_dev.md`; train-fitted threshold from the week-5 log); `peru-test` and `peru-2017-test` aggregates (`docs/phaseD_test.md`). Verify: every row has a non-empty `source`.
- [x] 2.2 Check the transcription. Recompute each dev mean ± sample std from its 3 seeds and compare it with the mean ± std in the log. Verify: all agree to the 4th decimal; list any difference in the METHODS_NOTES entry (task 4.1).

## 3. Notebook

- [x] 3.1 Write `notebooks/build_baseline_nb.py`, following `build_feature_engineering_nb.py` (nbformat + nbclient, Spanish). Sections:
  1. objective and how it links to weeks 4 and 5;
  2. experimental design (baseline, variants, splits, seeds, metric, protocol);
  3. dev table, per-seed figure, test table with CIs and precision/recall;
  4. optional check against `outputs/`;
  5. conclusions and limitations.

  Verify: the builder stops on a row with an empty source (spec scenario "Row without a source").
- [x] 3.2 Save the figures to `docs/figs/sprint6/*.png` and each displayed table to `docs/figs/sprint6/*.csv`. Verify: the files exist after the build.
- [x] 3.3 Run `uv run python notebooks/build_baseline_nb.py` without `outputs/`. Verify: `notebooks/baseline_variantes.ipynb` has all code cells executed with no error outputs, and the check cell reports that `outputs/` is absent. Also verify that no path under `data/` is read (grep the builder for `data/`).

## 4. Documentation of the notebook

- [x] 4.1 Add a dated entry to `docs/METHODS_NOTES.md` covering:
  - what was done and why (presentation of existing results, no new runs);
  - the choice of the experiment and the alternatives rejected;
  - the transcription check from task 2.2;
  - caveats: test values aggregate only, single spatial split;
  - the thesis chapter it feeds.

  Verify: the entry names `notebooks/results/sprint6_results.csv` and the notebook.
- [x] 4.2 Add an entry to `docs/EXPERIMENT_LOG.md` stating that there are no new runs, with the build command and output paths. Verify: the entry lists the build command exactly as it runs.

## 5. Sprint report

- [x] 5.1 Write `docs/informe_sprint_sem6.md` in Spanish, with these sections:
  1. objectives of the sprint;
  2. data (sources, splits, Peru v1 checksums);
  3. EDA summary (from `notebooks/eda.ipynb` section 9 and its figures);
  4. feature-engineering summary (from `docs/feature_engineering.md`);
  5. experiment, with design, results and figures from `docs/figs/sprint6/`;
  6. discussion;
  7. threats to validity;
  8. next steps;
  9. reproducibility (commands).

  Verify: every number appears in `sprint6_results.csv` or in a cited project document.
- [x] 5.2 Citations as BibTeX keys only if they exist in `thesis/3_3_BIBLIOGRAFIA/library.bib` (after `cd thesis && git pull`) or `docs/REFERENCES.md`; otherwise write "(referencia pendiente)". Verify: a grep of the bracketed keys in the report against `library.bib` finds no missing key.
- [x] 5.3 Check that the report converts to LaTeX. Verify: `pandoc -f markdown -t latex docs/informe_sprint_sem6.md` finishes without errors (install pandoc only with the user's OK), and the report contains no raw HTML (`rg '<[a-z]' docs/informe_sprint_sem6.md` is empty).

## 6. Branch wrap-up

- [x] 6.1 Add week 6 to the README "Course deliverables" table (branch, notebook, report). Verify: the links resolve to existing files.
- [x] 6.2 Integration check. `uv run pytest` passes. `git status` shows only the intended files. `configs/frozen.yaml` and `src/` are unchanged (`git diff --stat main -- configs src` is empty). Commit on `avance-sem6-sprint`.

## Workflow follow-up

- Push `avance-sem6-sprint` to GitHub only with the authors' approval.
- Merge into `main` like weeks 4 and 5, once the authors agree.
- Convert the report to LaTeX (separate step, in `thesis/` or the presentation repo).
- Archive this change after the deliverable is submitted.
