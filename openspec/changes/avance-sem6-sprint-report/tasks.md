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

## 7. Data-variant revision (2026-10-23, after the class slides on A/B experiments)

Groups 1–6 built the model comparison, which now becomes the annex. This group replaces the main experiment with the data A/B test (design decision 1).

- [x] 7.1 Extend `notebooks/results/sprint6_results.csv`:
  - rename the existing experiment `A` to `annex`;
  - add the columns `exp_id`, `change`, `run_dir` and `config`;
  - add experiment `AB` rows, with the source of each:
    - per-seed dev IoU of `UNet_s1f11_raw` (EXPERIMENT_LOG 2026-10-15) and of `UNet_s1f11_trHM` / `UNet_s1f11_trFDA_b0.01` (EXPERIMENT_LOG 2026-10-16);
    - dev mean ± std, precision and recall where recorded (METHODS_NOTES 2026-10-16 "Pod session 2 results");
    - `peru-test` and `peru-2017-test` for raw and trHM, plus the paired difference trHM − Peru-trained (`docs/phaseD_test.md`);
    - cost/latency rows: HM 0.11 and FDA 0.26 s/chip; 5.8 vs ≈ 25 s/epoch; source U-Net wall clock 336–648 s per run; $0.89/h.

  Verify: every row has a source, and each recomputed dev mean ± std equals the log to 4 decimals.
- [x] 7.2 Rewrite `notebooks/build_baseline_nb.py`. Main part:
  1. context and central metric;
  2. design (shared settings, plus the one change of each variant);
  3. standard A/B table;
  4. single chart;
  5. validation;
  6. decision;
  7. check against `outputs/` for every regime present.

  Annex: the existing model comparison, shortened. Verify: the notebook builds with no error outputs, the builder still refuses a row without a source, and `rg 'data/' notebooks/build_baseline_nb.py` shows no read of `data/`.
- [x] 7.3 Regenerate `docs/figs/sprint6/`: the A/B chart (`fig_ab_datos.png/.pdf`), the A/B tables, and the annex figures. Remove figures that are no longer used. Verify: every figure the report links exists, and no file in the folder is unreferenced.
- [x] 7.4 Rewrite `docs/informe_sprint_sem6.md` with the structure of design decision 7:
  - keep the EDA and FE summaries;
  - add the validation, decision and reproducibility sections, the latter with commands, data snapshot and library versions from `uv.lock`;
  - add risks and next steps, the demo script and the annex.

  Verify: every number appears in the CSV or a cited document; the citation keys exist in `docs/literature_thesis.bib`; `rg '<[a-z]'` finds no HTML.
- [x] 7.5 Update the documentation: amend the METHODS_NOTES and EXPERIMENT_LOG entries of 2026-10-23 (data A/B as the main experiment, models as annex, and the reason: class slides and the one-change rule), and reword the week-6 README row. Verify: the entries name the slides, the three experiment IDs and the rejected design.
- [x] 7.6 Integration check:
  - `pandoc -f markdown -t latex --natbib docs/informe_sprint_sem6.md` finishes without errors, with every table and figure in the output;
  - `uv run pytest` passes;
  - `git diff --stat main -- configs src` is empty;
  - `openspec validate avance-sem6-sprint-report --strict` passes.

  Commit on `avance-sem6-sprint`.

## Workflow follow-up

- Push `avance-sem6-sprint` to GitHub only with the authors' approval.
- Merge into `main` like weeks 4 and 5, once the authors agree.
- Convert the report to LaTeX (separate step, in `thesis/` or the presentation repo).
- Archive this change after the deliverable is submitted.
