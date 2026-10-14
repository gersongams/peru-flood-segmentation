# Next steps (handoff, 2026-10-14)

Status at commit `9eb2dfa`: phase C (Peru training, failure analysis, uncertainty) is done and pushed. All RunPod pods are deleted. Results and provenance are in `docs/EXPERIMENT_LOG.md`, `docs/failure_analysis_dev.md` and `docs/uncertainty_dev.md`. Open tasks are in OpenSpec change `plan-flood-research` (`openspec/changes/plan-flood-research/tasks.md`).

## Order of work

No thesis advisor is assigned yet (2026-10-14). The authors make the research decisions; a
course teacher gives general input. Earlier text that said "advisor" meant that teacher.

### 1. Phase-D decisions: done 2026-10-14 (task 6.5)
Recorded in `docs/failure_analysis_dev.md` (decision record) and `docs/METHODS_NOTES.md`:
phase D = flood mapping in Peru **without Peru labels** (adapt Sen1Floods11 models with
unlabelled Peru imagery); no Sentinel-1; v1 labels frozen plus a pre-registered
sensitivity row; timeline unknown (plan 3–6 months).
**Next: task 6.6**, open the phase-D OpenSpec change: literature check of self-training /
unsupervised domain adaptation for segmentation (from the literature base), pick 2–3 methods,
download unlabelled Peru imagery plan, protocol.

### 2. Close the open phase-C tasks (local, free, a few hours)
- ~~**6.4** per-stratum calibration, **5.4** reliability diagrams~~: done 2026-10-14
  (`docs/calibration_dev.md`). Predictions are now float32 logits in `outputs/logits_*`;
  AUROCs and temperatures before that date are superseded (`docs/METHODS_NOTES.md`).
  Key result: the uncertainty does not flag low-contrast/small-water misses.
- ~~**4.4** doubtful-label chips, **6.2** IoU drop and band statistics~~: done 2026-10-14
  (`docs/peru_doubtful_chips.md`, `docs/domain_shift_dev.md`).
- ~~**6.5** decision record~~: done. **6.6** open the phase-D OpenSpec change (see step 1).
- **5.2** MC-dropout (optional).

### 3. Remaining Sen1Floods11 tasks (RunPod, about $5–8)
- **3.4** learning-rate sweep; **3.5** repeat-run reproducibility check.
- **3.6** 3-seed Sen1Floods11 runs; **3.8** comparison with published numbers (needed for the paper's Sen1Floods11 row).
- Use `scripts/cloud/` (sync_data, setup_pod, run_queue, fetch_results). Delete the pod after verifying results locally.

### 4. Start writing the thesis (task 8.2)
Chapter III (data, Peru dataset, protocol) in Spanish, from `docs/peru_data_card.md`, the experiment log and the scripts. Edit in `thesis/` (separate Overleaf repo; `git pull` first); push to Overleaf only with the user's OK.

### Later
- **Section 7** final test evaluation (`test`, `peru-test`, `peru-2017-test`): only once, after phase D.
- **8.x** remaining chapters and the paper.
