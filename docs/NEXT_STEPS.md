# Next steps (handoff, 2026-10-15)

Status at commit `9eb2dfa`: phase C (Peru training, failure analysis, uncertainty) is done and pushed. All RunPod pods are deleted. Results and provenance are in `docs/EXPERIMENT_LOG.md`, `docs/failure_analysis_dev.md` and `docs/uncertainty_dev.md`. Open tasks are in OpenSpec change `plan-flood-research` (`openspec/changes/plan-flood-research/tasks.md`).

## Order of work

No thesis advisor is assigned yet (2026-10-14). The authors make the research decisions; a
course teacher gives general input. Earlier text that said "advisor" meant that teacher.

### 1. Phase-D decisions: done 2026-10-14 (task 6.5)
Recorded in `docs/failure_analysis_dev.md` (decision record) and `docs/METHODS_NOTES.md`:
phase D = flood mapping in Peru **without Peru labels** (adapt Sen1Floods11 models with
unlabelled Peru imagery); no Sentinel-1; v1 labels frozen plus a pre-registered
sensitivity row; timeline unknown (plan 3–6 months).
**6.6 done:** OpenSpec change `phase-d-peru-adaptation` (proposal, 3 specs, design, tasks;
`openspec validate --strict` passes). Literature: `docs/literature_phaseD.md`.
**Section 1 done 2026-10-15:** Sen1Floods11 source models on RunPod, LR sweep + 3 seeds:
U-Net (LR 3e-4) 0.8315 ± 0.0015, TerraMind (LR 5e-5) 0.8349 ± 0.0014 `valid` IoU; cost
$4.56; pods deleted (`docs/EXPERIMENT_LOG.md`, `docs/METHODS_NOTES.md` 2026-10-15).
Checkpoints: `outputs/s1f11_src_unet/`, `outputs/s1f11_src_terramind_lr5e-5/`.
**Next (local, free):** section 2 (target pool + guards), 3.1 (index baselines with tolerant
IoU), 3.2 (source-only rows on `peru-dev`). Prithvi source models wait for task 5.4.

### 2. Close the open phase-C tasks (local, free, a few hours)
- ~~**6.4** per-stratum calibration, **5.4** reliability diagrams~~: done 2026-10-14
  (`docs/calibration_dev.md`). Predictions are now float32 logits in `outputs/logits_*`;
  AUROCs and temperatures before that date are superseded (`docs/METHODS_NOTES.md`).
  Key result: the uncertainty does not flag low-contrast/small-water misses.
- ~~**4.4** doubtful-label chips, **6.2** IoU drop and band statistics~~: done 2026-10-14
  (`docs/peru_doubtful_chips.md`, `docs/domain_shift_dev.md`).
- ~~**6.5** decision record~~: done. **6.6** open the phase-D OpenSpec change (see step 1).
- **5.2** MC-dropout (optional).

### 3. Remaining Sen1Floods11 tasks
- ~~3.4 LR sweep, 3.5 reproducibility~~: done 2026-10-15. 3.6 done for U-Net + TerraMind;
  Prithvi pending (phase-D 5.4).
- **3.8** comparison with published numbers: re-check Prithvi-CAFE Table 1 in the PDF.

### 4. Start writing the thesis (task 8.2)
Chapter III (data, Peru dataset, protocol) in Spanish, from `docs/peru_data_card.md`, the experiment log and the scripts. Edit in `thesis/` (separate Overleaf repo; `git pull` first); push to Overleaf only with the user's OK.

### Later
- **Section 7** final test evaluation (`test`, `peru-test`, `peru-2017-test`): only once, after phase D.
- **8.x** remaining chapters and the paper.
