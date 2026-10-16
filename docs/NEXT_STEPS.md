# Next steps (handoff, 2026-10-16)

Status 2026-10-15: phase C is done; phase D is done up to the dev analysis (section 1 below). Local `main` is ahead of GitHub (not pushed). All RunPod pods are deleted. Results and provenance are in `docs/EXPERIMENT_LOG.md`, `docs/failure_analysis_dev.md` and `docs/uncertainty_dev.md`. Open tasks are in OpenSpec change `plan-flood-research` (`openspec/changes/plan-flood-research/tasks.md`).

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
**Done 2026-10-15 (local):** section 2 (target pool + guards), 3.1 (index baselines),
3.2 (source-only rows): `outputs/phaseD/dev_table.md`. Source-only U-Net 0.836 ± 0.022,
TerraMind 0.703 ± 0.091 dev IoU (seed 1 at 0.598); errors are false positives.
**Done 2026-10-15:** 3.3 AdaBN, 3.4 part 1 (test-time HM / FDA), section 4 self-training
(22 RunPod runs, ≈ $6.0, pod deleted), 5.1 gap-closed table. Write-up: `docs/phaseD_dev.md`.
Best per backbone on dev: **U-Net + AdaBN 0.897 ± 0.016**, **TerraMind + test-time HM
0.809 ± 0.006**. Self-training as pre-registered is harmful (the evaluation-time AdaBN pass
causes most of the TerraMind drop, post-hoc diagnostic); even without it, it stays below the
alignment baselines. The edge-distance correction is done (commit `6f1ad88`).
**Next (authors' decisions first):** (a) any new self-training variant (on HM inputs, fixed BN
rule) — new design, pre-register, pod ≈ $3–6; (b) training-time HM (3.4 part 2), cost
decision; (c) Prithvi (5.4). Free, local: 5.2 stratified analysis and 5.3 calibration of
AdaBN U-Net and HM TerraMind. Then section 6 (freeze, test once).

### 1b. Phase-D order of work (agreed 2026-10-15)
1. **Stratified analysis and calibration of the best methods (tasks 5.2, 5.3)**, local, free.
   Best methods: U-Net + AdaBN (0.897) and TerraMind + test-time histogram matching (0.809).
   Where do they still fail vs the Peru-trained models (near water edges, low water-index
   contrast, small water bodies)? Is their uncertainty still calibrated after adaptation?
   Reuses the phase-C scripts (`scripts/analyze_dev.py`, `calibration_dev.py`, `uncertainty_dev.py`).
   Before it: move today's inline phase-D diagnostics (paired deltas, water fractions, no-AdaBN
   grid scoring) into `scripts/phaseD_diagnostics.py` and check they reproduce.
2. **Prithvi (5.4) and training-time alignment (3.4 part 2): done 2026-10-16** (pod session 2,
   $9.68; ≈ $6.7 credit left). Training-time HM is the best label-free U-Net (0.912); TerraMind's
   pre-registered train-time row is FDA (0.837); Prithvi is best source-only (0.870), no method
   helped it. Strata/calibration (5.2/5.3) extended to Prithvi and the train-time rows.
   **Open (authors):** which method per backbone to freeze for section 6; recommendation in
   `docs/phaseD_dev.md` "Open decisions". No further pod session recommended.
3. **Freeze the phase-D methods and evaluate the test sets once (section 6).** Closes phase D.
4. **Start writing:** Chapter III (data, Peru dataset, protocol) and the phase-D results, in Spanish.

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
