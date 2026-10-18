# Next steps (handoff, updated 2026-10-17)

## Thesis status and pending work (2026-10-17)

Chapters I–II: drafted, three external reviews applied. Chapter III: §3.1.1–3.1.8, §3.2, §3.3
(HE1, HE2) drafted, audited and pushed, with 8 figures (`scripts/figures/`). Pending, in order:

1. **2017 optimism, quantified (small, local).** The 2017 labels keep only pixels where EMS and
   MNDWI agree. Chapter III states the caveat but gives no number (whole-thesis review,
   2026-10-17). Compute from existing data how much of the EMS-labelled area is dropped and how
   that could inflate the 2017 IoU; add one sentence + `% fuente:` to §3.1.7 (2017 subsection).
2. **The system (objective 3, HE3).** Built 2026-10-18 (`src/floodseg/system/`, `app/`; run:
   `uv run uvicorn floodseg.system.server:app --port 8000` + `cd app && pnpm dev`). Working with
   real data on 17MNP: RF1–RF4, RF5 (2023 only), RF6 (17MNP pre-event done), RF7. Left: authors'
   decisions in METHODS_NOTES 2026-10-18 (RF6 pre-event rules, sea in total water, 2017 event
   start); RF5 3-decimal check vs `score_frozen.py`; 2017 EMS reference; trHM comparison layer;
   pre-event scenes for the other 5 tiles; RF1 rectangle check in `server.log`; timing harness
   (`timing_runs.csv`, `networkQuality`, assert `archive_reused == false`, tagged commit); stale
   "no thesis result" docstrings in `scripts/viewer/`. Then measure RF1–RF7 and the time per tile with
   `docs/system_evaluation_protocol.md` (6 tiles × 3 repetitions, median ≤ 15 min per tile; report
   all times and the maximum). No result text before measuring (WRITING_RULES §11.11).
3. **Chapter III parts that wait for the system:** §3.1.9 (now a commented placeholder);
   §3.3.3 contrast of HE3 and of the general hypothesis; the HE3 row of Tabla 3.9; one line in
   the §3.2 synthesis. All marked `% PENDIENTE` in `Capitulo3.tex`.
   Chapter III review (2026-10-17) suggestions for the system part: split it into 3.1.9 development
   (architecture, modules, model choice), 3.1.10 full-scene processing (L1C download, DN → reflectance,
   bands, cloud/no-data masks, tiling, ensemble inference, mosaic, calibration, exports), 3.1.11
   functional and timing evaluation (RF1–RF7 evidence, 6 tiles × 3, keep tile IDs, hardware,
   connection, versions, cache state, logs), 3.2.6 discussion of the system; async jobs with
   progress for full tiles (~121 M pixels). **Decided 2026-10-18 (METHODS_NOTES):** the app shows seed std
   as uncertainty (as in the protocol); default model = Peru-trained U-Net, trHM as optional
   comparison layer; events limited to Yaku 2023 and Niño Costero 2017. The model choice is
   informed by the test results: declare it as such and do not present those results as an
   independent validation of the choice. UI: 6 screens (Visor, Nuevo análisis, Referencia, Línea
   de tiempo, Exportar, Evidencia), analyst-tool style, palette of `app/src/styles.css`. Optional, only with a
   suitable reference: a complementary full-scene evaluation (partitions are water-enriched).
4. **Authors:** rewrite §3.2 and the reasoning paragraphs of §3.3 (1b, 2a, 2b, HE2 summary) in
   their own words (WRITING_RULES §9); review the figures.
5. **Rest of the thesis:** Chapter IV (exactly 3 conclusions + 3 recommendations; the third
   depends on the system), Introducción, Resumen and Abstract (≤ 250 words, past tense),
   Annex 3 (reproducibility: `configs/frozen.yaml`, EXPERIMENT_LOG, protocol; include the AdaBN
   non-reproduction diagnostic, `outputs/phaseD/test/peru-dev_check_compare.csv`). Before
   submission: Annex 1 font size against UNI rules; final source audit (ISO 25010, paywalled items).


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
   **Decided 2026-10-16 (authors):** freeze as recommended (`docs/phaseD_dev.md` "Decided";
   METHODS_NOTES "Phase-D freeze"). No further pod session recommended.
3. **Freeze the phase-D methods and evaluate the test sets once (section 6).** Closes phase D.
   6.1 freeze: `configs/frozen.yaml` (`scripts/build_frozen.py`), 14 groups / 42 runs with sha256;
   gated evaluation `scripts/evaluate_frozen.py` + scoring `scripts/score_frozen.py`, checked on
   `peru-dev` first. Phase-D pass: 33 runs on `peru-test` and `peru-2017-test`. Section-7 pass
   (Sen1Floods11 → Peru fine-tuned on the Peru tests, Sen1Floods11 `test`) is frozen, run later
   (disk space).
   **6.2 done 2026-10-16:** `docs/phaseD_test.md`. 2023: U-Net trHM 0.798 (Peru-trained 0.828,
   source 0.576). 2017: order reverses (source-only best; labels score only clear-water recall).
   **Done 2026-10-16:** test calibration (frozen temperatures; near-edge ECE degrades on test for
   every row; the label-free T transfers; "U-Net trHM best calibrated" holds for all pixels, not
   near edges) and the section-7 pass (fine-tuned = Peru-only on test; Sen1Floods11 `test` U-Net
   0.833 / Prithvi 0.820 / TerraMind 0.819, papers reproduced within 0.5 IoU,
   `docs/published_comparison.md`). Plan 3.8, 7.3, 7.4 closed. Every test set has now been used
   once; no further test evaluation is planned. Next: writing.
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
- ~~**3.8** comparison with published numbers~~: done 2026-10-16 (`docs/published_comparison.md`).

### 4. Start writing the thesis (task 8.2)
Chapter III (data, Peru dataset, protocol) in Spanish, from `docs/peru_data_card.md`, the experiment log and the scripts. Edit in `thesis/` (separate Overleaf repo; `git pull` first); push to Overleaf only with the user's OK.

### Later
- ~~**Section 7** final test evaluation~~: done 2026-10-16 (frozen pass; `docs/phaseD_test.md`).
- **8.x** remaining chapters and the paper.
