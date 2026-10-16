## 1. Source models (Sen1Floods11, RunPod)

- [x] 1.1 Cost estimate for sections 1 and 4 (RunPod catalog read); get the authors' go
- [x] 1.2 LR sweep (3 values) on Sen1Floods11 `valid`, seed 0, for U-Net and TerraMind (= plan-flood-research 3.4)
- [x] 1.3 Repeat one run to record seed reproducibility on the 4090 (= 3.5)
- [x] 1.4 Train U-Net and TerraMind × 3 seeds on Sen1Floods11 with the chosen LR (= 3.6); copy back, verify, delete the pod
- [ ] 1.5 Compare `valid` IoU with published numbers (= 3.8); log runs in EXPERIMENT_LOG

## 2. Target pool and guards

- [x] 2.1 Target-pool datamodule: `peru-train` images only, no label paths, per-model normalization
- [x] 2.2 Guard: refuse dev/test/buffer/2017 chips and any Peru label read during adaptation; unit tests
- [x] 2.3 Pool manifest with checksums (default pool = `peru-train` images)
- [ ] 2.4 Optional, decided after 4.x: extra unlabelled 2023 imagery outside held-out blocks (spatial check)

## 3. Label-free baselines and input alignment

- [x] 3.1 MNDWI > 0 and Otsu-on-pool baselines (label-free) plus the dev-tuned MNDWI threshold (oracle row) on `peru-dev`; extend `scripts/index_baseline.py` with tolerant IoU, precision, recall
- [x] 3.2 Source-only rows: the 3-seed source models of 1.4 evaluated on `peru-dev` (plain and tolerant IoU, precision, recall)
- [x] 3.3 AdaBN for the U-Net (BN statistics re-estimated on the pool); `peru-dev` scores
- [x] 3.4 Per-band histogram matching and FDA transforms; source training with each, U-Net + TerraMind, seed 0 on dev; 3 seeds for the better one (test-time version done 2026-10-15: HM helps, FDA fails; training-time version only as transform × 3 seeds, after a cost decision; done 2026-10-16, pod session 2: trHM U-Net 0.912, trFDA TerraMind 0.837 (pre-registered seed-0 choices))

## 4. Self-training

- [x] 4.1 Adaptation LightningModule: EMA teacher, source + target batches, per-class top-q % pseudo-labels, CE on pseudo-labels, Dice on source, warm-up
- [x] 4.2 Binary DACS mixing (paste source water onto target chips) and student-only colour jitter / blur
- [x] 4.3 Water-fraction collapse warning (on the unlabelled pool); checkpoint every 500 iterations; run record with source checkpoint, budget, wall clock, peak memory; no Peru label read in the run
- [x] 4.4 Local MPS demo (U-Net, few iterations) to check that losses, mixing and EMA work; unit test for the pseudo-label thresholds
- [x] 4.5 Cloud demo: memory and speed for TerraMind (student + teacher + masked view) at batch 8 (measured in the runs themselves: 4 + 4, peak 6.8 GB / `mic` 8.9 GB on an RTX 4090; EXPERIMENT_LOG pod session)
- [x] 4.6 Default-config runs (pre-registered in design.md), seed 0, U-Net + TerraMind; score saved checkpoints on `peru-dev` offline
- [x] 4.7 Oracle-selection variant (q ∈ {20, 50, 80} × α ∈ {0.99, 0.999}, best dev checkpoint), reported only as the secondary row (seed 0; `outputs/phaseD/oracle_selection.md`)
- [x] 4.8 Masked-consistency variant (64 px patches, ratio 0.7) with the default config
- [x] 4.9 3 seeds of the default self-training and masked variants (and the oracle variant), U-Net + TerraMind (default and masked done; oracle variant 3 seeds not run: for U-Net the oracle cell is the default config; for TerraMind no grid cell, even without AdaBN (post-hoc), reaches test-time HM on seed 0 — METHODS_NOTES 2026-10-15 correction entry and addendum)
- [ ] 4.10 Optional: index-aware pseudo-labels (teacher confidence AND MNDWI vs pool-Otsu)

## 5. Analysis on peru-dev

- [x] 5.1 Gap-closed table: source-only → each method → Peru-trained upper bound, mean ± std, plain + tolerant IoU, precision, recall (`outputs/phaseD/dev_table.md`, `docs/phaseD_dev.md`)
- [x] 5.2 Stratified analysis of the best method (edge, NDWI band, water-body size) with the phase-C scripts (U-Net AdaBN, TerraMind HM: `scripts/phaseD_analysis.py`, `docs/phaseD_dev.md`; Prithvi rows after pod session 2)
- [x] 5.3 Calibration of adapted models (ECE near water, reliability) with `scripts/calibration_dev.py` (U-Net AdaBN, TerraMind HM: `scripts/phaseD_analysis.py`, `docs/phaseD_dev.md`; Prithvi rows after pod session 2)
- [x] 5.4 Decision on Prithvi (time and cost), then run its source models + best method if yes (done 2026-10-16: source 0.870; AdaBN, test-time HM and training-time HM all lower)

## 6. Final evaluation (once)

- [x] 6.1 Freeze phase-D configs in `configs/frozen.yaml`
- [x] 6.2 Evaluate all frozen phase-D models once on `peru-test` (full + both doubtful-excluded variants) and `peru-2017-test`
- [ ] 6.3 Tables and figures for the thesis (Chapters IV–V) and paper 2; METHODS_NOTES and EXPERIMENT_LOG entries
