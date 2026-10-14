## Context

See proposal.md (Why). Current state:
- Training runs through TerraTorch `SemanticSegmentationTask` configs (`configs/<model>.yaml`)
  with Dice loss and each model's official normalization.
- Predictions are saved as float32 logits (`SaveValPredictions`).
- Peru set v1 is frozen (`docs/peru_data_card.md`).
- Sen1Floods11 checkpoints exist for seed 0 only. Tasks 3.4–3.6 of `plan-flood-research`
  (LR sweep, reproducibility, 3 seeds) are open.
- Hardware: Mac M4 Max (MPS) for demos and CPU analysis; RunPod RTX 4090 for training
  (`scripts/cloud/`).

Facts that shape the design (all in `docs/`):
- The shift is the background: bright desert with NDWI near 0.
- UNOSAT labels are derived from an automated 20 m Sentinel-2 product (method unpublished;
  consistent with a SWIR index) with analyst review.
- A label-free MNDWI threshold reaches 0.49–0.57 IoU on dev; one tuned on dev reaches 0.92.
- TerraMind and Prithvi encoders use LayerNorm.

## Goals / Non-Goals

**Goals:**
- A small, fair benchmark of adaptation methods on one shared protocol: 2 backbones first,
  Prithvi last.
- One self-training recipe simple enough for 753 target chips and a few GPU-hours per run.

**Non-Goals:**
- Sentinel-1 or any new sensor.
- Adversarial alignment (AdaptSegNet): superseded by self-training. It is kept only if a
  reviewer-style comparison is needed later.
- Per-event test-time adaptation on `peru-test` images. That would be a transductive setting
  and is a possible later row, not the main protocol.
- New manual labels.

## Decisions

1. **Target pool = `peru-train` images, labels never read.** Using dev/test images, even
   unlabelled, would make the main result transductive. The buffer chips touch dev/test
   blocks. *Alternative:* download extra 2023 dates/areas; optional (task 2.4), only outside
   held-out blocks. Guard: a pool loader that only knows image paths, plus a check against
   the split lists (spec `peru-target-pool`).

2. **Source checkpoints: retrain on Sen1Floods11 with 3 seeds first.** U-Net and TerraMind,
   with the LR sweep (tasks 3.4–3.6 of `plan-flood-research`). Selection is on `valid` only.
   *Why:* adaptation results per seed need a source model per seed. Seed 0 alone cannot give
   mean ± std. Prithvi is done after the method is fixed (cost: about 2× TerraMind).

3. **Self-training as one LightningModule wrapping the TerraTorch model.**
   - Student = the task's model. Teacher = an EMA copy (α = 0.999 per step; tune {0.99, 0.999}).
   - Each step: a source batch (labelled Sen1Floods11, the same Dice loss as source training)
     plus a target batch (pseudo-labelled by the teacher).
   - Pseudo-label loss: pixel cross-entropy on confident pixels only.
   - *Why CE for pseudo-labels:* Dice over a mostly ignored mask is unstable, and CE is what
     DACS/DAFormer/MIC use.
   - Learning rate: 0.1× the source LR with linear warm-up over the first 10 % of iterations.
     DAFormer Table 4 (per the agent notes) shows warm-up matters for transformers.
   - EMA α = 0.99 (DAFormer/DACS default, per the agent notes).

4. **Per-class confidence thresholds (CBST-style) instead of one global cut.**
   - Within each target batch, keep the top q % most confident teacher pixels of each
     predicted class. q grows linearly from 20 % to 50 % over training (CBST's schedule
     starts at 20 % and grows, per the agent notes).
   - *Why:* water is rare (4.6 % of labelled train pixels), so one global threshold lets the
     dry class dominate, and binary self-training can collapse to "all dry".
   - *Alternative:* DAFormer's τ = 0.968 plus an image-level weight. Rejected, because with
     rare water the image weight barely protects the water class.

5. **Mixing: paste source water onto target images (DACS / ClassMix, binary).**
   - Paste the water pixels of a source image, with their labels, onto a target image, whose
     other pixels take teacher pseudo-labels. Colour jitter and blur on the student input only.
   - *Why:* it puts real water next to desert backgrounds, which is the failure mode.

6. **Masked consistency at image level for all backbones.**
   - Mask random 64 × 64 px patches (mask ratio 0.7) of the target image for the student
     (MIC Table 8 best setting, b = 64, r = 0.7, per the agent notes).
   - Target: the teacher's pseudo-label of the unmasked image.
   - *Why image-level, not token-level:* it is the same for the CNN and the ViTs, matches MIC's
     original formulation, and 64 px is a multiple of the 16 px ViT patch.

7. **Input-alignment baselines are run as source training with transformed inputs.**
   - Per-band histogram matching of each source chip to the target-pool CDF.
   - FDA per band, β = 0.01, with a random target-pool amplitude.
   - The transforms run before the model's own normalization.
   - AdaBN for the U-Net: one pass over the target pool in train mode, without gradients,
     to re-estimate BN statistics.

8. **Optional variant: index-aware pseudo-labels.** A pixel enters as water only if the
   teacher is confident and MNDWI > the target-pool Otsu threshold, and as dry only if the
   teacher is confident and MNDWI below it. *Why:* the labels are likely index-based, so this
   tests whether a physically motivated prior helps. It runs only after methods 3–6 work.

9. **Pre-registered defaults and selection without Peru labels.**
   - Every adaptation run uses the defaults above (α = 0.99, q 20 → 50 %, LR 0.1× source with
     10 % warm-up, 64 px / 0.7 masking) and a fixed 4,000 iterations at batch 8 (TerraMind) /
     16 (U-Net), each batch half source, half target, on a 4090. **The final checkpoint is the
     headline result.** No Peru label is read during the run.
   - *Why:* selecting on dev labels would leak Peru label information into a "no-label"
     method. A single MNDWI threshold tuned on dev already reaches 0.924 IoU, so dev labels are
     worth a lot.
   - Checkpoints are saved every 500 iterations and scored on `peru-dev` **offline** afterwards.
     A small dev-tuned variant (q ∈ {20, 50, 80} × α ∈ {0.99, 0.999}, best dev checkpoint) is
     reported only as an "oracle selection" row, next to the dev-tuned MNDWI threshold.
   - Final runs: 3 seeds of the default configuration (and of the oracle variant).

10. **Evaluation reuses the frozen-config gate of `plan-flood-research` section 7.** Phase-D
    configs are added to `configs/frozen.yaml`. Tables come from scripts that read run outputs
    (spec `adaptation-evaluation`).

## Risks / Trade-offs

- **Collapse to all-dry under self-training.** Mitigations: per-class thresholds, the source
  loss in every step, a water-fraction floor warning, and a CoTTA-style stochastic restore
  only if it still happens.
- **Defaults may be far from optimal for binary water.** The headline row may then
  under-sell the method; the oracle-selection row shows the ceiling. Both are reported.
  Every configuration tried is recorded in EXPERIMENT_LOG; `peru-test` and `peru-2017-test`
  stay untouched until the end.
- **Index-label circularity.** Evaluation labels may be SWIR-index based, so
  index-aware pseudo-labels may look good partly by construction. The Otsu-MNDWI baseline is
  reported next to them, and the 2017 EMS labels (a different provider) are the cross-check.
- **Memory.** Student + EMA teacher + masked view for TerraMind at batch 8 on 24 GB is not
  measured. Mitigation: measure in the first cloud demo; fall back to batch 4 with gradient
  accumulation.
- **Source checkpoints differ from the seed-0 ones used in phase C.** The phase-C
  Sen1Floods11-only numbers stay as reported. Phase-D tables use the new 3-seed source models
  for every row.
- **Time.** If the deadline tightens: drop Prithvi, keep U-Net + TerraMind and methods 3 + 6
  only.

## Migration Plan

Not applicable (new capability, nothing replaced). Rollback = do not freeze phase-D configs;
phase A–C results are unaffected.

## Open Questions

- Whether extra unlabelled 2023 imagery (other dates or areas) is worth downloading. Decide
  after the first self-training results on `peru-train` images; this does not change the
  method or protocol.
