## Why

A new Peruvian flood has no labels, and models trained on Sen1Floods11 lose 5–13 water-IoU
points on `peru-dev` (0.80–0.88 vs 0.92–0.94 for Peru-trained models), mainly because the
bright desert background overlaps low-contrast water (`docs/domain_shift_dev.md`). Phase C is
done and the authors chose phase D on 2026-10-14 (decision record in
`docs/failure_analysis_dev.md`). The other candidate, more spatial detail for the foundation
models, can gain at most ~0.5 IoU outside label noise. The literature check
(`docs/literature_phaseD.md`) found no unsupervised adaptation study for optical Sentinel-2
flood segmentation with geospatial foundation models, nor for desert backgrounds.

## What Changes

- Build an **unlabelled Peru target pool** from `peru-train` images (labels never read
  during adaptation), with a guard that refuses dev/test/buffer chips and any label access.
- Add **label-free baselines**: MNDWI > 0, MNDWI with an Otsu threshold from the unlabelled
  pool, and per-band input alignment (histogram matching / FDA-style amplitude transfer).
  Add AdaBN for the U-Net only.
- Add **self-training adaptation**: an EMA teacher, pseudo-labels on Peru chips with separate
  thresholds for water and not-water, and the supervised Sen1Floods11 loss kept in every
  step, plus a masked-consistency variant (MIC-style) for the ViT foundation models. Optional
  variant: pseudo-labels combining teacher confidence with MNDWI.
- Add an **adaptation evaluation protocol**:
  - three levels (source-only → adapted → Peru-trained upper bound) and the share of the gap
    closed;
  - precision and recall reported separately;
  - boundary-tolerant IoU next to plain IoU;
  - selection on `peru-dev` only;
  - ≥ 3 seeds;
  - final scoring on `peru-test` and `peru-2017-test` once, with the pre-registered
    label-sensitivity row.
- Backbones: U-Net (reference) and TerraMind v1-B first; Prithvi-EO-2.0 300M at the end. All
  start from Sen1Floods11-trained checkpoints with ≥ 3 seeds (Sen1Floods11 tasks 3.4–3.6 of
  `plan-flood-research`, run on RunPod).

## Capabilities

### New Capabilities
- `peru-target-pool`: the unlabelled Peru image pool used for adaptation, and the guard that
  keeps labels and held-out chips out of it.
- `domain-adaptation`: the adaptation methods (label-free baselines, input alignment, AdaBN,
  self-training, masked consistency) and their training contract.
- `adaptation-evaluation`: how adapted models are selected, compared and scored, including
  baselines, gap-closed reporting, tolerant IoU and the label-sensitivity row.

### Modified Capabilities
<!-- None: openspec/specs/ is empty; the phase A–C specs live in the unarchived change
`plan-flood-research`, whose evaluation-protocol rules (held-out sets touched once, ≥ 3
seeds, traceable numbers) this change follows without changing them. -->

## Impact

- New code: `src/floodseg/` (target-pool datamodule, adaptation LightningModule / callbacks,
  input-alignment transforms), `scripts/` (index baselines, adaptation runs, gap-closed tables),
  `configs/adapt/`.
- Compute: Sen1Floods11 source runs (3 seeds, U-Net + TerraMind first) and adaptation runs on
  RunPod RTX 4090, about $10–25 in total (estimate to confirm in task 1); local MPS for
  baselines and demos.
- Data: no new download required (the `peru-train` images are already on disk); optional extra
  unlabelled 2023 dates or areas away from dev/test.
- Docs: METHODS_NOTES entries, EXPERIMENT_LOG entries, thesis Chapters IV–V, paper 2.
