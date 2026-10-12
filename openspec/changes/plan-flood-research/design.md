## Context

See proposal.md for motivation and the specs for requirements. Current state: `scripts/download_sen1floods11.py`, `scripts/feasibility_mps.py` (TerraTorch on MPS, measured speeds in `docs/feasibility.md`), an EDA notebook, and EMSR199 vectors in `data/emsr199/raw/`. Constraints: an M4 Max (36 GB, MPS) for development, plus rented cloud GPUs (CUDA) for full runs when a local demo justifies them; two authors, Spanish thesis on Overleaf, a paper target. The feasibility run used TerraTorch's Sen1Floods11 normalization, not the models' official ones.

## Goals / Non-Goals

**Goals:**
- One code path (TerraTorch + Lightning) for all three models, so differences come from the model, not the pipeline.
- Peru set built in parallel with training, frozen before any Peru evaluation.
- Everything that feeds a table is a script, not a notebook.
- Reproduce published SOTA on Sen1Floods11 (within ~2 IoU) so the Peru failure analysis is credible.
- Keep `peru-test` untouched until phase D's method is frozen; all analysis uses `peru-dev`; only `peru-train` is ever trained on.
- Train on Peruvian data (advisor request): the main regime trains only on `peru-train`; Sen1Floods11-trained models are a comparison row.

**Non-Goals:**
- Claiming SOTA from a single seed or a `test`-tuned configuration: the SOTA claim uses mean ± std over seeds (and the ensemble reported separately).
- S1 / multimodal ablation (RQ2) and SegFormer-from-scratch: optional follow-up changes after advisor approval.
- Owning or maintaining GPU infrastructure: cloud GPUs are rented per run and released after.

## Decisions

1. **TerraTorch configs + thin `src/floodseg/` package.** Models, loss and trainer come from TerraTorch YAML in `configs/`; our package adds what TerraTorch lacks: official-normalization datamodules, Peru dataset, uncertainty, metrics, provenance. *Alternative:* hand-written PyTorch loops — more control, but re-implementing decoders and weight loading for two FMs is wasted time.
2. **Normalization from official configs.** TerraMind: raw DN with TerraMind pretraining means/stds (IBM/terramind `terramind_v1_base_sen1floods11.yaml`). Prithvi: 6 HLS bands with its own stats (terratorch `segmentation_sen1floods11.yaml`). U-Net: train-split stats over the 13 bands. *Alternative:* one shared normalization — simpler, but handicaps the FMs and breaks comparability with published numbers.
3. **Decoder: UNetDecoder for both FMs** (as in the official configs). UperNet crashes on MPS with default pool scales; avoid it.
4. **Fixed training budget, early stopping on `valid` water IoU.** Same max epochs (e.g. 100) and patience for every model; lr swept over a small grid (3 values) on seed 0 only, on `valid`. Prithvi uses batch 4 with gradient accumulation ×2–4 to approach the official effective batch.
5. **Ensemble = the seeds.** 5 seeds if time allows, else 3; the ensemble costs only inference. MC-dropout on the best single model (TerraMind is cheaper); temperature scaling fitted on `valid`.
6. **Peru set pipeline (no manual labelling):** UNOSAT 2023 vectors (WaterExtent, CloudObstruction, AnalysisExtent) → rasterize to the S2 grid at 10 m in UTM 17S → 512×512 chips of 13-band L1C from the CDSE Process API, for each label date → automatic label rules (1 water, 0 analysed dry, −1 cloud / outside / no-data) → overlay PNGs and a label-vs-water-index agreement score per chip → manifest CSV with checksums. EMSR199 2017 chips use S2 of 2017-04-07 with EMS/S2 disagreement set to −1. L1C to match Sen1Floods11 and TerraMind. *Alternative:* manual correction in QGIS — dropped, the authors have no time to label.
7. **Run logging:** Lightning CSV logger + a `run.json` (config, seed, commit, dirty flag, versions, timing) per run under `outputs/runs/<model>/<seed>/`. MLflow is optional. *Alternative:* W&B — needs network/account, not required.
8. **Freeze gate:** a `configs/frozen.yaml` lists checkpoints approved for final evaluation; the evaluate script refuses `test`/Peru for anything else.
9. **Where we try to beat SOTA: Peru, not Sen1Floods11.** Sen1Floods11 is used to *reproduce* SOTA. Improvement claims are made on `peru-test` against the best reproduced baseline (see the evaluation-protocol spec). Ensembles and test-time augmentation are baselines-plus, reported separately from single models.

10. **Geographic Peru split.** UNOSAT 2023 chips are grouped into spatial blocks (e.g. ~10×10 km); blocks are assigned to `peru-train` / `peru-dev` / `peru-test` (~70/15/15 by water pixels) with a one-chip buffer, and the same location is in one part for both dates. EMSR199 2017 is its own test, `peru-2017-test` (Salinas, Parachique, Piura; Trujillo excluded, almost no standing water on 2017-04-07). *Alternative:* random chip split — leaks spatial context and inflates scores.

11. **Failure strata from external layers.** Cloud/shadow from the S2 scene classification (or s2cloudless on L1C), land cover from ESA WorldCover (flooded vegetation, urban, bare), terrain from Copernicus DEM. Strata are computed per pixel, so one chip can contribute to several.
12. **Local demo first, then rent compute.** Every experiment first runs locally on MPS as a small demo: the same code and config, a few epochs, on a subset or at reduced batch. It goes to a rented GPU only if the demo (a) runs end-to-end without errors, (b) shows the loss falling and `valid` IoU rising, and (c) has a cost estimate (GPU-hours × price) that we accept. The same code must run on `mps` and `cuda` with only the device/precision changed in config. Data and checkpoints sync through one storage location (e.g. an S3/R2 bucket or the provider's volume); instances are shut down after each batch of runs. *Alternative:* everything on MPS — free, but Prithvi at batch 4 deviates from the official batch 16 and larger models do not fit.
13. **Bolivia:** not evaluated, per advisor. If a reviewer later asks, it can be added as an appendix without changing the protocol.

## Risks / Trade-offs

- [SOTA not reproduced] → investigate normalization, crop size and training budget before phase C; report the residual gap.
- [Peru set too small to split into dev and test] → if <30 water chips, use leave-one-AOI-out cross-validation for analysis and report it as such.
- [Phase C shows no clear dominant failure] → the paper becomes benchmark + analysis; phase D targets the largest stratum anyway.

- [UNOSAT labels are preliminary, not field-validated] → state it; per-chip agreement with a water index; optional 1-hour look at the worst chips.
- [EMSR199 labels are from SAR on 26–30 Mar; clear S2 is 7 Apr] → disagreement pixels set to −1; report the date gap and that this favours easy pixels.
- [Small Peru test/dev sets give noisy metrics] → report per-chip distribution and bootstrap confidence intervals, not just the mean.
- [L1C vs L2A mismatch if only L2A is available] → prefer L1C from CDSE; if forced to L2A, state it as a limitation and test the effect on `valid`.
- [Prithvi at batch 4 differs from the official batch 16] → gradient accumulation; document effective batch.
- [Compute cost on rented GPUs grows with seeds × models × sweeps] → the local demo gate plus a written cost estimate before each rental; U-Net and TerraMind-B can stay on MPS, so only Prithvi and large backbones need GPUs.
- [MPS vs CUDA numerical differences] → all reported runs of a given model use the same device type; the device is logged per run.

## Migration Plan

Update `docs/PLAN.md` (§3 RQ1/RQ4, §4, §7, §9) to replace Bolivia with Peru once this change is accepted. `scripts/feasibility_mps.py` is kept unchanged as a historical reference.

14. **Training on Peru.** Main regime (advisor): each model starts from its pretrained backbone (TerraMind / Prithvi foundation weights, U-Net ImageNet) and is trained only on `peru-train`, early stopping on `peru-dev`, ≥3 seeds. Comparison rows: Sen1Floods11-only models (already trained) and Sen1Floods11 → `peru-train` fine-tuning, to show whether other countries' data helps.

## Open Questions

- Defense deadline and paper venue (sets calendar dates, not the work).
- Final number of seeds (3 vs 5), decided after timing the first full TerraMind run.
- Whether WorldFloods v2 is added as extra training data (licence check pending); would be a separate change.
