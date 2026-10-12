## Context

See proposal.md for motivation and the specs for requirements. Current state: `scripts/download_sen1floods11.py`, `scripts/feasibility_mps.py` (TerraTorch on MPS, measured speeds in `docs/feasibility.md`), an EDA notebook, and EMSR199 vectors in `data/emsr199/raw/`. Constraints: an M4 Max (36 GB, MPS) for development, plus rented cloud GPUs (CUDA) for full runs when a local demo justifies them; two authors, Spanish thesis on Overleaf, a paper target. The feasibility run used TerraTorch's Sen1Floods11 normalization, not the models' official ones.

## Goals / Non-Goals

**Goals:**
- One code path (TerraTorch + Lightning) for all three models, so differences come from the model, not the pipeline.
- Peru set built in parallel with training, frozen before any Peru evaluation.
- Everything that feeds a table is a script, not a notebook.
- Reproduce published SOTA on Sen1Floods11 (within ~2 IoU) so the Peru failure analysis is credible.
- Keep `peru-test` untouched until phase D's method is frozen; all analysis uses `peru-dev`; only `peru-train` is ever trained on.
- Train on Peruvian data (advisor request): every model is reported both Sen1Floods11-only and fine-tuned on `peru-train`.

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
6. **Peru set pipeline:** EMS polygons → rasterize to the S2 grid at 10 m → tile to 512×512 chips with ≥ some water → manual correction in QGIS → `reviewed` flag in a chip manifest (CSV with checksums). Imagery from Planetary Computer (L2A available) or CDSE for **L1C**, since Sen1Floods11 and TerraMind use L1C; prefer L1C to avoid a processing-level shift. Agreement subset: ≥10 chips labelled by both authors before seeing each other's labels.
7. **Run logging:** Lightning CSV logger + a `run.json` (config, seed, commit, dirty flag, versions, timing) per run under `outputs/runs/<model>/<seed>/`. MLflow is optional. *Alternative:* W&B — needs network/account, not required.
8. **Freeze gate:** a `configs/frozen.yaml` lists checkpoints approved for final evaluation; the evaluate script refuses `test`/Peru for anything else.
9. **Where we try to beat SOTA: Peru, not Sen1Floods11.** Sen1Floods11 is used to *reproduce* SOTA. Improvement claims are made on `peru-test` against the best reproduced baseline (see the evaluation-protocol spec). Ensembles and test-time augmentation are baselines-plus, reported separately from single models.

10. **Geographic Peru split (train/dev/test).** Split by AOI so no area is in two parts: e.g. `peru-test` = Salinas, `peru-train` = Parachique, `peru-dev` = Piura (+ Piura Norte). If `peru-train` is too small, split Parachique + Piura into spatial blocks with a one-chip buffer instead, keeping Salinas whole for test; Trujillo excluded unless visual review shows standing water (the 7 Apr 2017 survey found almost none, `docs/peru_dataset_proposal.md` §3b). Final assignment after chips exist, balancing water pixels. *Alternative:* random chip split — leaks spatial context and inflates scores.

11. **Failure strata from external layers.** Cloud/shadow from the S2 scene classification (or s2cloudless on L1C), land cover from ESA WorldCover (flooded vegetation, urban, bare), terrain from Copernicus DEM. Strata are computed per pixel, so one chip can contribute to several.
12. **Local demo first, then rent compute.** Every experiment first runs locally on MPS as a small demo: the same code and config, a few epochs, on a subset or at reduced batch. It goes to a rented GPU only if the demo (a) runs end-to-end without errors, (b) shows the loss falling and `valid` IoU rising, and (c) has a cost estimate (GPU-hours × price) that we accept. The same code must run on `mps` and `cuda` with only the device/precision changed in config. Data and checkpoints sync through one storage location (e.g. an S3/R2 bucket or the provider's volume); instances are shut down after each batch of runs. *Alternative:* everything on MPS — free, but Prithvi at batch 4 deviates from the official batch 16 and larger models do not fit.
13. **Bolivia:** not evaluated, per advisor. If a reviewer later asks, it can be added as an appendix without changing the protocol.

## Risks / Trade-offs

- [SOTA not reproduced] → investigate normalization, crop size and training budget before phase C; report the residual gap.
- [Peru set too small to split into dev and test] → if <30 water chips, use leave-one-AOI-out cross-validation for analysis and report it as such.
- [Phase C shows no clear dominant failure] → the paper becomes benchmark + analysis; phase D targets the largest stratum anyway.

- [EMS labels are from SAR on 26–30 Mar; clear S2 is 7 Apr, water may have receded] → manual correction on S2 is mandatory; report the date gap; check S1 of late March as a visual aid.
- [Labelling a few hundred chips takes time] → the EMS polygons give the initial label, so most chips need correction, not drawing; split the AOIs between the two authors.
- [Small Peru test/dev sets give noisy metrics] → report per-chip distribution and bootstrap confidence intervals, not just the mean.
- [L1C vs L2A mismatch if only L2A is available] → prefer L1C from CDSE; if forced to L2A, state it as a limitation and test the effect on `valid`.
- [Prithvi at batch 4 differs from the official batch 16] → gradient accumulation; document effective batch.
- [Compute cost on rented GPUs grows with seeds × models × sweeps] → the local demo gate plus a written cost estimate before each rental; U-Net and TerraMind-B can stay on MPS, so only Prithvi and large backbones need GPUs.
- [MPS vs CUDA numerical differences] → all reported runs of a given model use the same device type; the device is logged per run.
- [Label quality disputes between authors] → written labelling guideline before the agreement subset.

## Migration Plan

Update `docs/PLAN.md` (§3 RQ1/RQ4, §4, §7, §9) to replace Bolivia with Peru once this change is accepted. `scripts/feasibility_mps.py` is kept unchanged as a historical reference.

14. **Peru fine-tuning.** Start from each model's best Sen1Floods11 checkpoint, fine-tune on `peru-train` with the same loss, lower lr and early stopping on `peru-dev`, ≥3 seeds. Alternative (ablation): train on Sen1Floods11 + `peru-train` jointly. Peru-only training from the pretrained backbone is reported as a secondary row.

## Open Questions

- Defense deadline and paper venue (sets calendar dates, not the work).
- Final number of seeds (3 vs 5), decided after timing the first full TerraMind run.
- Whether WorldFloods v2 is added as extra training data (licence check pending); would be a separate change.
