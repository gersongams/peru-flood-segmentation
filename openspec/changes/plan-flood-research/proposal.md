## Why

We are restarting the thesis from scratch and need a research plan that yields a strong thesis and at least one paper. Flood *forecasting* needs hydrological models and gauges and is out of reach; flood *detection* from satellite imagery is feasible and well supported by data. On the standard benchmark (Sen1Floods11) detection scores have plateaued around 82–83 water IoU (Prithvi-CAFE Table 1, arXiv 2601.02315 — to verify), so +0.3 IoU there is not a contribution. The gap is **generalization to new regions**: there is no public labelled flood benchmark for Peru, and a course teacher suggested evaluating on Peru instead of Bolivia. Our strategy: **reproduce SOTA → build a Peru benchmark → find where SOTA fails there → design a method that fixes that failure.**

## What Changes

Phases A–C are specified in this change. Phases D and E become their own changes once C gives evidence.

- **A. Reproduce SOTA.** Train U-Net (CNN baseline), TerraMind v1-B and Prithvi-EO-2.0 300M with TerraTorch on Sen1Floods11 official splits (train 252 / valid 89 / test 90), each with its official normalization, ≥3 seeds, and match published numbers within tolerance.
- **B. Peru dataset (no manual labelling).** Build Sentinel-2 L1C chips for Peru from expert flood maps: UNOSAT Cyclone Yaku 2023 (Lambayeque + Piura, labels drawn on the same Sentinel-2 images) split by space into `peru-train` / `peru-dev` / `peru-test`, plus Copernicus EMS EMSR199 2017 as a cross-event test `peru-2017-test`. Labels follow fixed rules; doubtful chips are flagged automatically. Sources: `docs/peru_label_sources.md`.
- **B2. Train on Peru (teacher's suggestion; Sen1Floods11 → Peru fine-tuning later changed dev IoU by ≤ 0.4 points).** Main regime: fine-tune each pretrained backbone on `peru-train` only. Comparison row: the Sen1Floods11-trained models (already run), with and without Peru fine-tuning.
- **C. Failure analysis.** Measure the in-distribution → Peru drop and break errors down by cloud, flooded vegetation, urban, desert/coastal terrain and water-body type; quantify uncertainty calibration (ECE, error-detection AUROC) as part of this.
- **D. Method (later change).** 3–4 approaches aimed at the main failure found in C — candidates: S1+S2 fusion robust to missing/cloudy optical input (possibly via TerraMind's cross-modal generation), test-time / unsupervised domain adaptation to Peru, uncertainty-guided correction. Whether to use Sentinel-1 is the authors' decision (task 6.5).
- **E. Flood susceptibility (optional later change).** Combine detected Peru flood maps with terrain (DEM, HAND) to map flood-prone zones.
- Compute policy: everything runs first as a small local demo on the M4 Max; full runs go to rented GPUs only when the demo works and the cost estimate is accepted.
- **BREAKING (vs `docs/PLAN.md`)**: Bolivia is dropped; Peru is the generalization test. The goal shifts from "beat SOTA on Sen1Floods11" to "beat SOTA on Peru / on the failure modes we identify".

## Capabilities

### New Capabilities
- `sen1floods11-data`: download, official splits, per-model normalization and loading of Sen1Floods11 hand-labelled chips.
- `peru-test-set`: construction from UNOSAT 2023 and EMSR199 maps with automatic label rules, geographic split, quality checks and documentation of the Peru dataset.
- `model-training`: reproducible multi-seed fine-tuning of U-Net, TerraMind v1-B and Prithvi-EO-2.0 300M, local demo then rented GPU.
- `uncertainty-estimation`: ensembles, MC-dropout, temperature scaling, uncertainty maps and calibration metrics.
- `evaluation-protocol`: metrics, held-out discipline, multi-seed aggregation, SOTA reproduction check, results tables.
- `failure-analysis`: stratified error analysis on `peru-dev` that decides which method phase D targets.

### Modified Capabilities
<!-- None: openspec/specs/ is empty. -->

## Impact

- New code: `configs/`, `src/floodseg/`, `scripts/` (train, evaluate, Peru dataset build, analysis, figures). `scripts/feasibility_mps.py` stays as reference.
- Data under `data/` (git-ignored): full Sen1Floods11, Sentinel-2 L1C for Peru (2023-03-21, 2023-04-01, 2017-04-07), UNOSAT 2023 and EMSR199 vectors, auxiliary layers (land cover, DEM) for stratification.
- Docs: rewrite `docs/PLAN.md` to this strategy; add a Peru data card and an experiment log.
- Thesis: Chapter I RQs reworded (Peru, failure analysis, method); Chapters III–V follow phases A–D.
- Paper plan: (1) Peru benchmark + SOTA failure analysis; (2) the phase-D method.
- Cost: rented GPU hours for Prithvi and later larger backbones / method experiments, each justified by a local demo.
