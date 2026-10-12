## Why

We are restarting the thesis from scratch and need a research plan that yields a strong thesis and at least one paper. Flood *forecasting* needs hydrological models and gauges and is out of reach; flood *detection* from satellite imagery is feasible and well supported by data. On the standard benchmark (Sen1Floods11) detection scores have plateaued around 82–83 water IoU (Prithvi-CAFE Table 1, arXiv 2601.02315 — to verify), so +0.3 IoU there is not a contribution. The gap is **generalization to new regions**: there is no public labelled flood benchmark for Peru, and the advisor asked us to evaluate on Peru instead of Bolivia. Our strategy: **reproduce SOTA → build a Peru benchmark → find where SOTA fails there → design a method that fixes that failure.**

## What Changes

Phases A–C are specified in this change. Phases D and E become their own changes once C gives evidence.

- **A. Reproduce SOTA.** Train U-Net (CNN baseline), TerraMind v1-B and Prithvi-EO-2.0 300M with TerraTorch on Sen1Floods11 official splits (train 252 / valid 89 / test 90), each with its official normalization, ≥3 seeds, and match published numbers within tolerance.
- **B. Peru benchmark.** Build a labelled Sentinel-2 flood dataset from Copernicus EMS EMSR199 (Niño Costero 2017, Piura + Trujillo), corrected by hand, with inter-annotator agreement and a data card. Split it **geographically** into `peru-train` (fine-tuning on Peruvian data, as the advisor asked), `peru-dev` (model selection, analysis, method design) and `peru-test` (touched once, at the end). Target: a few hundred reviewed chips, ≥150 with water for training.
- **B2. Fine-tune on Peru.** Fine-tune each Sen1Floods11 model on `peru-train` and compare with the Sen1Floods11-only model on `peru-dev` / `peru-test`: how much does Peruvian training data help?
- **C. Failure analysis.** Measure the in-distribution → Peru drop and break errors down by cloud, flooded vegetation, urban, desert/coastal terrain and water-body type; quantify uncertainty calibration (ECE, error-detection AUROC) as part of this.
- **D. Method (later change).** 3–4 approaches aimed at the main failure found in C — candidates: S1+S2 fusion robust to missing/cloudy optical input (possibly via TerraMind's cross-modal generation), test-time / unsupervised domain adaptation to Peru, uncertainty-guided correction. Needs advisor approval if it uses Sentinel-1.
- **E. Flood susceptibility (optional later change).** Combine detected Peru flood maps with terrain (DEM, HAND) to map flood-prone zones.
- Compute policy: everything runs first as a small local demo on the M4 Max; full runs go to rented GPUs only when the demo works and the cost estimate is accepted.
- **BREAKING (vs `docs/PLAN.md`)**: Bolivia is dropped; Peru is the generalization test. The goal shifts from "beat SOTA on Sen1Floods11" to "beat SOTA on Peru / on the failure modes we identify".

## Capabilities

### New Capabilities
- `sen1floods11-data`: download, official splits, per-model normalization and loading of Sen1Floods11 hand-labelled chips.
- `peru-test-set`: construction, labelling, geographic train/dev/test split, quality control and documentation of the Peru EMSR199 dataset.
- `model-training`: reproducible multi-seed fine-tuning of U-Net, TerraMind v1-B and Prithvi-EO-2.0 300M, local demo then rented GPU.
- `uncertainty-estimation`: ensembles, MC-dropout, temperature scaling, uncertainty maps and calibration metrics.
- `evaluation-protocol`: metrics, held-out discipline, multi-seed aggregation, SOTA reproduction check, results tables.
- `failure-analysis`: stratified error analysis on `peru-dev` that decides which method phase D targets.

### Modified Capabilities
<!-- None: openspec/specs/ is empty. -->

## Impact

- New code: `configs/`, `src/floodseg/`, `scripts/` (train, evaluate, Peru dataset build, analysis, figures). `scripts/feasibility_mps.py` stays as reference.
- Data under `data/` (git-ignored): full Sen1Floods11, Sentinel-2 (and possibly Sentinel-1) for Peru 2017, EMSR199 vectors, auxiliary layers (land cover, DEM) for stratification.
- Docs: rewrite `docs/PLAN.md` to this strategy; add a Peru data card and an experiment log.
- Thesis: Chapter I RQs reworded (Peru, failure analysis, method); Chapters III–V follow phases A–D.
- Paper plan: (1) Peru benchmark + SOTA failure analysis; (2) the phase-D method.
- Cost: rented GPU hours for Prithvi and later larger backbones / method experiments, each justified by a local demo.
