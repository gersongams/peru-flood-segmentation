## 1. Plan and repo setup

- [x] 1.1 Rewrite `docs/PLAN.md` to the A–E strategy (detection core, Peru benchmark, failure analysis, method, optional susceptibility)
- [x] 1.2 Create `configs/`, `src/floodseg/` (package in `pyproject.toml`), `outputs/runs/` layout
- [x] 1.3 Add a run-provenance helper writing `run.json` (config, seed, commit, dirty flag, versions, hardware, timing)
- [x] 1.4 Add the MPS memory guard (abort above `recommended_max_memory`) as a Lightning callback
- [x] 1.5 Make device and precision config-only (`mps` / `cuda`, fp32 / 16-mixed); add a `--demo` mode (few epochs, data subset)
- [ ] 1.6 Pick a GPU provider; script setup, data sync and teardown; write a cost estimate template (deferred until the Mac is the bottleneck; RunPod RTX 4090 preferred, AWS only with research credits)

## 2. Sen1Floods11 data

- [x] 2.1 Download all official-split S2Hand + LabelHand chips; verify counts 252/89/90 and that every file opens
- [x] 2.2 Write official split lists (replace the 48/8 feasibility subset lists)
- [x] 2.3 Implement per-model normalization (TerraMind, Prithvi HLS 6-band, U-Net train stats) and log it per run
- [x] 2.4 Test: ignore pixels (−1) excluded from loss and metrics; split sizes enforced; `bolivia` rejected

## 3. Training pipeline

- [x] 3.1 TerraTorch config for U-Net (ResNet encoder); one end-to-end run on seed 0
- [x] 3.2 TerraTorch config for TerraMind v1-B (UNetDecoder, official normalization); one run on seed 0
- [x] 3.3 TerraTorch config for Prithvi-EO-2.0 300M (batch 4 + grad accumulation); one run on seed 0
- [ ] 3.4 Learning-rate sweep (3 values) per model on `valid`, seed 0; fix hyperparameters
- [ ] 3.5 Check seed reproducibility on MPS (repeat one run) and record tolerance in the experiment log
- [ ] 3.6 Train all models × ≥3 seeds; decide 3 vs 5 seeds from measured timing
- [x] 3.7 Verify the SOTA reference number(s) in the source PDFs (table + page) and record them in `docs/REFERENCES.md`; note which inputs each used (the official TerraMind Sen1Floods11 config uses S2L1C + S1GRD, not S2 only)
- [ ] 3.8 Compare our `valid` results with published numbers; investigate any gap > 2 IoU before phase C

## 4. Peru dataset (no manual labelling)

- [x] 4.1 Find label sources that need no manual labelling; record them (`docs/peru_label_sources.md`); download UNOSAT 2023 vectors
- [ ] 4.2 Download S2 L1C (13 bands, 10 m, UTM 17S) chips for the UNOSAT 2023 analysed area on 2023-03-21 and 2023-04-01; local demo on a few chips first
- [ ] 4.3 Rasterize UNOSAT layers to labels with the automatic rules; write the chip manifest with provenance fields
- [ ] 4.4 Overlays and label-vs-water-index agreement per chip; list the doubtful chips
- [ ] 4.5 Build `peru-2017-test` from EMSR199 + S2 2017-04-07 with disagreement pixels set to −1
- [ ] 4.6 Spatial-block split into `peru-train` / `peru-dev` / `peru-test`; guard that refuses training on held-out parts
- [ ] 4.7 Freeze v1 (checksums) and write the data card (`docs/peru_data_card.md`)

## 4b. Training on Peru

- [ ] 4b.1 Peru-only training config per model (pretrained backbone → `peru-train`, early stopping on `peru-dev`); local demo first
- [ ] 4b.2 Run ≥3 seeds per model
- [ ] 4b.3 Comparison rows: Sen1Floods11-only models and Sen1Floods11 → `peru-train` fine-tuning, evaluated on `peru-dev`

## 5. Uncertainty

- [ ] 5.1 Ensemble inference over seed checkpoints (mean probability, entropy, mutual information)
- [ ] 5.2 MC-dropout inference on the best single model
- [ ] 5.3 Temperature scaling fitted on `valid`; store temperature per model
- [ ] 5.4 ECE, reliability diagrams and error-detection AUROC; validate on `valid` first
- [ ] 5.5 Export georeferenced probability and uncertainty rasters per chip

## 6. Failure analysis (phase C, on `peru-dev`)

- [ ] 6.1 Build stratum layers: cloud/shadow, WorldCover land cover, Copernicus DEM, permanent water, water-body size
- [ ] 6.2 Evaluate all models on `peru-dev`; IoU drop vs `valid`; band-distribution comparison
- [ ] 6.3 Stratified IoU / FN / FP tables (mean ± std over seeds)
- [ ] 6.4 Calibration and error-detection AUROC per stratum
- [ ] 6.5 Write the decision record: dominant failure mode(s) and 3–4 candidate methods for phase D
- [ ] 6.6 Open the phase-D OpenSpec change (and ask the advisor about Sentinel-1 if needed)

## 7. Final evaluation of baselines

- [ ] 7.1 Implement the evaluate script with split-level + per-chip metrics and the `frozen.yaml` gate
- [ ] 7.2 Freeze configurations and checkpoints in `configs/frozen.yaml`
- [ ] 7.3 Evaluate once on `test`, `peru-test` and `peru-2017-test` v1 for all models, all training regimes (Peru-only, Sen1Floods11-only, Sen1Floods11 → Peru) and uncertainty variants
- [ ] 7.4 Generate tables (mean ± std, test vs Peru, drop) and bootstrap CIs for `peru-test`
- [ ] 7.5 Generate figures: reliability diagrams, Peru prediction + uncertainty maps, failure cases

## 8. Thesis and paper

- [ ] 8.1 Chapters I–II draft during sections 2–3 (RQs reworded for Peru)
- [ ] 8.2 Chapter III (data, Peru set construction, protocol) after 4.8
- [ ] 8.3 Chapter IV (models, training, uncertainty) after 5
- [ ] 8.4 Chapters V–VI from sections 6–7 outputs; every number traced to a results file or cited table
- [ ] 8.5 Paper 1 draft: Peru benchmark + SOTA failure analysis (paper 2 comes with phase D)
