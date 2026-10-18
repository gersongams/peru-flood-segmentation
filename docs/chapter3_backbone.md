# Chapter III backbone: claims, sources and hypothesis verdicts

Started 2026-10-17. This file is the ledger for `thesis/2_CAPITULO3/Capitulo3.tex`. Every number in the chapter must appear here with its source and split, or come from a file listed here. The rules are in `docs/WRITING_RULES.md` (§4 integrity, §7 numbers, §11 claim precision).

## Outline

| Section | Content | Main sources |
|---|---|---|
| 3.1.1 Conjunto de datos peruano | construction rules, ignore rule, spatial blocks, split table, 2017 set | `docs/peru_data_card.md` |
| 3.1.2 Modelos y entrenamiento | three backbones, inputs, decoder, loss, optimizer, LR selection, regimes, seeds, hardware, reproducibility tolerance | `configs/*.yaml`; run `config.yaml` files; METHODS_NOTES 2026-10-15 "Sen1Floods11 source models", 2026-10-16 "Pod session 2 results"; EXPERIMENT_LOG 2026-10-13/14 |
| 3.1.3 Adaptación y protocolo de evaluación | pool and label guard, AdaBN, HM and FDA (test and training time), self-training, selection on dev, freeze and gate | METHODS_NOTES 2026-10-15 (pool, AdaBN, amendments, self-training details), 2026-10-16 (freeze) |
| 3.1.4 Sen1Floods11 | `valid` (selection) and `test` (frozen); published comparison | `docs/published_comparison.md`; `outputs/phaseD/test/test_groups.csv`, `bootstrap_diffs.csv` |
| 3.1.5 Peru dev: transfer and errors | regimes; transfer matrix; strata; domain shift | `docs/domain_shift_dev.md`, `docs/failure_analysis_dev.md`, `outputs/phaseD/analysis/error_shares.csv` |
| 3.1.6 Adaptation on dev | main dev table | `docs/phaseD_dev.md`, `outputs/phaseD/dev_table.md` |
| 3.1.7 Test (one pass) | `peru-test`, `peru-2017-test`, paired differences, drop from Sen1Floods11 | `docs/phaseD_test.md`, `outputs/phaseD/test/*.csv` |
| 3.1.8 Calibration and uncertainty | dev temperatures and ensembles; test ECE and AUROC | `docs/uncertainty_dev.md`, `docs/calibration_dev.md`, `outputs/phaseD/test/calibration.md` |
| 3.1.9 System | **pending**: commented placeholder only (WRITING_RULES §11.11) | — |
| 3.3 Contrastación | verdicts below | — |
| 3.2 Discusión | after the authors see the verdicts; the authors rewrite it (§9) | — |

## Number traps (decided)

- **AdaBN U-Net dev row.** Use 0.899 ± 0.012 (the reproducible recomputation). Footnote the original 0.897 ± 0.016 (METHODS_NOTES 2026-10-16 "Phase-D freeze", pre-test check).
- **Learning rates differ by regime.**
  - Source models: swept. U-Net 3e-4, TerraMind 5e-5, Prithvi 1e-4; 200-epoch cap.
  - Peru-only and fine-tuned runs: config LRs (U-Net 1e-4, TerraMind 2e-5, Prithvi 1e-4); 100-epoch cap.
  - Read from each run's `config.yaml`.
- **Weight decay.** U-Net 0.05, TerraMind 0.01, Prithvi 0.1 (run configs).
- **Prithvi `valid`:** 0.832 (from the full-precision mean 83.15).
- **Units.** Ours are proportions to 3 decimals; published values are given in % as published. ECE is in % with 2 decimals.
- **CIs.** For a difference between rows, cite `bootstrap_diffs.csv`. The per-row CI in `phaseD_test.md` is the chip CI of that row's seed mean. Never call `p_a_not_better` a p-value.
- **TerraMind primary row** stays training-time FDA. Training-time HM is reported as the secondary row.
- **Post-hoc labels:** self-training without AdaBN; 2017 recall by MNDWI; the error shares by stratum (`scripts/error_shares_dev.py`, 2026-10-17, derived from existing strata output, no new model run).
- **2017.** The main result uses 101 chips; give all 204 alongside, with the label caveat (concordant pixels only, precision ≈ 1, so the score is in effect recall).

## Hypothesis verdicts (proposed; for the authors before §3.3 is written)

### HE1. Foundation models vs U-Net

**(a)** "In Sen1Floods11 no significant difference is expected (95 % CI)."
- Evidence on Sen1Floods11 `test`:
  - U-Net − Prithvi = +0.013 [−0.022, 0.062];
  - U-Net − TerraMind = +0.015 [−0.018, 0.051] (`bootstrap_diffs.csv`, `test`).
- **Verdict: consistent with the hypothesis.** Both CIs include 0. Write "no se detectó una diferencia" (§11.8), not equivalence.

**(b)** "In Peru the foundation models show a smaller IoU drop than U-Net."
- Drop from `test` to `peru-test`, source-only: U-Net −0.257, TerraMind −0.192, Prithvi −0.080. No CI for the difference of drops: the test sets have different chips.
- Paired `peru-test` comparisons, source-only:
  - U-Net − Prithvi = −0.164 [−0.229, −0.102];
  - U-Net − TerraMind = −0.050 [−0.093, 0.002].
- **Verdict: supported for Prithvi.** On Sen1Floods11 no difference was detected, and in Peru Prithvi is clearly above U-Net.
- **Not established for TerraMind.** The point estimate goes the hypothesised way, but the Peru CI includes 0, and its margin (0.064; full precision 0.2568 − 0.1924) is about one TerraMind seed std on `peru-test` (0.055).
- 2017 (101 chips), source-only: U-Net − Prithvi −0.020 [−0.048, 0.002]; U-Net − TerraMind −0.017 [−0.066, 0.022]. Not separable.

### HE2. Errors, adaptation, calibration

**(a) Errors are concentrated in a few strata: cloud edges, flooded vegetation, desert or coastal terrain.**
Evidence: dev only, descriptive, post hoc (`error_shares.csv`, derived from the pre-specified strata), no CI. **No threshold for "concentrated" was fixed in advance.** The verdict depends on two choices the authors must make (below).

| Stratum (share of dev pixels) | Peru-trained, share of errors | Source-only, share of errors | Reading |
|---|---|---|---|
| ≤ 1 px label edge (2.9 %) | 58.6–66.3 % | U-Net 38.8, TerraMind 18.5, Prithvi 57.8 % | strongly concentrated (not a named stratum) |
| WorldCover permanent water (2.5 %) | 12.0–14.3 % | 3.3–8.1 % | concentrated (not named) |
| 5–20 px from thick cloud (8.3 %) | 7.6–9.4 % | U-Net 6.4, **TerraMind 14.6** (≈ 1.8×), Prithvi 6.4 % | proportional except for TerraMind source |
| herbaceous wetland, a flooded-vegetation proxy (0.2 %) | 0.9 % (≈ 5×) | 0.3–0.8 % | over-represented but negligible in size |
| bare/sparse, desert and coast (85.2 %) | 77.9–80.3 % | 84.7–91.5 % | about proportional; it is where the source models' false positives sit (FPR source 0.026 vs Peru 0.003, U-Net) |

Also strong, but not named in the hypothesis: low-contrast water (NDWI −0.1 to 0.1; band IoU 0.45–0.53 for the Peru-trained models) and water bodies < 1 ha (recall 45–58 %).

**Decisions (authors, 2026-10-17; post hoc operationalization, to be stated as such in §3.3):**
1. **Models:** the verdict refers to the **source-only models** (Sen1Floods11-trained, applied to Peru unadapted). Reason: HE2 compares adaptation against "el modelo sin adaptar" in the next sentence. Peru-trained and adapted rows are reported as complementary only.
2. **Threshold:** a stratum is **concentrated** when error share / pixel share **≥ 2**.

Mapping of the named strata (pre-specified strata families): cloud edges = distance to thick cloud < 20 px (bands < 5 px and 5–20 px together); flooded vegetation = WorldCover herbaceous wetland (proxy); desert or coast = WorldCover bare/sparse.

**Result under these decisions** (source-only, `error_shares.csv`; ratio = error share / pixel share):

| Named stratum (pixel share) | U-Net | TerraMind | Prithvi | Concentrated (≥ 2×)? |
|---|---|---|---|---|
| cloud edges < 20 px (8.4 %) | 0.78 | 1.80 | 0.77 | no (TerraMind closest) |
| herbaceous wetland (0.18 %) | 3.27 (0.6 % of errors) | 1.64 | 4.32 (0.8 %) | U-Net and Prithvi, but < 1 % of errors |
| bare/sparse (85.2 %) | 1.02 | 1.07 | 0.99 | no |
| *not named:* label edge ≤ 1 px (2.9 %) | 13.48 | 6.43 | 20.06 | yes, all |
| *not named:* permanent water (2.5 %) | 2.62 | 1.32 | 3.28 | U-Net and Prithvi |

The cloud-edge stratum is the pooled < 20 px band, as defined in `docs/failure_analysis_dev.md` Finding 4 and approved with the "bordes de nubes" wording; `scripts/error_shares_dev.py` now writes it as a "<20 px" row, and Tabla 3.4 uses it. The sub-bands are not separate strata: 5–20 px alone gives 0.77 / 1.76 / 0.77, and < 5 px alone 4.41 for TerraMind (0.11 % of pixels).

**Verdict: partly supported** (applying the approved ≥ 2× rule literally). Errors concentrate in few strata, mostly the label edge and permanent water, which are not named. Of the three named strata only flooded vegetation reaches 2×, for two of three models; that it holds under 1 % of the errors is interpretation, not part of the criterion. Desert or coast matters only as where the source models' false positives sit (FPR source 0.026 vs Peru 0.003, U-Net). Complementary: the Peru-trained models show the same pattern (edge 20.35–23.04×, permanent water 4.85–5.79×, herbaceous wetland 4.60–4.78×, cloud < 20 px 0.93–1.14×).

**(b) Unsupervised adaptation raises water IoU over the unadapted model (CI > 0) without reaching the Peru-trained model.**
- **2023 `peru-test`:**
  - U-Net trHM − source = +0.222 [0.132, 0.309]; trHM − Peru-trained = −0.030 [−0.048, −0.013]. **Supported.**
  - TerraMind trFDA − source = +0.082 [0.016, 0.156]; it stays below Peru-trained (0.708 vs 0.815). No paired CI against Peru-trained TerraMind; the gap is large. **Supported.**
  - Prithvi: no adaptation was frozen, because every method lowered its dev mean (AdaBN −0.003, test-time HM −0.052, training-time HM −0.010; AdaBN improved seed 0 alone). The prediction **could not be tested on test for Prithvi**. On dev it is not supported.
- **2017 (101 chips):** reversed.
  - U-Net trHM − source = −0.063 [−0.083, −0.046];
  - TerraMind trFDA − source = −0.103 [−0.175, −0.048].
  - **Refuted on 2017**, with the label caveat: 2017 labels reward recall of faint water.
- Overall: **partly supported**.

**(c) Uncalibrated probabilities are overconfident; temperature scaling and ensembles reduce ECE; error AUROC > 0.5.**
- **Overconfidence: supported.** Every fitted temperature is above 1:
  - Peru-trained models (dev, 2-fold): U-Net 2.35, TerraMind 1.49, Prithvi 3.33.
  - Frozen label-free rows, from their own Sen1Floods11 `valid` (`temperatures.json`):
    - U-Net trHM 1.82–2.22;
    - TerraMind trFDA 1.24–1.52;
    - Prithvi source 2.62–3.23.
  - In-domain source models: U-Net 2.02–2.12, TerraMind 1.13–1.36, Prithvi 2.80–3.23.
  - TerraMind is the closest to calibrated everywhere.
  - Exception in the reliability diagrams (Peru-trained, dev): TerraMind's two lowest-confidence bins lie above the diagonal (confidence 0.517 → accuracy 0.569; 0.550 → 0.566).
- **ECE reduction: supported on `peru-test`.** Ensemble plus Sen1Floods11 temperature lowers ECE against single models for every frozen row (all pixels and near edges; `calibration.md`). Point estimates only; no CIs (Chapter I §1.4).
- **AUROC > 0.5: holds overall for every frozen phase-D row on `peru-test` (0.598–0.783 single).**
  - Fails for Peru-trained Prithvi on `peru-test` (0.439 single, 0.456 ensemble).
  - Fails on dev in the low-contrast and small-water strata (0.41–0.56 for U-Net and Prithvi).
- **Verdict: mostly supported**, with these stated exceptions.

### HE3. System

**Pending.** The system has not been built or measured. The protocol was fixed 2026-10-17 (`docs/system_evaluation_protocol.md`). No result text before measurement (§11.11).
