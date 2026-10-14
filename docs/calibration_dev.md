# Calibration: reliability diagrams and calibration per stratum (tasks 5.3, 5.4, 6.4)

Date: 2026-10-14. Splits: Sen1Floods11 `valid` (check) and `peru-dev`. No test split used.
Script: `scripts/calibration_dev.py` (float32 logits in `outputs/logits_valid/`,
`outputs/logits_dev/`); tables: `scripts/report_calibration.py` →
`outputs/analysis_dev/calib_tables.md` (all strata, both regimes); raw numbers:
`outputs/analysis_dev/{calib_valid.csv,calib_strata.csv,reliability_bins.csv,temperatures_valid.json}`;
figures: `outputs/analysis_dev/figs/reliability_*.png`. Method details and design decisions:
`docs/METHODS_NOTES.md`; runs: `docs/EXPERIMENT_LOG.md`.

Definitions. Confidence = max(p, 1 − p). Reliability diagram and ECE: 15 equal-width bins on
[0.5, 1]; ECE = Σ n_b |accuracy_b − confidence_b| / N. Error-detection AUROC: exact
rank-based AUROC of −|logit| (single model) or of the ensemble's predictive entropy, for
separating wrong from right pixels. All labelled pixels are used. "near" = within 10 px of a
label water edge. Methods: **single** (each of 3 seeds, mean ± std), **temp** (temperature
scaling, T fitted 2-fold by chip on dev; for Sen1Floods11-only models T comes from `valid`),
**ensemble** (mean probability of 3 seeds), **ens+temp** (ensemble, then 2-fold temperature
scaling). No AUROC for 2-fold temperature scaling (it only measures the mismatch between the two
fold temperatures).

## 1. Check on Sen1Floods11 `valid` (in-domain)

Seed-0 Sen1Floods11 models on their own validation split (89 chips, 20.3 M labelled pixels).

| Model | T (all valid) | IoU | ECE all % raw → temp | ECE near % raw → temp | AUROC all | AUROC near |
|---|---|---|---|---|---|---|
| U-Net | 1.73 | 0.8188 | 1.74 → 0.68 | 5.89 → 3.87 | 0.780 | 0.770 |
| TerraMind | 1.01 | 0.8168 | 1.06 → 1.16 | 3.76 → 3.66 | 0.739 | 0.741 |
| Prithvi | 2.80 | 0.8327 | 1.73 → 0.75 | 5.98 → 3.43 | 0.813 | 0.822 |

The code checks pass (ECE from the reliability bins = `ece()`; IoU = logged `val/IoU_Flood`).
Even in-domain, U-Net and Prithvi are clearly overconfident (T 1.7 and 2.8; in the diagram,
pixels with 0.88–0.92 confidence are right only 60–63 % of the time, `reliability_bins.csv`); TerraMind is close to
calibrated (T 1.01), and temperature scaling does not help it. Hypothesis, not tested: all
three are trained with Dice loss, which pushes outputs to 0/1, but that does not explain why
TerraMind differs.

Applying the `valid` temperature to the same models on `peru-dev` (domain shift):

| Model | ECE all % raw → T_valid | ECE near % raw → T_valid |
|---|---|---|
| U-Net | 1.36 → 2.35 | 5.89 → 3.25 |
| TerraMind | 0.52 → 0.57 | 5.73 → 5.65 |
| Prithvi | 1.55 → 1.08 | 7.63 → 5.13 |

An in-domain temperature only partly transfers: it helps near water for U-Net and Prithvi,
but makes U-Net's overall ECE worse; it does not help TerraMind.

## 2. Reliability on `peru-dev` (Peru-only, 3 seeds pooled)

ECE % (`figs/reliability_perudev_peru.png`; Sen1Floods11 → Peru in
`reliability_perudev_s1f11_ft.png`):

| Model | all: single / temp / ensemble / ens+temp | near water: single / temp / ensemble / ens+temp |
|---|---|---|
| U-Net | 0.66 / 0.20 / 0.54 / 0.11 | 3.16 / 2.10 / 2.58 / 1.64 |
| TerraMind | 0.65 / 0.36 / 0.54 / 0.25 | 3.54 / 2.72 / 3.06 / 2.35 |
| Prithvi | 0.90 / 0.39 / 0.75 / 0.23 | 4.42 / 2.65 / 3.68 / 1.97 |

All raw models lie below the diagonal (overconfident) at every confidence level. Temperature
scaling and ensembling each help and add up: ens+temp halves the near-water ECE for U-Net
(3.16 → 1.64 %) and Prithvi (4.42 → 1.97 %), and cuts TerraMind's by a third
(3.54 → 2.35 %). Over 99 % of pixels sit in the top bin, which is why the overall ECE is small.

## 3. Calibration and error detection per stratum (`peru-dev`, Peru-only)

Selected cells; all strata and the Sen1Floods11 → Peru regime are in
`outputs/analysis_dev/calib_tables.md`. ECE: single (mean ± std over seeds) → ens+temp.
AUROC: single seed / ensemble. Strata defined from the inputs come first. Strata defined from
the label (distance to the label edge, water-only strata) are partly a restatement of the
error rate: in water-only strata every error is a miss.

**Inputs (NDWI over all pixels, land cover):**

| Stratum (errors, U-Net) | U-Net ECE % | U-Net AUROC | TerraMind ECE % | TerraMind AUROC | Prithvi ECE % | Prithvi AUROC |
|---|---|---|---|---|---|---|
| NDWI −0.1–0 (47,039) | 0.20 ± 0.00 → 0.32 | 0.707 / 0.777 | 0.14 ± 0.03 → 0.52 | 0.650 ± 0.066 / 0.710 | 0.24 ± 0.01 → 0.41 | 0.501 ± 0.053 / 0.524 |
| NDWI 0–0.1 (88,905) | 1.60 ± 0.04 → 0.53 | 0.685 / 0.685 | 1.73 ± 0.20 → 0.79 | 0.673 / 0.725 | 2.24 ± 0.03 → 0.87 | 0.715 / 0.751 |
| NDWI 0.1–0.3 (107,614) | 4.65 ± 0.22 → 2.28 | 0.777 / 0.800 | 5.96 ± 0.31 → 4.00 | 0.739 / 0.773 | 6.74 ± 0.15 → 2.69 | 0.753 / 0.789 |
| WorldCover permanent water (41,926) | 4.07 ± 0.07 → 3.11 | 0.844 / 0.865 | 4.72 ± 0.16 → 3.93 | 0.844 / 0.842 | 4.61 ± 0.14 → 2.74 | 0.575 / 0.579 |
| WorldCover bare/sparse (228,722) | 0.59 ± 0.02 → 0.16 | 0.738 / 0.769 | 0.58 ± 0.15 → 0.31 | 0.720 / 0.774 | 0.84 ± 0.02 → 0.30 | 0.739 / 0.773 |

**Label-defined strata:**

| Stratum (errors, U-Net) | U-Net ECE % | U-Net AUROC | TerraMind ECE % | TerraMind AUROC | Prithvi ECE % | Prithvi AUROC |
|---|---|---|---|---|---|---|
| edge ≤ 1 px (174,486) | 8.46 ± 0.35 → 4.16 | 0.733 / 0.744 | 10.29 ± 0.63 → 7.21 | 0.724 / 0.739 | 13.38 ± 0.37 → 5.86 | 0.698 / 0.716 |
| > 10 px from edge (40,811) | 0.12 ± 0.00 → 0.35 | 0.600 ± 0.110 / 0.607 | 0.12 ± 0.03 → 0.52 | 0.481 ± 0.064 / 0.571 | 0.15 ± 0.01 → 0.46 | 0.513 ± 0.074 / 0.549 |
| water, NDWI −0.1–0 (36,227) | 40.89 ± 0.73 → 35.88 | 0.561 ± 0.050 / 0.586 | 42.23 ± 1.78 → 38.98 | 0.707 / 0.685 | 45.04 ± 0.24 → 38.92 | 0.447 / 0.449 |
| water, NDWI 0–0.1 (67,404) | 36.18 ± 1.96 → 28.89 | 0.472 / 0.492 | 37.22 ± 1.19 → 33.60 | 0.636 / 0.634 | 38.75 ± 1.55 → 29.09 | 0.422 / 0.443 |
| water body < 1 ha (15,890) | 38.22 ± 4.10 → 27.55 | 0.533 / 0.555 | 47.18 ± 1.43 → 40.89 | 0.563 / 0.519 | 51.53 ± 1.32 → 37.79 | 0.413 / 0.430 |
| water body > 100 ha (81,345) | 1.59 ± 0.12 → 0.90 | 0.843 / 0.882 | 2.00 ± 0.11 → 1.48 | 0.943 / 0.958 | 2.18 ± 0.19 → 0.56 | 0.683 / 0.714 |

Single-seed AUROC std over seeds is shown where it is ≥ 0.05; all others are below 0.05 (full tables: `calib_tables.md`).

## What it shows

1. **Calibration is worst where the errors are.** ECE is 8–13 % on the 1-px edge band
   against ≤ 0.15 % more than 10 px from the edge; Prithvi is worst at the edge (13.4 %),
   U-Net best (8.5 %). Ens+temp halves the edge ECE for U-Net and Prithvi (4.2 %, 5.9 %),
   less for TerraMind (7.2 %).
2. **The uncertainty does not flag the dominant failure modes.** Missed low-contrast water
   (NDWI −0.1 to 0.1) and water bodies under 1 ha are missed with high confidence: error
   AUROC 0.41–0.56 for U-Net and Prithvi (below 0.5 means the misses are *more* confident
   than the hits), 0.56–0.71 for TerraMind. Ensembling barely changes this. The same pattern
   holds for Sen1Floods11 → Peru (Prithvi 0.35–0.44 there). An uncertainty map would not
   point a user to these areas. Caveat: in the turbid/shallow-water chips the label itself
   is doubtful (`docs/failure_analysis_dev.md`, Finding 2), so part of these "confident
   misses" may be confident disagreement with a doubtful label, the same confound as in point 3.
3. **Errors far from any label edge are not detected either** (AUROC 0.48–0.60). These are
   mostly confident disagreements with the label, consistent with the label errors of
   Finding 3 in `docs/failure_analysis_dev.md` (e.g. permanent water labelled dry).
4. **TerraMind's uncertainty ranks errors best inside the water strata** (e.g. AUROC
   0.94 on water bodies > 100 ha, 0.64–0.71 on low-contrast water), although its overall
   AUROC is the lowest (0.696). Prithvi's is worst there (0.41–0.58 on low-contrast and
   small water, 0.68 on > 100 ha). One reading, not tested: Prithvi's overall AUROC (0.725)
   comes more from separating easy from hard strata than from ranking errors within them.
5. **On permanent water (WorldCover)**, the stratum richest in label errors, U-Net and
   TerraMind reach AUROC 0.84, Prithvi 0.58.
6. Clouds and date change calibration little (raw ECE 0.5–0.9 % in every cloud-distance
   band from 5 px on; AUROC 0.54–0.76; the < 5 px band has under 1,000 errors).

For phase D, this argues that edge-aware methods should be judged on calibration as well
as IoU, and that low-contrast and small water need a better model (or better labels),
not a post-hoc uncertainty fix. These are hypotheses for the decision record (task 6.5), not tested here.
