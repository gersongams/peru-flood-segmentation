# Uncertainty on `peru-dev`: seed ensembles and temperature scaling (tasks 5.1, 5.3, 5.4)

Date: 2026-10-14. Split: `peru-dev` only. Script: `scripts/uncertainty_dev.py`; numbers
from `outputs/analysis_dev/uncertainty.csv`. Inputs: the per-chip water probabilities of
the 3 seeds per model (`outputs/preds_dev/`, see `docs/EXPERIMENT_LOG.md`).

- **single**: each seed alone, mean ± std over 3 seeds. Uncertainty = 1 − max(p, 1 − p).
- **temp**: each seed after temperature scaling. T is fitted (NLL) on the labelled pixels
  of half the dev chips and applied to the other half (2-fold by chip, fixed split), so no
  pixel is scored with a T fitted on it.
- **ensemble**: mean probability of the 3 seeds; uncertainty = predictive entropy, or
  mutual information (MI, the part of the uncertainty that comes from seeds disagreeing).

ECE: 15 equal-width bins. AUROC: exact, how well the uncertainty ranks wrong pixels above
right ones. "all" = every 7th labelled pixel; "near" = all labelled pixels within 10 px of a
label water edge (where nearly all errors are, see `docs/failure_analysis_dev.md`). IoU is
on the "all" sample. Caveats: logits are recovered from float16 probabilities, so they
saturate at |logit| ≈ 7.6; the fine-tuned seeds all start from one Sen1Floods11
checkpoint, so that ensemble is less diverse; T differs slightly between the two folds,
which can change AUROC a little even though scaling is monotone within a fold.

## Peru-only

| Method | Model | IoU | ECE all % | ECE near % | AUROC all | AUROC near |
|---|---|---|---|---|---|---|
| single | U-Net | 0.9416 ± 0.0018 | 0.66 ± 0.02 | 3.16 ± 0.10 | 0.661 ± 0.013 | 0.749 ± 0.005 |
| single | TerraMind | 0.9267 ± 0.0024 | 0.65 ± 0.15 | 3.54 ± 0.24 | 0.691 ± 0.035 | 0.758 ± 0.022 |
| single | Prithvi | 0.9220 ± 0.0006 | 0.90 ± 0.02 | 4.42 ± 0.11 | 0.655 ± 0.044 | 0.750 ± 0.023 |
| temp | U-Net | 0.9416 ± 0.0018 | 0.19 ± 0.05 | 2.12 ± 0.21 | 0.677 ± 0.048 | 0.745 ± 0.007 |
| temp | TerraMind | 0.9267 ± 0.0024 | 0.38 ± 0.01 | 2.71 ± 0.05 | 0.688 ± 0.038 | 0.757 ± 0.023 |
| temp | Prithvi | 0.9220 ± 0.0006 | 0.24 ± 0.00 | 2.92 ± 0.16 | 0.752 ± 0.025 | 0.750 ± 0.011 |
| ensemble (entropy) | U-Net | 0.9435 | 0.54 | 2.58 | 0.714 | 0.780 |
| ensemble (entropy) | TerraMind | 0.9285 | 0.54 | 3.06 | 0.747 | 0.789 |
| ensemble (entropy) | Prithvi | 0.9244 | 0.74 | 3.68 | 0.861 | 0.801 |
| ensemble (MI) | U-Net | 0.9435 | 0.54 | 2.58 | 0.748 | 0.788 |
| ensemble (MI) | TerraMind | 0.9285 | 0.54 | 3.06 | 0.758 | 0.793 |
| ensemble (MI) | Prithvi | 0.9244 | 0.74 | 3.68 | 0.845 | 0.790 |

## Sen1Floods11 → Peru

| Method | Model | IoU | ECE all % | ECE near % | AUROC all | AUROC near |
|---|---|---|---|---|---|---|
| single | U-Net | 0.9384 ± 0.0018 | 0.69 ± 0.02 | 3.34 ± 0.13 | 0.733 ± 0.015 | 0.777 ± 0.008 |
| single | TerraMind | 0.9284 ± 0.0008 | 0.41 ± 0.09 | 3.22 ± 0.12 | 0.727 ± 0.018 | 0.781 ± 0.010 |
| single | Prithvi | 0.9238 ± 0.0005 | 0.89 ± 0.01 | 4.33 ± 0.05 | 0.712 ± 0.023 | 0.734 ± 0.019 |
| temp | U-Net | 0.9384 ± 0.0018 | 0.20 ± 0.02 | 2.15 ± 0.20 | 0.744 ± 0.023 | 0.773 ± 0.011 |
| temp | TerraMind | 0.9284 ± 0.0008 | 0.39 ± 0.04 | 2.69 ± 0.01 | 0.720 ± 0.017 | 0.778 ± 0.009 |
| temp | Prithvi | 0.9238 ± 0.0005 | 0.24 ± 0.01 | 2.94 ± 0.11 | 0.703 ± 0.037 | 0.723 ± 0.025 |
| ensemble (entropy) | U-Net | 0.9396 | 0.58 | 2.80 | 0.836 | 0.840 |
| ensemble (entropy) | TerraMind | 0.9299 | 0.31 | 2.77 | 0.760 | 0.803 |
| ensemble (entropy) | Prithvi | 0.9251 | 0.79 | 3.83 | 0.845 | 0.787 |
| ensemble (MI) | U-Net | 0.9396 | 0.58 | 2.80 | 0.840 | 0.834 |
| ensemble (MI) | TerraMind | 0.9299 | 0.31 | 2.77 | 0.824 | 0.829 |
| ensemble (MI) | Prithvi | 0.9251 | 0.79 | 3.83 | 0.826 | 0.772 |

Fitted temperatures (mean ± std over seeds): Peru-only U-Net 2.31 ± 0.19, TerraMind
1.50 ± 0.24, Prithvi 2.70 ± 0.03; Sen1Floods11 → Peru U-Net 2.33 ± 0.09, TerraMind
1.20 ± 0.07, Prithvi 2.73 ± 0.02. T > 1 means the raw models are overconfident.

## What it shows

1. **All models are overconfident, Prithvi and U-Net most** (T 2.3–2.7, TerraMind 1.2–1.5).
2. **Temperature scaling cuts the calibration error near water by a quarter to a third**:
   Peru-only ECE 3.16 → 2.12 % (U-Net), 3.54 → 2.71 % (TerraMind), 4.42 → 2.92 % (Prithvi);
   overall ECE drops to 0.19–0.38 %. It does not change IoU. Near water it leaves error
   detection unchanged (AUROC within 0.01); the overall AUROC moves for Prithvi
   (0.655 → 0.752) only because the two folds get different temperatures.
3. **Seed ensembles detect errors better and add a little accuracy.** Peru-only ensemble
   IoU is +0.2 points over the mean single seed for every model (U-Net 0.9416 → 0.9435);
   error-detection AUROC near water rises from 0.75–0.76 to 0.78–0.80 (entropy or MI),
   and overall from 0.66–0.69 to 0.71–0.86. The ensemble improves ECE less than
   temperature scaling does.
4. Ranking by calibration after scaling: U-Net (2.1 %) < TerraMind (2.7 %) < Prithvi
   (2.9 %) near water; the foundation models are not better calibrated than the CNN.

Next: temperature-scale the ensemble too (fit T on the ensemble mean), MC-dropout on the
best single model (task 5.2), reliability diagrams, and per-stratum calibration (6.4).
