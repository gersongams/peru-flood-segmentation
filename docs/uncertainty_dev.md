# Uncertainty on `peru-dev`: seed ensembles and temperature scaling (tasks 5.1, 5.3, 5.4)

Corrected 2026-10-15: edge-distance numbers recomputed after the chip-frame fix (commit f849700; docs/METHODS_NOTES.md 2026-10-15). Old values: outputs/analysis_dev_before_edgefix_2026-10-15/.

Date: 2026-10-14 (revised the same day, see Correction). Split: `peru-dev` only. Script:
`scripts/uncertainty_dev.py`; numbers from `outputs/analysis_dev/uncertainty.csv`. Inputs:
the per-chip float32 water logits of the 3 seeds per model (`outputs/logits_dev/`, see
`docs/EXPERIMENT_LOG.md`). Method details: `docs/METHODS_NOTES.md`.

- **single**: each seed alone, mean ± std over 3 seeds. Uncertainty = −|logit| (same
  ranking as 1 − max(p, 1 − p)).
- **temp**: each seed after temperature scaling. T is fitted (NLL) on the labelled pixels
  of half the dev chips and applied to the other half (2-fold by chip, fixed split), so no
  pixel is scored with a T fitted on it. AUROC is not reported: scaling is monotone within
  a fold, so pooled over the folds it only measures the mismatch between the two Ts.
- **ensemble**: mean probability of the 3 seeds (computed as a logit, both tails in log
  space); uncertainty = predictive entropy, or mutual information (MI, the part of the
  uncertainty that comes from seeds disagreeing).

ECE: 15 equal-width bins. AUROC: exact, how well the uncertainty ranks wrong pixels above
right ones. "all" = every 7th labelled pixel; "near" = all labelled pixels within 10 px of a
label water edge (where nearly all errors are, see `docs/failure_analysis_dev.md`). IoU is
on the "all" sample. Caveat: the fine-tuned seeds all start from one Sen1Floods11
checkpoint, so that ensemble is less diverse.

**Correction (2026-10-14).** The first version of this file used float16 probabilities,
which saturate asymmetrically near 0 and 1; that biased every AUROC and temperature here
(e.g. Prithvi ensemble AUROC 0.861 was an artefact; the logit-based value is 0.758). IoU and
raw ECE were not affected. All numbers below come from the float32 logits.

## Peru-only

| Method | Model | IoU | ECE all % | ECE near % | AUROC all | AUROC near |
|---|---|---|---|---|---|---|
| single | U-Net | 0.9416 ± 0.0018 | 0.66 ± 0.02 | 5.42 ± 0.17 | 0.734 ± 0.037 | 0.814 ± 0.018 |
| single | TerraMind | 0.9267 ± 0.0024 | 0.65 ± 0.15 | 6.17 ± 0.32 | 0.695 ± 0.034 | 0.815 ± 0.002 |
| single | Prithvi | 0.9220 ± 0.0006 | 0.90 ± 0.02 | 7.60 ± 0.20 | 0.723 ± 0.046 | 0.801 ± 0.001 |
| temp | U-Net | 0.9416 ± 0.0018 | 0.20 ± 0.06 | 4.02 ± 0.38 | — | — |
| temp | TerraMind | 0.9267 ± 0.0024 | 0.36 ± 0.03 | 5.27 ± 0.07 | — | — |
| temp | Prithvi | 0.9220 ± 0.0006 | 0.38 ± 0.10 | 5.01 ± 0.44 | — | — |
| ensemble (entropy) | U-Net | 0.9435 | 0.54 | 4.42 | 0.760 | 0.833 |
| ensemble (entropy) | TerraMind | 0.9285 | 0.54 | 5.36 | 0.749 | 0.820 |
| ensemble (entropy) | Prithvi | 0.9244 | 0.74 | 6.33 | 0.758 | 0.810 |
| ensemble (MI) | U-Net | 0.9435 | 0.54 | 4.42 | 0.750 | 0.823 |
| ensemble (MI) | TerraMind | 0.9285 | 0.54 | 5.36 | 0.757 | 0.805 |
| ensemble (MI) | Prithvi | 0.9244 | 0.74 | 6.33 | 0.785 | 0.808 |

## Sen1Floods11 → Peru

| Method | Model | IoU | ECE all % | ECE near % | AUROC all | AUROC near |
|---|---|---|---|---|---|---|
| single | U-Net | 0.9384 ± 0.0018 | 0.69 ± 0.02 | 5.74 ± 0.23 | 0.805 ± 0.008 | 0.820 ± 0.022 |
| single | TerraMind | 0.9284 ± 0.0008 | 0.41 ± 0.09 | 5.76 ± 0.14 | 0.729 ± 0.018 | 0.815 ± 0.006 |
| single | Prithvi | 0.9238 ± 0.0005 | 0.89 ± 0.01 | 7.43 ± 0.09 | 0.743 ± 0.029 | 0.798 ± 0.010 |
| temp | U-Net | 0.9384 ± 0.0018 | 0.20 ± 0.04 | 4.11 ± 0.36 | — | — |
| temp | TerraMind | 0.9284 ± 0.0008 | 0.36 ± 0.04 | 5.23 ± 0.02 | — | — |
| temp | Prithvi | 0.9238 ± 0.0005 | 0.38 ± 0.06 | 4.75 ± 0.30 | — | — |
| ensemble (entropy) | U-Net | 0.9396 | 0.58 | 4.81 | 0.869 | 0.847 |
| ensemble (entropy) | TerraMind | 0.9299 | 0.31 | 5.02 | 0.762 | 0.827 |
| ensemble (entropy) | Prithvi | 0.9251 | 0.79 | 6.58 | 0.776 | 0.805 |
| ensemble (MI) | U-Net | 0.9396 | 0.58 | 4.81 | 0.887 | 0.846 |
| ensemble (MI) | TerraMind | 0.9299 | 0.31 | 5.02 | 0.817 | 0.819 |
| ensemble (MI) | Prithvi | 0.9251 | 0.79 | 6.58 | 0.787 | 0.803 |
Fitted temperatures (mean ± std over seeds): Peru-only U-Net 2.35 ± 0.22, TerraMind
1.49 ± 0.25, Prithvi 3.33 ± 0.09; Sen1Floods11 → Peru U-Net 2.34 ± 0.09, TerraMind
1.19 ± 0.07, Prithvi 3.83 ± 0.31. T > 1 means the raw model is overconfident.

## What it shows

1. **All models are overconfident, Prithvi most** (T 3.3–3.8), then U-Net (2.3),
   TerraMind least (1.2–1.5).
2. **Temperature scaling cuts the calibration error near water by 15–34 %**:
   Peru-only ECE near water 5.42 → 4.02 % (U-Net), 6.17 → 5.27 % (TerraMind),
   7.60 → 5.01 % (Prithvi); overall ECE drops to 0.20–0.38 %. It does not change IoU.
3. **Seed ensembles add a little accuracy; they detect errors better overall, but near water only slightly.** Peru-only
   ensemble IoU is +0.2 points over the mean single seed for every model (U-Net
   0.9416 → 0.9435). Error-detection AUROC near water rises only slightly, from 0.80–0.82 (single) to
   0.81–0.83 (entropy); with MI it rises for U-Net and Prithvi but falls for TerraMind
   (0.815 → 0.805). Overall it rises from 0.69–0.73 to 0.75–0.78. For Sen1Floods11 → Peru
   U-Net the gain is larger (0.805 → 0.869 overall). Ensembling alone improves ECE less than
   temperature scaling (exception: Sen1Floods11 → Peru TerraMind near water, 5.02 vs 5.23 %);
   both together are best (`docs/calibration_dev.md`).
4. Ranking by calibration near water after scaling: U-Net (4.02 %) < Prithvi (5.01 %) <
   TerraMind (5.27 %). The foundation models are not better calibrated than the CNN.

Reliability diagrams, ensemble + temperature scaling and per-stratum results:
`docs/calibration_dev.md`. Still open: MC-dropout (task 5.2).
