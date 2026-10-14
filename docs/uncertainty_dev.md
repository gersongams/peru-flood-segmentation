# Uncertainty on `peru-dev`: seed ensembles and temperature scaling (tasks 5.1, 5.3, 5.4)

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
| single | U-Net | 0.9416 ± 0.0018 | 0.66 ± 0.02 | 3.16 ± 0.10 | 0.734 ± 0.037 | 0.783 ± 0.004 |
| single | TerraMind | 0.9267 ± 0.0024 | 0.65 ± 0.15 | 3.54 ± 0.24 | 0.695 ± 0.034 | 0.766 ± 0.017 |
| single | Prithvi | 0.9220 ± 0.0006 | 0.90 ± 0.02 | 4.42 ± 0.11 | 0.723 ± 0.046 | 0.793 ± 0.016 |
| temp | U-Net | 0.9416 ± 0.0018 | 0.20 ± 0.06 | 2.10 ± 0.22 | — | — |
| temp | TerraMind | 0.9267 ± 0.0024 | 0.36 ± 0.03 | 2.72 ± 0.02 | — | — |
| temp | Prithvi | 0.9220 ± 0.0006 | 0.38 ± 0.10 | 2.65 ± 0.27 | — | — |
| ensemble (entropy) | U-Net | 0.9435 | 0.54 | 2.58 | 0.760 | 0.808 |
| ensemble (entropy) | TerraMind | 0.9285 | 0.54 | 3.06 | 0.749 | 0.792 |
| ensemble (entropy) | Prithvi | 0.9244 | 0.74 | 3.68 | 0.758 | 0.815 |
| ensemble (MI) | U-Net | 0.9435 | 0.54 | 2.58 | 0.750 | 0.804 |
| ensemble (MI) | TerraMind | 0.9285 | 0.54 | 3.06 | 0.757 | 0.794 |
| ensemble (MI) | Prithvi | 0.9244 | 0.74 | 3.68 | 0.785 | 0.819 |

## Sen1Floods11 → Peru

| Method | Model | IoU | ECE all % | ECE near % | AUROC all | AUROC near |
|---|---|---|---|---|---|---|
| single | U-Net | 0.9384 ± 0.0018 | 0.69 ± 0.02 | 3.34 ± 0.13 | 0.805 ± 0.008 | 0.815 ± 0.016 |
| single | TerraMind | 0.9284 ± 0.0008 | 0.41 ± 0.09 | 3.22 ± 0.12 | 0.729 ± 0.018 | 0.783 ± 0.008 |
| single | Prithvi | 0.9238 ± 0.0005 | 0.89 ± 0.01 | 4.33 ± 0.05 | 0.743 ± 0.029 | 0.786 ± 0.013 |
| temp | U-Net | 0.9384 ± 0.0018 | 0.20 ± 0.04 | 2.16 ± 0.19 | — | — |
| temp | TerraMind | 0.9284 ± 0.0008 | 0.36 ± 0.04 | 2.73 ± 0.02 | — | — |
| temp | Prithvi | 0.9238 ± 0.0005 | 0.38 ± 0.06 | 2.46 ± 0.15 | — | — |
| ensemble (entropy) | U-Net | 0.9396 | 0.58 | 2.80 | 0.869 | 0.857 |
| ensemble (entropy) | TerraMind | 0.9299 | 0.31 | 2.77 | 0.762 | 0.805 |
| ensemble (entropy) | Prithvi | 0.9251 | 0.79 | 3.83 | 0.776 | 0.804 |
| ensemble (MI) | U-Net | 0.9396 | 0.58 | 2.80 | 0.887 | 0.869 |
| ensemble (MI) | TerraMind | 0.9299 | 0.31 | 2.77 | 0.817 | 0.827 |
| ensemble (MI) | Prithvi | 0.9251 | 0.79 | 3.83 | 0.787 | 0.806 |
Fitted temperatures (mean ± std over seeds): Peru-only U-Net 2.35 ± 0.22, TerraMind
1.49 ± 0.25, Prithvi 3.33 ± 0.09; Sen1Floods11 → Peru U-Net 2.34 ± 0.09, TerraMind
1.19 ± 0.07, Prithvi 3.83 ± 0.31. T > 1 means the raw model is overconfident.

## What it shows

1. **All models are overconfident, Prithvi most** (T 3.3–3.8), then U-Net (2.3),
   TerraMind least (1.2–1.5).
2. **Temperature scaling cuts the calibration error near water by 23–40 %**:
   Peru-only ECE near water 3.16 → 2.10 % (U-Net), 3.54 → 2.72 % (TerraMind),
   4.42 → 2.65 % (Prithvi); overall ECE drops to 0.20–0.38 %. It does not change IoU.
3. **Seed ensembles add a little accuracy and detect errors somewhat better.** Peru-only
   ensemble IoU is +0.2 points over the mean single seed for every model (U-Net
   0.9416 → 0.9435). Error-detection AUROC near water rises from 0.77–0.79 (single) to
   0.79–0.82 (entropy or MI); overall from 0.69–0.73 to 0.75–0.78. For Sen1Floods11 → Peru
   U-Net the gain is larger (0.805 → 0.869 overall). Ensembling alone improves ECE less than
   temperature scaling; both together are best (`docs/calibration_dev.md`).
4. Ranking by calibration near water after scaling: U-Net (2.1 %) < Prithvi (2.65 %) ≈
   TerraMind (2.7 %). The foundation models are not better calibrated than the CNN.

Reliability diagrams, ensemble + temperature scaling and per-stratum results:
`docs/calibration_dev.md`. Still open: MC-dropout (task 5.2).
