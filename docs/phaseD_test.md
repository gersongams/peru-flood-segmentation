# Phase D on the held-out test sets (task 6.2)

One-time evaluation of the configurations frozen in `configs/frozen.yaml` (commit `f1f61d6`, made
before any test file was read; decision in `docs/METHODS_NOTES.md` 2026-10-16 "Phase-D freeze").
Method, checks and corrections: METHODS_NOTES 2026-10-16 "One-time test evaluation". Run record:
`docs/EXPERIMENT_LOG.md`. Tables: `outputs/phaseD/test/`.

## Headline

- **2023 event (`peru-test`, 172 chips):** U-Net trained on Sen1Floods11 with training-time
  histogram matching to unlabelled Peru imagery reaches **0.798 ± 0.005** water IoU (0.79848; written 0.799 until 2026-10-17), 0.030
  [0.013, 0.048] below the Peru-trained U-Net (0.828) and 0.22 above U-Net source-only (0.576). It is
  the best label-free row, 0.058 [0.013, 0.105] above the best source model (Prithvi, 0.741).
- **2017 event (`peru-2017-test`, 101 chips not touching 2023 train):** the order reverses. Every
  source-only model (0.917–0.937) beats every aligned row (0.831–0.887) and every Peru-trained model
  (0.878–0.892). The 2017 labels define dry only where MNDWI ≤ 0, so precision is ≈ 1 for all rows
  and the score is in effect recall; the gap sits in faint water (MNDWI 0–0.3), the recall that
  alignment and Peru training give up to remove false positives (dev strata and 2017 diagnostic).
- **No method wins both events.** The 2023 labels reward removing false positives; the 2017 labels
  can only reward keeping faint water.

## `peru-test` (water IoU, mean ± std over 3 seeds; 95 % chip-bootstrap CI of the seed mean)

| Row | IoU | 95 % CI | tolerant IoU | precision | recall | gap closed | dev IoU |
|---|---|---|---|---|---|---|---|
| U-Net trHM (primary) | 0.7985 ± 0.0048 (0.79848) | 0.704–0.865 | 0.848 | 0.952 | 0.832 | 88 % | 0.912 |
| TerraMind trFDA (primary) | 0.7085 ± 0.0200 | 0.618–0.779 | 0.764 | 0.928 | 0.750 | 44 % | 0.837 |
| Prithvi source-only (primary) | 0.7405 ± 0.0136 | 0.664–0.796 | 0.777 | 0.798 | 0.911 | — | 0.870 |
| TerraMind trHM (secondary) | 0.7209 ± 0.0053 | 0.609–0.802 | 0.769 | 0.896 | 0.787 | 50 % | 0.844 |
| U-Net AdaBN (secondary) | 0.7308 ± 0.0127 | 0.625–0.812 | 0.758 | 0.795 | 0.901 | 61 % | 0.899 |
| TerraMind test-time HM (secondary) | 0.7055 ± 0.0006 | 0.587–0.790 | 0.746 | 0.860 | 0.797 | 42 % | 0.809 |
| U-Net source-only | 0.5763 ± 0.0369 | 0.467–0.669 | 0.582 | 0.597 | 0.947 | — | 0.836 |
| TerraMind source-only | 0.6261 ± 0.0552 | 0.515–0.714 | 0.640 | 0.668 | 0.911 | — | 0.703 |
| Peru-trained U-Net | 0.8280 ± 0.0023 | 0.732–0.895 | 0.861 | 0.966 | 0.853 | — | 0.942 |
| Peru-trained TerraMind | 0.8149 ± 0.0023 | 0.718–0.885 | 0.852 | 0.960 | 0.843 | — | 0.927 |
| Peru-trained Prithvi | 0.8244 ± 0.0066 | 0.731–0.890 | 0.866 | 0.966 | 0.850 | — | 0.922 |

Wide CIs: a few large chips carry much of the water. Paired differences (same resampled chips) are
much tighter: see `outputs/phaseD/test/bootstrap_diffs.csv`. Without doubtful-label chips (166 or
169 chips) every row rises by 0.005–0.016 (all-blobs variant; 0.002–0.007 without near-cloud blobs). One ordering changes in both variants: TerraMind trFDA and test-time HM (0.003 apart) swap. Corrected 2026-10-17; the earlier text said "no ordering changes".

**Dev → test.** The primary and Peru-trained rows drop 0.10–0.13 (U-Net 0.941 → 0.828; recall
0.97 → 0.85), so the test blocks are harder or labelled differently; this is not an adaptation
effect. Other rows differ: U-Net AdaBN −0.169, U-Net source −0.260, TerraMind source −0.077 (corrected 2026-10-17; the earlier text said every row dropped 0.10–0.13). Gap closed rises for U-Net trHM (72 % → 88 %) mostly because source-only fell further.

**Pre-registered TerraMind rule (FDA over HM).** On `peru-test` trHM − trFDA = +0.013 [−0.022, 0.044],
not separable; on 2017 trHM is better (+0.035 [0.012, 0.066]). The primary stays trFDA (no choice is
changed after test); both are reported.

## `peru-2017-test`

| Row | IoU, 101 chips (main) | IoU, all 204 | recall MNDWI 0–0.1 / 0.1–0.3 / > 0.3 (204) |
|---|---|---|---|
| U-Net trHM | 0.854 ± 0.007 | 0.899 | 0.02 / 0.16 / 0.99 |
| TerraMind trFDA | 0.831 ± 0.006 | 0.835 | 0.08 / 0.22 / 0.91 |
| Prithvi source-only | 0.937 ± 0.006 | 0.957 | 0.36 / 0.85 / 1.00 |
| TerraMind trHM | 0.866 ± 0.012 | 0.904 | 0.05 / 0.28 / 0.99 |
| U-Net AdaBN | 0.887 ± 0.013 | 0.924 | 0.08 / 0.45 / 1.00 |
| TerraMind test-time HM | 0.875 ± 0.005 | 0.914 | 0.05 / 0.36 / 0.99 |
| U-Net / TerraMind source-only | 0.917 / 0.934 | 0.946 / 0.952 | 0.34 / 0.64 / 1.00; 0.35 / 0.84 / 1.00 |
| Peru-trained U-Net / TerraMind / Prithvi | 0.878 / 0.892 / 0.878 | 0.916 / 0.919 / 0.912 | 0.02–0.05 / 0.40–0.51 / 0.99–1.00 |

Gap closed is undefined here (Peru-trained ≤ source-only for every backbone). The recall-by-MNDWI
columns are a post-hoc diagnostic declared before it was run.

## Calibration on test

Method fixed before running (METHODS_NOTES 2026-10-16 "Test calibration of the frozen phase-D
rows"). Full tables: `outputs/phaseD/test/calibration.md`. ECE % with ensemble + Sen1Floods11
`valid` temperature (label-free), "all" / "near" (≤ 10 px from water edge):

| Row | `peru-test` | dev (same method) |
|---|---|---|
| U-Net trHM | 0.60 / 12.81 | 0.67 / 4.06 |
| U-Net AdaBN | 0.94 / 11.46 | 0.80 / 5.33 |
| TerraMind trFDA | 0.78 / 20.17 | 0.96 / 9.80 |
| Prithvi source-only | 1.09 / 15.12 | 1.07 / 7.97 |
| Peru-trained U-Net (T from dev labels) | 1.08 / 11.18 | 0.11 / 3.14 |

- **Near-edge calibration degrades on test for every row** (1.6–3.6×), Peru-trained included,
  in line with the harder test areas.
- **The label-free temperature transfers.** T fitted on Sen1Floods11 `valid` is as good as T fitted
  on Peru dev labels for most rows.
- **The dev claim holds only partly.** U-Net trHM has the lowest overall ECE of the label-free
  rows, but near water edges AdaBN U-Net is lower (11.46 vs 12.81 %). That AdaBN figure is within
  0.3 points of the Peru-trained U-Net.
- **Peru-trained Prithvi's confidence does not flag its errors on test** (error AUROC 0.44
  single, 0.46 ensemble; 0.72 on dev). Reported as found, not investigated on test.
- **2017:** source-only rows are the best calibrated, mirroring the IoU reversal. These labels
  cover only clear water and clear dry, so calibration there is measured on the easy cases only.

## Section-7 pass (deferred frozen rows)

METHODS_NOTES 2026-10-16 "Section-7 pass: results"; earlier rows reproduced exactly.

- **Sen1Floods11 → Peru fine-tuning = Peru-only training on test.**

  | Model | Fine-tuned − Peru-only on `peru-test` | 95 % CI |
  |---|---|---|
  | U-Net | +0.001 | −0.003 to 0.005 |
  | TerraMind | −0.001 | −0.006 to 0.003 |
  | Prithvi | −0.002 | −0.006 to 0.002 |

  This confirms on test the dev finding behind "train on Peru only".
- **Sen1Floods11 `test`, source-only:**

  | Model | water IoU | mIoU |
  |---|---|---|
  | U-Net | 0.833 ± 0.005 | 0.904 |
  | Prithvi | 0.820 ± 0.012 | 0.897 |
  | TerraMind | 0.819 ± 0.014 | 0.895 |

  - The three are not separable: U-Net − Prithvi +0.013 [−0.022, 0.062].
  - Prithvi and TerraMind reproduce their papers' `test` numbers within 0.5 IoU
    (`docs/published_comparison.md`).
- **The drop from Sen1Floods11 `test` to `peru-test`** (source-only):

  | Model | Drop |
  |---|---|
  | U-Net | −0.257 |
  | TerraMind | −0.192 |
  | Prithvi | −0.080 |

  Both foundation models lose less than U-Net. Prithvi loses by far the least; TerraMind's margin
  is within its seed spread (corrected 2026-10-16, METHODS_NOTES).
- Not evaluated (not frozen for it): Peru-trained, fine-tuned and training-time-aligned models on
  Sen1Floods11 `test`.
