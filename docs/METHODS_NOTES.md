# Methods notes (for writing the thesis and paper)

What was done and why, dated, newest at the bottom. Each entry gives the method in
enough detail for a Methods section, the alternatives rejected, the checks, results with
their source file, caveats and corrections, and where it goes in the thesis/paper.
Runs (commits, run dirs, costs) are in `docs/EXPERIMENT_LOG.md`; finished analyses have
their own write-up in `docs/`. Earlier work (before 2026-10-14 afternoon) is described in
`docs/peru_data_card.md`, `docs/failure_analysis_dev.md` and `docs/uncertainty_dev.md`.

## 2026-10-14 — How predictions are stored for uncertainty analysis (logits, not float16 probabilities)

**What and why.** Calibration and error-detection metrics are computed offline from
per-pixel predictions that `floodseg.callbacks.SaveValPredictions` saves during
`terratorch validate`. Until commit `583c018` it saved the water probability as float16.
float16 cannot represent confident probabilities symmetrically: in [0.99951, 1] it has
only the values 0.99951 and 1.0 (everything ≥ 0.99976 becomes 1.0), whereas near 0 it
still resolves values down to about 1e-7. About 99 % of `peru-dev` pixels are that
confident (Prithvi Peru-only seed 0: 11.8 % with p ≥ 0.9995, 87.1 % with p ≤ 0.0005), so
the ranking of confident pixels by uncertainty depended on which class was predicted.

**Effect found.** Error-detection AUROC and the fitted temperatures were biased; IoU,
recall/FPR and raw ECE were not (they depend only on the 0.5 threshold and the top
confidence bin). Example, Prithvi Peru-only seed 0, all labelled dev pixels: AUROC 0.607
(float16 as stored) vs 0.687 with symmetric clipping; seed ensemble 0.862 vs 0.771.
Source: `scripts/float16_diagnostic.py` (reads the old `outputs/preds_dev/`).

**Method now.** The callback saves z = logit_water − logit_other as float32 (p = sigmoid(z)).
All analysis is done on float64 logits (`scripts/uncertainty_dev.py` helpers):
- prediction: z > 0; confidence: sigmoid(|z|); single-model uncertainty for AUROC: −|z|;
- temperature scaling: z / T, T fitted by NLL (bounded search on log T ∈ [−3, 3]);
- seed ensemble: logit of the mean probability, log p̄ − log(1 − p̄), with p̄ and 1 − p̄ each
  averaged in log space (log-sum-exp), so no tail saturates; predictive entropy is a
  monotone function of |ensemble logit|, so it is ranked by −|logit|;
- mutual information = entropy(ensemble) − mean entropy(seeds), entropy computed stably
  from the logit: H = softplus(−|z|) + |z|·sigmoid(−|z|).

**Alternatives rejected.** Symmetric clipping of the stored float16 probabilities at
0.99951: cheap, but it ties ~99 % of pixels, so AUROC could only ask whether an error is in
the least-confident 1 %; it cannot rank the confident errors that Finding 3 (label errors)
is about. Used only as a diagnostic.

**Checks.** All 24 predictions recomputed from the same checkpoints; every run's water IoU
equals its training log to 4 decimals (`docs/EXPERIMENT_LOG.md`). Helpers checked on
synthetic logits: ensemble logit and entropy match the naive formulas to ~1e-13; ECE of a
perfectly calibrated synthetic sample ≈ 0.2 % (sampling noise). Raw ECE on dev is unchanged
by the switch (U-Net Peru-only 0.66 %, near water 3.16 %), as expected.

**Thesis use.** Chapter IV (uncertainty methods): state that logits are stored and why;
a footnote or appendix on the float16 pitfall is a useful practical lesson. Every AUROC and
temperature reported before this date is superseded.

## 2026-10-14 — Calibration check on Sen1Floods11 `valid` (task 5.4, "validate on valid first")

**What and why.** Before interpreting calibration on `peru-dev`, check the reliability-
diagram/ECE code on the split the Sen1Floods11 models were selected on (in-domain), and
fit a temperature there for the Sen1Floods11-only models (task 5.3 says "fitted on
`valid`"). Peru-trained models keep the 2-fold-by-chip fit on dev, because `peru-dev` is
their validation split.

**Method.** Seed-0 Sen1Floods11 checkpoints, all labelled `valid` pixels (89 chips).
Reliability diagram: 15 equal-width bins of confidence max(p, 1 − p) on [0.5, 1]; per bin
accuracy vs mean confidence; ECE = Σ_bins n_b |acc_b − conf_b| / N. "near" = labelled pixels
within 10 px of a label water edge (as in the failure analysis). T for `valid` scores is
fitted 2-fold by chip (no pixel scored with a T fitted on it); T applied to `peru-dev` is
fitted on all `valid` pixels. Script: `scripts/calibration_dev.py`, outputs
`outputs/analysis_dev/calib_valid.csv`, `temperatures_valid.json`,
`figs/reliability_valid_s1f11.png`.

**Checks.** ECE recomputed from the reliability bins equals `ece()` (asserted in the
script); IoU from the saved predictions equals the logged `val/IoU_Flood`.

**Observation (logit-based run).** Temperatures on `valid`: U-Net 1.73, TerraMind 1.01,
Prithvi 2.80 (`temperatures_valid.json`). So U-Net and Prithvi are overconfident even
in-domain, TerraMind is nearly calibrated. Hypothesis, not tested: all three are trained
with Dice loss (`configs/*.yaml`, `loss: dice`), which pushes probabilities to 0/1; it does
not explain the difference between models (TerraMind also uses Dice). Any literature
claim about Dice and calibration needs a citation from the literature base (page/table).

**Thesis use.** Chapter IV (calibration method), Chapter V (calibration results: in-domain
vs Peru). Final numbers: `docs/calibration_dev.md` once written.

## 2026-10-14 — Per-stratum calibration and reliability diagrams on `peru-dev` (tasks 5.4, 6.4)

**What and why.** The failure analysis showed errors concentrate at water edges,
low-contrast water and label errors; task 6.4 asks whether calibration and error detection
also degrade there, i.e. whether the uncertainty is useful exactly where the model fails.

**Method.** `scripts/calibration_dev.py`, all labelled dev pixels (174 chips), strata from
`scripts/analyze_dev.py` (WorldCover land cover, NDWI bands, distance to thick cloud,
date, distance to label water edge, water-body size, water type). Per stratum value: pixel
and error counts, ECE, error-detection AUROC. Methods: single seed (3 seeds, mean ± std);
temperature-scaled (2-fold by chip on dev; Sen1Floods11-only: T from `valid`); seed
ensemble; ensemble + temperature. Reliability bins pooled over the 3 seeds. Tables:
`scripts/report_calibration.py` → `outputs/analysis_dev/calib_tables.md`.

**Design decisions.**
- AUROC is not reported for 2-fold temperature scaling: scaling is monotone within a fold,
  so a pooled AUROC only measures the mismatch between the two folds' temperatures (this
  produced a spurious 0.655 → 0.752 jump for Prithvi in the first uncertainty table).
- Cells where a model has < 1,000 errors are marked (†): no AUROC, ECE unreliable.
- Input-defined strata (NDWI over all pixels, land cover, cloud distance, date) are
  presented first. Label-defined strata (edge distance; water-only strata such as water
  NDWI, water-body size, water type) partly restate the error rate, because they are
  conditioned on the label; in water-only strata every error is a miss.

**Results** (logit-based run, `outputs/analysis_dev/calib_tables.md`; write-up
`docs/calibration_dev.md`):
- Raw models are overconfident at every confidence level. Ensemble + temperature scaling
  halves the near-water ECE for U-Net (3.16 → 1.64 %) and Prithvi (4.42 → 1.97 %), and cuts
  TerraMind's by a third (3.54 → 2.35 %).
- Calibration is worst where errors are: ECE 8.5–13.4 % on the 1-px edge band (Prithvi
  worst) against ≤ 0.15 % more than 10 px away.
- The uncertainty does not flag the dominant failure modes: low-contrast water
  (NDWI −0.1 to 0.1) and water bodies < 1 ha are missed confidently (single-seed AUROC
  0.41–0.56 for U-Net and Prithvi, 0.56–0.71 for TerraMind); ensembling barely helps.
  Errors far from label edges (likely label errors) are not detected either (0.48–0.60).
- TerraMind ranks errors best inside the water strata despite the lowest overall AUROC.
- An in-domain (`valid`) temperature transfers only partly to Peru.

**Checks.** Single-seed raw ECE equals the earlier analysis (`uncertainty_dev.md`); ECE
from bins = `ece()`; pooled reliability bins and per-stratum counts come from the same
pixels as the overall rows.

**Caveats.** Two definitions of "near water" exist: `near_water()` (≤ 10 px from a label
edge; `uncertainty_dev.py`, `calibration_dev.py`) and the edge-distance stratum of
`analyze_dev.py` (bands up to 10.5 px), so the near-water ECE differs slightly between
`failure_analysis_dev.md` (U-Net 3.14 %) and the uncertainty/calibration docs (3.16 %). Use
the ≤ 10 px values in the thesis. Strata are not independent (low-contrast water is mostly at edges); AUROC
within a stratum is computed only among that stratum's pixels; the Sen1Floods11 → Peru
seeds share one starting checkpoint (less diverse ensemble). Runtime ≈ 1 h on the Mac
(exact AUROC on continuous float64 logits; ~12–14 GB RAM).

**Thesis use.** Chapter V (failure analysis + calibration by stratum); paper 1 results.
