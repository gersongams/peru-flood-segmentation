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

## 2026-10-14 — Doubtful-label chips in the UNOSAT 2023 set (task 4.4)

**What and why.** Finding 3 of the failure analysis showed that part of the "errors" on
`peru-dev` are label errors. Task 4.4 turns that into a model-free, reproducible list of
chips whose UNOSAT label disagrees clearly with the image, for every split, so the thesis
can describe label quality and a possible v2 can decide what to do with them.

**Method.** `scripts/label_qa.py`, all 1,291 UNOSAT chips (train, dev, test, buffer).
Index: MNDWI = (B03 − B11)/(B03 + B11), the same index as the builder's `agreement()` and
the manifest's `mndwi_iou`. Pixels are counted only if labelled and more than 2 px from a
label water edge, so slight misregistration along shores does not create blobs.
- Omission blob: labelled dry and MNDWI > 0.5.
- Commission blob: labelled water and MNDWI < −0.1.
- Blobs are 4-connected components of ≥ 100 px (1 ha).
- Each blob is tagged "near cloud" (< 20 px from thick cloud, using the builder's rule:
  shadow or haze possible), "vegetated" (mean NDVI > 0.3: irrigated fields or flooded
  vegetation) or "open".
- Chip flags: doubtful if blobs cover ≥ 5 ha, minor if 1–5 ha.

**How the thresholds were set (and one revision).** The rule uses the image and label only.
- First version: ±0.2. It flagged 45 % of chips. Its blobs had median MNDWI 0.25, half of
  their area was next to cloud (haze), and 99 % of labelled dry pixels lie below 0.21, so
  0.2 sat inside the tail of dry land.
- Revised from the label-conditional MNDWI distributions of 300 random train/dev chips
  (every 20th labelled pixel). Labelled water: median 0.72, 1st percentile 0.05, 73 %
  above 0.5. Labelled dry: median −0.16, 99th percentile 0.21, 0.03 % above 0.5.
- Thresholds were fixed before any comparison with model errors, so the label-error
  conclusion is not self-fulfilling.

**Results** (`outputs/label_qa/chips.csv`; list in `docs/peru_doubtful_chips.md`):
- 32 doubtful chips (2.5 %) and 56 minor. Train 16 / 40, dev 8 / 6, test 6 / 6, buffer 2 / 4.
- Without the blobs next to thick cloud: 15 doubtful (train 8, dev 4, test 3) and 57 minor.
- 128 omission blobs (534 ha) against only 7 commission blobs (18 ha). More than half the
  omission area (281 ha) is next to thick cloud, where shadow, haze or bright surf can fool
  the index. The largest dev case is the sea edge just inside UNOSAT's analysed extent
  (`2023-04-01_E517120_N9308160`): ocean labelled dry, with surf triggering the cloud rule.
- Clear open cases exist, e.g. a dark rectangular pond labelled dry
  (`2023-04-01_E588800_N9328640`, train). A flag therefore means a clear label/MNDWI
  disagreement, not a proven label error.

**Check against the models** (after the rule was fixed; `--check`, `outputs/label_qa/check_dev.txt`):
- On dev, the 9 Peru-only runs call omission-blob pixels water 83.2 % (not near cloud) and
  87.3 % (near cloud) of the time, against 0.4 % of other dry pixels.
- But they also call 90.6 % of the dry, MNDWI > 0.5 pixels that the rule left out (blobs
  under 1 ha or in the edge band) water. Omission pixels are high-MNDWI by construction, so
  this check mostly shows that the models follow the index, whether or not the label is
  wrong. It cannot separate label errors from index failures (shadow, surf).
- Same limitation for the per-chip Spearman between omission share and false-positive rate
  (0.36, p < 1e-6, n = 174). Commission vs miss rate: 0.13, p = 0.10, too few blobs.
- What the check does show: models trained on these labels still call these pixels water,
  which is consistent with label error. The direct evidence is visual (overlays in
  `outputs/label_qa/figs/`).
- Correction (same day): the first version of this entry said the check "supports" label
  error; it compared against all other dry pixels (mostly desert, MNDWI ≈ −0.16), which is
  not a fair baseline.

**Limitation.** The rule is conservative and misses turbid/shallow water. Of the 10
Finding-3 chips it flags 3 (2 doubtful, 1 minor). The clearest missed case
(`2023-04-01_E517120_N9349120`, water labelled dry) has MNDWI 0.30–0.47 in the disputed
area, the same as that chip's own labelled water (median 0.39), so a global 0.5 threshold
cannot see it. A chip-relative threshold could catch it. It was not added, because it
would be tuned to a known case. An unflagged chip is therefore not guaranteed clean.

**Decisions.**
- The list annotates the frozen v1 data and removes nothing. Exclusion would be a v2
  decision, taken before section 7 and applied to all models alike.
- Test chips are flagged mechanically, with no overlays.
- EMSR 2017 is not screened: its labels already set EMS/index disagreement to −1, so the
  check would be circular.
- Nothing is added to the frozen manifests (that would break `freeze_peru.py --check`).

**Thesis use.** Chapter III (dataset, label quality and known issues); Chapter V (label
noise as a cause of errors; dev IoU is a lower bound); a v2 / phase-D label-noise method.

## 2026-10-14 — Sen1Floods11 vs Peru band distributions (task 6.2, part 1)

**What and why.** To explain why models transfer poorly between Sen1Floods11 and Peru,
measure how the input distributions differ, per band and per class.

**Method.** `scripts/band_stats.py`. Datasets: Sen1Floods11 S2Hand `train` (252 chips),
`peru-train` (753), `peru-dev` (174); no `peru-test`.
- Pixels: labelled pixels only (label −1 and all-zero no-data skipped), split into water
  and dry, sampling every 37th labelled pixel of each chip.
- Per band: median and percentiles.
- Shift in units of the Sen1Floods11 train std of the band (what the U-Net sees after
  normalization): standardized mean difference (SMD) and 1-D Wasserstein distance (W1).
- MNDWI and NDWI percentiles per class; water share per dataset.
- Prithvi's 6 HLS bands are marked.
- Outputs: `outputs/band_stats/{summary.md,bands.csv,indices.csv,figs/hist_*.png}`.

**Check.** Radiometric harmonization (Peru L1C after the −1000 DN offset fix): over water
the low percentiles of B11/B12 are lower in Peru (p5 74 / 43 DN) than in Sen1Floods11
(130 / 75), so there is no leftover offset that would fake a shift.

**Results.**
- **Dry land differs much more than water.** Peru dry land (Sechura desert, bare soil) is
  far brighter in the visible bands and SWIR: SMD vs Sen1Floods11 +0.8 to +0.9 (peru-train)
  and +1.4 to +1.7 (peru-dev) for B02–B04 and B12, +0.5 / +0.9 for B11. NIR changes
  little (B08, B8A: −0.03 to +0.18).
- **Water is similar.** |SMD| ≤ 0.4 in every band. Peru water is darker in NIR and red-edge
  (B8A −0.22 / −0.40) and slightly brighter in green.
- **Peru water is clearer by MNDWI** (median 0.72 vs 0.49 for Sen1Floods11). Peru dry land
  has NDWI much closer to 0 (median −0.15 train, −0.06 dev, vs −0.38), because it is bare
  and unvegetated.
- So in Peru, dry land overlaps the low-contrast water range (NDWI around 0). This fits
  Finding 2: the Sen1Floods11-only models have false-positive rates of 5–7 % on
  Peru dry pixels with NDWI 0–0.1, and low-contrast water is the weakest stratum for every
  model.
- Water share of labelled pixels: Sen1Floods11 train 9.5 %, peru-train 4.6 %, peru-dev 12.4 %
  (Sen1Floods11 `valid`: 11.0 %).
- peru-dev's dry land is much brighter than peru-train's (SMD +1.4–1.7 vs +0.8–0.9 for the
  visible bands and B12), because dev blocks were drawn only from blocks with water, in the
  desert coastal plain. Dev is not a random sample of the Peru background.

**Caveats.**
- The pixel sample is not stratified by event or chip, so large chips dominate.
- Sen1Floods11 covers many biomes; the "Sen1Floods11 distribution" is a mixture.
- SMD and W1 are 1-D, per band; they do not capture band interactions.

**Thesis use.** Chapter III (dataset description: how Peru differs) and Chapter V (domain
shift as the explanation for the transfer results); paper 1 (benchmark motivation).

## 2026-10-14 — Transfer matrix: training regime × evaluation set (task 6.2, part 2)

**What and why.** Task 6.2 asks for the IoU drop between Sen1Floods11 and Peru. One model
scored on two sets is confounded by how hard each set is, so all regimes are evaluated on
both sets and compared **within** each evaluation set (in-domain vs transferred training).

**Method.** Water IoU (`val/IoU_Flood`) from `terratorch validate` logs. The 18 new runs are
the Peru-only and Sen1Floods11 → Peru checkpoints on Sen1Floods11 `valid`, each with its own
model normalization; the other cells come from existing logs. Script
`scripts/transfer_matrix.py`; write-up `docs/domain_shift_dev.md`.

**Results.**
- Sen1Floods11 → Peru costs 4.6–13.1 IoU points on `peru-dev` (Prithvi least, TerraMind most).
- Peru → Sen1Floods11 costs 12.9–18.0 points on `valid` (U-Net least).
- After Peru-only training, U-Net generalizes best to `valid` (0.690 vs 0.656 / 0.653).
- Peru fine-tuning drops the Sen1Floods11 checkpoints by 12–15 points on `valid`
  (forgetting), yet they stay 1–4 points above Peru-only training.

**Caveats.**
- In-domain cells are optimistic, because each checkpoint was selected on that split. This
  slightly inflates the measured gaps.
- Sen1Floods11-only rows have one seed.
- Cross-column comparisons are not meaningful: label sources, water share (`valid` 11.0 %,
  `peru-dev` 12.4 %) and difficulty differ. Gaps in different columns are measured from
  different in-domain baselines (0.82–0.83 vs 0.92–0.94), so "Peru → Sen1Floods11 costs
  more" is a hedged statement; as a share of the in-domain IoU the gaps are 5–14 % on
  `peru-dev` and 16–22 % on `valid`.

**Thesis use.** Chapter V (generalization RQ); paper 1 main table (regime × set), with the
test-set version after section 7.

## 2026-10-14 — Edge-tolerant IoU: is the foundation models' edge loss real? (input to task 6.5)

**What and why.** The draft phase-D recommendation was "full-resolution detail for the
foundation models", because they lose to the U-Net in the 1-px edge band (Finding 1). But
UNOSAT labels are rasterized polygons, least reliable exactly at edges. If the U-Net's lead
vanishes once the edge band is ignored, that method would mostly fit label noise.

**Method.** `scripts/edge_tolerance.py` → `outputs/analysis_dev/edge_tolerance.{md,csv}`.
- Water IoU from the float32 logits, recomputed after excluding labelled pixels within k px
  of a label water edge, k = 0 (plain IoU), 1, 2, 3.
- Edge definition as in `analyze_dev.py`; k px excludes distance < k + 0.5, so k = 1 is the
  "edge (≤ 1 px)" stratum.
- Runs: Peru-only and Sen1Floods11 → Peru, 3 seeds each.

**Results** (Peru-only, mean over 3 seeds; U-Net minus foundation model, IoU points):

| | plain | excl. ≤ 1 px | ≤ 2 px | ≤ 3 px |
|---|---|---|---|---|
| U-Net − TerraMind | +1.50 | +0.40 | +0.42 | +0.46 |
| U-Net − Prithvi | +1.97 | +0.28 | +0.24 | +0.25 |
| U-Net − TerraMind (Sen1Floods11 → Peru) | +1.01 | −0.01 | −0.01 | +0.01 |
| U-Net − Prithvi (Sen1Floods11 → Peru) | +1.47 | +0.45 | +0.44 | +0.46 |

- 75–100 % of the U-Net's lead lies in the 1-px label-edge band. The rest (0.25–0.46
  points, Peru-only) is stable from k = 1 to 3, so it is a small but real difference.
- The small-water recall gap (< 1 ha: U-Net 58 % vs foundation models 45–46 %) is not an edge
  effect and remains a real resolution deficit, but it covers only 37,799 dev pixels.

**Conclusion.** Full-resolution detail for the foundation models can gain at most about
0.5 IoU points outside the label-noise band. That is too small to carry phase D, so it is
demoted from the main candidate. Report a boundary-tolerant IoU next to plain IoU in the
thesis: plain IoU ranks models partly on how they draw label edges.

**Thesis use.** Chapter V (model comparison; metric choice), the phase-D decision record.

## 2026-10-14 — Phase-D decisions (task 6.5)

**Context.** No thesis advisor is assigned; the authors decide (a course teacher gives only
general input). Decisions taken with the evidence above, recorded in full in the decision
record of `docs/failure_analysis_dev.md`.

**Decisions.**
- **Phase D = mapping floods in Peru without Peru labels.** Adapt Sen1Floods11-trained
  models to Peru using only unlabelled Peru Sentinel-2 imagery (self-training or unsupervised
  domain adaptation). UNOSAT labels are used only for selection (`peru-dev`) and evaluation.
  - Why: the measured headroom is 5–13 IoU points (Sen1Floods11-only 0.80–0.88 vs
    Peru-trained 0.92–0.94 on dev). The alternative "spatial detail for the foundation
    models" has at most ~0.5 points outside label noise (edge-tolerance entry).
  - Practical framing: a new Peruvian flood has no labels.
  - The Peru-trained models remain the main supervised result and the upper bound.
- **No Sentinel-1** in Peru: little cloud impact, a costly new pipeline, a date mismatch
  with S2-drawn labels. Future work. The authors asked to keep the focus on Peru.
- **Labels: frozen v1 plus a pre-registered sensitivity row** on `peru-test` without the
  "doubtful" chips (6 with all blobs, 3 without near-cloud blobs), fixed before seeing any
  test result. Plain and boundary-tolerant IoU (≤ 1 px label edge excluded) are both
  reported.
- **Uncertainty is a secondary result.** MC-dropout (task 5.2) is optional; the models have
  dropout only in the head.
- **Timeline unknown:** plan for 3–6 months.

**Thesis use.** Chapter I (contribution and RQs, phase D), Chapter IV (method), Chapter V
(evaluation protocol: sensitivity row, tolerant IoU).
