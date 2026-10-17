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

## 2026-10-14 — Literature check for phase D and index-threshold baselines (task 6.6, part 1)

**What and why.** Phase D adapts Sen1Floods11 models to Peru without Peru labels. The old
literature base has no domain-adaptation papers, so a new search was needed to (a) see whether
the topic is open, (b) pick methods and baselines, (c) copy an evaluation protocol.

**How.** Two research agents searched the web in parallel (general UDA/test-time adaptation
for segmentation; adaptation and generalization in Earth observation, floods, foundation
models, Peru). They read arXiv PDFs and publisher full texts and recorded a URL and a
verification status per paper (`docs/literature/`). Spot-checks here: Crossref/OpenAlex
metadata of CIM-TTDA and Portalés-Julià 2025, and the VFM-UDA arXiv page. Publisher pages
were not readable from here, so the agents' numbers stay marked "re-check in the PDF".

**Findings** (synthesis: `docs/literature_phaseD.md`):
- Closest work: CIM-TTDA (2026), test-time self-training for **SAR** flood mapping. Nothing
  found for optical S2 flood adaptation with geospatial foundation models, for desert
  backgrounds, or deep-learning flood segmentation of the Peruvian coast / Yaku. The claim
  to make is therefore narrow (optical + GFM + desert shift), not "first DA for floods".
- Method choice: self-training with an EMA teacher, per-class (CBST-style) thresholds and
  masked consistency (MIC/VFM-UDA) fits ViT foundation models. AdaBN reaches only the U-Net,
  because TerraMind/Prithvi encoders use LayerNorm. TENT is weak for transformers (CoTTA
  Table 5, agent-read). Adversarial methods are superseded.
- An MNDWI baseline is expected by the literature (Portalés-Julià 2025: MNDWI close to deep
  learning on unseen datasets).

**Index baselines measured here** (`scripts/index_baseline.py` → `outputs/analysis_dev/index_baseline.md`):

| Baseline | peru-dev water IoU | Sen1Floods11 valid |
|---|---|---|
| MNDWI > 0 | 0.486 (precision 0.49, recall 1.00) | 0.599 |
| MNDWI > Otsu of unlabelled peru-train (t = 0.039) | 0.572 | — |
| best MNDWI threshold picked on the split itself | 0.924 (t = 0.25) | 0.745 (t = 0.15) |
| best NDWI threshold picked on the split itself | 0.844 (t = 0.15) | 0.710 (t = −0.05) |

- Without labels the index is badly placed in Peru: hazy desert top-of-atmosphere pixels
  have MNDWI > 0.
- A threshold tuned on dev labels matches the Peru-trained Prithvi (0.922); it is an
  optimistic, single-parameter fit.
- Open question: did UNOSAT draw its water layer with an index threshold? If so, dev IoU
  partly measures agreement with an index-derived product. Check UNOSAT's product notes.
- Correction in the same session: an interpolated "≈ 0.55" for the Otsu baseline was replaced
  by the exact 0.572 once the computation was put in the script.

**Consequences.** Phase-D baselines include label-free index methods (MNDWI > 0, Otsu) and
index-based pseudo-labels. Pseudo-labels that combine model confidence with MNDWI are a
candidate ingredient. Report precision and recall separately, and the share of the
source→oracle gap closed.

**Thesis use.** Chapter II (related work: domain adaptation, flood generalization, Peru),
Chapter IV (phase-D method), Chapter V (baselines). All papers must go into Zotero with
notes before being cited.

## 2026-10-14 — How the UNOSAT 2023 labels were made (open question from the literature check)

**Question.** A single MNDWI threshold tuned on dev (0.924 IoU) matches the Peru-trained
foundation models. Were the UNOSAT polygons produced by an index threshold?

**Evidence.**
1. UNOSAT's methodology note in the HDX metadata of both Lambayeque/Piura datasets (read via
   the HDX CKAN API, field `methodology_other`): "For flood extractions, landcover mapping and
   similar analyses, a variety of automated remote sensing techniques are used to extract the
   relevant information which is then reviewed and revised as necessary by the analyst."
   Dataset notes: "satellite-detected water using Sentinel-2 … preliminary analysis … not yet
   validated in the field".
2. Polygon geometry (`scripts/unosat_provenance.py` → `outputs/label_qa/unosat_geometry.csv`,
   all layers reprojected to UTM 17S):
   - WaterExtent / FloodExtent layers (22 Mar, 1 Apr, 21 Apr): 87–91 % of vertices on a 20 m
     grid, 92–93 % axis-aligned edges, median edge exactly 20 m. They were vectorized from a
     20 m raster aligned to the Sentinel-2 grid.
   - CloudObstruction: on a 10 m grid (89–99 %), from a 10 m raster.
   - AnalysisExtent: 1–3 % on the grid, median edge 181 m. Hand-drawn.
   - Tumbes layers (not used): axis-aligned ~10 m edges but not on our grid.
3. The ArcGIS geoprocessing history in the `.shp.xml` files logs only attribute edits
   (AddField / CalculateField), not the extraction step.

**Conclusion.** The labels come from an automated extraction on a 20 m Sentinel-2 grid, plus
analyst review. The method is unpublished. 20 m fits the SWIR bands (B11/B12) that MNDWI
uses, but red-edge bands and Sen2Cor's SCL (with a water class) are also 20 m, so
"index-based" is consistent with the evidence, not proven.

**Consequences.**
- `peru-dev` / `peru-test` IoU partly measures agreement with an automated 20 m product. That
  fits the strong tuned-MNDWI baseline, and it must be stated in the evaluation chapter.
- Label edges are 20 m steps rasterized to 10 m, so the 1-px edge band is inherently
  uncertain. This supports reporting the boundary-tolerant IoU (edge-tolerance entry).
- For phase D, index-based pseudo-labels are close to how the evaluation labels were made.
  That makes them a strong baseline, and a method that beats it is more convincing.

**Thesis use.** Chapter III (dataset: label provenance and resolution), Chapter V (metric
interpretation), Chapter VI (limitations).

## 2026-10-14 — Phase-D plan (task 6.6): OpenSpec change `phase-d-peru-adaptation`

**What.** The plan for phase D, as specs and a design (`openspec/changes/phase-d-peru-adaptation/`).

**Key design decisions and why** (full list in its `design.md`):
- **Target pool = `peru-train` images, labels never read.** Using dev/test images, even
  unlabelled, would make the main result transductive. Buffer chips touch held-out blocks.
- **New 3-seed Sen1Floods11 source models** (U-Net, TerraMind first; Prithvi last). Per-seed
  adaptation needs per-seed sources.
- **Self-training:** EMA teacher; per-class top-q % pseudo-labels (CBST-style) against
  collapse to "all dry" with rare water; CE on pseudo-labels, the source Dice loss in every
  step; binary DACS mixing that pastes source water onto desert chips (the failure mode);
  LR warm-up.
- **Masked consistency at image level** (64 px patches, ratio 0.7) for all backbones: uniform
  for the CNN and the ViTs, and 64 px is a multiple of the 16 px ViT patch. (Corrected
  2026-10-15: this line first said 32 px; `design.md` and the defaults below use 64 px, MIC
  Table 8 per the agent notes.)
- **Baselines on the same backbones:** MNDWI > 0, Otsu-MNDWI on the pool, histogram matching,
  FDA, AdaBN (U-Net only, because the ViT encoders use LayerNorm).
- **Protocol: the headline uses no Peru labels at all.** Hyperparameters are pre-registered
  from the literature (EMA α = 0.99, class-wise share q 20 → 50 %, LR 0.1× source with 10 %
  warm-up, 64 px / 0.7 masking, 4,000 iterations, final checkpoint). Selecting on dev labels
  would leak Peru label information into a "no-label" method: one MNDWI threshold tuned on dev
  already reaches 0.924. Dev-tuned runs are a secondary "oracle selection" row next to the
  dev-tuned MNDWI threshold. Dev is scored offline from saved checkpoints. Also:
  gap-closed reporting; plain and tolerant IoU, precision, recall; final `peru-test` (with the
  pre-registered sensitivity row) and `peru-2017-test` once.
- **Rejected:** adversarial alignment (superseded), per-event test-time adaptation on test
  images (transductive; possible later row), Sentinel-1.

**Thesis use.** Chapter IV (method), Chapter V (protocol).

## 2026-10-15 — Sen1Floods11 source models: LR choice, epoch cap, reproducibility (phase-D tasks 1.1–1.4)

**What and why.** Phase D adapts Sen1Floods11-trained models to Peru, and per-seed adaptation
results need one source model per seed. So U-Net and TerraMind v1-B were retrained on
Sen1Floods11 with a learning-rate sweep and 3 seeds (the seed-0 phase-C checkpoints had an
untuned LR). Runs and costs: `docs/EXPERIMENT_LOG.md` (2026-10-14/15 entry).

**Method.**
- Data: official Sen1Floods11 hand-labelled splits, train 252 chips, selection on `valid` (89).
  13 Sentinel-2 L1C bands, each model's official normalization, same Dice loss, AdamW
  (weight decay 0.05), ReduceLROnPlateau on `val/loss` (factor 0.5, patience 5), batch and
  augmentation as in `configs/<model>.yaml`. Early stopping on `val/IoU_Flood` with patience 20;
  the checkpoint with the best `val/IoU_Flood` is kept.
- LR sweep, seed 0, 100-epoch cap: U-Net {3e-5, 1e-4, 3e-4, 1e-3} (4 values), TerraMind
  {1e-5, 2e-5, 5e-5} (3 values). Chosen by best `valid` water IoU: **U-Net 3e-4 (0.8325),
  TerraMind 5e-5 (0.8354)**.
- Final: 3 seeds (0, 1, 2) per model, epoch cap raised to 200. RTX 4090, 16-bit mixed precision.

**Results** (`valid` water IoU, mean ± std over 3 seeds; recomputed from `metrics.csv` in
`outputs/s1f11_src_unet/` and `outputs/s1f11_src_terramind_lr5e-5/`):
U-Net 0.8315 ± 0.0015; TerraMind 0.8349 ± 0.0014. Both above the phase-C seed-0 models trained
at the default LR on the Mac (U-Net 0.8188, TerraMind 0.8168).

**Decisions and alternatives.**
- *Epoch cap 200 instead of 100* (authors approved "add more epochs if that improves it"):
  U-Net 3e-4 and TerraMind 2e-5 peaked at the last sweep epoch (95 and 99 of 100). With 200,
  best epochs were 52–89, so the cap did not bind in the final runs; early stopping ended
  every run.
- *TerraMind 5e-5 is the edge of its grid.* A higher LR was not tried; the optimum may be
  above 5e-5. Accepted because 3 seeds at 5e-5 are stable (std 0.0014) and above every other
  value, and the remaining budget goes to adaptation.
- *Selection bias.* LR, early stopping and the checkpoint are all chosen on `valid`, so these
  `valid` numbers are optimistic. They only rank configurations; Sen1Floods11 `test` is not
  used here (scored once at the end).

**Checks.**
- All runs clean commit `8a8fee4`; all copies verified file by file.
- `run.json` shows `lr: 0.001` (an unused task field); the LR used is
  `optimizer.init_args.lr` in each run's `config.yaml` — checked for every run.

**Correction / mistake.** The TerraMind final seeds were first launched at 2e-5, chosen
before the 5e-5 sweep run had finished (0.8141 ± 0.0151). They were rerun at 5e-5. The 2e-5
runs are logged and not used.

**Reproducibility (plan 3.5).** Rerunning U-Net seed 0 with identical settings gave 0.8161
(best epoch 34, stopped at 54) instead of 0.8330 (best 83): 1.7 points. GPU training here is
not deterministic (cuDNN kernels, mixed precision) and early stopping amplifies small early
differences. **Implication:** the seed std (≈ 0.15 points) understates the run-to-run
variation; differences between methods below ~2 points from 3 seeds should be stated with
care. Thesis: report this tolerance next to the seed std.

**Published reference (plan 3.8, not yet closed).** Prithvi-CAFE (arXiv 2601.02315) Table 1
lists TerraMind v1-B at 82.90 water IoU on Sen1Floods11; its protocol differs (inputs,
splits used for reporting). Re-check the table in the PDF and add the paper to Zotero before
citing; no comparison claim beyond "same range" until then.

**Thesis use.** Chapter IV (training setup, hyperparameter selection), Chapter V (source
models of phase D, reproducibility caveat).

## 2026-10-15 — Phase-D target pool and label guards (phase-D tasks 2.1–2.3)

**What and why.** Phase D may use Peru *images* but no Peru *labels*. The pool and its
guards make that a property of the code, not only of intent (spec `peru-target-pool`).

**Method.**
- Pool = the Sentinel-2 images of all `peru-train` chips of Peru set v1: 753 images
  (2023-03-22: 392, 2023-04-01: 361). Manifest `configs/pools/peru-train.csv` (stem, sha256),
  written by `scripts/build_target_pool.py`; every sha256 equals the `s2_sha256` of the Peru v1
  data manifest. Versioned in git so a run can name the exact pool.
- `src/floodseg/target_pool.py`:
  - `TargetPoolDataset` knows only image paths and returns only `{"image"}`. It reads
    `dataset_bands`, selects `output_bands`, float32, NaN → `no_data_replace`, and normalizes
    with each model config's means/stds.
  - `check_pool` refuses any `peru-dev`, `peru-test`, buffer or `peru-2017-test` chip, and any
    chip outside the Peru v1 split (extra imagery would first need the spatial check of task
    2.4). Called when a manifest is read and when the dataset is built.
  - `forbid_peru_labels()` / `guard_worker_init`: while active, any `rasterio.open` of a
    file under a Peru label root (`data/peru/{unosat2023,emsr2017}/label`, raw
    `data/unosat2023`, `data/emsr199`) raises an error naming the file. rioxarray and TerraTorch
    go through `rasterio.open`, so their reads are caught too. Spawned DataLoader workers do
    not inherit the patch, hence the `worker_init_fn`.

**Alternatives rejected.** Reusing TerraTorch's `GenericNonGeoSegmentationDataset` for the
pool: it requires label paths and reads masks, the opposite of the guarantee. Using dev/test
images unlabelled: transductive (design decision 1).

**Checks** (`tests/test_target_pool.py`, 13 tests, all pass): held-out and unknown chips
refused, naming the chips; the default pool equals `peru-train` (753); for U-Net, TerraMind
and Prithvi a pool image equals TerraTorch's normalized model input for the same chip to
1e-5; the guard stops TerraTorch's own mask read and works in a spawned DataLoader worker;
the guard is removed after the block.

**Caveat.** The guard covers `rasterio.open` only; code that reads labels another way (e.g.
GDAL directly, or geopandas on the raw shapefiles) is not intercepted. Adaptation code must
use the pool dataset; the run record lists every file it opened from `data/` (task 4.3).

**Thesis use.** Chapter IV (adaptation protocol: what data the method may see).

## 2026-10-15 — Rule fixed before a diagnostic: source checkpoints are chosen on `valid` only

Written before the superseded TerraMind 2e-5 source runs were scored on `peru-dev`. Those dev
scores are a **diagnostic** (does the source LR change transfer to Peru?). Whatever they
show, the phase-D source models stay the ones selected on Sen1Floods11 `valid`
(U-Net 3e-4, TerraMind 5e-5, best-`valid` checkpoint per seed). Choosing the source LR or seed
by `peru-dev` would put Peru-label information into the "no-label" headline; if that is ever
done, it is an "oracle selection" row (spec `adaptation-evaluation`).

## 2026-10-15 — Phase-D starting point on `peru-dev`: index baselines, source-only models, metric module (tasks 3.1, 3.2)

**What and why.** The first rows of the phase-D table: what is reached without Peru labels
before any adaptation, and the Peru-trained ceiling, all scored by one metric function.

**Method.**
- `src/floodseg/metrics.py`: pooled water IoU, precision, recall over labelled pixels
  (label ≠ −1); tolerant versions leave out labelled pixels within 1 px (Euclidean distance
  < 1.5) of a label water/not-water edge (spec `adaptation-evaluation`). Unit tests
  `tests/test_metrics.py`.
- `scripts/phaseD_dev_table.py`: index rows from MNDWI = (B03 − B11)/(B03 + B11):
  MNDWI > 0; MNDWI > Otsu threshold of the unlabelled pool (753 `peru-train` images from the
  manifest, every 50th pixel with a non-zero band, read under the label guard); MNDWI > t
  tuned on `peru-dev` (t ∈ [−0.30, 0.60], step 0.01, best plain IoU; oracle row). Model rows
  from saved logits (water if z > 0), mean ± std over seeds.

**Results** (`outputs/phaseD/dev_table.md`; water IoU / tolerant IoU / precision / recall):

| Row | IoU | tolerant | precision | recall |
|---|---|---|---|---|
| MNDWI > 0 | 0.4858 | 0.4763 | 0.4869 | 0.9955 |
| MNDWI > 0.0391 (Otsu on pool) | 0.5718 | 0.5692 | 0.5740 | 0.9933 |
| MNDWI > +0.27 (tuned on dev, oracle) | 0.9251 | 0.9581 | 0.9681 | 0.9542 |
| Source-only U-Net (3 seeds) | 0.8364 ± 0.0224 | 0.8815 ± 0.0240 | 0.8475 ± 0.0242 | 0.9847 ± 0.0017 |
| Source-only TerraMind (3 seeds) | 0.7027 ± 0.0913 | 0.7243 ± 0.1070 | 0.7139 ± 0.0951 | 0.9789 ± 0.0033 |
| Peru-trained U-Net (upper bound) | 0.9415 ± 0.0019 | 0.9721 ± 0.0006 | 0.9735 ± 0.0029 | 0.9662 ± 0.0012 |
| Peru-trained TerraMind (upper bound) | 0.9265 ± 0.0025 | 0.9682 ± 0.0015 | 0.9651 ± 0.0018 | 0.9586 ± 0.0021 |

**Findings.**
1. **Source-only models over-predict water in Peru.** Recall stays ≥ 0.98 for every
   source-only run; precision is what drops (0.60–0.90). This matches the domain-shift
   finding (bright desert background read as water) and is what self-training must fix: the
   errors are false positives, not misses.
2. **TerraMind's transfer to Peru depends strongly on the seed; its `valid` score does not
   show it.** On `valid` the three seeds differ by 0.0026 IoU (std 0.0014); on `peru-dev` by
   0.17 (0.598–0.764). Seed 1 is the worst at both LRs (5e-5: 0.598; superseded 2e-5:
   0.579), and the LR changes the mean little (0.703 vs 0.692). So it is not the LR choice;
   something fixed by the seed (decoder initialization, data order) decides how the model
   treats desert. U-Net varies less (0.811–0.854).
3. The phase-C seed-0 TerraMind (Mac, LR 2e-5) scored 0.7951; the cloud seed-0 run at the
   same LR 0.7314. Together with the 1.7-point `valid` repeat difference, single runs say
   little about Peru transfer.
4. The label-free index baselines stay far below the source models (0.49–0.57); the
   dev-tuned threshold (0.925) matches the Peru-trained models, as found before
   (0.924 at the coarser 0.05 step; now t = +0.27 at step 0.01).

**Implications for phase D.** The adaptation gap is large (U-Net 0.84 → 0.94, TerraMind
0.70 → 0.93) and per-seed source quality varies, so adaptation results must be reported per
seed and as gap closed per seed, not only as means. With TerraMind's spread, 3 seeds may
be too few for tight claims; decide 3 vs 5 after the first self-training results (cost).
Source checkpoints stay fixed by `valid` (rule recorded before the diagnostic).

**Correction (edge definition).** `scipy.ndimage.binary_erosion` treats outside the chip as
background by default, so `scripts/edge_tolerance.py` and `scripts/analyze_dev.py` counted
the 1-px chip frame as label edge (with 1-px tolerance a 2-px frame was excluded: 590,398
dev pixels, 1.55 % of tolerant pixels). `floodseg.metrics` uses `border_value=1`, so only
real water edges count. Recomputed for all 18 phase-C Peru runs: tolerant IoU changes by
≤ 0.0001 (e.g. Peru-only U-Net 0.9722 → 0.9721), so the edge-tolerance conclusions stand.
The **edge stratum of the failure analysis** (`analyze_dev.py`, `docs/failure_analysis_dev.md`)
also contains the frame; its per-stratum numbers are not yet recomputed (open; see NEXT_STEPS).

**Thesis use.** Chapter V (phase-D baselines and starting point), Chapter IV (metric
definitions). Finding 2 is a result in itself for the discussion: in-domain validation does
not predict out-of-domain behaviour of the foundation model.

## 2026-10-15 — AdaBN baseline for the U-Net (phase-D task 3.3)

**What and why.** The cheapest adaptation: keep every weight of the Sen1Floods11 U-Net and
replace the BatchNorm statistics by Peru statistics (AdaBN, Li et al. 2016; not yet in
Zotero, add before citing). Only the U-Net has BatchNorm in its encoder; TerraMind and Prithvi
encoders use LayerNorm, so AdaBN is a U-Net-only row (design decision 7).

**Method** (`scripts/adabn.py`, commit `d48fc3f`). Source = the three phase-D U-Net
checkpoints (best `valid`). All 46 `BatchNorm2d` layers (encoder and decoder) reset, set to
an exact cumulative average (`momentum=None`) and re-estimated in one pass over the 753
unlabelled pool images (`configs/pools/peru-train.csv`), batch 16, no augmentation, no
gradients, under the Peru-label guard; all other modules in eval mode (the U-Net has no
dropout). Then eval mode; `peru-dev` logits saved (`outputs/logits_dev/adabn/unet/seed<s>/`,
with `run.json`) and scored offline by `scripts/phaseD_dev_table.py`. No hyperparameter was
tuned (one pass, all layers: the original recipe). MPS, 43–66 s per seed.

**Check.** The checkpoint loaded outside the TerraTorch CLI reproduces the saved source-only
dev logits exactly (max |Δz| = 0 on 4 chips), so the AdaBN gain is not an artefact of a
different inference path.

**Results** (`outputs/phaseD/dev_table.md`, mean ± std over 3 seeds):

| U-Net | IoU | tolerant IoU | precision | recall | gap closed |
|---|---|---|---|---|---|
| Source-only | 0.8364 ± 0.0224 | 0.8815 ± 0.0240 | 0.8475 ± 0.0242 | 0.9847 ± 0.0017 | — |
| AdaBN | 0.8969 ± 0.0161 | 0.9491 ± 0.0172 | 0.9189 ± 0.0204 | 0.9741 ± 0.0041 | 58 % |
| Peru-trained | 0.9415 ± 0.0019 | 0.9721 ± 0.0006 | 0.9735 ± 0.0029 | 0.9662 ± 0.0012 | 100 % |

Per seed: 0.8433 → 0.9052, 0.8545 → 0.9071, 0.8114 → 0.8783. Every seed improves; the gain
comes from precision (fewer desert false positives) at a small recall cost.

**Interpretation.** Much of the U-Net's shift is first-order feature statistics: re-centring
the features on Peru removes most desert false positives. This sets a strong bar for
self-training on the U-Net: it must beat 0.897, not 0.836. It also suggests that a
statistics-alignment step for the LayerNorm models (TerraMind) is worth a look; not planned
(LayerNorm normalizes per token, so there are no stored statistics to replace).

**Caveat.** The pool holds both 2023 dates, like `peru-dev`; whether pool statistics from a
different date or area transfer is not tested here (relevant to the `peru-2017-test` event).

**Thesis use.** Chapter V (phase-D baselines).

## 2026-10-15 — Design amendments recorded before scoring: test-time input alignment, decoder AdaBN for TerraMind

Written before any of these rows was scored.

1. **Input alignment first as a test-time transform (task 3.4, part 1).** The design ran
   histogram matching and FDA as *source training* on transformed inputs. Given the
   run-to-run noise now measured (1.7 IoU points on `valid` for an identical rerun; TerraMind
   dev IoU 0.60–0.76 across seeds), a retrained model differs from its source by training
   noise that can exceed the transform's effect, and picking "the better transform" from
   seed 0 on dev would pick noise. So the paired, training-free version comes first: the
   existing 6 source checkpoints, unchanged, on transformed dev images.
   - **Histogram matching:** per band, one global monotone map fitted on raw DN:
     v → F_S⁻¹(F_P(v)), F_P = empirical CDF of the unlabelled pool (753 images), F_S = that
     of the Sen1Floods11 `train` images (252; images only), each from every 50th pixel with a
     non-zero band, 1,001 quantiles, linear interpolation. Fitted once on the pool, never on
     dev or on a single dev image (label-free, not transductive). Applied to dev images
     before the model's normalization; no-data pixels stay 0.
   - **FDA (Yang & Soatto 2020), per band:** replace the low-frequency amplitude
     (|f| within β = 0.01 of the centred spectrum, i.e. a (2·⌊0.01·512⌋+1)² = 11 × 11 square
     for 512 × 512 chips) of each dev image by the **mean** amplitude of the Sen1Floods11
     `train` images (deterministic; a random source image would add noise), keep the phase,
     inverse FFT, clip to ≥ 0. β fixed in advance (the paper's small-β setting), not tuned.
   - The training-time version (source trained on pool-styled inputs) stays in the plan,
     but only as transform × 3 seeds × both models if run at all, after a cost estimate.
2. **Decoder AdaBN for TerraMind (extra row).** TerraMind's encoder has no BatchNorm, but
   its UNet decoder and neck have 9 `BatchNorm2d` layers. AdaBN on those 9 layers, same
   recipe as the U-Net (one cumulative pass over the pool, no tuning).
3. **Rule for BatchNorm during self-training (section 4), fixed now because AdaBN is worth
   ≈ 6 points:** the student's BN layers run in train mode on its batches; the EMA teacher's
   BN buffers are copied from the student at every EMA step (not averaged); before every
   offline evaluation of a self-trained U-Net or TerraMind checkpoint, one AdaBN pass on the
   pool re-estimates the BN statistics (same recipe as `scripts/adabn.py`). The comparison
   row for self-training is therefore "AdaBN alone", and the self-training gain is reported
   against it as well as against source-only.

## 2026-10-15 — Self-training details fixed before implementation (phase-D tasks 4.1–4.3)

Details the design left open, fixed before any self-training run (they are part of the
pre-registered default configuration; nothing here is tuned on dev):
- **Batches.** Each step: half a batch of labelled Sen1Floods11 `train` chips (the source
  config's D4 augmentation and normalization) and half a batch of pool images (D4
  augmentation). U-Net 8 + 8, TerraMind 4 + 4 (design: batch 16 / 8).
- **Source loss:** the source task's own criterion (TerraTorch multiclass Dice, ignore −1).
- **Pseudo-labels:** teacher softmax on the un-augmented-by-colour target images; per
  predicted class, keep the top q % most confident pixels of that class in the batch, others
  ignored. q rises linearly from 20 % to 50 % over the run. The two per-class confidence cut-offs
  are logged every step (spec: separate, recorded thresholds).
- **Mixing (binary DACS):** for each target image, a random source image of the batch; its
  labelled water pixels (with label 1) are pasted onto the target; the other pixels keep the
  pseudo-label. Loss: pixel cross-entropy with ignore, mean over non-ignored pixels.
- **Student-only augmentation on the mixed image** (in normalized units, all bands alike):
  with p = 0.8, brightness shift U(−0.2, 0.2) and contrast factor U(0.8, 1.2) per image;
  with p = 0.5, Gaussian blur with σ ~ U(0.1, 1.0) (kernel 5). No colour jitter per band
  (would change band ratios, i.e. the water signal itself).
- **Masked variant (MIC):** an extra loss term, weight 1: the target image with random
  64 × 64 patches set to 0 (the normalized mean) at ratio 0.7, CE against the same
  pseudo-labels.
- **Forward passes:** source and mixed (and masked) batches go through the student in
  separate forward passes (as DACS/DAFormer code), so BN batch statistics are per domain.
- **EMA teacher:** α_t = min(1 − 1/(t + 1), 0.99) (the DACS/DAFormer ramp, so the teacher
  follows the student at the start); BN buffers copied from the student each step (rule of
  the earlier entry).
- **Optimizer:** AdamW, weight decay 0.05 (as source), LR = 0.1 × source LR (U-Net 3e-5,
  TerraMind 5e-6), linear warm-up from 0 over the first 400 of 4,000 iterations, then
  constant. No decay: one fewer choice, and the final checkpoint is the headline.
- **Model evaluated:** the **EMA teacher** at the final iteration, after the AdaBN pass on the
  pool (earlier entry). The teacher averages over iterations, so the unselected final
  checkpoint is less noisy than the student. Student weights are saved too.
- **Collapse watch:** the teacher's predicted water fraction on target batches, smoothed
  (EMA 0.98); floor = 0.5 × its mean over the first 50 iterations. Falling below logs a warning
  and records the iteration in the run record; the run continues.
- **Run record:** source checkpoint path and its commit (from the source `run.json`), seed,
  method, full config, iterations, wall clock, peak memory, collapse events, the pool
  manifest name, and the label-guard status (the whole fit runs inside
  `forbid_peru_labels()`; DataLoader workers use `guard_worker_init`).

## 2026-10-15 — Test-time input alignment and TerraMind decoder AdaBN on `peru-dev` (task 3.4 part 1, extra row)

Pre-registered in the "design amendments" entry above (same date), scored afterwards with
`scripts/phaseD_dev_table.py` → `outputs/phaseD/dev_table.md`. Runs: `scripts/adabn.py
terramind_v1_base`, `scripts/input_alignment_dev.py` (commit `4903b5f`, clean; Mac MPS);
fitted parameters `outputs/phaseD/alignment/params.npz` (pool and Sen1Floods11 `train`
quantiles, mean Sen1Floods11 amplitude).

| Row (3 seeds) | U-Net IoU | U-Net P / R | TerraMind IoU | TerraMind P / R |
|---|---|---|---|---|
| Source-only | 0.8364 ± 0.0224 | 0.848 / 0.985 | 0.7027 ± 0.0913 | 0.714 / 0.979 |
| AdaBN (U-Net all 46 BN; TerraMind decoder + neck 9 BN) | 0.8969 ± 0.0161 | 0.919 / 0.974 | 0.6563 ± 0.1019 | 0.665 / 0.981 |
| Test-time histogram matching | 0.8734 ± 0.0052 | 0.918 / 0.948 | **0.8089 ± 0.0064** | 0.853 / 0.940 |
| Test-time FDA (β = 0.01) | 0.0116 ± 0.0010 | 0.632 / 0.012 | 0.0104 ± 0.0016 | 0.487 / 0.011 |

Tolerant IoU in the table file. Per seed, TerraMind HM: 0.8152 / 0.8091 / 0.8024 (source
0.7639 / 0.5978 / 0.7464); decoder AdaBN: 0.7333 / 0.5407 / 0.6948.

**Findings.**
1. **Histogram matching helps TerraMind most and removes its seed sensitivity.** Mean
   0.703 → 0.809 (47 % of the gap), and the seed spread drops from ± 0.091 to ± 0.006: the
   weak seed 1 goes 0.598 → 0.809. So TerraMind's seed-dependent failure is a reaction to the
   input distribution (the bright desert), and moving inputs back into the Sen1Floods11 range
   removes it. For the U-Net, HM (0.873) is below AdaBN (0.897): the feature-level fix is
   better there. HM raises precision and costs recall (0.98 → 0.94–0.95).
2. **Decoder-only AdaBN hurts TerraMind** on every seed (−3 to −6 points). Re-normalizing the
   decoder while the encoder features stay shifted does not help; whatever TerraMind needs
   happens in the encoder (LayerNorm, no stored statistics).
3. **Per-band FDA destroys the water signal.** Not a code bug (checked on a dev chip): the
   transform does what was specified. Replacing the low-frequency amplitude includes the zero
   frequency, i.e. each band's mean. Peru chips are much brighter than the Sen1Floods11
   average (e.g. B03 mean 3312 vs 1364, B11 3076 vs 2031 on the first dev chip), so each band
   is shifted by a different amount and the B03 − B11 difference that marks water shrinks by
   ~900 DN; the models then see almost no water (recall 0.01). FDA was designed for RGB
   photographs; its per-band use on multispectral data was our extrapolation
   (`docs/literature_phaseD.md`), and this is why it fails. Reported as pre-registered; an
   FDA variant that keeps the zero frequency was **not** run (it would be a post-hoc change).

**Thesis use.** Chapter V (baselines); the discussion: TerraMind's seed sensitivity is an
input-distribution effect; spectral data needs alignment that preserves band relationships
(HM is monotone per band and kept the water/dry ordering; FDA's mean shift did not).

## 2026-10-15 — Self-training implementation and local demos (tasks 4.1–4.4)

`src/floodseg/self_training.py` (module), `scripts/self_train.py` (runner),
`src/floodseg/adapt_eval.py` + `scripts/eval_adapted.py` (offline dev evaluation: AdaBN pass,
then logits), as pre-registered in the "self-training details" entry. Changes found while
implementing, before any real run:
- **Pseudo-label ranking by logit margin, exact top-k.** A unit test showed that a
  probability threshold keeps more than q % when confidences tie; in float32 the softmax ties
  at 1.0 for logit margins above ~17, which is common for confident pixels (same trap as the
  float16 bug). The margin |z_water − z_dry| ranks pixels identically for two classes and
  does not saturate; exactly ⌈q·n⌉ pixels per class are kept. The logged cut-offs are margins.
- **The teacher stays in eval mode** (Lightning's `train()` would otherwise switch on dropout
  and BN batch statistics in the teacher).
- **Collapse warnings** recorded once per episode.
Unit tests `tests/test_self_training.py` (per-class top-q, ties, mixing pastes only labelled
source water, masking ratio and block structure, augmentation). Demos (`--demo`: 20
iterations, batch 2 + 2, MIC variant, Mac MPS fp32): U-Net 40 s, TerraMind 71 s (11.7 GB MPS
allocation); losses finite, q schedule 0.20 → 0.50, teacher between source and student weights
(EMA working), no collapse warning; teacher water fraction on pool batches ≈ 0.05 (U-Net)
and ≈ 0.15 (TerraMind). The offline evaluation ran on the demo checkpoint (174 dev files).
Demo outputs in `outputs/phaseD/self_train/mic_demo/` are not results.

## 2026-10-15 — Cloud protocol for the self-training runs (fixed before the runs; tasks 4.5–4.7, 4.9)

- **Authors' decision:** run the default and masked variants with 3 seeds and the oracle grid
  (seed 0) in one pod session (stages A + B together).
- **Jobs** (`scripts/cloud/jobs_phaseD_<model>.txt`), per backbone: `st` and `mic` with the
  pre-registered defaults, seeds 0–2 (6 runs); oracle grid at seed 0, `st`, α ∈ {0.99, 0.999}
  × q ∈ {20, 50, 80} %, where q is the **final** share of the q schedule (start stays 20 %,
  so q = 20 % is constant); the default (0.99, 50 %) is shared with the headline runs, so 5
  extra runs. 11 runs per backbone, 22 in total. The 3-seed follow-up of the oracle-best
  configuration needs dev labels to choose it, so it is a later, separate session.
- **Checkpoint interval 1,000 iterations** (was 500 in design.md), changed before any run to
  keep pod disk and transfer small; it only changes the resolution of the oracle checkpoint
  choice (iterations 1,000 / 2,000 / 3,000 / 4,000). The headline stays the final checkpoint.
- **Dev predictions on the pod, scoring on the Mac.** After each run, the pod runs the AdaBN
  pass and saves dev predictions (`scripts/eval_adapted.py --all`): packed binary maps for
  intermediate checkpoints, float32 logits for the final one. The pod holds dev and pool
  *images* but **no Peru label file** (`scripts/cloud/sync_phaseD.sh` copies images and split
  lists only and checks that no label folder exists); scoring happens on the Mac. Intermediate
  weights are deleted after their predictions are saved; final teacher and student kept.
- **Source checkpoints uploaded weights-only** (`scripts/cloud/stage_phaseD.py`; state dict
  identical, checked).
- **Two queues share one GPU** (U-Net and TerraMind in parallel) to use the GPU fully; the
  recorded wall clock per run is therefore under shared load and is reported as such (peak
  memory is per process).

## 2026-10-15 — Correction: label-edge distance counted the chip frame (phase-C analysis recomputed)

**What was wrong.** The phase-C scripts (`analyze_dev.py`, `uncertainty_dev.py`,
`edge_tolerance.py`) found label edges with `scipy.ndimage.binary_erosion`, whose default
treats everything outside the chip as background. So every chip's 1-px frame counted as a
"label water edge": the "edge (≤ 1 px)" band and the "≤ 10 px from a water edge" (near water)
region also contained a frame around every chip, mostly easy desert. Found while writing the
phase-D metric module (a unit test on a toy chip). A second, related problem: for chips with
no edge at all, scipy's distance transform has no zero pixel and measures to a phantom point
outside the corner, marking corner pixels as edge (hidden before, because the frame always
gave an edge).

**Fix** (commit `f849700`): one shared `floodseg.metrics.edge_distance` with
`border_value=1` (the frame is not an edge) and +∞ for chips without edges; tests in
`tests/test_metrics.py`. The phase-C scripts now use it, and only the phase-C regimes
(commit after `f849700`). The whole analysis was re-run (`outputs/analysis_dev/run_edgefix.log`,
Mac CPU, ≈ 50 min); old outputs kept in `outputs/analysis_dev_before_edgefix_2026-10-15/`.

**What changed** (Peru-only, mean ± std over 3 seeds; U-Net / TerraMind / Prithvi):
- Pixels: edge band 1,721,974 → 1,131,488; near-water region 6,911,817 → 3,961,359.
- Edge-band water IoU: 0.761 / 0.695 / 0.661 → 0.740 / 0.667 / 0.631 (ranking unchanged).
- Near-water ECE (single models): 3.16 / 3.54 / 4.42 % → 5.42 / 6.17 / 7.60 %; after
  ensembling + temperature: 1.64 / 2.35 / 1.97 % → 3.14 / 4.52 / 3.76 %. The models are
  roughly twice as miscalibrated near water as reported; the relative effect of temperature
  scaling and ensembling (largest for Prithvi and U-Net, smallest for TerraMind) is unchanged.
- Near-water error AUROC (single): 0.783 / 0.766 / 0.793 → 0.814 / 0.815 / 0.801.
- Edge-tolerant IoU: changes ≤ 0.0002; the conclusion that most of the U-Net's lead lies in
  the 1-px edge band stands (Peru-only lead 1.50 → 0.39 and 1.97 → 0.27 points).
- Not affected: overall IoU, ECE and AUROC; temperatures; NDWI, water-body-size, land-cover
  and other strata that do not use the edge distance.
Source: `outputs/analysis_dev/{tables.md,calib_tables.md,edge_tolerance.md}` vs the old copies.
The write-ups `docs/failure_analysis_dev.md`, `docs/calibration_dev.md` and
`docs/uncertainty_dev.md` were updated to the new numbers (each has a dated note). Earlier
METHODS_NOTES entries quoting near-water ECE/AUROC or edge-band IoU (2026-10-14) are
superseded by these values; they are left as written for the record.

**Thesis use.** Chapter V (calibration and failure analysis): use only the corrected values.

## 2026-10-15 — Self-training results on `peru-dev` (tasks 4.5–4.7): negative result

> **Partly corrected the same day** (next entry, "Correction: the evaluation-time AdaBN pass
> causes most of the self-training drop"). The numbers below stand; the *mechanism* paragraph
> and the AdaBN caveat were wrong: without the AdaBN pass, TerraMind `st` scores 0.758 ± 0.006,
> and its predicted water area does not grow. The TerraMind **oracle-grid** conclusions below
> (damage by iteration 1,000, no setting rescues it, slower teacher degrades more slowly) were
> also drawn from AdaBN-evaluated predictions and are uninformative about self-training itself;
> the grid's final checkpoints without AdaBN are in the correction entry.

**What was done.** The 22 pre-registered self-training runs (cloud-protocol entry above) were
scored on `peru-dev` on the Mac: per backbone, `st` (teacher–student with binary DACS) and
`mic` (+ masked consistency), pre-registered defaults, seeds 0–2, final EMA teacher after the
AdaBN pass on the pool (the pre-registered evaluated model); and the seed-0 oracle grid
α ∈ {0.99, 0.999} × final q ∈ {20, 50, 80} % at iterations 1,000–4,000. Pixels: all 174 dev
chips pooled, as every phase-D row. Sources: `outputs/phaseD/dev_table.md`,
`dev_table_runs.csv`, `oracle_selection.md`, `paired_deltas.csv`, `water_frac_trajectory.csv`.

**Headline (water IoU, mean ± std over 3 seeds; gap closed = (IoU − source)/(upper − source)):**

| Row | U-Net | TerraMind |
|---|---|---|
| Source-only | 0.8364 ± 0.0224 | 0.7027 ± 0.0913 |
| AdaBN (U-Net all BN; TerraMind decoder BN) | 0.8969 ± 0.0161 (58 %) | 0.6563 ± 0.1019 (−21 %) |
| Test-time histogram matching | 0.8734 ± 0.0052 (35 %) | 0.8089 ± 0.0064 (47 %) |
| Self-training `st` | 0.8184 ± 0.1102 (−17 %) | 0.4841 ± 0.0273 (−98 %) |
| Self-training `mic` | 0.8186 ± 0.0927 (−17 %) | 0.5149 ± 0.0302 (−84 %) |
| Peru-trained upper bound | 0.9415 ± 0.0019 | 0.9265 ± 0.0025 |

Precision/recall: `st` TerraMind 0.486 / 0.992, `mic` TerraMind 0.516 / 0.995 (source-only
0.714 / 0.979). For comparison, MNDWI > 0 is 0.487 / 0.996 (IoU 0.4858): TerraMind after
self-training is no better than the zero-threshold index.

**Paired per-seed deltas vs the same source seed** (IoU points; the means hide them):
- U-Net `st`: +0.071 / −0.011 / −0.114 (seeds 0/1/2); `mic`: +0.016 / +0.030 / −0.099.
  AdaBN: +0.062 / +0.053 / +0.067 (every seed). HM: +0.032 / +0.013 / +0.066.
- TerraMind `st`: −0.294 / −0.131 / −0.231; `mic`: −0.282 / −0.058 / −0.223. HM: +0.051 / +0.211 / +0.056.
Self-training is either harmful or seed-dependent on both backbones; no seed of either
backbone beats AdaBN (U-Net) or HM (TerraMind) from the same source seed, except U-Net seed 0
`st` (0.914 vs AdaBN 0.905).

**Oracle selection (task 4.7; uses dev labels, optimistic by construction, reported only next
to the dev-tuned MNDWI threshold 0.9251).** Best (configuration, checkpoint) over the seed-0 grid:
- U-Net: α 0.99, final q 50 %, iteration 4,000 — i.e. the default seed-0 run — IoU 0.9138
  (tolerant 0.9653). Same seed AdaBN: 0.9052. The whole U-Net grid at 4,000 iterations lies in
  0.890–0.914, so on seed 0 the hyperparameters barely matter; seeds 1–2 show that the
  default can also lose 0.01–0.11.
- TerraMind: α 0.999, final q 80 %, iteration 1,000, IoU 0.6353, below its own source-only seed
  0 (0.7639). Every TerraMind grid cell at every checkpoint is below source-only
  (0.42–0.64): no setting in the grid rescues it, and the damage happens by iteration 1,000.
  A slower teacher (α 0.999) degrades more slowly (0.63 → 0.44–0.57).

**Mechanism: confirmation bias toward false-positive water.** The source models' Peru error
is false positives on bright desert/coast surfaces (recall ≈ 0.98, precision 0.71–0.85;
2026-10-15 starting-point entry). Self-training turns the teacher's confident false positives
into pseudo-labels, so the student learns them and the teacher's water area grows:
- teacher water fraction on pool batches (`target/water_frac_ema`, smoothed 0.98; from
  `metrics.csv`), iteration 100 → 4,000: TerraMind `st` 0.147/0.131/0.104 → 0.175/0.194/0.131,
  `mic` 0.167/0.148/0.119 → 0.232/0.217/0.213; U-Net stays at ≈ 0.04–0.07 throughout;
- predicted water fraction on dev (final model, all dev pixels): dev labels 0.107; TerraMind
  source 0.229/0.270/0.205 → `st` 0.322/0.353/0.271; U-Net source 0.132/0.131/0.137 → `st`
  0.116/0.129/0.162 (seed 2, the seed that lost 0.11 IoU, is the one that grew);
- recall rises to ≈ 0.99 while precision falls (table above).
U-Net starts with fewer false positives (precision 0.85), so the loop has less to amplify,
and the outcome depends on the seed. TerraMind starts at precision 0.71 (seed 1: 0.60) and
diverges on every seed. Per-class top-q selection does not protect against this: the
false positives are among the most confident water pixels.

**Checks.** All 22 runs exit 0, 4,000 iterations, commit `e8a5841` clean; transfer verified
file by file (4,048 files); every scored folder has 174 dev logit files (asserted by the
table script); the Peru-label guard was active in every fit, and the pod held no label file.

**Caveats.**
- **The collapse watch was one-sided.** It warns only if the teacher's target water fraction
  falls below half its initial value (pre-registered); the failure observed is the opposite
  (expansion). "0 collapse events" therefore says nothing about stability. For a later
  design: a two-sided band, or a precision proxy that needs no labels (e.g. agreement with
  an MNDWI threshold).
- The AdaBN pass before evaluation (pre-registered) is known to hurt TerraMind source-only
  (decoder AdaBN −0.047 mean). ~~It cannot explain the drop by itself~~ **Wrong, see the
  correction entry: it explains most of the TerraMind drop.** The headline rows stay as
  pre-registered.
- The oracle grid is seed 0 only; the 3-seed follow-up of the oracle configuration (4.7 part
  2) is not run. For U-Net the oracle configuration is the default, so the 3 headline seeds
  already are that follow-up.
- Gap closed on means is misleading when the source mean is driven by a weak seed
  (TerraMind seed 1); the paired deltas are the primary comparison.

**Alternatives not taken.** No new configuration was tried after seeing dev results (e.g.
self-training on histogram-matched inputs, a lower LR, fewer iterations): any of those
would be selected on dev and is a new design decision for the authors, to be pre-registered
first if done.

**Conclusion for the thesis.** On Peru, simple statistics alignment beats teacher–student
self-training: AdaBN for the BN-based U-Net (+0.06 on every seed, 58 % of the gap) and test-time
histogram matching for TerraMind (+0.05 to +0.21 on every seed, 47 % of the gap, and removes
its seed sensitivity). Self-training, as specified a priori from the DACS/DAFormer/MIC
recipe, amplifies the source model's dominant error (desert false positives) instead of
correcting it. Feeds: thesis chapter 5 (results, unsupervised adaptation) and 6
(discussion: why self-training fails when the source error is confident and systematic);
paper section "Adapting without Peru labels".

## 2026-10-15 — Correction: the evaluation-time AdaBN pass causes most of the self-training drop (post-hoc diagnostic)

**What was done and why.** The previous entry kept the pre-registered evaluation (final EMA
teacher, then one AdaBN pass on the pool) and attributed the TerraMind collapse to
confirmation bias. Because decoder AdaBN already hurt TerraMind source-only (−0.047), the
same final teachers were re-evaluated **without** the AdaBN pass, i.e. with the BN buffers the
teacher had at the end of training (copied from the student every step: running statistics,
momentum 0.1, over the separate source, mixed and — for `mic` — masked forward passes).
`scripts/eval_adapted.py --no_adabn` (new flag), regimes `st_noadabn`, `mic_noadabn` in
`outputs/logits_dev/` and in `outputs/phaseD/dev_table.md` (labelled "post-hoc diagnostic").
Both backbones, all 3 seeds, final checkpoint only (the intermediate weights of all runs were deleted on the
pod after their AdaBN-evaluated predictions; the final teachers of the oracle grid are re-evaluated
in the addendum below).
**This is post-hoc**: chosen after seeing dev results; not a headline row.

**Results (water IoU, seeds 0/1/2; mean ± std):**

| | source | AdaBN only | HM only | `st` (+AdaBN, headline) | `st`, no AdaBN | `mic` (+AdaBN, headline) | `mic`, no AdaBN |
|---|---|---|---|---|---|---|---|
| U-Net | 0.843/0.855/0.811 | 0.905/0.907/0.878 | 0.875/0.868/0.878 | 0.914/0.844/0.698 | 0.900/0.891/0.841 (0.8774 ± 0.0319) | 0.859/0.884/0.713 | 0.796/0.877/0.857 (0.8435 ± 0.0424) |
| TerraMind | 0.764/0.598/0.746 | 0.733/0.541/0.695 | 0.815/0.809/0.802 | 0.470/0.467/0.516 | 0.751/0.761/0.761 (0.7576 ± 0.0058) | 0.482/0.540/0.523 | 0.619/0.704/0.672 (0.6648 ± 0.0432) |

TerraMind `st` without AdaBN: precision 0.771 ± 0.004, recall 0.978 (source 0.714 / 0.979);
predicted dev water fraction 0.205/0.239/0.167 vs source 0.229/0.270/0.205 (labels 0.107; all
fractions over every dev pixel, ignore pixels (−1) counted as non-water, same denominator for
all; the primary evidence is the higher precision at the same recall),
i.e. the water area **shrinks** slightly; with AdaBN it grew to 0.322/0.353/0.271.
`mic` without AdaBN: 0.277/0.262/0.256 (grows); U-Net `st` no AdaBN 0.112/0.120/0.131.

**What this corrects.**
1. The TerraMind headline collapse (0.48) is mostly caused by the evaluation-time AdaBN pass
   (0.758 → 0.484 on the same weights), not by self-training itself. The "confirmation bias:
   water area grows" mechanism of the previous entry is **withdrawn for `st`**: the evidence
   for it (dev precision falling, dev water fraction growing) came from AdaBN-evaluated
   teachers. The training-time teacher's water fraction on pool batches does rise (TerraMind
   iteration 100 → 4,000: 0.147/0.131/0.104 → 0.175/0.194/0.131), but the pool has no labels
   and dev shows no expansion, so this is not evidence of false positives.
2. For `mic`, the water area does grow even without AdaBN and IoU falls below source on seeds
   0 and 2. Hypothesis (not tested): the masked forward pass (70 % of pixels set to 0) enters
   the BN running statistics the teacher inherits, so the masked variant's BN buffers describe
   a distribution no real image has.
3. Why AdaBN after self-training is harmful (hypothesis): during self-training the weights
   co-adapt with BN statistics of a *mixture* (source batches, DACS-mixed batches); replacing
   them with pure-pool statistics moves every BN layer at once, away from what the weights
   were trained with. AdaBN is a fix for a model trained on source statistics only; applying
   it on top of a model that already saw target data in training is not justified. The
   pre-registered BN rule (2026-10-15, design amendments) did not anticipate this; we also did
   not revisit it after the decoder-AdaBN result for TerraMind (−0.047), which came before the
   pod runs — a missed warning, recorded as such.

**What does not change.** Even with the evaluation choice made post hoc in its favour,
self-training does not beat the simple alignment baselines on any seed:
- U-Net `st` no AdaBN 0.900/0.891/0.841 < AdaBN 0.905/0.907/0.878 (paired: −0.005/−0.016/−0.037);
- TerraMind `st` no AdaBN 0.751/0.761/0.761 < HM 0.815/0.809/0.802 (−0.064/−0.048/−0.042).
Paired vs source, `st` without AdaBN is positive on every U-Net seed (+0.057/+0.036/+0.030) and
mixed for TerraMind (−0.013/+0.163/+0.014): mostly it repairs the weak seed 1 and removes the
seed sensitivity (std 0.006), as HM does, but less. The thesis conclusion "simple statistics
alignment beats teacher–student self-training on Peru" stands; the explanation of *why*
self-training fails changes from "confirmation bias" to "small gains that do not reach the
alignment baselines, made worse by an evaluation rule (AdaBN after self-training) that
destroys TerraMind and destabilizes U-Net".

**Reporting rule.** Headline rows stay as pre-registered (with AdaBN). The no-AdaBN rows are
reported as a post-hoc diagnostic, with the reason above; the oracle grid (all with AdaBN) is
reported with this caveat, since its TerraMind cells are depressed by the same pass; only the
iteration-4,000 column can be re-run without AdaBN (addendum).

**Checks.** 174 dev logit files per folder (asserted); run.json of each diagnostic folder
records `adabn_bn_layers: 0`; same weights file (`iter4000.pt`, teacher) as the headline rows.

Feeds: thesis chapter 5 (results; the diagnostic row) and 6 (discussion: BN statistics and
self-training; pre-registration caught an evaluation rule that was wrong, and the honest
report keeps it); paper section "Adapting without Peru labels".

**Addendum (same day): oracle grid, final teachers without AdaBN (post-hoc).** The 5 extra
seed-0 grid runs per backbone re-evaluated at iteration 4,000 with `--no_adabn`
(`outputs/phaseD/oracle_grid_noadabn.md`; logits in `outputs/logits_dev/st_a*_noadabn/`).
Water IoU (precision), seed 0:

| α, final q | U-Net | TerraMind |
|---|---|---|
| 0.99, 20 % | 0.9009 (0.940) | 0.7917 (0.813) |
| 0.99, 50 % (default) | 0.9002 (0.947) | 0.7510 (0.766) |
| 0.99, 80 % | 0.9028 (0.939) | 0.6873 (0.701) |
| 0.999, 20 % | 0.8875 (0.930) | **0.7955** (0.806) |
| 0.999, 50 % | 0.8967 (0.946) | 0.7840 (0.793) |
| 0.999, 80 % | 0.8884 (0.950) | 0.6841 (0.689) |
| reference, seed 0: source / AdaBN / HM | 0.8433 / 0.9052 / 0.8748 | 0.7639 / 0.7333 / 0.8152 |

For TerraMind, fewer pseudo-labels are better (final q 20 % > 50 % > 80 % at both α): the
more of the teacher's predictions are trusted, the worse, which is consistent with
confirmation bias at high q. The best cell (0.7955), chosen with dev labels, is +0.032 over
source-only but still below test-time HM on the same seed (0.8152), which needs no training. For
U-Net no cell reaches AdaBN on seed 0 (0.9052). So even oracle-selected self-training, with the
evaluation choice made post hoc in its favour, does not beat the simple alignment baselines on
seed 0. This weakens the case for a further self-training session on the current recipe.

## 2026-10-15 — Reproducibility: phase-D diagnostics moved from inline code into a script

The paired deltas, water-fraction trajectories, dev water fractions and the no-AdaBN oracle-grid
scores of the three entries above were first computed with inline Python. They are now produced
by `scripts/phaseD_diagnostics.py` (definitions in its docstring) → `outputs/phaseD/{paired_deltas.csv,
water_frac_trajectory.csv, dev_water_fraction.csv, oracle_grid_noadabn.md}`. Check: re-running it
reproduced the earlier files exactly (trajectory CSV byte-identical, grid table identical, every
paired delta and dev water fraction quoted above unchanged; `paired_deltas.csv` gains the
no-AdaBN columns). No number changes. Feeds: reproducibility appendix.

## 2026-10-15 — Pod session 2, pre-registered before launch: Prithvi source models and training-time input alignment (tasks 5.4, 3.4 part 2)

**Authors' decision:** the authors added $10 of RunPod credit and asked to run what fits
(balance ≈ $16.4). Everything below is fixed before any run; dev results are not used for
any choice in this session except where stated (and then only the pre-registered rule).

**Queue A — Prithvi-EO-2.0 300M source models on Sen1Floods11 (task 5.4).** Why: Prithvi
was the best Sen1Floods11-only model on Peru in phase C (seed 0, dev 0.8758), so it is the most
relevant missing backbone. Config `configs/prithvi_eo_v2_300.yaml` unchanged (6 HLS bands,
UNet decoder, batch 4, AdamW wd 0.1, ReduceLROnPlateau, early stopping patience 20), 16-mixed
on an RTX 4090, official Sen1Floods11 splits.
- LR sweep, seed 0, 100-epoch cap: {3e-5, 1e-4, 3e-4} (half-decade steps around the config's
  1e-4). Winner = highest `valid` water IoU (`val/IoU_Flood`, max over epochs).
- **Edge rule** (TerraMind's best LR sat on its grid edge): if the winner is 3e-5, add 1e-5;
  if 3e-4, add 1e-3; re-select over all four.
- Final: seeds 0–2 at the winner, 200-epoch cap (as section 1). Checkpoint = best `valid`.
- Selection runs automatically on the pod (`scripts/cloud/queue_phaseD2.sh A`), so finals cannot
  be queued with a wrong LR (the section-1 mistake). `valid` labels are on the pod; no Peru data.
- Afterwards, on the Mac: `peru-dev` logits (source-only), test-time histogram matching (as
  for U-Net/TerraMind), decoder AdaBN if the decoder has BN layers (noting it hurt TerraMind,
  −0.047), `valid` logits for the temperature fit. **No self-training** for Prithvi
  (`docs/phaseD_dev.md`).

**Queue B — training-time input alignment (task 3.4 part 2, design decision 7).** The source
model is trained on Sen1Floods11 chips transformed towards the Peru pool, then applied to
**unchanged** Peru images.
- `floodseg.train_alignment.HistMatchToPool`: per band, v → F_pool⁻¹(F_s1f(v)), the inverse of
  the test-time map, from the same cached quantiles (`outputs/phaseD/alignment/params.npz`:
  1,001 quantiles of every 50th non-no-data pixel of the 252 Sen1Floods11 `train` images and of
  the 753 pool images). Check: mapping all 252 train chips reproduces the pool quantiles exactly
  (5/25/50/75/95 % of GREEN, NIR, SWIR1, SWIR2).
- `FDAToPool`: per band, the centred low-frequency amplitude square (β = 0.01, half-width 5 px)
  replaced by that of a random pool image (phase kept); validation uses the mean pool amplitude
  (deterministic, as the test-time FDA). Pool amplitude windows: (753, 13, 11, 11), built by
  `scripts/build_train_alignment.py` under the Peru-label guard; only this npz (5 MB) goes to the
  pod, no Peru image. FDA is run as designed even though test-time FDA failed (it shifts each
  band's mean and so changes B03 − B11); expectation recorded: likely harmful.
- Applied to raw DN after TerraTorch's NaN → 0 and band selection, before D4 and normalization;
  pixels with all bands 0 stay 0 (checked). Also applied to `valid`, so checkpoint selection sees
  the same input style as training (the model is meant for Peru-styled inputs).
- Runs: U-Net (LR 3e-4) and TerraMind (LR 5e-5, the section-1 LRs) × {HM, FDA} × seeds 0–2,
  200-epoch cap; then Prithvi × HM × seeds 0–2 at queue A's LR (after queue A is done: two
  Prithvi runs may not fit in 24 GB). All 3 seeds are run up front (cheap, ≈ 10 min each) to
  avoid a dev-dependent pause; the pre-registered choice between HM and FDA per backbone
  (task 3.4) is still made on **seed-0 dev IoU**, and both are reported.
- Evaluation on the Mac: dev logits on unchanged Peru images; rows `trhm`, `trfda` in the dev table.

**Budget rule.** Pod RTX 4090 Secure ($0.89/h), CUDA ≥ 13.0. Estimate from measured 4090 times
(phase-C Peru runs: Prithvi 2,056 s / 69 epochs over 753 chips ≈ 0.04 s per chip-epoch → ≈ 10 s per
Sen1Floods11 epoch; TerraMind Sen1Floods11 470 s / 73 epochs; U-Net 602 s / 104): queue A ≈ 2–3 h,
queue B ≈ 2–3 h in parallel, ≈ 3.5–4.5 h of pod time ≈ $3–4 plus setup. Ceiling: stop the pod if
the session reaches $9 (keeps ≥ $7 for later). Priority if the budget runs short: A (Prithvi) >
B U-Net/TerraMind HM > FDA > Prithvi HM.

**Checks before launch:** unit tests `tests/test_train_alignment.py` (nodata, quantile map,
band subset, FDA shape); Mac smoke tests (1 epoch, few batches) of U-Net + HM, U-Net + FDA,
Prithvi + HM all train and validate. One first U-Net + HM attempt with 4 DataLoader workers
stalled at 0 % CPU and was killed; it did not reproduce (two later runs with workers finished),
so the pod watcher also checks that logs keep growing.

Feeds: thesis chapter 5 (Prithvi rows; input alignment at training vs test time), paper section
"Adapting without Peru labels".

## 2026-10-15 — Where the best label-free methods still fail, and their calibration (tasks 5.2, 5.3)

**What was done and why.** For the two best label-free methods (U-Net + AdaBN, TerraMind +
test-time histogram matching, HM), compared with their source-only models and the Peru-trained
upper bound: (1) errors per stratum, to see which part of the Peru gap alignment closes and
which remains; (2) calibration, to see whether the uncertainty stays usable after adaptation,
and whether a temperature can be fitted **without Peru labels**. `scripts/phaseD_analysis.py`
(definitions in its docstring) → `outputs/phaseD/analysis/{strata.csv, strata_runs.csv,
calibration.csv, calibration_runs.csv, temperatures.json}`. Both alignment methods were run
for both backbones; only the best one per backbone is discussed below.

**Method.** 174 dev chips, labelled pixels only (label ≠ −1), 3 seeds per regime.
- Strata: the phase-C strata (`scripts/analyze_dev.strata_for`, edge distance after the
  chip-frame fix). TP/FP/FN/TN per seed and stratum value → IoU, recall, false-positive rate (FPR),
  mean over seeds. For water-only strata (water NDWI, water-body size) IoU equals recall.
- Calibration: ECE (15 bins) and error AUROC of −|z| on every 7th pixel ("all") and all pixels
  ≤ 10 px from a label water edge ("near"), as `scripts/uncertainty_dev.py`. Temperatures:
  `temp_s1f` = T fitted by NLL on Sen1Floods11 `valid` (all 89 chips, labelled pixels) for the
  same source model and seed. This is **label-free with respect to Peru**: AdaBN and HM do not
  change the model's `valid` behaviour, so the source model's `valid` T is used (U-Net
  2.12/2.11/2.02, ensemble 1.94; TerraMind 1.13/1.36/1.23, ensemble 1.16). `temp_dev` = 2-fold-by-
  chip T on dev labels (the phase-C method; reference only, since it uses Peru labels).
  `valid` logits come from `scripts/phaseD_valid_logits.py`, which reproduces each source run's
  logged best `val/IoU_Flood` exactly (6/6).
- Check: the Peru-trained U-Net near-water ECE (5.42 %) and the dev IoUs equal the corrected
  phase-C numbers and the dev table.

**Results — strata (IoU or recall/FPR, seed means; source → method → Peru-trained):**
- *Edges.* Neither method fixes the label edge. U-Net edge (≤ 1 px) IoU 0.593 → 0.611 → 0.740;
  TerraMind 0.571 → 0.580 → 0.667. Far from edges (> 10 px) U-Net goes 0.890 → 0.959 (Peru-trained 0.984).
- *Desert false positives are what alignment fixes.* FPR on bare/sparse land: U-Net 0.026 →
  0.012 (Peru-trained 0.003); TerraMind 0.063 → 0.024 (0.004). This stratum holds 33.5 M of the
  40 M labelled pixels, so it drives the IoU gain.
- *Low-contrast and small water is lost.* Recall of water with NDWI < 0.1: U-Net AdaBN 0.28–0.75
  (it keeps more of it than the Peru-trained model in 0–0.1: 0.747 vs 0.608). TerraMind HM
  collapses: 0.02 (NDWI < −0.1), 0.08 (−0.1–0), 0.32 (0–0.1), vs source 0.38/0.82/0.81. Water
  bodies < 1 ha: U-Net 0.601 → 0.486 (0.580); TerraMind 0.558 → 0.334 (0.460).
- *HM creates a new error for TerraMind:* dry pixels with NDWI 0.1–0.3 have FPR 0.511 under HM
  vs 0.176 source and 0.071 Peru-trained, and IoU > 10 px from edges is 0.810 vs 0.979. HM
  sharpens the index contrast: ambiguous mid-NDWI land is pushed to water, and faint water is
  pushed to dry. The net is positive only because bare desert dominates the pixel count.
- *Dry pixels with NDWI > 0.3* (43 k pixels): every model calls most of them water (FPR
  0.65–0.93). These are probably label omissions or wet soil; the Peru-trained models learned
  the label convention (0.649 / 0.695), the label-free ones cannot.

**Results — calibration (near-water ECE %, mean ± std over seeds; error AUROC near water):**

| | source single | method single | method + T(S1F11 valid) | method ens + T(valid) | method + T(dev), reference | Peru-trained single |
|---|---|---|---|---|---|---|
| U-Net, AdaBN | 12.98 ± 1.18 | 9.49 ± 0.39 | 7.64 ± 0.42 | 5.33 | 7.71 ± 0.29 | 5.42 ± 0.18 |
| TerraMind, HM | 11.91 ± 1.12 | 7.96 ± 0.27 | 7.04 ± 0.06 | 6.62 | 5.60 ± 0.04 | 6.17 ± 0.32 |

AUROC near water: U-Net AdaBN 0.755 ± 0.014 (Peru-trained 0.814); TerraMind HM 0.785 ± 0.005 (0.815).
AUROC over all pixels: U-Net AdaBN 0.645, TerraMind HM **0.522 ± 0.015** (≈ chance; source 0.770).
- Alignment itself improves calibration near water (ECE −3.5 and −4.0 points).
- A label-free temperature (from Sen1Floods11 `valid`) helps further and, combined with the
  3-seed ensemble, brings near-water ECE to the Peru-trained single-model level (U-Net 5.33 vs
  5.42; TerraMind 6.62 vs 6.17). For U-Net the `valid` temperature is as good as one fitted on
  Peru dev labels (7.64 vs 7.71).
- But **the remaining errors are confident**: over all pixels the uncertainty of TerraMind + HM
  does not rank errors above correct pixels (AUROC 0.52). Most of its remaining errors are
  the confident mid-NDWI false positives above, far from water edges. Calibration is fine on
  average (ECE all 0.84 % with T), while the uncertainty is uninformative for these errors.

**Caveats.** Dev only; strata are those of phase C (WorldCover, Copernicus DEM, cloud mask);
NDWI bins use (B03 − B08)/(B03 + B08) as stored in the strata (×100 in the code bins); water-only
strata say nothing about false positives. `temp_s1f` assumes the model's `valid` behaviour is
unchanged by the method — true by construction for HM (input transform only on Peru) and for
AdaBN (BN statistics re-estimated only on the pool; on `valid` the source statistics apply).
The ensemble rows reuse the same 3 seeds (no extra training).

**For the thesis.** Label-free alignment closes the Peru gap mainly by removing desert false
positives. Label edges, small water bodies and low-contrast water stay as they were, or get
worse (TerraMind + HM). A Sen1Floods11-fitted temperature with a 3-seed ensemble gives
near-water calibration at the Peru-trained level without Peru labels, but the remaining errors
of TerraMind + HM are confident, so uncertainty cannot be used to flag them. Feeds chapter 5
(stratified results, calibration) and chapter 6 (limits of label-free adaptation; why
histogram matching trades one error for another); paper sections "Where adaptation fails" and
"Uncertainty under shift".

## 2026-10-16 — Pod session 2 results: Prithvi source models and training-time input alignment on `peru-dev` (tasks 5.4, 3.4 part 2)

**What was run** (protocol: entry "Pod session 2, pre-registered"; runs and deviations: EXPERIMENT_LOG
2026-10-15/16 "Pod session 2"). All 21 runs exited 0 (3 sweep, 3 Prithvi finals, 9 trHM, 6 trFDA); copy-back verified file by file (186 files);
pod deleted. Dev logits on **unchanged** Peru images with each model's base config and its best
`valid` checkpoint (`scripts/phaseD2_dev_logits.sh`, the procedure of the phase-D source rows);
scored by `scripts/phaseD_dev_table.py` → `outputs/phaseD/dev_table.md`, `dev_table_runs.csv`;
paired deltas by `scripts/phaseD_diagnostics.py` → `outputs/phaseD/paired_deltas.csv`. Prithvi
test-time rows on the Mac with the existing scripts, Prithvi added (`adabn.py`: 9 BN layers in its
UNet decoder; `input_alignment_dev.py`: HM and FDA on all 13 bands, then Prithvi's 6 selected).

**Prithvi LR (automatic, pre-registered rule).** Max `val/IoU_Flood`, seed 0, 100-epoch cap:
3e-5 0.8247, 1e-4 0.8257, 3e-4 0.8170 → 1e-4, interior, no edge run. The three are within 0.009,
so the LR choice matters little. Finals (200-epoch cap) best `valid`: 0.8330 / 0.8263 / 0.8354
(best epochs 74 / 42 / 111). The seed-0 final and the seed-0 sweep run at 1e-4 differ (0.8330
vs 0.8257, both best at epoch 74): GPU 16-mixed training is not bit-reproducible, and the two
ran next to other jobs. Mac `valid` logits of the finals reproduce the logged values exactly
(also a check that stripping the optimizer state left the weights intact).

**Results, `peru-dev` water IoU (3 seeds, mean ± std; gap closed vs source and Peru-trained):**

| Row | U-Net | TerraMind | Prithvi |
|---|---|---|---|
| Source-only | 0.8364 ± 0.0224 | 0.7027 ± 0.0913 | **0.8701 ± 0.0011** |
| AdaBN | 0.8969 ± 0.0161 (58 %) | 0.6563 ± 0.1019 (−21 %) | 0.8673 ± 0.0094 (−5 %) |
| Test-time HM | 0.8734 ± 0.0052 (35 %) | 0.8089 ± 0.0064 (47 %) | 0.8185 ± 0.0104 (−100 %) |
| Test-time FDA | 0.0116 (fails) | 0.0104 (fails) | 0.0076 (fails) |
| **Training-time HM** | **0.9118 ± 0.0026 (72 %)** | 0.8440 ± 0.0284 (63 %) | 0.8597 ± 0.0098 (−20 %) |
| Training-time FDA | 0.6151 ± 0.1122 (−211 %) | **0.8365 ± 0.0010 (60 %)** | not run |
| Peru-trained (upper) | 0.9415 ± 0.0019 | 0.9265 ± 0.0025 | 0.9218 ± 0.0006 |

Per seed (IoU − same-seed source; the train-time models are separate training runs with the same
seed, so the pairing is by seed only): U-Net trHM +0.072 / +0.055 / +0.100; TerraMind trHM
+0.053 / +0.244 / +0.127, trFDA +0.071 / +0.240 / +0.090; Prithvi trHM −0.000 / −0.009 / −0.022,
AdaBN +0.005 / −0.013 / −0.000, test-time HM −0.060 / −0.040 / −0.054.

**Pre-registered HM-vs-FDA choice (seed-0 dev IoU, task 3.4).** U-Net: HM 0.9148 vs FDA 0.7251 →
**HM**. TerraMind: FDA 0.8353 vs HM 0.8170 → **FDA**. By the rule, TerraMind's training-time row is
FDA (0.8365 ± 0.0010). HM has the higher 3-seed mean (0.8440 ± 0.0284), but the difference is
within HM's seed spread, and FDA is far more stable; the HM mean is reported, as post-hoc
information, not as the choice. The FDA result contradicts the expectation recorded before launch
("likely harmful", from test-time FDA): at training time the random pool amplitude acts as an
augmentation over Peru-like low-frequency styles, while test-time FDA replaced a Peru image's
amplitude by one mean Sen1Floods11 amplitude and destroyed the B03 − B11 contrast. For U-Net it
is harmful (0.615, high variance), as expected.

**Findings.**
1. **Training-time HM is the best label-free method for U-Net** (0.912, every seed above AdaBN's
   same seed: 0.915/0.910/0.911 vs 0.905/0.907/0.878), closing 72 % of the gap, with low seed
   variance. It raises precision (0.962 vs source 0.848) at some cost in recall (0.946 vs 0.985).
2. **For TerraMind, training-time alignment (HM or FDA) beats every test-time method** (0.84 vs
   test-time HM 0.809), and removes the seed sensitivity of its source models (FDA std 0.001).
3. **Prithvi transfers best without any adaptation** (source-only 0.870 ± 0.001, above every
   adapted TerraMind row), and **every adaptation hurts it**: AdaBN on its decoder −0.003,
   test-time HM −0.052, training-time HM −0.010. Its residual error is still false positives
   (precision 0.887, recall 0.978). One reading: it sees only 6 bands (B02, B03, B04, B8A, B11, B12),
   and its pre-training on harmonized HLS imagery may make it less sensitive to the
   Sentinel-2 L1C intensity shift that HM corrects. This is a hypothesis, not tested.
4. **No label-free method reaches the Peru-trained models** (best: U-Net trHM 0.912 vs 0.942).

**Checks.** Dev logits: 174 files per run; table IoU = TerraTorch-logged dev IoU for the 18
`terratorch validate` runs (max |diff| 2.7e-8; one-off comparison of the logs with
`dev_table_runs.csv`, not part of the script; the Prithvi AdaBN/HM/FDA rows have no TerraTorch log). Source-run reproducibility check of
Prithvi `valid` logits passed (above).

**Caveats.** Dev only; test splits untouched. The train-time rows' `valid` IoUs (U-Net HM
0.813–0.821, FDA 0.716–0.727; TerraMind HM 0.829–0.832, FDA 0.764–0.776; Prithvi HM
0.831–0.834) are on *transformed* `valid` and are not comparable with the source models' `valid`
IoUs. Budget deviation: the session cost $9.68 (billing read 2026-10-16, GPU $9.51 + disk $0.16;
may still rise slightly) against the pre-registered $9 ceiling. The estimate (3.5–4.5 h) was
≈ 2.5× too low because the CPU transforms made queue B input-bound (TerraMind FDA ≈ 1¾ h per
seed). At 08:20 UTC (≈ $8.2) the authors' instruction to use the added credit was taken to
override the conservative ceiling; the full priority list was then completed, nothing was cut,
and no result had been looked at. Prithvi HM was started early, in parallel with TerraMind FDA,
with 4 DataLoader workers instead of queue B's 8 (speed only). Session 1 billing read $6.19
(2026-10-16; the earlier balance drop implied ≈ $6.98, not reconciled). Remaining balance ≈ $16.4 − $9.68 ≈ $6.7 (not read from the account).

**Which method per backbone is frozen for the test evaluation (section 6) is a separate decision**
for the authors; recommendation in `docs/phaseD_dev.md`.

Feeds: thesis chapter 5 (phase-D results table, Prithvi column; training- vs test-time alignment),
paper section "Adapting without Peru labels".

## 2026-10-16 — Strata and calibration extended to Prithvi and the training-time rows (tasks 5.2, 5.3, addendum)

**What and why.** The 2026-10-15 analysis covered the then-best label-free rows (U-Net AdaBN,
TerraMind test-time HM). Pod session 2 changed the best rows, and the pre-registration promised
Prithvi rows, so `scripts/phaseD_analysis.py` now covers regimes s1f11_src, adabn, hm, peru, trhm,
trfda × U-Net, TerraMind, Prithvi (17 combinations; trfda not run for Prithvi). Strata, metrics,
pixels and methods are unchanged from the 2026-10-15 entry.

**Temperature for the train-time rows.** The label-free `temp_s1f` must come from the same
model: for trhm/trfda, the train-time checkpoint's own Sen1Floods11 `valid` logits on the
transformed `valid` it was selected on (model config + its `configs/phaseD/train_*.yaml` overlay;
`scripts/phaseD2_valid_logits.sh` → `outputs/logits_valid/<trhm|trfda>/`). Using the source
model's T would have silently calibrated one model with another's temperature. Check: each
validate pass reproduces the run's logged best `val/IoU_Flood` within 2e-5 (Mac fp32 vs pod
16-mixed), all 15 runs (`outputs/phaseD/pod2/valid_logits.txt`). The adabn/hm rows keep the source
model's T (their weights are the source weights). `temperatures.json` is now keyed by logit folder
then model. Temperatures (seeds 0/1/2, ensemble): Prithvi source 2.80/2.88/3.23, 2.62; trHM U-Net
1.91/2.22/1.88, 1.82; trHM TerraMind 1.15/1.47/1.18, 1.15; trHM Prithvi 2.92/2.85/3.12, 2.68;
trFDA TerraMind 1.38/1.52/1.24, 1.24. Check: the 2026-10-15 U-Net/TerraMind numbers reproduce
exactly (e.g. U-Net AdaBN near ECE 9.49, ens+T 5.33; TerraMind HM all-pixel AUROC 0.522).

**Results** (`outputs/phaseD/analysis/strata.csv`, `calibration.csv`; means over seeds):

| | U-Net trHM | U-Net AdaBN | TerraMind trFDA | TerraMind trHM | TerraMind HM | Prithvi source | Prithvi trHM |
|---|---|---|---|---|---|---|---|
| dev IoU | 0.912 | 0.897 | 0.837 | 0.844 | 0.809 | 0.870 | 0.860 |
| bare/sparse FPR (Peru: U 0.003, T 0.004, P 0.005) | **0.005** | 0.012 | 0.007 | 0.016 | 0.024 | 0.018 | 0.016 |
| edge ≤ 1 px IoU (Peru: 0.740, 0.667, 0.631) | 0.613 | 0.611 | **0.426** | 0.566 | 0.580 | 0.558 | 0.595 |
| 2–3 px IoU (Peru: 0.922, 0.923, 0.917) | 0.882 | 0.875 | 0.695 | 0.848 | 0.859 | 0.787 | 0.875 |
| water < 1 ha IoU (Peru: 0.580, 0.460, 0.449) | 0.349 | 0.486 | 0.233 | 0.350 | 0.334 | **0.565** | 0.435 |
| water recall, NDWI −0.1–0 / 0–0.1 (Peru U 0.58/0.61) | 0.23 / 0.39 | 0.66 / 0.75 | 0.54 / 0.31 | 0.06 / 0.30 | 0.08 / 0.32 | 0.69 / 0.83 | 0.23 / 0.56 |
| dry FPR, NDWI 0.1–0.3 (Peru U 0.058) | 0.103 | 0.174 | 0.081 | 0.357 | 0.511 | 0.163 | 0.315 |
| near ECE %, single | 7.62 | 9.49 | 13.49 | 8.32 | 7.96 | 13.44 | 9.47 |
| near ECE %, ens + T_valid | **4.06** | 5.33 | 9.80 | 6.60 | 6.62 | 7.97 | 5.36 |
| error AUROC all pixels, single / ens | 0.805 / 0.858 | 0.645 / 0.733 | 0.751 / 0.806 | 0.606 / 0.728 | 0.522 / 0.591 | 0.791 / 0.883 | 0.655 / 0.740 |

Peru-trained, near ECE single / 3-seed ensemble: U-Net 5.42 / 4.42, TerraMind 6.17 / 5.36, Prithvi 7.60 / 6.33 (ens + dev-label T: 3.14 / 4.52 / 3.76, reference only).

**Findings.**
1. **U-Net trHM removes the desert false positives almost completely** (bare FPR 0.005 vs 0.003
   Peru-trained) and keeps mid-NDWI dry land dry (FPR 0.103). That explains its IoU gain over
   AdaBN. Like test-time HM, it pays on faint water: recall 0.23 / 0.39 for NDWI −0.1–0 / 0–0.1
   (AdaBN 0.66 / 0.75), and small water bodies (< 1 ha IoU 0.349 vs 0.486). Edges unchanged (0.613).
2. **U-Net trHM is also the best-calibrated label-free model**: with T from its own transformed
   `valid` and the 3-seed ensemble, near-water ECE is 4.06 %, below the Peru-trained U-Net 3-seed ensemble without T (4.42 %; with a dev-label T 3.14 %, reference only) and every Peru-trained single
   model. Its errors are rankable (all-pixel AUROC 0.805 single, 0.858 ensemble).
3. **TerraMind trFDA (the pre-registered choice) is stable but has a boundary problem.** It has
   the best TerraMind interior (> 10 px IoU 0.946) and the fewest false positives, but the worst
   edges (≤ 1 px 0.426, 2–3 px 0.695), the lowest small-water IoU (0.233), and missed water even at
   high NDWI (recall 0.950 for NDWI > 0.3; every other row 0.996–1.000). Its calibration is poor
   (near ECE 13.5 % single, 9.8 % ens + T). TerraMind trHM has better edges and calibration
   (6.60 %) but HM's faint-water trade-off (recall 0.06 / 0.30). This is relevant to the
   freezing decision; the pre-registered rule (task 3.4) stands as recorded.
4. **Prithvi source-only keeps faint and small water** (recall 0.69 / 0.83; < 1 ha IoU 0.565,
   above its Peru-trained model's 0.449) and has the most rankable errors of all rows (all-pixel
   AUROC 0.791 single, 0.883 ensemble). Its loss is false positives (bare FPR 0.018, mid-NDWI 0.163)
   and edges (0.558). Training-time HM moves it the HM way (fewer false positives, faint water
   lost), which nets −0.010 IoU.
5. Across all rows, every alignment that removes desert false positives loses faint water; no
   label-free row matches the Peru-trained models at the edge (≤ 1 px).

**Caveats.** The faint-water and small-water strata are where the UNOSAT labels are least
certain (`docs/failure_analysis_dev.md`, label QA), so part of the "lost" recall may be label
noise; dry pixels with NDWI > 0.3 have FPR 0.54–0.93 for every model, consistent with label
omissions. Dev only.

Feeds: thesis chapter 5 (where adapted models still fail; calibration of adapted models), paper
section "Adapting without Peru labels".

## 2026-10-16 — Phase-D freeze for the one-time test evaluation (task 6.1; authors' decision)

**Decision.** The authors approved the recommendation in `docs/phaseD_dev.md` ("Open
decisions", now "Decided") on 2026-10-16. All choices use `peru-dev` (and Sen1Floods11 `valid`
for checkpoints and temperatures) only; no test file had been read for any phase-D model. The
freeze is the commit that adds `configs/frozen.yaml`; that hash is the evidence that every choice
below predates the test evaluation.

| Role | Backbone | Row (logit folder) | Dev water IoU (3 seeds) | Why |
|---|---|---|---|---|
| primary | U-Net | training-time HM (`trhm`) | 0.9118 ± 0.0026 | best label-free row; also U-Net's pre-registered train-time choice (seed-0 dev HM 0.9148 vs FDA 0.7251) |
| primary | TerraMind | training-time FDA (`trfda`) | 0.8365 ± 0.0010 | pre-registered choice (seed-0 dev FDA 0.8353 vs HM 0.8170) |
| primary | Prithvi | source-only (`s1f11_src`) | 0.8701 ± 0.0011 | no method improved it on dev |
| secondary | TerraMind | training-time HM (`trhm`) | 0.8440 ± 0.0284 | higher mean than FDA within its seed spread; better edges and calibration than FDA (strata/calibration addendum above) |
| secondary | U-Net | AdaBN (`adabn`) | 0.8969 ± 0.0161 | best test-time row for U-Net |
| secondary | TerraMind | test-time HM (`hm`) | 0.8089 ± 0.0064 | best test-time row for TerraMind |
| reference | U-Net, TerraMind | source-only (`s1f11_src`) | 0.8364, 0.7027 | gap-closed denominator |
| reference | all three | Peru-trained (`peru`) | 0.9415, 0.9265, 0.9218 | upper bound, gap-closed denominator |

**Scope of the freeze (procedural, follows from the spec).** The gap-closed column on test needs
the source-only and Peru-trained rows on `peru-test` too, and those are section-7 baselines of
`plan-flood-research`. Freezing phase-D rows now and baselines later would freeze baselines after
the test set had been seen, so `configs/frozen.yaml` lists the baselines as well:
- evaluated in the phase-D pass: the 11 groups above (33 runs; first written as 10 groups / 30 runs, a miscount, corrected 2026-10-16) on `peru-test` and
  `peru-2017-test`;
- frozen now, evaluated in the section-7 pass: Sen1Floods11 → Peru fine-tuned models
  (`s1f11_ft`, 9 runs; dev 0.938 / 0.928 / 0.924) on the Peru test sets, and the Sen1Floods11
  `test` split for the Sen1Floods11-trained rows. Deferred only for disk space (below); their
  checkpoints and hashes are fixed now.
- not frozen (not reported on test): self-training rows (harmful on dev), test-time FDA (fails,
  dev IoU ~0.01), U-Net training-time FDA, Prithvi AdaBN / test-time HM / training-time HM (all
  below Prithvi source on dev), phase-C seed-0 Sen1Floods11 models (superseded by the 3-seed
  phase-D source models), TerraMind LR 2e-5 source runs (superseded).

**Frozen recipe (each entry of `configs/frozen.yaml`).** Checkpoint path and sha256 (the
optimizer-stripped files for the Prithvi source runs, since those are what load; state-dict
equality was checked when stripping); seed; the training run's commit from its run record.
Prediction: water if the water logit z = logit_water − logit_other > 0, no threshold tuning.
Fitted artifacts, none fitted on a test split:
- AdaBN: BN statistics re-estimated on the 753 `peru-train` pool images
  (`configs/pools/peru-train.csv`, sha256 in the file), exact cumulative average, batch 16, as on dev.
  Deterministic given the pool, so the statistics are recomputed, not stored.
- Test-time HM: the global per-band map pool → Sen1Floods11 `train` in
  `outputs/phaseD/alignment/params.npz` (sha256 in the file), fitted on the 2023 pool images. It is
  applied unchanged to `peru-2017-test` images as well (the 2017 event has no pool of its own; a
  2017-specific fit would use test images).
- Training-time HM/FDA: no test-time transform (Peru images unchanged), as on dev.
- Temperatures for calibration: `outputs/phaseD/analysis/temperatures.json` (Sen1Floods11 `valid`
  NLL fits, per model, seed and ensemble; sha256 in the file).

**Test sets and scores (spec `adaptation-evaluation`).** `peru-test`: 172 UNOSAT 2023 chips (Yaku
event), full set and the two doubtful-excluded variants from `scripts/label_qa.py`
(`outputs/label_qa/chips.csv`, built in task 4.4 from labels and MNDWI, no model; test counts
doubtful 6 all blobs / 3 without near-cloud blobs). `peru-2017-test`: EMSR199 2017 event, main
number on the 101 chips that do not touch any 2023 `peru-train` location, all 204 alongside
(`docs/peru_data_card.md`). Metrics as on dev: pooled plain and boundary-tolerant water IoU,
precision, recall (`floodseg.metrics`), mean ± std over seeds, gap closed against the same
backbone's test-set source and Peru-trained rows. Per-chip counts are kept for bootstrap CIs.

**Gate.** `scripts/evaluate_frozen.py` refuses `peru-test` / `peru-2017-test` for any run not in
the committed `configs/frozen.yaml`, if the checkpoint hash differs, if the working tree is
dirty, or if the run's test logits already exist. Before the test pass it runs on `peru-dev`
through the same code and must reproduce the dev table logits.

**Disk.** Float32 logits are 1.0 MB per chip; the phase-D pass writes 33 runs × 376 chips ≈ 12.4 GB
with 21 GB free. The section-7 rows are deferred to keep ≥ 8 GB free until `outputs/` is backed up.

**Alternatives rejected.** TerraMind trHM as the primary row: it would switch the pre-registered
rule after seeing its consequences; it is reported as a secondary row instead, so both are on
test. Freezing only the phase-D rows: rejected for the gap-closed reason above.

**After the test pass.** No model choice changes. An evaluation-code bug found later is fixed,
rerun and logged as a correction with its reason.

Feeds: thesis chapter 3 (protocol: freeze before test), chapters 4–5 (test results), paper
section "Evaluation protocol".

**Pre-test check on `peru-dev` (done before the freeze commit).** `scripts/evaluate_frozen.py
peru-dev` ran all 33 phase-D runs through the gated code path (Mac MPS fp32) into
`outputs/frozen_check/logits_dev/`; `scripts/score_frozen.py peru-dev --compare` compared them with
the dev-table logits (`outputs/phaseD/test/peru-dev_check_compare.csv`). **31 of 33 runs are
bit-identical** (first written as 28 of 30, a miscount) (max |Δz| = 0, IoU equal), including the runs whose dev logits came from
`terratorch validate` (source, Peru-trained, trHM, trFDA) and the test-time HM runs.
**Two AdaBN U-Net runs differ**: seed 0 max |Δz| 2.2, 8,286 pixels change sign, IoU 0.9056 vs
0.9052; seed 2 max |Δz| 14.3, 71,405 pixels, IoU 0.8852 vs 0.8783; seed 1 identical.
Investigation: same checkpoint (unchanged since 14 Oct 17:59–18:20, before the AdaBN run), same
pool manifest (sha256-verified on read), same algorithm (`adapt_bn` is a refactor of the
`d48fc3f` code), no dependency change since `d48fc3f`. Today the AdaBN pass is deterministic:
two MPS passes give identical BN statistics, and CPU agrees within 3.8e-6 (means) / 7.7e-7
(relative variance) (scratch script, seed 2). The smallest re-estimated running variance is
8.7e-13, so a BN channel with near-zero target variance amplifies any accumulation-order
difference; this is a plausible mechanism for the 14 Oct difference but not proven. **Cause
not identified.** Consequence: the AdaBN U-Net dev row becomes 0.8993 ± 0.0122 (gap closed 60 %)
with today's reproducible recomputation, vs 0.8969 ± 0.0161 (58 %) as reported; no freeze choice
depends on it (it is a secondary row, below U-Net trHM 0.9118 either way). The test pass uses
the frozen recipe (recompute on the pool), i.e. today's statistics; tables that show the AdaBN
dev row next to its test row use 0.8993 and footnote the original value.

## 2026-10-16 — One-time test evaluation of the frozen configurations (task 6.2)

**What.** After the freeze commit `f1f61d6` (pushed), `scripts/evaluate_frozen.py peru-test` and
`peru-2017-test` ran the 33 phase-D runs once (Mac MPS fp32; every run.json records commit
`f1f61d6`, `git_dirty: false`, the frozen sha256s; logs `outputs/phaseD/test/eval_*.log`; 172 / 204
logit files per run checked). Scored with `scripts/score_frozen.py` (first reading of any test
label) and `scripts/bootstrap_frozen.py` (2,000 chip-bootstrap replicates, paired across rows,
seed-mean IoU, chip sampling only). Sources: `outputs/phaseD/test/{peru-test,peru-2017-test}.md`,
`*_groups.csv`, `*_runs.csv`, `*_chip_counts.csv`, `bootstrap_{groups,diffs}.csv`.

**Correction to the scoring script (before any write-up).** The first scoring printed gap-closed
ratios of 117–354 % on `peru-2017-test`: there the Peru-trained rows are *below* source-only for
every backbone, so the denominator is negative and the ratio meaningless (e.g. U-Net trHM 0.854 <
source 0.917 showed "+162 %"). `score_frozen.py` now prints "n/a (upper ≤ source)" when IoU_Peru ≤
IoU_source; rescored from the saved logits (no inference rerun). peru-test denominators are large
(U-Net 0.25, TerraMind 0.19), unaffected.

**Results, `peru-test` (172 chips, 2023 Yaku event; water IoU, mean ± std over 3 seeds, 95 % chip-bootstrap CI):**

| Row | IoU | 95 % CI | precision | recall | gap closed | dev IoU |
|---|---|---|---|---|---|---|
| U-Net trHM (primary) | 0.7985 ± 0.0048 | 0.704–0.865 | 0.952 | 0.832 | 88 % | 0.912 |
| TerraMind trFDA (primary) | 0.7085 ± 0.0200 | 0.618–0.779 | 0.928 | 0.750 | 44 % | 0.837 |
| Prithvi source-only (primary) | 0.7405 ± 0.0136 | 0.664–0.796 | 0.799 | 0.911 | — | 0.870 |
| TerraMind trHM (secondary) | 0.7209 ± 0.0053 | 0.609–0.802 | 0.896 | 0.787 | 50 % | 0.844 |
| U-Net AdaBN (secondary) | 0.7308 ± 0.0127 | 0.625–0.812 | 0.795 | 0.901 | 61 % | 0.899 |
| TerraMind test-time HM (secondary) | 0.7055 ± 0.0006 | 0.587–0.790 | 0.860 | 0.797 | 42 % | 0.809 |
| U-Net / TerraMind source-only | 0.5763 / 0.6261 | | 0.597 / 0.668 | 0.947 / 0.911 | — | 0.836 / 0.703 |
| Peru-trained U-Net / TerraMind / Prithvi | 0.8280 / 0.8149 / 0.8244 | | 0.966 / 0.960 / 0.966 | 0.853 / 0.843 / 0.850 | — | 0.942 / 0.927 / 0.922 |

Paired differences (seed-mean IoU, 95 % CI): U-Net trHM − Peru U-Net −0.030 [−0.048, −0.013];
U-Net trHM − source U-Net +0.222 [0.132, 0.309]; U-Net trHM − Prithvi source +0.058 [0.013, 0.105];
U-Net trHM − AdaBN +0.068 [0.005, 0.138]; TerraMind trFDA − source +0.082 [0.016, 0.156]; TerraMind
trHM − trFDA +0.013 [−0.022, 0.044] (not separable); Prithvi source − Peru Prithvi −0.084
[−0.134, −0.034]. Doubtful-excluded variants (166 / 169 chips) shift every row by +0.004 to
+0.016 and change no ordering (`peru-test.md`).

**Results, `peru-2017-test` (EMSR199 2017 event; main subset 101 chips not touching 2023 train):**
source-only U-Net 0.917, TerraMind 0.934, Prithvi 0.937; Peru-trained 0.878 / 0.892 / 0.878; U-Net
trHM 0.854, AdaBN 0.887; TerraMind trFDA 0.831, trHM 0.866, HM 0.875. **The order reverses**: every
alignment method and every Peru-trained model is below its source-only model (U-Net trHM − source
−0.064 [−0.083, −0.046]; TerraMind trFDA − source −0.103 [−0.175, −0.048]; Prithvi source − Peru
+0.059 [0.039, 0.082]). TerraMind trHM beats trFDA on 2017 (+0.035 [0.012, 0.066]). All 204 chips:
same pattern. Precision is ≥ 0.988 for every row; the differences are recall.

**Reading.**
1. On the 2023 event, label-free training-time HM brings U-Net to 0.03 IoU of the Peru-trained
   U-Net, and is the best label-free row, ahead of the best source model (Prithvi) by 0.058. The dev
   ordering of rows holds on test for the primaries.
2. Every row drops ≈ 0.10–0.13 from dev to test, the Peru-trained ones too (U-Net 0.942 → 0.828,
   recall 0.97 → 0.85): the test blocks are harder or labelled differently, not an adaptation
   effect. The higher gap closed on test (88 % vs 72 % on dev) comes mostly from source-only falling
   further (0.836 → 0.576); the paired differences are the better summary.
3. On the 2017 event no label-free method beats source-only, and Peru training does not either.
   The 2017 labels define dry only where MNDWI ≤ 0 and ignore MNDWI > 0 pixels outside EMS water
   (`docs/peru_data_card.md`), so this set cannot penalise the over-prediction that phase D removes
   (precision ≈ 1 for all) and in effect measures recall of spectrally clear water — where the dev
   strata showed alignment and Peru training are conservative. **No method wins both events.**
4. Pre-registered TerraMind rule (FDA over HM, seed-0 dev): on `peru-test` the two are not
   separable; on 2017 HM is clearly better. The primary stays trFDA (no change after test); both rows
   are reported.

**Calibration on test:** not computed yet; the dev claim "U-Net trHM is the best-calibrated
label-free model" is dev-only until it is (frozen temperatures and saved logits allow it with no refit).

**Post-hoc diagnostic, declared before running (changes no reported number).** To test reading 3,
on `peru-2017-test` (all 204 chips and the 101-chip subset), per row: recall on label-water pixels
binned by MNDWI (0–0.1, 0.1–0.3, > 0.3), and the share of ignored (−1) pixels with MNDWI > 0 that
the row predicts as water. Expected if reading 3 holds: the recall gap between source-only and the
aligned / Peru-trained rows sits mostly in the low-MNDWI bins.

Feeds: thesis chapters 4–5 (test results, two-event contrast), paper section "Results on held-out
events"; the dev→test drop and the 2017 label construction go in the limitations.

**Post-hoc diagnostic result** (`scripts/diag_2017_recall.py` → `outputs/phaseD/test/diag_2017_recall.csv`;
all 204 chips; seed means; label-water pixels per bin 0.60 M / 0.65 M / 11.47 M, ignored MNDWI > 0 7.37 M):

| Row | recall MNDWI 0–0.1 | 0.1–0.3 | > 0.3 | ignored MNDWI > 0 predicted water |
|---|---|---|---|---|
| source-only U-Net / TerraMind / Prithvi | 0.344 / 0.349 / 0.356 | 0.642 / 0.836 / 0.851 | 0.998 / 0.999 / 0.999 | 0.711 / 0.722 / 0.750 |
| Peru-trained U-Net / TerraMind / Prithvi | 0.017 / 0.054 / 0.049 | 0.401 / 0.511 / 0.399 | 0.998 / 0.997 / 0.994 | 0.628 / 0.719 / 0.619 |
| U-Net trHM / AdaBN | 0.019 / 0.082 | 0.158 / 0.451 | 0.988 / 0.996 | 0.619 / 0.613 |
| TerraMind trFDA / trHM / HM | 0.084 / 0.053 / 0.047 | 0.223 / 0.275 / 0.355 | 0.912 / 0.985 / 0.992 | 0.461 / 0.627 / 0.674 |

Reading 3 holds: on clear water (MNDWI > 0.3, 91 % of 2017 label water) every row except TerraMind
trFDA has recall ≥ 0.985; the 2017 gap between source-only and the aligned / Peru-trained rows is
in the faint-water bins (0–0.3), where source-only keeps 0.34–0.85 and the others 0.02–0.51 — the
same trade-off the dev strata showed. TerraMind trFDA additionally misses clear water (0.912), as on
dev (0.950), which is why it is the lowest 2017 row. The ignored MNDWI > 0 pixels (EMS says dry, or
image/label disagree) cannot be scored; source-only rows call more of them water. Whether that is
correct depends on whether those pixels were wet on 2017-04-07, which the labels cannot tell.
The 2017 result therefore says the label-free methods and Peru training trade faint-water recall
for fewer false positives, and the 2017 labels score only the first half of that trade.

## 2026-10-16 — Feature engineering for a pixel classifier (course progress, week 5; exploratory, dev-only)

**What and why.** The course asked for a feature-engineering progress report. The deep
models take raw L1C reflectance, so this measures what hand-made features (spectral indices,
local context) add, using a classical pixel classifier. It runs on branch
`avance-sem5-feature-engineering` only and is not merged into main. It was run **after** the
one-time test evaluation, so it reads no test split (asserted in the script) and changes
nothing in the frozen comparison.

**Method.** `scripts/feature_engineering.py`.
- **Features**, on r = DN/10,000:
  - 13 bands;
  - 8 indices: NDWI (B03, B08), MNDWI (B03, B11), MNDWI2 (B03, B12), NDVI, NDMI, AWEI_nsh,
    AWEI_sh, WRI;
  - 5 context features: MNDWI local mean 3×3 and 7×7, MNDWI local std 7×7, NIR and NDVI local
    mean 7×7, via `uniform_filter` on the full chip.
- **Feature sets:** bands; indices; bands+indices; all 26.
- **Training:** sklearn HistGradientBoosting (200 iterations, lr 0.1, no early stopping), on
  random labelled pixels with the natural class prior:
  - 600 per `peru-train` chip (450,403 px, water share 0.040);
  - or 2,000 per Sen1Floods11 `train` chip (501,892 px, water share 0.093).
- **Seeds:** 3; the seed sets the sample and the model.
- **Evaluation:** `peru-dev`, every 2nd row and column; pooled water IoU, p > 0.5.
- **Baseline:** MNDWI > t, with t fitted by best pooled IoU over all labelled pixels of the
  training set (0.005 bins): t = 0.28 (`peru-train`), 0.15 (Sen1Floods11 `train`).
- **Extras:** univariate AUROC per feature; permutation importance (Jaccard, 3 repeats) of
  the 26-feature model, seed 0, on 1,200 random dev pixels per chip.

**Results** (`outputs/feature_engineering/{results,auroc,importance}.csv`, `summary.md`).

Trained on `peru-train` (peru-dev IoU):

| Row | IoU |
|---|---|
| bands | 0.9300 ± 0.0013 |
| indices | 0.9127 ± 0.0040 |
| bands+indices | 0.9256 ± 0.0033 |
| all 26 | 0.9290 ± 0.0046 |
| MNDWI > 0.28 | 0.9237 |

Trained on Sen1Floods11 `train` (peru-dev IoU):

| Row | IoU |
|---|---|
| bands | 0.7359 ± 0.0181 |
| indices | 0.7348 ± 0.0092 |
| bands+indices | 0.7233 ± 0.0070 |
| all 26 | 0.7126 ± 0.0148 |
| MNDWI > 0.15 | 0.8358 (precision 0.848, recall 0.983) |

- The Sen1Floods11-trained classifiers have recall ≥ 0.985 and precision 0.72–0.74, the same
  desert false-positive error as the source-only deep models.
- Permutation importance: the Peru model relies on AWEI_nsh, MNDWI2, MNDWI, B12; the
  Sen1Floods11 model on B12, AWEI_nsh, B08, B06.
- Visible-band AUROC (water vs dry) is 0.56–0.62 on Sen1Floods11 and 0.24–0.32 on peru-train,
  i.e. the direction flips: water is darker than the Peru desert. MNDWI is 0.983 / 0.994 /
  0.997 (Sen1Floods11 train / peru-train / peru-dev).

**Findings.**
1. With in-domain labels, engineered features add nothing to a tree ensemble. A
   train-fitted MNDWI threshold (0.924) almost equals the dev-tuned oracle (0.9251, entry
   2026-10-15). The oracle concern is therefore small for Peru 2023: the threshold is stable
   between train and dev.
2. Cross-domain, the single invariant index transfers better than any flexible model, and
   adding indices or context to the flexible model does not help. The bottleneck is the
   dry-land shift, not missing features.
3. MNDWI > 0.15 fitted on Sen1Floods11 is a label-free baseline equal to source-only U-Net on
   dev (0.8358 vs 0.8364 ± 0.0224). It was defined after the test pass, so it is not evaluated
   on test and is not added to the frozen table.

**Caveats.**
- Dev pixels are subsampled 1/4, so comparisons with the deep-model rows (all pixels) are
  approximate.
- The pixel classifier has no learned spatial context.
- Permutation importance splits credit among correlated features.

**Alternatives rejected (time; course deadline).** Retraining U-Net with an extra MNDWI
channel (it would also break the foundation models' pretrained band embeddings; future work);
SHAP.

**Thesis use.** Chapter III (justifies raw-reflectance inputs; dry-land shift as the
bottleneck); Chapter V discussion (label-free index baseline, exploratory). Course write-up:
`docs/feature_engineering.md`, `notebooks/feature_engineering.ipynb` (Spanish).

## 2026-10-16 — Test calibration of the frozen phase-D rows: method fixed before running (task 6.2 calibration part; plan 7.4)

**What and why.** The dev calibration claims ("U-Net trHM best calibrated label-free", entry
2026-10-16 "Strata and calibration extended") are dev-only until checked on the held-out sets.
This entry fixes the method **before** any calibration number on a test split is computed. It
uses only the logits already written by the one-time pass (`outputs/logits_test`,
`outputs/logits_2017`); no model is rerun and nothing is refitted on test.

**Method** (`scripts/calibration_frozen.py`). The same definitions as `scripts/phaseD_analysis.py`
on dev (`scripts/uncertainty_dev.py` helpers):
- **Metrics:**
  - ECE: 15 equal-width confidence bins on [0.5, 1].
  - Error-detection AUROC of −|z|.
  - IoU on the same pixels, as a sanity check.
- **Pixel sets:**
  - "all": every 7th labelled pixel;
  - "near": all labelled pixels ≤ 10 px from a label water edge.
- **Rows:** the 11 phase-D groups of `configs/frozen.yaml` × 3 seeds.
- **Splits:**
  - `peru-test` (172 chips);
  - `peru-2017-test` on `no_2023_train_overlap` (101, the main 2017 number) and `all` (204).
- **Label-free methods:**
  - `single`: raw logits.
  - `temp_s1f`: z / T, with T from the frozen `outputs/phaseD/analysis/temperatures.json`
    (Sen1Floods11 `valid` NLL fit per model and seed; sha256 checked against frozen.yaml). The
    T source is the same as on dev: `s1f11_src` temperatures for s1f11_src / adabn / hm, and the
    row's own for trhm / trfda. Not defined for `peru`.
  - `ensemble`: mean probability of the 3 seeds.
  - `ens+temp_s1f`: the ensemble with the frozen ensemble T.
- **Reference that uses Peru labels:**
  - `temp_dev` / `ens+temp_dev`: one T per run, fitted by NLL on every 7th labelled `peru-dev`
    pixel of that run's dev logits (`outputs/logits_dev`), then applied to test.
  - On dev this was a 2-fold-by-chip fit. For test, the whole of dev is the fitting set, which
    is the natural held-out version.
  - It is a reference for what Peru-labelled calibration reaches, not a label-free method.
- **AUROC** is reported for `single` and `ensemble` only, because a global T is monotone.

**Caveats fixed in advance.**
- `peru-2017-test` labels call a pixel dry only where MNDWI ≤ 0, and most faint water is
  missing from the labelled set. Calibration there measures the clear-water / clear-dry
  population, not the hard cases. Its "near" set is therefore not comparable to 2023's.
- The AdaBN U-Net dev logits of seeds 0 and 2 differ slightly from today's reproduction
  (freeze entry). `temp_dev` uses the stored `outputs/logits_dev`, the ones behind the dev table.

**Outputs.** `outputs/phaseD/test/calibration_runs.csv`, `calibration.csv`, `calibration.md`.

## 2026-10-16 — Section-7 pass: Sen1Floods11 → Peru fine-tuned rows on the Peru tests, source rows on Sen1Floods11 `test` (plan 7.3)

**What and why.** This closes the "evaluate once at the end" requirement for the rows that the
freeze deferred for disk space. The groups and hashes are unchanged from `configs/frozen.yaml`
(commit `f1f61d6`). About 29 GB is free; the pass needs about 3.6 GB.

**Scope, as frozen.**
- `s1f11_ft` × {U-Net, TerraMind, Prithvi} × 3 seeds on `peru-test` and `peru-2017-test`, with
  `--pass section7`.
- Sen1Floods11 official `test` (90 chips) for the Sen1Floods11-trained rows.

**Decision on "Sen1Floods11-trained rows".** The phrase is read as the source-only groups
`s1f11_src` × 3 models (9 runs), method none.
- Reason: `test` exists in the plan to report the in-domain benchmark (RQ1, comparison with
  published numbers).
- `trhm` / `trfda` are also trained on Sen1Floods11, but on transformed inputs. A `test` score
  for them would need the training-time transform applied to `test` images: a design choice that
  was not made before the freeze.
- They are left out and not evaluated. Their in-domain cost is already on `valid` (selection).
- Peru-trained and `s1f11_ft` models on Sen1Floods11 `test` were not frozen for that split
  (plan 7.3 lists "all regimes"; the freeze did not). They are not reported. This is stated as
  a deviation from plan 7.3.

**Code change (evaluation code, after the phase-D pass, no effect on it).**
- `scripts/evaluate_frozen.py` gains split `test`:
  - images are the official `test` ids (`floodseg.sen1floods11.official_ids`), read as in
    `scripts/phaseD_valid_logits.py` (raw DN → the config's normalization);
  - output goes to `outputs/logits_s1f11_test`;
  - the same one-look and hash gate applies;
  - the group filter is regime `s1f11_src` with method none, instead of `--pass`.
- `scripts/score_frozen.py` gains split `test` (subset `full`, LabelHand labels, the same pooled
  metrics).
- Tests pass (50).

**Outputs.**
- `outputs/logits_test/s1f11_ft/`, `outputs/logits_2017/s1f11_ft/`, `outputs/logits_s1f11_test/`.
- Scores in `outputs/phaseD/test/{peru-test,peru-2017-test,test}*`. The Peru splits are
  rescored with the extra rows; the existing rows' numbers must not change (check).

## 2026-10-16 — Test calibration of the frozen phase-D rows: results

Method as fixed in the entry above (commit `f2ad639`); run
`uv run python scripts/calibration_frozen.py` → `outputs/phaseD/test/calibration{.md,.csv,_runs.csv,_temperatures_dev.json}`.
ECE in %, mean over 3 seeds (ensemble rows: one value). "all" = every 7th labelled pixel;
"near" = labelled pixels ≤ 10 px from a label water edge.

**`peru-test` (172 chips), ECE all / near**

| Row | single | temp_s1f | ens+temp_s1f | ens+temp_dev (uses dev labels) | dev ens+temp_s1f (all / near) |
|---|---|---|---|---|---|
| U-Net trHM | 1.96 / 16.88 | 0.48 / 14.09 | 0.60 / 12.81 | 1.14 / 13.68 | 0.67 / 4.06 |
| U-Net AdaBN | 3.06 / 17.29 | 1.39 / 14.56 | 0.94 / 11.46 | 1.34 / 12.03 | 0.80 / 5.33 |
| TerraMind trFDA | 2.11 / 23.36 | 0.73 / 21.22 | 0.78 / 20.17 | 1.27 / 20.91 | 0.96 / 9.80 |
| TerraMind trHM | 2.31 / 20.63 | 1.35 / 19.47 | 1.34 / 18.57 | 1.16 / 18.35 | 0.38 / 6.60 |
| TerraMind HM | 2.55 / 19.32 | 1.60 / 18.20 | 1.53 / 17.47 | 1.09 / 15.81 | 0.71 / 6.62 |
| Prithvi source | 3.09 / 23.35 | 1.59 / 19.06 | 1.09 / 15.12 | 1.19 / 15.74 | 1.07 / 7.97 |
| U-Net source | 6.67 / 27.82 | 4.54 / 24.52 | 3.17 / 20.90 | 3.69 / 21.40 | 0.70 / 8.22 |
| TerraMind source | 3.45 / 18.30 | 2.34 / 17.14 | 1.73 / 15.74 | 1.56 / 15.48 | 0.93 / 9.91 |
| Peru-trained U-Net / TerraMind / Prithvi | 1.77 / 14.22, 1.77 / 14.11, 1.82 / 14.90 | — | — | 1.08 / 11.18, 1.07 / 12.13, 1.01 / 11.23 | (ens+temp_dev) 0.11 / 3.14, 0.25 / 4.52, 0.22 / 3.76 |

Error-detection AUROC of −|z| on `peru-test`, all pixels, single → ensemble:
- U-Net trHM 0.712 → 0.809;
- U-Net AdaBN 0.730 → 0.850;
- Prithvi source 0.783 → 0.902;
- Peru-trained U-Net / TerraMind / Prithvi 0.618 / 0.632 / **0.439** → 0.704 / 0.685 / **0.456**.

**Findings.**
1. **Every row is less calibrated on test than on dev, most of all near water edges.** Near-edge
   ECE rises 1.6–3.6× for every row (ens+temp, dev → test), Peru-trained included (U-Net ens+temp_dev 3.14 → 11.18 %). This
   matches the IoU drop: the test areas are harder for every model.
2. **The label-free temperature transfers.** T fitted on Sen1Floods11 `valid` (`temp_s1f`) is as
   good as or better than T fitted on Peru dev labels for most rows (U-Net trHM ens: 0.60 vs
   1.14 % all, 12.81 vs 13.68 % near). Temperature scaling and ensembling lower ECE for every
   label-free row, as on dev.
3. **The dev claim "U-Net trHM is the best-calibrated label-free row" holds only for the "all"
   pixel set on test.**
   - U-Net trHM is lowest on "all": temp_s1f 0.48 %, ens+temp_s1f 0.60 %.
   - Near edges, AdaBN U-Net is lower (11.46 vs 12.81 %, ens+temp_s1f).
   - The best label-free near-edge ECE (11.46 %) is within 0.3 points of the Peru-trained
     U-Net's (11.18 %, which uses dev labels for T).
   - The thesis must state the claim per pixel set.
4. **Peru-trained Prithvi's confidence does not flag its errors on test.** Its AUROC is below 0.5
   (0.439 single, 0.456 ensemble): its errors are more confident than its correct pixels. On dev
   it was 0.723.
   - Its near-edge AUROC is 0.706: the bad ranking comes from confident errors away from edges,
     consistent with confidently wrong regions (e.g. label gaps or doubtful areas).
   - It is reported as found. It is not investigated on test, because that would be test-driven
     analysis.

**`peru-2017-test`.**
- On the 101-chip subset, source-only rows are best calibrated: ens+temp_s1f ECE all 0.21–0.85 %,
  near 9.5–11.0 %. Aligned and Peru-trained rows sit at 2.2–3.9 % and 20–26 %.
- This mirrors the IoU reversal: the 2017 labels contain only clear water and clear dry, so the
  aligned rows' lost faint-water confidence shows up as confident misses.
- The caveat fixed in advance applies: 2017 calibration covers only the clear population.

**Caveats.** ECE depends on the pixel set and on label quality. On 2023, the doubtful-label
chips are included ("full", as declared). There is no bootstrap on calibration.

**Thesis use.** Chapter V (calibration of the label-free methods on the held-out sets; RQ3);
corrects the dev-only calibration claim.

## 2026-10-16 — Section-7 pass: results

Run as fixed in the entry "Section-7 pass" above (commits `d42c861` for `peru-test`, `78844bb` for
`peru-2017-test` and `test`; every run.json `git_dirty: false`). Scores:
`outputs/phaseD/test/{peru-test,peru-2017-test,test}{.md,_groups.csv,_runs.csv}`. CIs:
`bootstrap_{groups,diffs}.csv`.

**Checks.**
- The rescored `peru-test` / `peru-2017-test` tables reproduce all 55 earlier group rows exactly
  (max |Δ| 0.0).
- The extended bootstrap reproduces all 44 earlier group CIs and all 28 paired differences
  exactly. Chip weights are drawn per subset, before any group.
- `score_frozen.py` now also writes two-class mIoU (commit `78844bb`, before scoring). Water IoU
  is unchanged.

**Sen1Floods11 → Peru fine-tuning vs Peru-only training (paired, seed means, 95 % CI)**

| | `peru-test` IoU, ft / Peru-only | ft − Peru | 2017 (101 chips) ft − Peru |
|---|---|---|---|
| U-Net | 0.8291 / 0.8280 | +0.0010 [−0.0031, 0.0054] | −0.0154 [−0.0331, −0.0014] |
| TerraMind | 0.8138 / 0.8149 | −0.0011 [−0.0062, 0.0034] | −0.0059 [−0.0137, 0.0001] |
| Prithvi | 0.8226 / 0.8244 | −0.0018 [−0.0062, 0.0018] | +0.0019 [0.0003, 0.0036] |

- On 2023 test, Sen1Floods11 pretraining before Peru fine-tuning changes nothing. All CIs contain
  0 and span less than 0.01. This confirms the dev finding (≤ 0.4 points) behind training on Peru
  only.
- On 2017, the differences are within 0.016 and of mixed sign.

**Sen1Floods11 `test` (90 chips), source-only models**

| Model | water IoU (95 % CI) | mIoU | precision | recall | `valid` IoU (selection) |
|---|---|---|---|---|---|
| U-Net | 0.8331 ± 0.0049 (0.760–0.889) | 0.9042 | 0.940 | 0.880 | 0.8315 |
| Prithvi | 0.8203 ± 0.0115 (0.751–0.873) | 0.8966 | 0.924 | 0.880 | 0.8315 |
| TerraMind | 0.8185 ± 0.0136 (0.754–0.869) | 0.8954 | 0.910 | 0.890 | 0.8349 |

- U-Net − Prithvi +0.013 [−0.022, 0.062]; U-Net − TerraMind +0.015 [−0.019, 0.051]. The three
  are not separable in-domain (RQ1: no meaningful gap in-distribution, as hypothesised).
- Peru-2023 test vs Sen1Floods11 test, source-only: U-Net 0.833 → 0.576, TerraMind 0.819 →
  0.626, Prithvi 0.820 → 0.741. Prithvi loses the least, so "foundation models lose less on Peru"
  holds for Prithvi only.

**Deviation from plan 7.3.** Peru-trained and `s1f11_ft` models were not evaluated on
Sen1Floods11 `test` (not frozen for it). Neither were `trhm` / `trfda` (scope entry above).

**Thesis use.** Chapter V (RQ1 on test; fine-tuning vs Peru-only on test); the published-number
comparison (next entry).

## 2026-10-16 — Published-number comparison on Sen1Floods11 (plan 3.8, phase-D 1.5)

**What and why.** Task 3.8: check our Sen1Floods11 results against published numbers and
investigate any gap > 2 IoU. Write-up: `docs/published_comparison.md`. The `valid` part was
drafted read-only from the PDFs in `sources/papers/`. The test-vs-test part was added after the
section-7 pass.

**Method.**
- Published numbers come from tables, with page numbers. TerraMind Table 8 (p. 8: v1-B, S-2,
  IoU_water 82.26, mIoU 89.70) was re-checked in the PDF.
- Metric definitions were matched: pooled water IoU and two-class mIoU, the same as TerraTorch
  logs and the papers report. F1 was left out: ours is macro F1, Prithvi-EO-2.0 reports micro
  F1. Bonafilia et al. average IoU per chip on S1, so they are not comparable.

**Results (`test` vs `test`, ours vs published, water IoU / mIoU).**
- Prithvi 300M: 82.03 / 89.66 vs 82.2 / 89.7 (Prithvi-EO-2.0 Table IV): −0.17 / −0.04.
- TerraMind v1-B: 81.85 / 89.54 vs 82.26 / 89.70 (TerraMind Table 8): −0.41 / −0.16.
- U-Net: 83.31 / 90.42 vs the two secondary quotes in Prithvi-CAFE Table 1: 84.03 ([19],
  −0.72) and 80.69 ([16], +2.62).
- No like-for-like gap is above 2. The only gap above 2 is against a secondary U-Net quote with
  an unknown setup.
- Plan 3.8 and phase-D 1.5 are closed.

**Corrections found (also fixed in `docs/REFERENCES.md`).**
- The phase-A reproduction target for TerraMind was 76.3. That number is v1-B-single (Table 12),
  a different model; the setup-matched target is Table 8 (82.26).
- Prithvi-CAFE Table 1 pairs Prithvi 600M's non-TL IoU (82.5) with the TL mIoU (90.30;
  Prithvi-EO-2.0 Table IV gives 89.9 for non-TL).
- Prithvi-CAFE's TerraMind row (82.90 / 90.60) has no source.
- Prithvi-CAFE feeds more than the 6 HLS bands (its CNN branch takes the remaining channels).

**Caveats.**
- TerraMind Table 8 states neither the split nor the S2 processing level.
- The U-Net references [16] and [19] were not read; their setups are unverified.
- Our training recipes differ: UNet decoder and dice loss, versus UPerNet and cross-entropy in
  Prithvi-EO-2.0.

**Thesis use.** Chapter IV (reproduction of SOTA, phase A); cite tables, not abstracts.

## 2026-10-16 — EDA notebook (course progress, week 4)

**What and why.** The course syllabus (MIA-403 v1.1, weeks 3–4) asks for a decision-oriented
EDA notebook covering leakage, class imbalance and drift. The project's EDA existed as separate
analyses: the data card, `scripts/band_stats.py`, `scripts/label_qa.py` and the domain-shift
write-up. `notebooks/eda.ipynb` (Spanish, executed; built by `notebooks/build_eda_nb.py`) gathers
them into one notebook.

**Method.**
- Labels are read only for `peru-train`, `peru-dev` and Sen1Floods11 `train`/`valid`. The test
  splits appear only as metadata: location, date, counts.
- **Inventory:** from the manifest and split list.
- **Spatial map:** chip centres in UTM 17S by split. Check: no location is in two splits (0
  found).
- **Class balance:** from the manifest fractions (Peru) and LabelHand (Sen1Floods11).
- **Band statistics, SMD and index percentiles:** from the existing `outputs/band_stats/`
  (entry 2026-10-14).
- **Label QA flags:** from `outputs/label_qa/chips.csv`, train and dev only.
- **New:** Pearson correlation between the 13 bands on `peru-train`. Sample: 60 chips drawn
  with seed 0, every 50th valid pixel (311,919 pixels).

**Results.**
- **Water share of labelled pixels:**

  | Set | Water | Chips without water |
  |---|---|---|
  | `peru-train` | 4.6 % | 64.5 % |
  | `peru-dev` | 12.4 % | 36.2 % |
  | Sen1Floods11 `train` | 9.5 % | 13.1 % |
  | Sen1Floods11 `valid` | 11.0 % | 13.5 % |

- **Doubtful-label chips:** 16 of 753 (train) and 8 of 174 (dev).
- **Band correlation:** within groups ≥ 0.97 (visible B02–B05; red-edge/NIR B06–B8A; SWIR B11–B12).
  Between groups 0.70–0.91. Cirrus B10 is uncorrelated (|r| ≤ 0.09, except 0.50 with B09).
- The other figures repeat earlier entries.

**Thesis use.** Chapter III (data description). Course deliverable: branch
`avance-sem4-eda`, also merged into main.
