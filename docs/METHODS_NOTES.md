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
  0.626, Prithvi 0.820 → 0.741. Both foundation models lose less than U-Net (drops −0.257 U-Net, −0.192 TerraMind, −0.080
  Prithvi). Prithvi loses by far the least. TerraMind's advantage is within its large seed spread on
  `peru-test` (±0.055). (Corrected 2026-10-16: first written as "holds for Prithvi only", which
  misread TerraMind's drop. See the correction entry below.)

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

## 2026-10-16 — Thesis writing rules and template restructure (authors' decision)

**What and why.** Writing starts now, so the rules were fixed before any chapter is drafted.
The Overleaf template turned out to be an **undergraduate** FIIS template: its readme says
"plantilla de tesis de pregrado", it has 6 chapters, and its cover reads "INGENIERO DE
SISTEMAS". It matches neither EPG-UNI guide.

**Sources.**
- `thesis-guidelines/`: Guía N°1 (plan), Guía N°2 (thesis) and the two MIA-403 syllabi.
- The guides are scanned PDFs. They were read page by page, and the full extraction with
  quotes is in `docs/guide_rules_extracted.md`.
- Key pages were re-checked by eye:
  - G2 p. 4: "en forma estricta".
  - G2 pp. 9–10: the structure of a Maestría de Investigación thesis.
  - G2 p. 20: the cover.
  - G2 p. 27: the consistency matrix.
  - G1 pp. 43–44: the format standards.
- Guía N°2 was approved on 28-03-2023 by the EPG Consejo Directivo N°05, with the FIIS
  graduate-unit director among the signatories.

**Decisions (authors, 2026-10-16).**
1. Follow Guía N°2's 4-chapter structure and restructure the template.
2. Use 3 specific objectives and 3 hypotheses. This fits G2 p. 19 (exactly 3 conclusions and 3
   recommendations). The hypotheses come from the research questions in `docs/PLAN.md`, written
   before the results.
3. Write a new title of at most 20 words. Whether changing the approved plan's title needs a
   formal request is still to be checked.
4. Claude drafts and the authors review and edit. The S1 LLM-detector clause is flagged, and the
   argumentative sections should be rewritten by the authors.

**Rules.** The binding rules are in `docs/WRITING_RULES.md`. Defaults set where the guides are
silent:
- impersonal voice;
- a Spanish terminology table;
- metric proportions with 3 decimals and a decimal point;
- APA 7 captions;
- a `% fuente:` comment after every number.

**Template (thesis repo).**
- Structure:
  - front matter in G2 order;
  - Introducción;
  - Cap. I Protocolo, II Marco teórico, III Desarrollo, IV Conclusiones y recomendaciones;
  - Referencias;
  - Anexos: consistency matrix (G2 p. 27), operationalization matrix (required by the course),
    reproducibility (required by the course).
- Format:
  - A4; margins 3/3/2.5/2.5 cm; Arial 11, falling back to Liberation Sans or TeX Gyre Heros
    with a log warning; 1.5 spacing.
  - Page number only, top right. Lowercase Roman numerals in the front matter, with the cover
    counted but not printed. Arabic numerals from the Introducción, not printed on the
    Introducción or on chapter-opening pages.
  - The cover follows G2 p. 20, with Poppins (OFL) bundled in `fonts/`.
  - References double-spaced with a hanging indent.
- Removed sections: Permisos, Prólogo, Símbolos, and the separate Conclusiones and
  Recomendaciones chapters.
- Compiled locally with XeLaTeX: 17 pages, no errors. The pages were inspected as rendered
  images.


## 2026-10-16 — Correction: "foundation models lose less on Peru" is not Prithvi-only

The Section-7 results entry and `docs/phaseD_test.md` first said that only Prithvi "loses less"
from Sen1Floods11 `test` to `peru-test`. The numbers in the same entry say otherwise.

| Model (source-only) | Sen1Floods11 `test` | `peru-test` | Drop |
|---|---|---|---|
| U-Net | 0.8331 | 0.5763 | −0.257 |
| TerraMind | 0.8185 | 0.6261 | −0.192 |
| Prithvi | 0.8203 | 0.7405 | −0.080 |

(`outputs/phaseD/test/{test,peru-test}_groups.csv`.)

- Both foundation models drop less than U-Net. TerraMind's margin over U-Net (0.065) is about one
  seed std of TerraMind on `peru-test` (0.055).
- No CI was computed for the difference of drops. The two test sets have different chips, so a
  paired bootstrap does not apply; an independent bootstrap could be added.
- Found while mapping the PLAN hypotheses to the results for Chapter I. The statement was
  reported to the authors in the same session and is now corrected in both places.

## 2026-10-16 — Literature and Peru context for Chapters I–II (agent search)

**What.** A background agent gathered sourced material for §1.1 (Peru flood impacts),
§2.1 (antecedentes) and §2.2 (bases teóricas). Outputs:
- `docs/literature_thesis.md`, sorted oldest to newest, each item with its method and a result
  with table/page;
- `docs/literature_thesis.bib`, 127 entries, each with a `% verified:` DOI/URL.

Entries move into `thesis/3_3_BIBLIOGRAFIA/library.bib` one at a time, only when cited
(`docs/WRITING_RULES.md` §8).

**Results.**
- **Antecedentes:** 31 in the main list. Including the supplementary list there are 24
  Peruvian/Latin-American studies (17 Peru, 5 of them theses).
- **Closest Peruvian precedent:** Shen et al. (2022), a Sen1Floods11-trained UNet++ applied to
  Piura 2017, water IoU 0.69 (their Table 3). Any claim of a "first deep-learning flood map of
  Peru" is ruled out.
- **2017 impacts:** INDECI 2017 Compendio, pp. 13–16: 285 453 damnificados, 1 454 051 afectados,
  138 deaths. The 2023 edition revises these to 283 137 / 1 644 879 / 169. Cite one edition.
- **2017 losses:** about US$ 3.1 bn (French et al., 2020, Table 1). The BCRP S/ 25 655 M figure is
  the reconstruction budget, not losses.
- **Yaku 2023:** 44 deaths and 107 780 damnificados (INDECI Compendio 2024, Tabla 6, p. 86).

**Open checks.**
- UNOSAT's product page gives a **flood** extent of 985.9 km² (22 March) and 992.8 km² (1
  April). `docs/peru_data_card.md` says "UNOSAT reports about 1,023 km²" for the water extent
  (flood plus existing water, the layer we rasterized). These are probably different quantities.
  To be reconciled before either number goes into §1.1.
- **Unverified:** Portalés-Julià 2025 Table 3 (download blocked), the Quintana-Ortiz 2021 full
  text, journal indexing, the World Bank/CEPAL loss figures, the ENFEN 2017 magnitude.
- **Follow-ups for `docs/published_comparison.md` / `docs/REFERENCES.md`:**
  - the "U-Net Base 84.03" row is Li et al. 2023, Tables 2–3;
  - the Badillo-Rivera AUC question is resolved by its Table 5.

**Privacy note.** During the search, one sub-agent sent the user's email address to the
Unpaywall API (an open-access lookup that requires an email parameter). The address is not
stored in any output file. It was reported to the user.

## 2026-10-16 — Thesis reframed around a system (AI application); new hypothesis HE3 fixed before measurement

**Decision (authors, 2026-10-16).**
- The general objective becomes *developing* an AI flood-analysis system for Peru, built on the
  segmentation models. This is close to the approved plan title, "Sistema de análisis satelital
  de inundaciones basado en modelos de segmentación profunda…", whose new ending is "para el
  mapeo de áreas inundadas en el Perú".
- Specific objectives:
  - **OE1:** compare and select the system's model.
  - **OE2:** adaptation without Peru labels, plus calibration.
  - **OE3:** implement the system as a web application and evaluate it **technically only**
    (authors' choice; no user study).
- Text: `docs/chapter1_backbone.md` v2. Written into thesis Chapter I and Annexes 1–2.

**Integrity.**
- **HE1 and HE2 keep the pre-registered wording.** HE1 comes from `docs/PLAN.md` (2026-10-11).
  HE2 merges the former HE2 and HE3, from PLAN and the phase-D proposal of 2026-10-14. Their
  outcomes are already known, so they are not reworded.
- **HE3 is new and is fixed today, before the system exists:**
  - The system meets 100 % of the functional requirements RF1–RF7.
  - It produces the water and uncertainty map of a full Sentinel-2 scene (about 110 × 110 km)
    in **≤ 15 min including download**, on a desktop machine (Apple M4 Max).
  - **RF1:** select an event or date and an area.
  - **RF2:** get the S2 L1C image from CDSE, or load an already processed one.
  - **RF3:** water-probability map from the selected model (3-seed ensemble).
  - **RF4:** uncertainty plus clouds and no-data.
  - **RF5:** reference overlay (UNOSAT/EMS) with agreement metrics.
  - **RF6:** flooded area per date on a timeline.
  - **RF7:** export as GeoTIFF and GeoJSON.
- The 15 min threshold is an operational requirement of the authors, not a literature value.
- The requirements and the threshold must not change after the measurement. The commit of this
  entry is the evidence of when they were fixed.

**Consequences.**
- The flood viewer, paused locally in `.viewer-wip/`, becomes a thesis component. Its earlier
  rule ("produces no thesis results") is replaced:
  - the RF checklist and the time measurements are thesis results, logged like any run;
  - model-quality numbers still come only from the frozen test pass.
- RF2 and the time hypothesis need on-demand processing (a backend that fetches a scene and runs
  the model). This is new work.
- Chapter III §3.1 gains the system's development and its evaluation.

**Thesis use.** Chapter I (objectives, hypotheses, requirements table); Chapter III (system);
Chapter IV (third conclusion).

## 2026-10-17 — Open items from the Chapter I audit: UNOSAT areas reconciled, RF6 defined, HE3 protocol fixed

**1. UNOSAT 985.9 vs ~1,023 km² (reconciled).** `scripts/unosat_reconcile.py` →
`outputs/unosat_reconcile.txt`. Areas measured on the UNOSAT 2023-03-22 geometry in UTM 17S:

| Layer | Area | Notes |
|---|---|---|
| FloodExtent ("New Water / Water Increase") | 986.1 km² | UNOSAT table: 985.9; layer attribute: 986.8 |
| WaterExtent (flood + pre-existing water) | 1,023.4 km² | the data card's "about 1,023" |
| Pre-existing water (WaterExtent − FloodExtent) | 37.3 km² | |
| Project rasterized labels (668 chips) | 1,016.6 km² | the difference comes from grid coverage and ignore pixels |

- Both UNOSAT figures are correct; they are different quantities. The data card now states the
  layers explicitly.

**2. RF6 "área inundada" (defined before implementation).**
- **Definition:** water (p ≥ 0.5) at date *t*, minus water on a pre-event reference image of the
  same tile; valid pixels only.
- **Reference image:** the clearest S2 acquisition in the 60 days before the event start.
- This mirrors UNOSAT's own FloodExtent logic.
- **Alternative rejected:** a static ESA WorldCover 2021 class-80 mask. It covers 173.0 km² of the
  analysed extent, of which **105.9 km² fall inside UNOSAT's 2023 flood** and only 22.6 km² inside
  its pre-existing water. Subtracting it would remove about 11 % of the mapped flood.
- Wording: "área inundada" is used only for this quantity, otherwise "agua detectada"
  (WRITING_RULES §11.12).

**3. HE3 evaluation protocol (fixed before the system exists).** `docs/system_evaluation_protocol.md`:
- **Escena:** one S2 L1C MGRS tile and date.
- **Descarga:** the OData product download from CDSE, including authentication. The Process API is
  excluded for its processing-unit cost.
- **Conditions:**
  - warm system with models loaded; empty image cache;
  - timer from the backend receiving the request to the GeoTIFFs written and the map available;
  - the 6 tiles covering the UNOSAT extent on 2023-03-22 (17MNP, 17MNQ, 17MPN, 17MNN, 17MPP,
    17MPQ);
  - 3 repetitions each;
  - network capacity logged with `networkQuality`;
  - external failures retried at most 3 times and reported.
- **Decision rule:** the median is ≤ 15 min for each of the 6 tiles. A cold start is reported
  separately.
- **Requirements:** each RF1–RF7 has a pass/fail test with evidence. RF3 must agree with the
  frozen pass on the overlapping `peru-test` chips: mean |Δp| ≤ 0.02 and mask IoU ≥ 0.95.
- **Rule:** nothing in the protocol may change after the first measurement.

**Thesis use.** Chapter III (system evaluation method; data description of the UNOSAT layers);
WRITING_RULES §11 open items closed.

## 2026-10-17 — Chapter I wording corrections after external review: HE1, "sin etiquetas", RF6

**Context.** An external AI-assisted review of the Chapter I PDF raised two methodological points.
The rule applied is to fix the statement itself, not only explain it afterwards (WRITING_RULES
§11.8–11.9). Authors' approval: 2026-10-17.

**HE1, rewording made after the results are known (recorded as such).**
- **Before:** "En Sen1Floods11 no existe una diferencia apreciable … (el IC 95 % de la diferencia
  incluye el cero)".
- **Now:** "En Sen1Floods11, los modelos fundacionales no superan a la U-Net en IoU de agua: no
  se detecta una diferencia (el IC 95 % de la diferencia incluye el cero)".
- The pre-registered prediction, "No meaningful gap in-distribution" (PLAN, 2026-10-11), is
  unchanged in direction.
- The CI criterion itself was first written on 2026-10-16, when the test results were already
  known. The new wording only removes the implied equivalence claim, which a CI containing 0
  cannot support. It does not make the hypothesis easier to pass.
- **No equivalence margin was introduced:** choosing one now, after seeing the CIs, would be
  fitting to the results.

- **Second correction (same day, after a second review):**
  - "no superan" still asserted more than a CI containing 0 can show.
  - HE1 now reads: "En Sen1Floods11 no se espera detectar una diferencia estadísticamente
    significativa en IoU de agua entre los modelos fundacionales y la U-Net, según los
    intervalos de confianza al 95 % de la diferencia. En el Perú, los modelos fundacionales
    presentan una menor caída de IoU de agua que la U-Net respecto de su desempeño en
    Sen1Floods11."
  - This is a testable prediction about the test outcome; the direction is the same as PLAN.
  - The reviewer's draft dropped "que la U-Net"; it was kept, because the comparison is the
    hypothesis.
- **Also corrected:** §1.4.1 "es de tipo aplicada" → "es una investigación aplicada".
- **Authors' rule from now on:** no further style-only edits to Chapter I. Remaining effort goes
  to making Chapters II–IV support exactly what Chapter I claims.

**"Sin etiquetas peruanas" made precise in every statement.** The phrase becomes "sin etiquetas
peruanas de entrenamiento" in:
- the general problem;
- PE2;
- the practical justification;
- OE2, which now adds "la selección de configuraciones usa la partición de validación peruana";
- the general hypothesis and HE2;
- stage 4;
- the variables table;
- Annexes 1–2.

No claim changes; the statements now say from the start what the method does.

**RF6.** Equation 1.1 is followed by "representa la nueva superficie de agua detectada en la
imagen, no una inundación verificada en campo".

**Thesis.** Chapter I; Annexes 1–2.

## 2026-10-17 — Chapter II drafted and audited against primary sources

**What.**
- Chapter II (Marco teórico) was written following the `thesis-writing` skill:
  - §2.1 antecedentes: 17 international (2021–2026) and 6 national or Latin-American studies,
    chronological, each with its method and a result with table/page; a comparison matrix
    (Tabla 2.1) and the research gap;
  - §2.2 bases teóricas, organised by the thesis variables;
  - §2.3 sixteen term definitions.
- 49 references were added to the thesis bibliography, only as cited.
- Three independent agents then audited every claim against the primary sources (procedure:
  `thesis-claim-audit`). Record: `docs/chapter2_citation_audit.md`.

**Selection.**
- Excluded: studies without a verifiable result (Quintana-Ortiz 2021, paywalled; Portalés-Julià
  2025 Table 3, not re-read) and studies known only from an abstract.
- Gonzalez et al. (2020) is the only pre-2021 antecedent, with a one-sentence justification.

**Substantive corrections from the audit.**
- **Gap paragraph.** Under leave-one-region-out, the foundation model did not beat the U-Net in
  Bolivia, Colombia or Paraguay (Kaushik et al. 2026, Table VI). Their Table III is a random
  split, and its "mIoU" is the flood-class IoU.
- **Prithvi-EO-2.0.** The variant used here (300M, non-TL) has no time/location embeddings.
- **Pech-May (2024)** reports no Tabasco evaluation.
- **Several locators and attributions corrected:** Englert Table 1; ECE formula attributed to Guo
  eq. 3; the Handbook's 100 km² vs SentiWiki's 110 km tiles.
- **Water threshold made consistent with the frozen rule (z > 0, p > 0.5)** in Chapter I eq. 1.1,
  in Chapter II and in `docs/system_evaluation_protocol.md` (previously "≥ 0.5"). This is a
  consistency correction made before any system measurement.

**Not verifiable (paywalled), used only for definitions:** McFeeters 1996, Otsu 1979, Jaccard
1912, Gonzalez & Woods 2018, Efron 1979, Efron & Tibshirani 1993. Xu 2006 was checked from its
abstract only. To open via the UNI library before submission.

**Thesis.** Chapter II; Chapter I eq. 1.1 wording.

## 2026-10-17 — Chapter II: external review applied (thesis commit 302922d, not pushed)

**What and why.** An external review (ChatGPT, of `new_thesis_capitulo2.pdf`) raised 8 technical
and 3 layout points. Each was checked against the code, results and primary sources before
deciding.

**Accepted and applied (Chapter II):**
- *HLS vs L1C (§2.2.4).* New paragraph: HLS S30 is surface reflectance, nadir-normalized
  (BRDF), resampled to 30 m from 10/20/60 m, with most bands adjusted to Landsat (Ju et al. 2026,
  HLS User Guide v2.0, §1 p. 6, Table 2 p. 10). This research feeds Prithvi six L1C TOA bands
  (B2, B3, B4, B8A, B11, B12; `configs/prithvi_eo_v2_300.yaml`) at 10 m, the Sen1Floods11 format
  (13 bands sampled to 10 m, Bonafilia 2020 p. 4). Stated as a factor that may affect Prithvi's
  transfer; no effect size claimed. TerraMind was pretrained on S2 L1C and L2A (Jakubik 2025,
  arXiv v5 §8, p. 13 of PDF) and we use its S2L1C modality. Sen1Floods11's processing level is
  not stated in Bonafilia, so the text does not claim it.
- *AdaBN applicability (§2.2.5).* AdaBN only re-estimates BN statistics (Li 2018). ViT encoders
  use LayerNorm before every block (Dosovitskiy 2021 §3.1), which computes the same at train and
  test time (Ba 2016, abstract p. 1), so in the foundation models AdaBN can only act on decoder BN
  layers. This matches what was run (phaseD_dev: "decoder BN only"); the results stay in Ch. III.
- *ECE variant (§2.2.7).* Stated exactly what was computed: top-label confidence max(p, 1−p),
  15 equal-width bins on [0.5, 1], pixels (`scripts/uncertainty_dev.py::ece`,
  `docs/calibration_dev.md`), and that reliability diagrams were also produced (Guo 2017 §2).
  Not class-wise calibration of p; no new metric added after the fact.
- *System evaluation (new §2.2.9).* Short basis for OE3/HE3: pass/fail requirement tests fixed in
  advance; end-to-end time from request to map; its components (download, band reading and
  resampling, normalization, inference ~1.21e8 px per 110 km tile at 10 m, writing); repetitions
  summarized by the median. Consistent with `docs/system_evaluation_protocol.md`. **Pending:** a
  normative citation (ISO/IEC 25010, functional suitability / time behaviour) could not be
  verified (iso.org 403, standard paywalled); left as `% pendiente`, not paraphrased from
  secondary sites.
- *UDA definition (§2.3).* "métodos que buscan reducir" (the methods are what is being tested).
- *Mateo-Garcia (§2.1.1).* Removed "El trabajo establece la comparación con índices espectrales
  como línea base necesaria": an interpretation, not a finding of the paper.
- *Layout.* Tabla 2.1 moved after the gap paragraph, `\FloatBarrier` (placeins) before §2.2, so it
  no longer splits a sentence; table note now states each value keeps its source's scale (IoU/mIoU
  % or proportion, ECE proportion) instead of converting the numbers.

**Rejected:**
- *Threshold "≥ 0.5" in both chapters.* The reviewer read an older Chapter I. Implementation is
  strict (z > 0 ⇔ p > 0.5; `ece()`, `score_frozen.py`); both chapters already say "mayor que 0.5"
  (thesis 815c9ba).
- *Add an NDWI/MNDWI baseline experiment.* Not added: the Sen1Floods11-fitted MNDWI > 0.15 was
  defined after the test pass (entry above) and test is evaluated once. The label-free MNDWI rows
  already in phase D dev (MNDWI > 0, Otsu, oracle) are reported in Ch. III.

**Escalated to the authors (not applied):** Chapter I HE2 lists "nubes" as an error stratum, but
cloudy pixels are excluded; the stratum actually analysed is *proximity to thick cloud* (within
20 px, `docs/failure_analysis_dev.md` Finding 4). Proposed wording "bordes de nubes" /
"proximidad a nubes". HE2 is pre-registered, so a wording change after results needs the
authors' decision and its own log entry.

**Checks.** XeLaTeX build: 0 errors, 0 undefined references, `main.blg` 0 warnings; pages 20–30
rendered and inspected. Primary sources downloaded and grepped for every new claim.

**Feeds.** Thesis Ch. II (§2.1.1, §2.1.3, §2.2.4, §2.2.5, §2.2.7, §2.2.9, §2.3); Ch. III
methodology (decoder-only AdaBN, Prithvi input path); paper Methods.

## 2026-10-17 — HE2 wording: "nubes" → "bordes de nubes" (authors approved)

**What and why.** Follow-up to the Chapter II review entry above. HE2 listed "nubes" as an error
stratum, but pixels under thick cloud are excluded from every evaluation, so they cannot carry
errors. The stratum that was actually analysed is proximity to thick cloud (within 20 px;
`docs/failure_analysis_dev.md`, Finding 4). The authors approved "bordes de nubes" on 2026-10-17.

**Scope.** Wording only: it names the stratum that the pre-registered analysis already measured.
The hypothesis, the strata and the analysis are unchanged. It was corrected after the dev results
were known, and it is recorded here for that reason (as with the HE1 corrections). Changed in
thesis Chapter I (HE2) and in `docs/chapter1_backbone.md`; the consistency matrix (annex) uses the
short form "pocos estratos" and needed no change.

**Feeds.** Thesis Ch. I §1.5 (hypotheses); Ch. III error analysis.

## 2026-10-17 — Chapter II, second external review

**Accepted and applied:**
- *Timer start (§2.2.9).* "desde que el servidor recibe la tesela y la fecha" read as if the image
  arrived with the request. Now: the request names tile and date but carries no image; the download
  from Copernicus Data Space is inside the time when the image is not cached. Matches
  `docs/system_evaluation_protocol.md` §2 (timer start = request received; cache empty).
- *Decision statistic for the 15-min limit.* Chapter II now says the statistic is fixed before
  measuring and points to Chapter I; Chapter I (after the specific hypotheses) now states the rule
  already fixed in the protocol on 2026-10-17, before the system exists: 6 tiles × 3 repetitions,
  warm system, image not cached, met if the median of the 3 runs ≤ 15 min for each of the 6 tiles;
  all times and the maximum reported. Nothing has been measured, so this is not a post-hoc change.
- *ECE bins terminology.* "intervalos de confianza" → "intervalos … según su confianza predicha"
  (§2.2.7) and "intervalos de confianza predicha del modelo; no son intervalos de confianza
  estadísticos" (§2.3), to keep "intervalo de confianza" for the bootstrap CI.
- *Prithvi bands phrase.* "seleccionadas entre las 13 bandas que Sen1Floods11 distribuye
  muestreadas a 10 m".

**Checked, no text change:**
- *Decoders contain BatchNorm.* `scripts/adabn.py`: U-Net 46 BN layers; TerraMind 9 BN layers in its
  UNet decoder and neck; Prithvi 9 BN layers in its decoder (entry 2026-10-16, Mac re-run). So the
  §2.2.5 sentence holds for all three models.
- *ECE bins in code.* `ece()`: 15 equal-width bins on [0.5, 1] of max(p, 1−p), as written.

**Rejected:**
- *Threshold "≥ 0.5" in Chapter I eq. (1.1).* Not present: a search of all `.tex` files finds no
  "mayor o igual" or ≥; Chapter I and II both say "mayor que 0.5" (since thesis 815c9ba). The
  reviewer compared against an older Chapter I PDF.
- *ISO/IEC 25023:2016 citation.* Paywalled, not verified in the primary source; the `% pendiente`
  note stays. Not cited from secondary descriptions.

**Checks.** XeLaTeX 0 errors, 0 undefined references; Chapters I (pp. 11–20) and II (pp. 21–31)
rendered and inspected. **Feeds.** Thesis Ch. I §1.3.2, Ch. II §2.2.7, §2.2.9, §2.3; Ch. III system
evaluation.

## 2026-10-17: Chapter II third review

**What was done.** A third external review of Chapter II, against the second-review PDF. One change was accepted; the other points were already resolved or rejected.

- **Accepted: "memoria local" → "almacenamiento local (caché)".** "Memoria" can be read as RAM, but the protocol means the on-disk product cache. Ch. I §1.3.2 now reads "sin la imagen en su almacenamiento local (caché vaciada antes de cada repetición)", matching `docs/system_evaluation_protocol.md` §2 (cache cleared before each run). Ch. II §2.2.9 reads "almacenamiento local (caché)". Timed runs always include the download; the cached path is only tested functionally (RF2 b) and is not timed.
- **Already resolved: decision statistic for 15 min.** The reviewer said Ch. I §1.3.2 does not name the statistic. It does: since the second review, Ch. I says the requirement is met if the median of three repetitions is ≤ 15 min in each of the six tiles, and all times and the maximum are reported. The `\ref` resolves to §1.3.2. The reviewer seems to have checked an older Ch. I. The rule was fixed on 2026-10-17 in `docs/system_evaluation_protocol.md`, before the system exists and before any measurement, so "se fija antes de medir" is accurate.
- **Rejected, for the third time: "≥ 0.5 in Ch. I".** No `.tex` file contains "mayor o igual" or ≥ for the water rule. The rule is p > 0.5 (z > 0), as in `configs/frozen.yaml`.
- **Not done: optional page-break tweak between §2.2.9 and §2.3.** The float/page layout is acceptable, and forcing breaks would make the layout fragile.
- **Confirmed (no change): AdaBN layers.** The counts were checked earlier in code: U-Net 46 BN layers; TerraMind 9 (decoder and neck); Prithvi 9 (decoder).

**Checks.** XeLaTeX: 0 errors, 0 undefined references. **Feeds.** Ch. I §1.3.2, Ch. II §2.2.9.

## 2026-10-17: Whole-thesis review (compiled PDF, 47 pp.)

**What was done.** An external review of the full compiled thesis. It confirms all Chapter I–II fixes from the previous reviews:
- the 15-minute median rule;
- "bordes de nubes";
- p > 0.5 in both chapters;
- "almacenamiento local (caché)".

It asks for no new text changes in Chapters I–II and recommends stopping editorial polishing. No edits were made.

**Open items it raises, all already planned:**
- Chapters III–IV (results and hypothesis tests) are still empty.
- Annex 3 (reproducibility) is still empty. Its source material exists:
  - `configs/frozen.yaml` (sha256 of the checkpoints), which backs the Annex 2 phrase "congelada con suma de verificación";
  - `docs/EXPERIMENT_LOG.md` (commits, seeds, run dirs);
  - `docs/system_evaluation_protocol.md`.
- Report HE3 as "mediana por tesela ≤ 15 min" and publish all 18 times and the maximum. The protocol already requires this.
- Keep "agua detectada" separate from "área inundada" (WRITING_RULES §11.12). The title says "mapeo de áreas inundadas". It is left to the authors, since "área inundada" is the defined RF6 quantity.
- Quantify the possible optimism from evaluating 2017 only on EMS/MNDWI-concordant pixels. Ch. I already states the caveat (Capitulo1.tex:330–332); the number goes in Ch. III.
- Annex 1 matrix font is about 8 pt. Check it against the UNI format rules before submission.
- Final source audit of cited numbers. Chapter II has had three independent passes; ISO 25010 and the paywalled items are still pending.

**Feeds.** Ch. III, Ch. IV, Annex 3.

## 2026-10-17: Chapter III §3.1 drafted; number audit and corrections to the result docs

**What was done.**
- §3.1.1–3.1.8 of Chapter III were drafted: data, models, adaptation protocol, and the Sen1Floods11, dev, test and calibration results. Thesis commits `218d84e` and `6cd161d`, plus the corrections below.
- The claim ledger and the proposed hypothesis verdicts are in `docs/chapter3_backbone.md`.
- §3.2 and §3.3 are not written yet. The verdicts go to the authors first.
- §3.1.9 (system) is a commented placeholder until the system is measured (WRITING_RULES §11.11).

**New derived numbers, post hoc.**
- `scripts/error_shares_dev.py` → `outputs/phaseD/analysis/error_shares.csv`: share of pixels and of errors per stratum on `peru-dev`, from the existing `strata.csv`. No model was run.
- Errors per stratum are computed as water_px·(1 − recall) + dry_px·FPR, using seed-mean rates.
- They feed Tabla 3.4 and the HE2 strata verdict.

**Audit.**
- An independent pass recomputed about 400 numbers and claims in the draft from the primary files: result CSVs, manifests, run configs and `temperatures.json`, not the summary docs.
- It found 21 items: 11 rounding errors, 1 false claim and 9 wording issues. All were fixed in the thesis.
- Report: session scratchpad `audit_c3.md`. The main cause was double rounding: a 4-decimal value in the docs rounded again to 3 decimals.
- Examples:
  - Peru-trained U-Net dev 0.94147 → 0.941, not 0.942;
  - U-Net trHM `peru-test` 0.79848 → 0.798, not 0.799;
  - TerraMind trFDA dev 0.83646 → 0.836;
  - Prithvi Sen1Floods11 `test` CI low 0.7505 → 0.750.

**Errors corrected in the result docs**, so later writing does not inherit them:
- **`docs/phaseD_test.md`:**
  - The headline now gives 0.798 for U-Net trHM.
  - The Peru-trained U-Net CI low is 0.732 (was 0.733).
  - Prithvi source precision is 0.798 (was 0.799).
  - "Every row drops 0.10–0.13 from dev to test" was false. Only the primary and Peru-trained rows drop 0.097–0.130; U-Net AdaBN drops 0.169, U-Net source 0.260 and TerraMind source 0.077.
  - "Without doubtful chips … no ordering changes" was false. TerraMind trFDA and test-time HM (0.003 apart) swap in both variants. The rises are 0.005–0.016 (6 chips excluded) and 0.002–0.007 (3 chips).
- **`docs/published_comparison.md`:** Prithvi `test` CI 75.0–87.3 (was 75.1).
- **`docs/calibration_dev.md`, not edited, noted here:** "All raw models lie below the diagonal at every confidence level" does not hold for TerraMind's two lowest-confidence bins (0.517 → accuracy 0.569; 0.550 → 0.566; `reliability_bins.csv`).
- **The Sen1Floods11 in-domain temperatures quoted in the chapter are now the final source models'** (`temperatures.json`, s1f11_src): U-Net 2.02–2.12, TerraMind 1.13–1.36, Prithvi 2.80–3.23. The earlier 1.73 / 1.01 / 2.80 belong to the pre-sweep seed-0 models.
- **Chapter I and Chapter II do not contain the affected values** (checked by search).

**Feeds.** Chapter III §3.1; §3.3 verdicts (`docs/chapter3_backbone.md`).

## 2026-10-17: HE2(a) operationalized (authors' decision, post hoc)

**What was decided.** HE2(a) ("los errores de los modelos en el Perú se concentran en pocos estratos: bordes de nubes, vegetación inundada, terreno desértico o costero") had no pre-specified model group or threshold. The authors chose, after seeing the dev error shares:
1. **Model group:** the source-only models (Sen1Floods11-trained, unadapted). HE2 compares adaptation with "el modelo sin adaptar", so the errors it refers to are those models' errors. Peru-trained and adapted rows are complementary.
2. **Threshold:** a stratum is concentrated when error share / pixel share ≥ 2.

**Alternatives considered.** Peru-trained models (what is hard in Peru even with Peru labels); one verdict per group (longer, less clear); > 1× (too permissive: any over-representation counts); the named strata together holding > 50 % of errors (rejected: bare/sparse is 85 % of the pixels, so it holds almost always and says little).

**Mapping of named strata** to the pre-specified strata families: cloud edges = distance to thick cloud < 20 px (< 5 and 5–20 px bands pooled); flooded vegetation = WorldCover herbaceous wetland (proxy); desert or coast = WorldCover bare/sparse.

**Result** (`outputs/phaseD/analysis/error_shares.csv`, source-only, ratio U-Net / TerraMind / Prithvi):
- cloud edges < 20 px: 0.78 / 1.80 / 0.77 (5–20 px band alone 0.77 / 1.76 / 0.77; < 5 px band alone TerraMind 4.41 but 0.11 % of pixels, 0.47 % of errors);
- herbaceous wetland: 3.27 / 1.64 / 4.32, holding 0.6 / 0.3 / 0.8 % of errors;
- bare/sparse: 1.02 / 1.07 / 0.99;
- not named: label edge ≤ 1 px 13.48 / 6.43 / 20.06; permanent water 2.62 / 1.32 / 3.28.

**Verdict.** Not supported for the named strata: errors concentrate in few strata, but in the label edge and permanent water; only flooded vegetation reaches 2× (two models) and it holds < 1 % of errors. Peru-trained models show the same pattern. Ledger: `docs/chapter3_backbone.md`.

**Caveats.** Dev only, descriptive, no CI; criterion fixed after seeing results, to be reported as post hoc in §3.3.

**Feeds.** Chapter III §3.3 (HE2a), §3.2.

## 2026-10-17: Chapter III §3.3 (Contrastación) drafted; HE2(a) verdict corrected

**What was done.** §3.3 was written in `thesis/2_CAPITULO3/Capitulo3.tex` with a summary table (`tab:contrastacion`). Each specific hypothesis is split into the claims it contains (1a, 1b, 2a, 2b, 2c), each contrasted separately. Verdict terms are defined once: *se confirma*, *no se confirma*, *se refuta* (CI excludes 0 in the opposite direction), *no pudo contrastarse*. HE3 and the general hypothesis: one sentence saying they are pending the system measurement, plus a commented placeholder (WRITING_RULES §11.11).

**Correction to the entry "HE2(a) operationalized" above.**
- The verdict recorded there ("not supported for the named strata") did not apply the approved ≥ 2× rule literally. Herbaceous wetland is 3.27× (U-Net) and 4.32× (Prithvi), so one named stratum meets the rule for two models. Corrected verdict: **partly supported**. The "< 1 % of errors" point is interpretation, not a criterion.
- The note that the < 5 px cloud band "does not change the verdict" was wrong under ≥ 2× (TerraMind 4.41). The cloud-edge stratum is the pooled < 20 px band, fixed before this analysis (`docs/failure_analysis_dev.md` Finding 4; "bordes de nubes" entry, 2026-10-17). Pooled ratios 0.78 / 1.80 / 0.77.
- `scripts/error_shares_dev.py` now adds the pooled "<20 px" row to `error_shares.csv`. Tabla 3.4 and the §3.1.5 text now use it: 8.4 % of pixels; Peru-trained 9.6 / 8.0 / 7.8 % of errors, source 6.5 / 15.1 / 6.5 % (was the 5–20 px band, 8.3 %).

**Other number checks.**
- HE1b TerraMind vs U-Net drop difference is 0.064 at full precision (0.2568 − 0.1924), not 0.065; ledger fixed.
- TerraMind trFDA − Peru-trained TerraMind on `peru-test` is 0.106 (0.8150 − 0.7085); no paired CI computed, and none was added after the test pass.
- HE2c on `peru-test`: temperature alone and ensemble alone each lower ECE versus single in all six frozen rows, all pixels and near edges. The combination is not always lowest (U-Net trHM 0.60 vs 0.48 % with temperature alone). Frozen-row temperatures (per seed and ensemble) 1.82–2.22 / 1.24–1.52 / 2.62–3.23 (`temperatures.json`).

**Verdicts in the thesis.** 1a coherent (no difference detected; CIs reach 0.062 / 0.051, no equivalence margin); 1b partly (Prithvi yes, TerraMind not established); 2a partly (post hoc criterion, dev); 2b confirmed on 2023 for U-Net and TerraMind, refuted on 2017 (label caveat), not testable for Prithvi; 2c confirmed for the frozen rows, with exceptions outside them (Peru-trained Prithvi AUROC 0.439; dev low-contrast and < 1 ha strata); HE2 overall partly; HE3 pending.

**Feeds.** Chapter III §3.3; §3.2 next.

## 2026-10-17: Chapter III §3.2 (Discusión) drafted

**What was done.** §3.2 written in G2 order: synthesis; comparison with other studies (agreeing and contradicting); findings and the authors' interpretation; limitations; implications. Only references already cited in Chapter II, with the same locators and numbers. No new test analyses.

**Comparisons and how they are bounded.**
- Agreeing: Sen1Floods11 parity (Li 2023, PANGAEA, Jakubik 2023); Prithvi OOD advantage (Li 2023, Bolivia, Prithvi-100M named as such); overconfidence and temperature/ensemble ECE reduction (Guo 2017, Lakshminarayanan 2017).
- Contradicting: Kaushik 2026 leave-region-out (Clay not above U-Net) alongside TerraMind not separable from U-Net in Peru; self-training worked in LoveDA, Englert 2024 and Chen 2026 but not here (scenario differences stated; evaluation-time AdaBN pass from the post hoc diagnostic; rest of the cause "no se aisló").
- Qualitative only: Kolluru 2026 (same error direction, high recall / low precision; numbers not comparable: per-event, HLS 30 m, GFM excludes permanent water); Shastry 2023 (not contradicted; under-cloud pixels are excluded here); Banerjee 2026 ECE stated as not comparable; Ovadia 2019 applied only to the source→Peru temperature transfer, with the caveat that the degradation versus the source domain was not measured.

**Checks.** Peru-trained rows on `peru-test` (0.828 / 0.815 / 0.824) have no paired differences in `bootstrap_diffs.csv`, so the text gives point estimates and says they were not compared. The implication "train with Peru labels" is scoped to the 2023 test (all three Peru-trained means above every label-free row; on 2017 the source rows are higher). The label-free selection caveat (selection on labelled `peru-dev`, §11.9) is stated in limitations and implications.

**Also.** The §3.3 verdict terms used `\emph`; replaced with «» (WRITING_RULES §9: no italics except foreign words). Table 3.6 group headers changed from italic to bold for the same reason.

**Feeds.** Chapter III §3.2. The authors rewrite §3.2 in their own words (§9).

## 2026-10-17: Claim audit of Chapter III §3.2–3.3 and fixes

**What was done.** An independent audit recomputed every number in §3.2 and §3.3 from the primary files and checked every citation against Chapter II and the literature notes. Numbers: no double rounding, no sign errors, no CI from the wrong row. It found 9 claim-scope errors and 7 wording issues; all were fixed (thesis commit after `8ed04d2`).

**Errors fixed.**
- Prithvi "no method improved it": only AdaBN and histogram matching (test and training time) were run on Prithvi; now "los métodos probados".
- Fine-tuned vs Peru-only "indistinguishable in every model" holds on the 2023 test only; on 2017 (101 chips) U-Net −0.015 [−0.033, −0.001] and Prithvi +0.002 [0.0003, 0.0036] exclude 0. Now scoped to 2023.
- HE2a "largest concentrations … and permanent water": false by ratio (herbaceous wetland and the 2–3 px edge band are higher; TerraMind permanent water 1.32 < 2). Rewritten: edge ≤ 1 px is the largest concentration (13.48 / 6.43 / 20.06; 38.8 / 18.5 / 57.8 % of errors); permanent water described by error share (6.5 / 3.3 / 8.1 %) with its ratios.
- Dry-soil brightness 1.38–1.68 SD is peru-dev only, visible bands and B12; scoped.
- High-confidence errors (AUROC 0.41–0.56) are the Peru-trained U-Net and Prithvi only; scoped.
- Self-training vs literature: without the evaluation-time AdaBN pass, self-training beat the source model in both backbones (0.877 vs 0.836; 0.758 vs 0.703), so the "rest of the drop" sentence had no referent. Rewritten: the contradiction is mostly the AdaBN pass; the remaining gain was smaller than alignment.
- Shastry: their 74 % is area underestimated against a hydraulic model and their "vegetation" is NLCD classes, not herbaceous wetland. Rewritten as "not testable here" (under-cloud pixels excluded; image-derived labels do not record hidden water).
- Kolluru: the same model under-detected against DSWx-HLS (Chapter II), so the error direction depends on the reference; rewritten.
- HE2c exceptions: Peru-trained Prithvi also fails on 2017 (AUROC 0.390, ensemble 0.419; `calibration.md`, no_2023_train_overlap); added with the label caveat.

**Wording fixed.** 2017 label caveat added in the §3.2 synthesis, the HE1b 2017 sentence, the HE2 summary and the `tab:contrastacion` note; "una U-Net de otro estudio" (Kaushik); ensembles tied to Lakshminarayanan's uncertainty claim, not ECE; "otra agrupación" (Banerjee) removed as unsupported; "no siempre se traslada"; AUROC range labelled as individual models; units (%) on published values; PANGAEA locator matches Chapter II; HE1b now states the conservative bound (−0.022 and 0.102 → 0.080), labelled as an approximation and not a CI.

**Feeds.** Chapter III §3.2–3.3.

## 2026-10-17: Thesis figures (Chapter III)

**What was done.** The authors asked for figures. Seven were made, in a shared style (`scripts/figures/style.py`: Arial like the text, Paul Tol colour-blind-safe palette, A4 text width 15.5 cm, vector PDF in `thesis/E_IMAGENES/`, PNG previews in git-ignored `outputs/figures_preview/`). Each figure has its own script in `scripts/figures/`; every plotted number is read from result files, none typed by hand.

- **Figura 3.1, flow diagram** (`E_IMAGENES/fig_metodologia.tex`, TikZ): data → training → adaptation and selection → single evaluation. Counts from Chapter I §1.4 and Tabla 3.1. Needed `\usetikzlibrary{babel}` (Spanish babel shorthands break TikZ arrows).
- **`fig_brecha`**: peru-test IoU, origen → adapted (U-Net trHM, TerraMind trFDA) → Peru-trained, with per-row bootstrap CIs (`bootstrap_groups.csv`) and `gap_closed` (88 %, 44 %). Caption warns that per-row CIs do not compare rows.
- **`fig_comparacion_modelos`**: error maps (TP/FP/FN/ignored) for 8 columns × 4 test chips; 3-seed ensemble (mean probability > 0.5), raw logits. **Selection rule** (fixed before looking at tiles): among the 37 test chips with ≥ 5 % water, the chip nearest the 0.125/0.375/0.625/0.875 quantiles of Δ = IoU(U-Net trHM) − IoU(U-Net source), each location once; doubtful chips kept as in the `full` table. A spatial-block constraint was tried and rejected (it pushed the 0.875 pick to a chip with negative Δ). Per-tile IoU is per chip, not comparable with pooled IoU; a self-check reproduces `peru-test_chip_counts.csv` seed-0 counts.
- **`fig_incertidumbre`**: U-Net trHM, chips at the 0.1/0.5/0.9 quantiles of its per-chip IoU; uncertainty = mutual information between seeds, H(p̄) − (1/3)Σ H(pₛ), bits, colour scale capped at 0.5 (4.2 % of row b above). Predictive entropy was rejected because it repeats the probability panel. In row a (IoU 0.040) the Peru-trained U-Net also fails (0.001), so the label there may be at fault; stated in the note.
- **`fig_validacion_prueba`**: per configuration, IoU on validation (`peru-dev_check_groups.csv`, which holds the recomputed AdaBN U-Net 0.8993), 2023 test (full) and 2017 test (101 chips); `s1f11_ft` rows omitted.
- **`fig_fiabilidad`**: a) reliability diagrams of the Peru-trained models on dev (`reliability_bins.csv`, single, seeds pooled; from the 2026-10-14 calibration pass, same checkpoints, dev IoU 0.9416 vs 0.9415); b) test ECE single / temperature / ensemble+temperature for the Tabla 3.8 rows. Temperature = Sen1Floods11-fitted (`temp_s1f`), except Peru-trained U-Net (`temp_dev`, marked *).
- **`fig_prueba_ic`**: paired differences from `bootstrap_diffs.csv`, a − b as in the file; `p_a_not_better` not shown. The file has no TerraMind trFDA − Peru-trained row; the note says so.

**Grey-scale check (WRITING_RULES §8).** The first error palette made TP/FN and FP/ignored indistinguishable in grey. Changed to TP #1A4B8C, FP #F2A33A, FN #D6457A, ignored #E4E4E4 (grey levels 0.27 / 0.68 / 0.46 / 0.89).

**Feeds.** Chapter III §3.1.1, §3.1.7, §3.1.8, §3.3; defence slides.
- **Addendum, Figura 3.2 `fig_area_estudio`** (same day): study-area map. Background: true-colour S2 L1C (B04/B03/B02, TOA) mosaic of the project chips at 80 m, one stretch for all bands; per location the less cloudy date (share of pixels with B02 > 0.30 or empty): 381 locations 2023-04-01, 287 2023-03-22, 25 only-2017 locations 2017-04-07. Hillshade from Copernicus DEM GLO-30 (`data/aux/cop_dem`), Natural Earth coastline, ocean, departments and places (downloaded to git-ignored `data/aux/naturalearth/`). Legend counts recortes / ubicaciones from `split.csv`: 753/392, 174/87, 172/90, buffer 192/99, 2017 204/204 (179 of the 2017 locations are also 2023 locations). Caveat: mixed dates and TOA imagery leave visible seams and haze.

## 2026-10-17: Term "buffer" for the unused separation chips (authors' decision)

**What was decided.** The 192 unused chips that separate the training blocks from the validation and test blocks were called "de separación" in Chapter I and "amortiguamiento" in Chapter III. The authors chose the field's technical term, *buffer* (as in buffered spatial block cross-validation), in italics (`\ingles{buffer}`), defined at first use in Chapter I §1.4 ("recortes sin uso que separan los bloques de entrenamiento de los de evaluación") and in §2.3 Definición de términos. **Alternatives:** "zona de amortiguamiento" (standard translation of *buffer zone* but ecological connotation); "de separación" (clear Spanish, recommended by WRITING_RULES §6's prefer-Spanish default). The authors accepted the §6 condition: English term in italics with a Spanish gloss at first use. Applied in Chapter I, Chapter II §2.3, Chapter III (text, Tabla 3.1) and the Figura 3.2 legend; WRITING_RULES §6 table updated. `style.py` now sets mathtext to Arial so italic words in figure labels match the text font.

## 2026-10-17: Chapter III external review — new post hoc analyses and corrections

**Source.** A review of the Chapter III PDF (24 pp.) by the authors' reviewer. It audited the text, not the code. Every point was checked against the files before acting.

**New analyses (post hoc; no model run, no selection changed; read the existing per-chip counts):**
1. **HE1b direct CI** (`scripts/bootstrap_drop_diff.py` → `outputs/phaseD/test/bootstrap_drop_diff.csv`). D = (IoU_F,Peru − IoU_F,S1F) − (IoU_U,Peru − IoU_U,S1F), source-only models. Each replicate resamples the 90 Sen1Floods11 `test` chips and, independently, the Peru chips; paired across models within each domain; seed-mean pooled IoU; 2,000 replicates, seed 0. Chip sampling only (no seed variation, no spatial dependence). 2023: Prithvi D 0.177 [0.102, 0.253]; TerraMind 0.064 [0.001, 0.120]. 2017 (101 chips): 0.033 [−0.011, 0.086]; 0.031 [−0.021, 0.089]. Per-seed drops (pooled IoU per seed): U-Net 0.235 / 0.231 / 0.305; TerraMind 0.149 / 0.272 / 0.156; Prithvi 0.101 / 0.066 / 0.073. **Verdict:** Prithvi confirmed; TerraMind not conclusive (CI lower bound 0.001, seed 1 drop inside the U-Net range); HE1b partly confirmed, as before but now on the right estimand. Replaces the two-step argument and the "conservative bound" approximation.
2. **2017 vs 2023 spatial overlap** (`scripts/overlap_2017_2023.py` → `outputs/phaseD/analysis/overlap_2017_2023.csv`; same cell rule as `build_peru_emsr2017.py`, 5,120 m cell and 8 neighbours). Of the 101 main 2017 chips: same cell as a 2023 dev location 57, as a test location 29; same or adjacent dev 77, test 50, buffer 35, train 0. The 2017 test is cross-event over largely the same area as the selection split, not cross-region.
3. **2017 sensitivity without dev-near chips** (`scripts/sensitivity_2017_dev.py` → `outputs/phaseD/test/sensitivity_2017_dev.csv`). Adapted − source: U-Net trHM −0.117 [−0.164, −0.080] and TerraMind trFDA −0.068 [−0.114, −0.023] on 44 chips (no same cell as dev); −0.102 [−0.127, −0.072] and −0.089 [−0.109, −0.071] on 24 chips (no same or adjacent). The 2017 reversal holds. (On these subsets source U-Net − source TerraMind becomes positive, +0.045 [0.016, 0.087] on 44 chips; not used for any verdict.) Note: selection on dev at the same locations would, if anything, favour the adapted rows, and they still lose.

**Clarifications from existing files:**
- AUROC score: single model = −|z| (ranks like predictive entropy and 1 − confidence); ensemble = −|z̄| with z̄ the logit of the mean probability (predictive entropy of the ensemble). Mutual information (Figura 3.7) had dev AUROC 0.750–0.785 for the Peru-trained ensembles (`outputs/analysis_dev/uncertainty.csv`). Now stated in §3.1.8. The score the app will show is a system design decision (pending).
- AdaBN non-reproduction: the investigation already in METHODS_NOTES 2026-10-16 "Phase-D freeze" (same checkpoint, pool, algorithm; deterministic today; min running variance 8.7e-13 as plausible, unproven mechanism) is now summarised in §3.1.3.

**Text corrections:** "cota inferior" → "puede subestimar o sobreestimar"; partition water prevalence (4.6 / 12.4 / 10.4 %) and the full-scene extrapolation limit in §3.1.1 and limitations; 2017 wording "worse against that reference; not extrapolated to an independent, complete reference" (§3.2 finding 3, §3.3 2b, Tabla 3.9); 2a reported as "respaldo exploratorio parcial", separating the anticipated strata from those found in the analysis; "la elección del modelo pesa poco" → descriptive similarity, no paired comparison; §3.3 intro lists three exceptions to the pre-specified evidence.

**Not changed:** tables that split a sentence across a page break (pp. 26–27, 30–31, 44–45) — normal float placement that will move as text changes; fix at final layout. The system suggestions (3.1.9–3.1.11, 3.2.6, model choice for the app declared as informed by the test results) go to `docs/NEXT_STEPS.md`.

**Feeds.** Chapter III §3.1.1, 3.1.3, 3.1.5, 3.1.7, 3.1.8, 3.2, 3.3.

## 2026-10-18: Chapter III second review (comparative) — fixes

**Source.** The reviewer compared the new Chapter III PDF with the previous one: most points resolved; four remaining.

**Applied.**
- **HE1b wording:** the drop-difference contrast is now described as not in the initial plan and not pre-registered confirmatory evidence; Prithvi "respaldada", TerraMind "sensible a la variación entre semillas y a la dependencia espacial … no concluyente"; Tabla 3.9 verdict "Se confirma parcialmente (Prithvi; intervalo post hoc)".
- **Self-training causal wording (§3.2.2):** "se debe sobre todo a esa pasada" → "en el protocolo de esta investigación, el diagnóstico sugiere que la pasada de AdaBN contribuyó de forma importante…; no se extrapola a los otros trabajos".
- **Layout:** Tabla 3.9 split a sentence. A `\FloatBarrier` after the §3.3 intro gave two half-empty float pages, so Figura 3.8 now sits `[H]` after the 1b paragraph and Tabla 3.9 `[H]` at the end of §3.3.2 (before §3.3.3), both at paragraph breaks.

**Kept (authors' decisions or already in place).** *Buffer* stays (authors chose it 2026-10-17; the reviewer calls it optional). 2017 nuance ("worse against that reference"), 2a as exploratory and the AdaBN cause as an unproven hypothesis are already in the text. The uncertainty map the app shows is a system decision (NEXT_STEPS). Annex 3 should keep the AdaBN diagnostic artefacts (`outputs/phaseD/test/peru-dev_check_compare.csv`, METHODS_NOTES 2026-10-16 "Phase-D freeze").

## 2026-10-18: System UI design and three system decisions (authors)

**What was done.** Before writing app code, the authors reviewed clickable prototypes of the
system (6 desktop screens, Spanish UI; design canvas, not in the repo). Audience chosen: an
analyst/operator tool (map-first, dense), not a guided demo or a public viewer; visual direction:
the warm palette already in `app/src/styles.css`. Screens and the requirements they cover:
Visor (RF3, RF4), Nuevo análisis + job progress with timer against 15 min (RF1, RF2, HE3),
Referencia with swipe comparison and IoU / precision / recall for the view, the UNOSAT extent or
the `peru-test` chips (RF5), Línea de tiempo with "área inundada" vs "agua detectada" and the
pre-event image (RF6), Exportar GeoTIFF / GeoJSON + `metadatos.json` with model sha256 and source
product (RF7), Evidencia with the frozen test IoU and the known limits.

**Decisions (authors, 2026-10-18).**
1. **Default model: Peru-trained U-Net** (`peru/unet`, 3-seed ensemble, frozen checkpoints in
   `configs/frozen.yaml`). It is the best `peru-test` row (0.828 ± 0.002; `docs/phaseD_test.md`).
   U-Net trHM (0.798 ± 0.005, best label-free row) is offered as an optional comparison layer.
   - Alternatives: U-Net trHM as default (suggested in NEXT_STEPS; fits the label-free story but
     is 0.030 lower on 2023); Prithvi source-only (best on 2017, 0.937, but 0.741 on 2023, where
     most of the system's target extent lies).
   - **Integrity:** the choice was made with the test results known. The thesis and the app's
     Evidencia screen must say so and must not present those results as an independent validation
     of the choice (NEXT_STEPS, Chapter III review 2026-10-17).
2. **Uncertainty shown: seed standard deviation** of the probability, as already fixed in
   `docs/system_evaluation_protocol.md` §1 (2026-10-17). This closes the open item "predictive
   entropy or mutual information". Alternatives: predictive entropy of the ensemble (the score
   behind the reported AUROC; better ranking of errors but mixes in aleatoric uncertainty);
   mutual information (dev AUROC 0.750–0.785 for the Peru-trained ensembles,
   `outputs/analysis_dev/uncertainty.csv`). Kept seed std because the protocol was fixed with it
   before implementation, it is directly interpretable (σ of p across seeds), and changing it now
   would alter a pre-registered RF4 test. Caveat for the text: the AUROC figures in §3.1.8 are for
   entropy / MI, not for σ; the app must not quote them as the quality of its uncertainty layer.
3. **Events limited to Yaku 2023 and Niño Costero 2017.** An "other event" option (any start
   date, no reference map) was dropped: it is outside the RF1 test and has no reference for RF5.

**UI wording rules carried into the app.** "Área inundada" only for the RF6 quantity, otherwise
"agua detectada" (WRITING_RULES §11.12); every on-screen fact carries its source; map metrics are
labelled as not the thesis numbers (those come from the frozen pass); limits stated on screen
(research prototype, preliminary UNOSAT labels, faint-water misses, low uncertainty on
low-contrast water, worse calibration near edges).

**Feeds.** Chapter III §3.1.9 (development: modules, model choice, uncertainty layer), §3.2.6.

## 2026-10-18: Full-scene pipeline of the system; inference windows aligned to the chip grid (authors' decision)

**What was built.** `src/floodseg/system/` (backend of the web app) and `app/` (frontend):
- `cdse.py`: catalog lookup (CDSE OData) and streaming download of the whole L1C product. The
  Sentinel Hub client-credentials token is accepted by the OData download service (checked
  2026-10-18), so the existing credentials are used; the Process API is not used (protocol §1).
- `scene.py`: reads the 13 JP2 bands from the zip (`/vsizip/`, 13 threads), upsamples 20/60 m to
  10 m by pixel replication (= Process API default NEAREST, since the grids share the tile origin)
  and applies `RADIO_ADD_OFFSET` per band from `MTD_MSIL1C.xml` (DN + offset, clipped at 0, DN 0
  kept as no data), which reproduces the Process API `harmonizeValues` used for the training chips.
  Cloud / no-data mask: the project thick-cloud rule (B02 > 4500 and B11 < 1.15·B02, dilated 5 px);
  no data where all bands are 0.
- `infer.py`: frozen `peru/unet` ensemble (sha256 checked at load), probability = seed mean of
  sigmoid(z1 − z0), uncertainty = seed std.
- `pipeline.py`: timed stages, COG GeoTIFF outputs (probability and std as uint16 × 10 000, nodata
  65535; mask uint8; RGB preview), `meta.json` with product, offsets, checkpoints, commit, times.
- `server.py` (FastAPI): models loaded at start-up (warm system), jobs with per-stage progress,
  request/stop timestamps in `outputs/system/server.log` and `jobs.jsonl`, cache (RF2b), dynamic
  map tiles from the COGs (rio-tiler), pixel readout.

**Check: input path.** For 8 `peru-test` chips of 17MNP the tile stack equals the training chip
bit for bit in all 13 bands. The 2023 product is processing baseline N0510 (offset −1000).

**Decision: chip-aligned inference windows.** The first version used overlapping 512 px windows
(64 px overlap, tent blending). RF3 (protocol §3) gave MAD 0.0075 (pass) but IoU 0.934 (< 0.95,
fail) on the 49 `peru-test` chips of 17MNP. Since the inputs are identical, the difference comes
only from the window context: in the outer 48 px of each chip IoU was 0.887, inside 0.958. On the
73 `peru-dev` chips of the same tile both layouts score the same against the labels (0.9532
overlapping vs 0.9523 chip windows), so neither is better. The authors chose non-overlapping
512 px windows aligned to the chips' 5 120 m UTM grid, i.e. exactly the evaluated configuration:
on every chip the map equals the frozen pass. RF3 then gives MAD 1.8e-5 and IoU 0.99994.
- **Alternatives rejected:** (a) keep overlap and amend RF3 before the formal measurement — a
  protocol change made after a failing development run; (b) keep overlap and report RF3 as failed.
- **Integrity note:** the dev labels were used only to check that the two layouts are equally
  accurate; no test label was read. The RF3 test and thresholds are unchanged. This was a
  development run on an uncommitted tree, not the protocol measurement.
- **Caveat:** non-overlapping windows can leave faint seams every 5.12 km where the model's
  predictions depend on context; to be inspected on the maps.

**Timing (development, 17MNP 2023-03-22, M4 Max).** Download 303.9 s of 371.6 s total (771 MB at
about 2.5 MB/s); without download 56.7 s (read 12.3, mask 2.4, inference 31.9, write 7.8). The
15 min threshold of HE3 therefore depends mainly on the CDSE download rate; the protocol records
`networkQuality` before each run. Source: `docs/EXPERIMENT_LOG.md` 2026-10-18.

**Also decided while building:** the earlier viewer note ("the viewer produces no thesis result",
2026-10-16 draft, never committed) is superseded by the reframing of 2026-10-16; `app/` and
`scripts/viewer/` are now tracked in git so the system can be tagged (protocol §2). The chip-based
PMTiles path of the paused viewer is not used: map tiles are cut on the fly from the full-tile COGs.

**Feeds.** Chapter III §3.1.9 (development), §3.1.10 (full-scene processing), §3.1.11 (RF3).

## 2026-10-18: System RF5–RF7 implemented (reference metrics, flooded area, export); choices made while building

**RF5, reference and metrics** (`src/floodseg/system/reference.py`). UNOSAT is rasterized on the
scene grid with the Peru-label rule (`scripts/build_peru_unosat.py`): water = the date's
FloodExtent + pre-existing water of 2023-03-22; ignore = outside the analysed extent or inside
UNOSAT's cloud-obstruction layer. Scored pixels: valid in the system mask and not ignored. Water:
p > 0.5. Scopes: map view, whole tile, or one Peru-split partition (pixels of the `peru-test`
chips of that date).
- **Why the partition scope:** the tile contains training chips. On 17MNP 2023-03-22, 51 % of the
  scored pixels are `peru-train` (dev 18 %, test 12 %, buffer 18 %); IoU is 0.925 on the whole
  tile, 0.957 on train pixels and 0.701 on test pixels. The app therefore shows the partition
  share and a warning, and offers the test-only scope. These on-screen numbers are system
  outputs, not thesis results; they select nothing (the thesis numbers stay the frozen pass).
- **Not yet done:** the RF5 check "on peru-test chips the metrics match `scripts/score_frozen.py`
  to 3 decimals" (needs the same pooling as score_frozen); the 2017 EMS reference.

**RF6, flooded area** (`src/floodseg/system/preevent.py`). Definition as in the protocol §4.
Pre-event image: candidates = all L1C acquisitions of the tile in the 60 days before the event
start; each ranked on a 200 m Process API preview (B02, B11, dataMask, harmonized DN) by its
thick-cloud share among valid pixels. Two operational details not spelled out in the protocol,
implemented provisionally and **pending the authors' confirmation** (no measurement yet): (1) candidates with < 50 % of the tile valid are skipped —
they are the adjacent orbit, which covers only about 25 % of 17MNP, and would count as "clear"
with almost no data; (2) ties go to the date closest to the event. 17MNP: 23 candidates
(2023-01-03 to 2023-03-03); chosen 2023-01-26 (thick cloud 0.04 %, valid 94 %), narrowly ahead
of 2023-03-02 (0.05 %), so the reference image is 37 days before the event rather than 2; record
`outputs/system/scenes/preevent_17MNP_2023-03-04.json`. Preview cost: 23 requests at 549 × 549 px,
3 bands (a few processing units in total). The pre-event scene is processed like any scene and
cached; it is outside the HE3 timer.
- Results on 17MNP: flooded area 786.2 km² (2023-03-22) and 846.7 km² (2023-04-01); valid in both
  dates 84 % and 87 % of the tile. Source: `/api/flood/<scene>` (`preevent.flood`).
- **Caveat found:** "agua detectada" (total water) includes the sea inside the tile (e.g. 1 390.7 km²
  on 2023-03-22). The flooded area is not affected (the sea is wet on both dates). The UI labels
  it "incluye mar". Whether to mask the sea for total water is open (no protocol change made).

**RF7, export** (`src/floodseg/system/export.py`). ZIP with probability and uncertainty (uint16,
×1e-4, nodata 65535), flood (uint8: 1 flooded, 0 not, 255 cloud / no data), mask, flood polygons
(GeoJSON, EPSG:4326, 8-connected `rasterio.features.shapes`, no simplification) and
`metadatos.json` (product, pre-event date, checkpoints with sha256, commit, attribution).
Whole tile or map view. Check on 17MNP 2023-03-22: rasters open in rasterio with EPSG:32717 and
the exact tile bounds (499 980, 9 290 200, 609 780, 9 400 000); GeoJSON opens in geopandas, 3 718
polygons, area 786.2 km² = the RF6 flooded area. Build time 22 s, 48 MB.

**Feeds.** Chapter III §3.1.9–3.1.11.


## 2026-10-18: System checks after review — RF5 reference consistency, seams, archive reuse, tile generation

**RF5 reference consistency.** On the 49 `peru-test` chips fully inside 17MNP (2023-03-22), the
system map scores IoU 0.6171 against the chip label files and 0.6171 against the tile-rasterized
UNOSAT reference (with the system mask): the tile rasterization reproduces the labels. The frozen
per-seed IoU on the same chips is 0.6152 / 0.6162 / 0.6123 (mean 0.6146). The test-partition
scope of the app gives 0.7007 because it also includes the in-tile parts of border test chips
(0.53 M px, IoU 0.998, mostly lagoon). The gap to the thesis 0.828 (172 chips, two dates,
per-seed pooled) is subset difficulty, not an error. The RF5 3-decimal check against
`scripts/score_frozen.py` still has to be implemented with score_frozen's pooling.

**Seams of the chip-aligned windows (measured).** On 17MNP 2023-03-22, the mean absolute change
of probability between adjacent pixels is 0.018 across window edges vs 0.0035 elsewhere (columns;
rows 0.018 vs 0.0033); uncertainty 0.013 vs 0.0023. The water mask (p > 0.5) changes across a
window edge at 1.5 % of edge pixels vs 0.35 % elsewhere. Small but measurable; the cost of the
2026-10-18 window decision. To report in the thesis with the decision.

**Archive reuse.** `pipeline.run` reused a product archive left on disk (development
convenience). It is now opt-in (`--reuse-archive`, CLI only); the app path always downloads unless
the scene outputs are cached, and `meta.json` records `archive_reused`. The timing harness must
assert `archive_reused == false`.

**"Tile generation" inside the HE3 timer.** The protocol's timed path includes "tile generation
for the map". In this system it is the build of the COG overviews inside the `write` stage; map
tiles are then rendered on request from the COGs (rio-tiler), as for any cached scene. The timer
stops when the COGs are closed, i.e. when the map can be served.

**Open for the authors:** RF6 pre-event rules (50 % valid, tie → latest); whether "agua
detectada" should exclude the sea; the 2017 event start date for RF6 (the server's 2017-03-15 is
the timeline range, not a sourced event start; the Niño Costero rains began earlier, so a 60-day
window ending on 2017-03-14 could pick a flooded image).

## 2026-10-18: 3D terrain spike (visualization only, no thesis result)

At the authors' request, a Three.js view of Bajo Piura / Sechura (`app/terrain.html`, data from
`scripts/system/terrain_export.py`). Zone: UTM 17S 515 000–565 000 E, 9 335 000–9 385 000 N
(50 × 50 km, inside 17MNP, around the Sechura depression). Relief: Copernicus DEM GLO-30
(`data/aux/cop_dem`), bilinear to a 512 × 512 grid (~98 m), sea clamped to 0 m; elevation range
0–101 m, shown with adjustable vertical exaggeration (default ×30). Ground texture: Sentinel-2
true colour of the clear pre-event scene (2023-01-26). Water per date: system water map
(p > 0.5, valid pixels) of 2023-01-26, 2023-03-22 and 2023-04-01; 38.8 / 400.7 / 401.3 km² inside
the zone. The panel's "aumento" is the difference of totals, not the RF6 flooded area (which is
pixel-wise and valid in both dates); it is labelled so. GLO-30 is a surface model (DSM), so
dunes and vegetation are part of the relief. Not part of RF1–RF7 or HE3.
Addendum 2026-10-20: textures 4096 px (~12 m); the water texture holds the probability (0 on
cloud / no-data) and the page draws its 0.5 contour, so the drawn water is the same p > 0.5 map
with sub-pixel edges (km² unchanged). Hover picking now marches the ray over the height grid
instead of raycasting the mesh (frames over 50 ms in a scripted orbit/zoom/hover test: 179 → 0).

## 2026-10-18: RF5 chip check passes; HE3 timing harness written; trHM comparison layer

**RF5 (chips).** `floodseg.system.reference.frozen_chip_metrics` counts tp / fp / fn with its own
code (water if z > 0, labelled pixels, pooled over chips per seed, mean over seeds, as
`scripts/score_frozen.py`). On all 172 `peru-test` chips with the frozen `peru/unet` logits:
IoU 0.828048, precision 0.966298, recall 0.852683, equal to the thesis evaluator to 6 decimals
(`outputs/system_eval/rf/rf5_chips.json`). The app shows these numbers in Referencia ("Todos los
recortes de prueba"). The overlay part of RF5 (reference on the map with metrics) was shown
working earlier the same day; RF5 is complete for 2023. 2017 (EMS) is still missing.

**HE3 timing harness** (`scripts/system_timing.py`), written to the protocol §2: 6 tiles × 3
repetitions interleaved (≥ 10 min between runs of the same tile), cache cleared before each run,
`networkQuality -s -c` before each run (downlink in Mbps), job sent to the warm server as the app
does, timer = server timestamps (request received → GeoTIFFs written), external failures retried
at most 3 times and logged (`timing_failures.csv`), any other failure stops the run as a system
failure, `archive_reused` must be false, refuses an untagged or dirty tree. Output
`outputs/system_eval/timing_runs.csv` with the protocol columns plus stage times and product.
Not run yet. Observation for planning: on this machine `networkQuality` reports ~560 Mbps while
CDSE downloads ran at 2–8 MB/s (download 86–361 s per tile), so the CDSE side sets the time.

**trHM comparison layer** (authors' decision 2026-10-18: default model + optional comparison).
Opt-in per request ("Opciones avanzadas"). The second ensemble (`trhm/unet`, sha256 checked,
loaded on first use) runs on the same in-memory stack **after** the HE3 timer stops (the stop
stamp is written when the selected model's GeoTIFFs are closed) and writes
`probability_compare.tif`; `meta.json` records it under `comparison`. Shown in purple as "IA sin
etiquetas peruanas", off by default. 17MNP 2023-03-22: comparison water 1 437.7 km² vs 1 390.7 km²
for the main model (valid pixels, sea included). Display only; no metric or selection uses it.

## 2026-10-20: System RF5 for 2017 (EMS scene reference) and RF1 check

**Why.** RF5 asks for the expert reference on the map with agreement metrics; for 2017 the
reference is Copernicus EMS EMSR199. The app had it only for UNOSAT 2023.

**Scene reference (`floodseg.system.reference`, `_ems`).** Same rule as the 2017 test labels
(`scripts/build_peru_emsr2017.py`), because EMS was drawn on radar of 23–30 Mar and the image is
of 7 Apr: water = inside EMS water (crisis_information + hydrography polygons) and MNDWI > 0;
dry = outside EMS water and MNDWI ≤ 0; ignore = disagreement or outside the AOIs (Salinas,
Parachique, Piura). MNDWI = (B03 − B11) / (B03 + B11 + 1e-6) on the scene's own 13-band stack
(offset-harmonized DN, B11 nearest-upsampled to 10 m, as the chips). The system's cloud /
no-data mask is applied at scoring time, as for 2023. One difference from the chips: the chip
builder used only the first AOI's polygons and extent where AOIs overlap; the scene uses the
union of the three AOIs. Building the reference needs the product archive (B03, B11), so the two
2017 scenes were run with `--keep-archive`; the reference is cached as `reference.tif` and the
archives were deleted. Partitions: every 2017 chip is test; chips that touch a 2023 `peru-train`
location get their own code (`test_near_train`), so the app's "Recortes de prueba" scope on 2017
equals the main 2017 subset (`no_2023_train_overlap`).

**Checks.** (1) Scene reference vs chip labels (`scripts/system_ref2017_check.py`): 159 chips on
17MNP and 41 on 17MNQ; agreement 1.0 on all pixels scored by both (32.3 M and 8.4 M). Pixels scored
only by the scene: 1 266 025 (17MNP) and 125 772 (17MNQ), from the AOI-union difference above;
scored only by the chips: 133 and 162 (cloud-mask edge). This also confirms the L1C stack of the
reprocessed 2017 product (baseline N0500, offset −1000) matches the Process API chips.
(2) Chip metrics (`scripts/system_rf5_check.py`): the system's counting code on the 101
no-overlap chips with the frozen logits gives IoU 0.878004, precision 0.999948, recall 0.878044,
identical to `score_frozen.py` (`outputs/system_eval/rf/rf5_chips.json`). (3) Scene-level on the
17MNP no-overlap chips: IoU 0.881 (ensemble mean probability, union AOIs; not a thesis number).
(4) A straight-edged water body south of Sechura looked like sea on the map; Natural Earth 10 m
ocean puts 0.0 % of the 2017 chip reference water (and 0.0008 km² of the scene's) in the sea. It
is the temporary 2017 Sechura lagoon (La Niña), cut by the AOI boundary.

**App.** Referencia is event-aware: EMS labels, the MNDWI-agreement caveat ("solo se evalúan los
píxeles donde EMS y la imagen coinciden; el acuerdo tiende a ser alto"), the 2017 chip numbers in
"Todos los recortes de prueba", and a note that no 2017 pixel was used for training (40 % of the
pixels in the default view are in chips near 2023 training locations).

**RF1.** In the app: event Yaku 2023, tile 17MNP, Shift-drag rectangle, date 2023-03-22, process.
`server.log`: `job 258b4948ad1e received tile=17MNP date=2023-03-22 bounds=[-80.7779, -6.1611,
-80.2658, -5.6517] cached=True`. The rectangle in UTM 17S (524 573, 9 318 928, 581 309, 9 375 290)
is inside the 17MNP footprint (`outputs/system_eval/rf/rf1.json`). Pass.

**Feeds:** thesis §3.1.9 (system development and RF checks), Tabla of RF results.

