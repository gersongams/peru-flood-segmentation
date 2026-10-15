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
