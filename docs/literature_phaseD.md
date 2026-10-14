# Literature check for phase D: flood mapping in Peru without Peru labels (task 6.6, part 1)

Date: 2026-10-14. Phase D (decision record in `docs/failure_analysis_dev.md`): adapt
Sen1Floods11-trained models to Peru using only unlabelled Peru Sentinel-2 imagery; UNOSAT
labels only for selection (`peru-dev`) and evaluation.

The old literature base (23 papers, `~/Projects/masters-degree-thesis/docs/papers/INDEX.md`)
has nothing on domain adaptation, so this is a new web search, done by two research agents
and spot-checked here. Full notes with the URL of each source and a verification status per
paper:
- `docs/literature/uda_segmentation_methods.md`: general UDA / test-time adaptation for
  semantic segmentation (16 papers, arXiv PDFs read).
- `docs/literature/uda_earth_observation.md`: adaptation and generalization for flood/water
  and remote-sensing segmentation, geospatial foundation models, Peru.

**Verification status.**
- Checked here: existence and metadata of the two key papers via Crossref/OpenAlex,
  CIM-TTDA (doi:10.1029/2025JH001145) and Portalés-Julià et al. 2025 (doi:10.1016/j.rse.2025.114882),
  and the VFM-UDA arXiv page (2406.09896).
- Read by the agents in full text, not re-read here: the numbers quoted from those papers.
  Re-check each number in the PDF before it goes into the thesis (CLAUDE.md: cite tables, not
  abstracts). None of these papers is in Zotero yet; add them before citing.

## 1. What exists, and the gap

| Work | What | Relevance |
|---|---|---|
| Chen, Wang, Zhao & Li 2026, *CIM-TTDA*, JGR: Machine Learning and Computation | Test-time self-training (EMA teacher, class-balanced loss, CoTTA-style restore) of **SAR** flood models on unlabelled post-event scenes; baselines CBST, AdaptSegNet, TENT, CSFDA | Closest prior work. **SAR only**, small CNN/SegFormer, humid coastal events; a Prithvi-EO-2.0 baseline was not adapted |
| Portalés-Julià, Mateo-García & Gómez-Chova 2025, RSE 328:114882 | Cross-dataset evaluation of S1/S2 flood models (no adaptation) | MNDWI thresholding comes close to deep learning on unseen datasets; models over-predict under shift. Protocol to copy (per-event metrics, PR curves) |
| Patel et al. 2021 (arXiv 2111.10079) | FixMatch semi-supervised learning on Sen1Floods11 S1, unseen test regions | Consistency baseline |
| Wang et al. 2021, *LoveDA*, NeurIPS D&B | Standard remote-sensing UDA benchmark | Self-training recovers most of the water-class gap (per the agent; table not re-read) |
| Englert et al. 2024, *VFM-UDA*, CVPR 2024 Workshops (arXiv 2406.09896) | UDA with vision foundation models | Self-training + masking works with large pretrained ViTs; DAFormer extras (FD, HRDA) do not help ViTs |
| Akiva et al. 2021 *H2O-Net* (WACV); Zhao et al. 2025 *UUCP* (RSE) | Adaptation / label-free water segmentation | Rules out "first DA for water" |
| Prithvi-CAFE (arXiv 2601.02315) | Bolivia hold-out results | No adaptation; its Bolivia rows compare different papers |
| Peru: Gonzalez et al. 2019 (ML4D); Badillo-Rivera et al. 2025; Alvan Romero & Suarez Alayza 2024 | Water bodies with PeruSAT-1; S1 index/change detection for 2017 and Yaku | **No deep-learning flood segmentation for the Peruvian desert coast or Cyclone Yaku found** |

**Gap (defensible claim):** the first study of unsupervised adaptation / self-training for
**optical (Sentinel-2)** flood segmentation with **geospatial foundation models**, under a
documented **desert-background shift**, evaluated on expert maps of a Peruvian event. Do
not claim "first domain adaptation for floods" (CIM-TTDA) or "first for water" (H2O-Net,
UUCP).

## 2. Method families, and what fits our setting

Setting: binary segmentation, ~750 unlabelled Peru training chips, ViT foundation models
(LayerNorm encoders) plus a CNN, a few GPU-hours per run.

| Family | Examples | Fit |
|---|---|---|
| Input statistics / style alignment | per-band histogram matching, FDA (Yang & Soatto, CVPR 2020) | **Cheap baseline.** Matches our finding that the shift is the background spectra. FDA was designed for RGB; per-band use on S2 is our extrapolation |
| Normalization-statistics adaptation | AdaBN | **U-Net only.** TerraMind/Prithvi encoders use LayerNorm; AdaBN reaches only neck/decoder BN |
| Entropy minimization at test time | TENT, CoTTA | Weak for transformers: per CoTTA Table 5 (agent-read), BN-adapt and TENT fell below the source model on a SegFormer. CoTTA's stochastic restore is useful as a guard |
| **Self-training with a teacher** | CBST, DACS, DAFormer, MIC, VFM-UDA | **Main candidate.** EMA teacher, pseudo-labels on Peru chips with **per-class thresholds** (CBST-style), source loss kept in every step, ClassMix-style mixing, LR warm-up; **+ masked consistency (MIC)** as token masking, natural for MAE-style TerraMind/Prithvi |
| Adversarial alignment | AdaptSegNet, ADVENT | Not recommended (unstable, superseded); possibly one row as the classic baseline, as CIM-TTDA does |
| Source-free / multi-stage | SHOT, ProDA | Not needed (source data available) or too expensive |

Binary-specific risk: pseudo-labels collapsing to "all dry". Guard with per-class
thresholds, the source loss, and monitoring of the predicted water fraction on `peru-dev`.

## 3. A strong baseline from our own data: index thresholding

The literature makes an MNDWI baseline mandatory. Our numbers
(`scripts/index_baseline.py`, `outputs/analysis_dev/index_baseline.md`):

| Baseline on `peru-dev` (water IoU) | Labels needed? | IoU |
|---|---|---|
| MNDWI > 0 (standard) | no | 0.486 (precision 0.49, recall 1.00) |
| MNDWI > Otsu threshold of unlabelled `peru-train` (t = 0.039) | no | 0.572 (precision 0.57, recall 0.99) |
| Sen1Floods11-only models (seed 0) | source labels only | 0.795–0.876 |
| MNDWI > 0.25, threshold picked **on dev itself** | yes (optimistic) | **0.924** |
| Peru-trained models (3 seeds) | Peru labels | 0.922–0.942 |

On Sen1Floods11 `valid` the best single MNDWI threshold reaches only 0.745 (vs 0.82–0.83 for the
models). So in Peru one well-chosen threshold rivals the trained models, while without labels
the threshold is badly placed (hazy desert top-of-atmosphere pixels have MNDWI > 0).

**Consequences for phase D.**
1. The no-label target is real: the label-free index baselines (0.49–0.57) and the
   Sen1Floods11-only models (0.80–0.88) leave a clear gap to 0.92–0.94.
2. A method that learns a better water/desert boundary from unlabelled Peru images competes
   with "pick the right MNDWI threshold". Index-based pseudo-labels (as in UUCP / H2O-Net)
   are therefore a natural, cheap baseline **and** a possible ingredient (combine index and
   model confidence for pseudo-labels).
3. Open question: did UNOSAT draw its water layer with an index threshold on Sentinel-2? If so,
   `peru-dev` IoU partly measures agreement with an index-derived product, which would explain
   why a tuned MNDWI threshold matches the deep models. Check UNOSAT's product notes before
   writing the evaluation chapter.

## 4. Protocol to copy (from CIM-TTDA, Portalés-Julià 2025, LoveDA, Patel 2021)

- Three levels: source-only → each adaptation method → Peru-trained upper bound; report the
  **share of the gap closed**.
- Re-run every baseline on the **same backbone** under our protocol (do not mix numbers across
  papers or backbones).
- Hyperparameters fixed in advance or selected on `peru-dev` only; never on test.
- Water IoU pooled and per event; **precision and recall separately** (shifted models
  over-predict); precision-recall curves (links to the calibration results).
- ≥ 3 seeds; report adaptation time and memory.
- Fix the adaptation unit before the final test: one adaptation on the 2023 unlabelled pool,
  then evaluate on `peru-test` and `peru-2017-test` (CIM-TTDA found per-event adaptation better;
  a per-event variant can be a secondary row).

## 5. Proposed candidate set for the phase-D OpenSpec change (to confirm)

Baselines: source-only; MNDWI > 0 and Otsu-MNDWI; per-band histogram matching / FDA;
AdaBN (U-Net only); index-based pseudo-labels (train on MNDWI-Otsu masks of unlabelled Peru
chips).
Main methods: (1) mean-teacher self-training with per-class thresholds (DACS/DAFormer-style,
simplified); (2) the same + masked consistency (MIC-style token masking); optional (3)
pseudo-labels that combine teacher confidence with the MNDWI index.
Backbones: U-Net reference + TerraMind first (cheaper, best-calibrated); Prithvi at the end.
Unlabelled Peru pool: `peru-train` images (labels ignored) plus, optionally, more 2023 dates
or areas away from the dev/test blocks.
