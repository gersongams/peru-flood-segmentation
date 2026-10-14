# Research plan: flood detection in Peru with geospatial foundation models

Status: 2026-10-12. Supersedes the 2026-09-29 draft. The specification lives in `openspec/changes/plan-flood-research/` (proposal, specs, design, tasks); this file is the short human-readable version.
**Open inputs: defense deadline, paper venues. No thesis advisor is assigned yet (2026-10-14): the authors make the research decisions; a course teacher gives general input.**

## 1. One-line pitch

Flood detection models score well on the standard benchmark but are untested on Peru. We **reproduce SOTA**, **build a Peru flood dataset from expert maps (no manual labelling) and train on it**, **measure where SOTA fails there and why**, and **design a method that fixes that failure**.

## 2. Why this framing

- **Detect, not forecast.** Forecasting needs hydrological models, rainfall forecasts and river gauges: a different field. Detection from satellite imagery has labelled data, pretrained models and expert maps for Peru.
- **The standard benchmark is saturated.** In Prithvi-CAFE's Table 1 (arXiv 2601.02315, p. 6), water IoU on Sen1Floods11 `test` is 82.2–83.4 for the foundation models and **84.03 for a U-Net**, the best in the table. +0.3 IoU there is not a contribution. Verified numbers and caveats: `docs/REFERENCES.md`, "SOTA reference numbers".
- **Generalization is where the gap is.** The same paper reports a 10.8 IoU gap over U-Net on the Bolivia holdout. Peru has no public labelled flood benchmark we know of (`docs/peru_dataset_proposal.md`).
- **Teacher's input (general):** drop Bolivia; train and evaluate on Peruvian data. Kept because the data back it: Sen1Floods11 → Peru fine-tuning changed dev IoU by at most 0.4 points (`docs/failure_analysis_dev.md`).

## 3. Research questions

| # | Question | Hypothesis |
|---|---|---|
| RQ1 | Do Transformer foundation models (TerraMind, Prithvi) beat a CNN (U-Net) on Sen1Floods11 and on Peru, and how much does fine-tuning on Peruvian data help? | No meaningful gap in-distribution; the foundation models lose less on Peru and need fewer Peru chips. |
| RQ2 | Where and why do SOTA models fail on Peru? | Most errors concentrate in a few strata: clouds, flooded vegetation, desert/coastal terrain. |
| RQ3 | Are the models' uncertainties calibrated, and do they flag the failures? | Raw softmax is over-confident; ensembles + temperature scaling lower ECE; high uncertainty concentrates errors. |
| RQ4 | Can a method targeted at the dominant failure beat SOTA on Peru? | Defined after RQ2 (phase D). |

## 4. Phases

| Phase | Work | Output | Paper |
|---|---|---|---|
| **A. Reproduce SOTA** | U-Net, TerraMind v1-B, Prithvi-EO-2.0 300M on Sen1Floods11 official splits, ≥3 seeds | Results within ~2 IoU of published | — |
| **B. Peru dataset** | UNOSAT Cyclone Yaku 2023 maps (drawn on Sentinel-2) → automatic labels on S2 L1C chips; spatial split `peru-train` / `peru-dev` / `peru-test`; EMSR199 2017 as `peru-2017-test` | Dataset + data card + label-quality check | 1 |
| **B2. Train on Peru** | Pretrained backbones trained only on `peru-train`, ≥3 seeds; Sen1Floods11 models as comparison | Peru-only vs Sen1Floods11 on `peru-dev` | 1 |
| **C. Failure analysis** | Stratified errors and calibration on `peru-dev` | Decision record: dominant failure, 3–4 candidate methods | 1 |
| **D. Method** (own change) | Candidates: S1+S2 fusion robust to clouds, domain adaptation to Peru, uncertainty-guided correction | Beat best baseline on `peru-test` | 2 |
| **E. Susceptibility** (optional) | Detected Peru floods + terrain (DEM, HAND) → flood-prone zones | Extra chapter | — |

B runs in parallel with A. Chapters I–II are written during A–B, not at the end.

## 5. Data

| Set | Role |
|---|---|
| Sen1Floods11 hand-labelled, official splits (train 252 / valid 89 / test 90) | Training, model selection (`valid`), in-distribution test (`test`). Never re-split. `bolivia` is not used. |
| UNOSAT 2023 (Lambayeque + Piura, ~17 900 km² analysed) + S2 L1C of 2023-03-21 and 2023-04-01 | `peru-train` / `peru-dev` / `peru-test`, split by spatial blocks. |
| EMSR199 2017 (Salinas, Parachique, Piura) + S2 L1C of 2017-04-07 | `peru-2017-test`: cross-event test, never trained on. |
| ESA WorldCover, Copernicus DEM, cloud masks | Strata for failure analysis. |

## 6. Models and compute

- U-Net (ResNet encoder), TerraMind v1-B, Prithvi-EO-2.0 300M via TerraTorch, shared loss and trainer, each with its official input normalization.
- **Local demo first, then rent GPUs.** Every experiment runs as a small demo on the M4 Max. It goes to a rented GPU only if the demo works and the cost estimate is accepted. Measured local speeds: `docs/feasibility.md`.

## 7. Protocol

- Metrics: water IoU (primary), mIoU, F1, ECE, error-detection AUROC; split-level and per-chip.
- Mean ± std over ≥3 seeds. A method "beats" a baseline only if its mean − std exceeds the baseline mean.
- Select on `valid` / `peru-dev` only. `test` and `peru-test` are evaluated once per frozen configuration.
- Every run logs config, seed, git commit, device and versions. Every number in the thesis traces to a results file or a cited table.

## 8. Thesis ↔ work

| Chapter | Content |
|---|---|
| I Planteamiento | Problem, RQ1–4, objectives, scope (detection, Peru) |
| II Fundamento teórico | Remote sensing, S1/S2, CNNs vs Transformers, foundation models, uncertainty, domain shift; state of the art (`docs/REFERENCES.md`) |
| III Método | Sen1Floods11, Peru dataset construction, protocol, metrics |
| IV Desarrollo | Models, training, uncertainty, phase-D method |
| V Resultados | Phases A–D |
| VI Discusión | Hypotheses vs results; comparison with published numbers |

## 9. Risks

| Risk | Mitigation |
|---|---|
| UNOSAT labels preliminary; EMSR199 labels a week before the S2 image | Automatic agreement check; EMSR199 disagreement pixels → −1; report as limitations |
| Peru training set small or noisy | ~600+ chips per date from UNOSAT; Sen1Floods11 comparison rows show whether more data would help |
| SOTA not reproduced | Fix normalization/budget before phase C; report residual gap |
| No dominant failure in C | Paper 1 is benchmark + analysis; D targets the largest stratum |
| Phase D needs Sentinel-1, outside the plan | Authors decide in the phase-D decision record (task 6.5) |

## 10. Decisions needed

1. Defense deadline and paper venues (to date the phases).
2. ~~Sentinel-1 in phase D~~: decided 2026-10-14, not used (decision record in `docs/failure_analysis_dev.md`). Phase D = flood mapping in Peru without Peru labels.
3. Optional: someone spends ~1 h looking at the most doubtful Peru chips.
