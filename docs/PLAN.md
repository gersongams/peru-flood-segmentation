# Research plan: flood detection in Peru with geospatial foundation models

Status: 2026-10-11. Supersedes the 2026-09-29 draft. The specification lives in `openspec/changes/plan-flood-research/` (proposal, specs, design, tasks); this file is the short human-readable version.
**Open inputs: defense deadline, paper venues, advisor OK on Sentinel-1 for phase D.**

## 1. One-line pitch

Flood detection models score well on the standard benchmark but are untested on Peru. We **reproduce SOTA**, **build the first labelled Peru flood benchmark**, **measure where SOTA fails there and why**, and **design a method that fixes that failure**.

## 2. Why this framing

- **Detect, not forecast.** Forecasting needs hydrological models, rainfall forecasts and river gauges: a different field. Detection from satellite imagery has labelled data, pretrained models and expert maps for Peru.
- **The standard benchmark is saturated.** In Prithvi-CAFE's Table 1 (arXiv 2601.02315, p. 6), water IoU on Sen1Floods11 `test` is 82.2–83.4 for the foundation models and **84.03 for a U-Net**, the best in the table. +0.3 IoU there is not a contribution. Verified numbers and caveats: `docs/REFERENCES.md`, "SOTA reference numbers".
- **Generalization is where the gap is.** The same paper reports a 10.8 IoU gap over U-Net on the Bolivia holdout. Peru has no public labelled flood benchmark we know of (`docs/peru_dataset_proposal.md`).
- **Advisor:** drop Bolivia; evaluate generalization on Peru.

## 3. Research questions

| # | Question | Hypothesis |
|---|---|---|
| RQ1 | Do Transformer foundation models (TerraMind, Prithvi) beat a CNN (U-Net) on Sen1Floods11 and on Peru? | No meaningful gap in-distribution; the foundation models lose less on Peru. |
| RQ2 | Where and why do SOTA models fail on Peru? | Most errors concentrate in a few strata: clouds, flooded vegetation, desert/coastal terrain. |
| RQ3 | Are the models' uncertainties calibrated, and do they flag the failures? | Raw softmax is over-confident; ensembles + temperature scaling lower ECE; high uncertainty concentrates errors. |
| RQ4 | Can a method targeted at the dominant failure beat SOTA on Peru? | Defined after RQ2 (phase D). |

## 4. Phases

| Phase | Work | Output | Paper |
|---|---|---|---|
| **A. Reproduce SOTA** | U-Net, TerraMind v1-B, Prithvi-EO-2.0 300M on Sen1Floods11 official splits, ≥3 seeds | Results within ~2 IoU of published | — |
| **B. Peru benchmark** | EMSR199 (Niño Costero 2017) → hand-corrected Sentinel-2 chips; geographic split `peru-dev` / `peru-test` | Dataset + data card + annotator agreement | 1 |
| **C. Failure analysis** | Stratified errors and calibration on `peru-dev` | Decision record: dominant failure, 3–4 candidate methods | 1 |
| **D. Method** (own change) | Candidates: S1+S2 fusion robust to clouds, domain adaptation to Peru, uncertainty-guided correction | Beat best baseline on `peru-test` | 2 |
| **E. Susceptibility** (optional) | Detected Peru floods + terrain (DEM, HAND) → flood-prone zones | Extra chapter | — |

B runs in parallel with A. Chapters I–II are written during A–B, not at the end.

## 5. Data

| Set | Role |
|---|---|
| Sen1Floods11 hand-labelled, official splits (train 252 / valid 89 / test 90) | Training, model selection (`valid`), in-distribution test (`test`). Never re-split. `bolivia` is not used. |
| Peru EMSR199 (Salinas, Parachique, Piura; Piura Norte weak; Trujillo likely dropped) + Sentinel-2 L1C of 7 Apr 2017 | `peru-dev` for analysis/method design; `peru-test` evaluated once, at the end. Split by area, no area in both. |
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
| EMS labels are from radar (26–30 Mar); clear S2 is 7 Apr, water may have receded | Manual correction on S2; report the date gap |
| Peru set small | Bootstrap CIs; leave-one-area-out if too small to split |
| SOTA not reproduced | Fix normalization/budget before phase C; report residual gap |
| No dominant failure in C | Paper 1 is benchmark + analysis; D targets the largest stratum |
| Phase D needs Sentinel-1, outside the approved plan | Ask the advisor before C ends |

## 10. Decisions needed

1. Defense deadline and paper venues (to date the phases).
2. Advisor OK for Sentinel-1 in phase D.
3. Who labels which Peru areas; both authors label the shared agreement subset.
