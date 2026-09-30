# Research plan: flood extent segmentation with geospatial foundation models

Status: draft, 2026-09-29. **Open inputs: defense deadline, paper venue, advisor sign-off** (see the end of this file).

## 1. One-line pitch

Fine-tune Transformer-based geospatial foundation models (TerraMind, Prithvi-EO-2.0) to segment flood water in Sentinel-2 imagery. Test whether they beat a CNN baseline **on regions they never saw** (Bolivia holdout, Peru), and ship **calibrated uncertainty maps** with every prediction.

## 2. Why this framing

Checked against the literature in September 2026:

- In-distribution Sen1Floods11 scores have plateaued. In Prithvi-CAFE's Table 1 (arXiv 2601.02315, Jan 2026), water IoU is 82.50 for Prithvi 2.0 600M and 82.90 for TerraMind v1-B. The same paper reports that a U-Net ties or slightly beats foundation models in-distribution. Chasing +0.5 IoU on the standard test split is not a contribution.
- The gap opens **out of distribution**. On the Bolivia holdout, Prithvi-CAFE beats the U-Net baseline by 10.8 IoU (same paper). Generalization to a new region is where Transformers are claimed to matter, and **Peru has no public labelled flood benchmark we know of**.
- The approved thesis plan commits to Sentinel-2, Transformer vs CNN, and uncertainty quantification. This framing keeps all three.

## 3. Research questions → hypotheses

| # | Question | Hypothesis |
|---|---|---|
| RQ1 | Do Transformer foundation models beat a CNN baseline for S2 flood segmentation? | No meaningful gap in-distribution. The FMs are better on held-out regions (Bolivia, Peru). |
| RQ2 | Does adding Sentinel-1 SAR help? (ablation) | It helps most where S2 is cloudy or water is under vegetation. Needs advisor OK, since the approved plan says S2 only. |
| RQ3 | Are the models' uncertainties calibrated, and do they flag errors? | Raw softmax is over-confident. Ensembles or MC-dropout plus temperature scaling lower the ECE, and high-uncertainty pixels concentrate the errors. |
| RQ4 | Do the models transfer to real Peruvian flood events? | Performance drops versus Sen1Floods11. The uncertainty maps localize where the models fail. |

## 4. Data

| Set | Role | Notes |
|---|---|---|
| **Sen1Floods11, hand-labelled** | train / val / test / OOD | Official splits: train 252, valid 89, test 90, bolivia 15 (counted from the split CSVs, see `docs/feasibility.md`). S2 chips are **L1C**, 13 bands. Use the official splits, never re-split, so numbers are comparable with published tables. |
| WorldFloods v2 (ml4floods) | optional extra training / second OOD test | Check the licence before use. |
| **Peru test set (new)** | OOD case study, the paper's contribution | Candidate events: Piura / Niño Costero 2017 and Tumbes / Cyclone Yaku 2023. First check the Copernicus EMS activation list for Peru rapid-mapping products (EMSR codes). If none are usable, hand-label about 30–60 chips at 512×512 with S1 as the reference. That is real labelling work: budget for it and have both authors label a shared subset to measure agreement. |

Imagery for Peru comes from free sources (Copernicus Data Space, Planetary Computer, or Earth Engine). CDSE free tier: 40,000 processing units per month.

## 5. Models

All models go through TerraTorch so they share the same data pipeline, loss and trainer.

1. **Baseline CNN: U-Net** with a ResNet encoder. Required by the plan (comparison with conventional CNNs) and by reviewers. One config file.
2. **TerraMind v1-B** (103M params). The main model. It runs on MPS at batch size 8, about 38 s per epoch (`docs/feasibility.md`).
3. **Prithvi-EO-2.0 300M.** Second foundation model. Batch size 4 on MPS, about 96 s per epoch.
4. Optional, if time allows: a Transformer trained from scratch (e.g. SegFormer). It separates "Transformer architecture" from "pretraining", which strengthens RQ1.

**Fix before accuracy runs:** match each model's official input normalization. The feasibility run did not (see `docs/feasibility.md`).

## 6. Uncertainty (RQ3)

- **Deep ensemble:** the 3–5 seeds we train anyway double as ensemble members, so it costs no extra training.
- **MC-dropout** at inference on the best single model.
- **Temperature scaling** fitted on the validation split.
- **Metrics:** ECE plus reliability diagrams, and AUROC of uncertainty for predicting per-pixel error.

## 7. Evaluation protocol

- **Metrics:** water IoU (the main metric), mIoU, F1, and ECE.
- Report **mean ± std over at least 3 seeds.** In-distribution gaps between models are below 1 IoU, so single-seed tables are not evidence.
- Choose checkpoints on `valid` only. `test`, `bolivia` and Peru are touched once, at the end.
- Report **in-distribution (test)** and **OOD (bolivia, Peru)** side by side. That contrast is the paper.
- Log every run's config, seed and git commit (for example with Lightning's CSV logger or MLflow).

## 8. Repository layout (target)

```
peru-flood-segmentation/
├── configs/        # TerraTorch YAML, one per model/experiment
├── src/floodseg/   # data modules, uncertainty, metrics, Peru inference
├── scripts/        # download, train, evaluate, figures
├── notebooks/      # exploration only, no logic that results depend on
├── docs/           # PLAN.md, feasibility.md, REFERENCES.md, experiment log
├── data/           # git-ignored
├── outputs/        # git-ignored (runs, checkpoints, figures)
└── thesis/         # Overleaf repo (separate git, git-ignored here)
```

## 9. Phases

Durations are estimates. Map them onto dates once the deadline is known.

| Phase | Work | Output | Est. |
|---|---|---|---|
| 0 | Full Sen1Floods11 download, correct normalization, reproducible train/eval scripts | 1 run end-to-end | 1 wk |
| 1 | U-Net + TerraMind + Prithvi, 3 seeds each, S2 only | RQ1 in-distribution + Bolivia table | 2 wk |
| 2 | Uncertainty: ensembles, MC-dropout, temperature scaling, ECE | RQ3 results + figures | 1–2 wk |
| 3 | Peru test set: EMS search, then labelling if needed | Labelled Peru chips + data card | 2–3 wk (runs in parallel with 1–2) |
| 4 | Peru evaluation + S1 ablation (if approved) | RQ2, RQ4 results | 1–2 wk |
| 5 | Thesis chapters III–VI + paper draft | Full draft | 3–4 wk |
| 6 | Revisions, defense slides | — | 2 wk |

Write Chapter I (problem) and II (theory / state of the art) during phases 0–2, not at the end.

## 10. Thesis ↔ work mapping

| Chapter | Content |
|---|---|
| I Planteamiento | Problem, RQ1–4, hypotheses, objectives (keep the approved plan's objectives), scope |
| II Fundamento teórico | Remote sensing, S1/S2, CNNs vs Transformers, foundation models, uncertainty; state-of-the-art review (see `docs/REFERENCES.md`) |
| III Método | Data, splits, protocol, metrics, Peru dataset construction |
| IV Desarrollo | Architectures, fine-tuning setup, uncertainty methods, implementation |
| V Resultados | RQ1–4 tables and figures |
| VI Discusión | Hypotheses tested against the results; comparison with published numbers |

**Paper:** the same RQ1 + RQ3 + Peru contribution in a compact form. Choose the venue once the deadline is known, and check the venue's own call for papers for dates.

## 11. Risks

| Risk | Mitigation |
|---|---|
| No usable EMS product for Peru | Hand-label a smaller set; report inter-annotator agreement |
| Clouds over Peru during flood peaks (S2) | S1 ablation; report the cloud fraction per chip; uncertainty maps |
| Flooded vegetation invisible to optical imagery (`shastry_mapping_2023`: 74% of misses) | State it as a limitation; S1 ablation |
| Prithvi batch-size limits on MPS | Close the VM/Docker that holds about 16 GB; gradient accumulation; Kaggle/Colab only if needed |
| Deviating from the approved plan (S1, title) | Get advisor sign-off before phase 4 |

## 12. Decisions needed from the authors

1. Defense deadline and paper venue, which set the dates in §9.
2. Advisor approval for the S1 ablation (RQ2) and any change to the title.
3. How labelling (phase 3) and chapters are split between the two authors.
