# Flood mapping in Peru with geospatial foundation models

Master's thesis, Maestría en Ciencias con mención en Inteligencia Artificial, Universidad
Nacional de Ingeniería (UNI). Authors: Gerson Andre Garrido Mansilla, Renzo Elvis Cerron Tome.
The thesis is written in Spanish; the repository documentation is in English.

**Question.** Flood-extent segmentation in Sentinel-2 imagery scores well on the standard
benchmark (Sen1Floods11), but nobody has tested it on Peru. We:
- compare two Transformer geospatial foundation models (TerraMind v1-B, Prithvi-EO-2.0 300M)
  with a U-Net;
- build a labelled Peru flood dataset from expert maps, without manual labelling;
- measure where the models fail in Peru and why;
- adapt Sen1Floods11-trained models to Peru without Peru labels.

The research questions and phases are in [`docs/PLAN.md`](docs/PLAN.md).

## Data

| Set | Source | Use |
|---|---|---|
| Sen1Floods11 hand-labelled, Sentinel-2 L1C, 13 bands | Bonafilia et al. 2020 | official splits (train 252 / valid 89 / test 90); never re-split |
| Peru 2023 (Cyclone Yaku, Lambayeque and Piura) | UNOSAT expert flood maps → automatic labels on Sentinel-2 chips | `peru-train` (753), `peru-dev` (174), `peru-test` (172), spatial split |
| Peru 2017 (Niño Costero) | Copernicus EMS EMSR199 maps | `peru-2017-test` (204 chips) |

The data card is [`docs/peru_data_card.md`](docs/peru_data_card.md). Neither the data nor the
checkpoints are in the repository (`data/` and `outputs/` are git-ignored). They are rebuilt with
`scripts/download_sen1floods11.py`, `scripts/build_peru_unosat.py` and
`scripts/build_peru_emsr2017.py`. The Peru builders download imagery from the Copernicus Data Space (CDSE) Process API and need CDSE credentials, which
they read from environment variables.

## Protocol

- Models are selected on Sen1Floods11 `valid` and on `peru-dev` only.
- The test sets (`test`, `peru-test`, `peru-2017-test`) are evaluated **once**, from configurations
  frozen with checkpoint hashes in [`configs/frozen.yaml`](configs/frozen.yaml).
  [`scripts/evaluate_frozen.py`](scripts/evaluate_frozen.py) refuses to run otherwise.
- At least 3 seeds per configuration, reported as mean ± std. Each model uses its official input
  normalization.
- Every run is logged with its commit in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).
  Every method choice, check and correction is dated in
  [`docs/METHODS_NOTES.md`](docs/METHODS_NOTES.md).

## Results so far

Detailed write-ups:

| Topic | Document |
|---|---|
| Phase-D (adaptation without Peru labels) on `peru-dev` | [`docs/phaseD_dev.md`](docs/phaseD_dev.md) |
| One-time test evaluation | [`docs/phaseD_test.md`](docs/phaseD_test.md) |
| Failure analysis | [`docs/failure_analysis_dev.md`](docs/failure_analysis_dev.md) |
| Domain shift | [`docs/domain_shift_dev.md`](docs/domain_shift_dev.md) |
| Calibration and uncertainty | [`docs/calibration_dev.md`](docs/calibration_dev.md), [`docs/uncertainty_dev.md`](docs/uncertainty_dev.md) |
| Feature engineering | [`docs/feature_engineering.md`](docs/feature_engineering.md) |

Headline from `docs/phaseD_test.md`:

- **2023 event (`peru-test`).** U-Net trained on Sen1Floods11 with histogram matching to
  unlabelled Peru imagery reaches 0.798 ± 0.005 water IoU.
  - That is 0.030 below the Peru-trained U-Net (0.828) and 0.22 above the unadapted U-Net (0.576).
  - It is the best result without Peru labels.
- **2017 event (`peru-2017-test`).** The order reverses: the unadapted models score best.
  - The 2017 labels only measure recall of clear water.
  - Adaptation trades faint-water recall for fewer desert false positives, so it scores lower
    here.
  - No method wins both events.

## Repository layout

| Path | Contents |
|---|---|
| `src/floodseg/` | Library: datasets, normalization, metrics, input alignment (histogram matching, FDA), AdaBN, self-training |
| `configs/` | TerraTorch model configs; `frozen.yaml` (test-time freeze); pools |
| `scripts/` | Data building, training helpers, analyses, frozen evaluation and scoring; `scripts/cloud/` holds the RunPod helpers |
| `tests/` | pytest suite (`uv run pytest`) |
| `notebooks/` | Executed notebooks, e.g. [`feature_engineering.ipynb`](notebooks/feature_engineering.ipynb) (Spanish) |
| `docs/` | Plan, references, data card, experiment log, methods notes, write-ups |
| `openspec/` | Specifications and task lists (OpenSpec) |

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/). The core library is
[TerraTorch](https://github.com/IBM/terratorch).

```
uv sync
uv run pytest
```

Development ran on an Apple M4 Max (MPS); training runs that were too slow there ran on RunPod
GPUs. Notes on MPS speed and pitfalls are in [`docs/feasibility.md`](docs/feasibility.md).

## Course deliverables

Weekly progress reports for the master's course are kept on their own branches and are also
in `main`. The notebooks are in Spanish and saved with their outputs:

| Week | Branch | Notebook | Report |
|---|---|---|---|
| 4: exploratory data analysis (EDA) | `avance-sem4-eda` | [`notebooks/eda.ipynb`](notebooks/eda.ipynb) | — |
| 5: feature engineering | `avance-sem5-feature-engineering` | [`notebooks/feature_engineering.ipynb`](notebooks/feature_engineering.ipynb) | [`docs/feature_engineering.md`](docs/feature_engineering.md) |
| 6: sprint report (EDA + feature engineering + A/B test of the training data: U-Net on raw vs histogram-matched vs FDA Sen1Floods11 images; model comparison as annex) | `avance-sem6-sprint` | [`notebooks/baseline_variantes.ipynb`](notebooks/baseline_variantes.ipynb) | [`docs/informe_sprint_sem6.md`](docs/informe_sprint_sem6.md) |
