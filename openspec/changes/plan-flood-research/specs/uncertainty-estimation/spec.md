## Purpose

Produce calibrated per-pixel flood probabilities and uncertainty maps, and measure whether uncertainty identifies prediction errors.

## ADDED Requirements

### Requirement: Uncertainty methods
The system SHALL provide, for each model: (a) single-model softmax, (b) a deep ensemble of the seed models, (c) MC-dropout at inference, and (d) temperature scaling fitted on `valid`.

#### Scenario: Ensemble from seeds
- **WHEN** the ensemble prediction is computed
- **THEN** it averages the probabilities of the same ≥3 seed checkpoints used in the main results, with no extra training

#### Scenario: Temperature fitted on valid only
- **WHEN** temperature scaling is applied to `test` or Peru
- **THEN** the temperature was fitted on `valid` and is recorded with the results

### Requirement: Uncertainty maps
For every evaluated chip the system SHALL output the water probability and a per-pixel uncertainty map (predictive entropy; mutual information for ensemble and MC-dropout).

#### Scenario: Map export
- **WHEN** a model is evaluated on Peru
- **THEN** a georeferenced probability and uncertainty raster is written per chip

### Requirement: Calibration and error-detection metrics
The system SHALL report ECE (fixed binning, bin count recorded), reliability diagrams, and the AUROC of uncertainty for predicting per-pixel errors.

#### Scenario: Ignore pixels excluded
- **WHEN** calibration metrics are computed
- **THEN** pixels labelled −1 are excluded
