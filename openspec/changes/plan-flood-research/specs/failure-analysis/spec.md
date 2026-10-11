## Purpose

Find out where and why SOTA flood segmentation models fail on Peru, so the method phase targets a measured failure instead of a guess.

## ADDED Requirements

### Requirement: Analysis on peru-dev only
Failure analysis SHALL use Sen1Floods11 `valid` and `peru-dev` only; `peru-test` and Sen1Floods11 `test` SHALL NOT inform any analysis or design decision.

#### Scenario: Attempt to analyse peru-test
- **WHEN** an analysis script is pointed at `peru-test`
- **THEN** it refuses to run unless the configuration is frozen for final evaluation

### Requirement: Stratified error breakdown
The analysis SHALL report water IoU, false negatives and false positives per stratum: cloud / cloud-shadow fraction, flooded vegetation, urban, bare/desert coastal terrain, permanent vs flood water, and water-body size; strata definitions and data sources SHALL be documented.

#### Scenario: Breakdown table
- **WHEN** the analysis runs for a model
- **THEN** it writes a table with one row per stratum, the pixel count, and IoU / FN rate / FP rate for each model (mean ± std over seeds)

### Requirement: Domain-shift measurement
The analysis SHALL quantify the drop from Sen1Floods11 `valid` to `peru-dev` per model and compare input statistics (per-band distributions) between the two.

#### Scenario: Drop reported
- **WHEN** all models are evaluated
- **THEN** the report lists per-model IoU on both sets, the drop, and band-distribution plots

### Requirement: Uncertainty vs errors
The analysis SHALL report whether high-uncertainty pixels coincide with errors on `peru-dev` (error-detection AUROC and calibration per stratum).

#### Scenario: Uncertainty by stratum
- **WHEN** uncertainty maps are available
- **THEN** ECE and error-detection AUROC are reported per stratum

### Requirement: Decision record
The analysis SHALL end with a written decision naming the dominant failure mode(s), with evidence, and the 3–4 candidate methods to test in phase D.

#### Scenario: Phase D kickoff
- **WHEN** phase D starts
- **THEN** a decision record exists that links each candidate method to a measured failure
