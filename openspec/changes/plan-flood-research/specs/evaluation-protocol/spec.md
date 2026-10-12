## Purpose

Define how models are compared so that thesis and paper numbers are reproducible, honest about variance, and separate in-distribution from out-of-distribution performance.

## ADDED Requirements

### Requirement: Metrics
Evaluation SHALL report water IoU (primary), mIoU, F1 and ECE, computed over all valid pixels of a split (not averaged per chip), with per-chip values also saved.

#### Scenario: Metric output
- **WHEN** a checkpoint is evaluated on a split
- **THEN** a machine-readable file with split-level and per-chip metrics is written alongside the run

### Requirement: Held-out sets touched once
`test` and `peru-test` SHALL be evaluated only after all models, hyperparameters and uncertainty settings are frozen, and only once per frozen configuration.

#### Scenario: Premature evaluation
- **WHEN** a test or Peru evaluation is requested for a configuration not marked frozen
- **THEN** the evaluation refuses to run

### Requirement: Multi-seed aggregation
Results tables SHALL report mean ± std over at least 3 seeds per model.

#### Scenario: Single-seed result
- **WHEN** a table row has fewer than 3 seeds
- **THEN** it is flagged and excluded from the main comparison

### Requirement: In-distribution vs OOD reporting
The main results table SHALL show Sen1Floods11 `test` and `peru-test` side by side for every model, and the drop from `test` to Peru, for two training regimes: Sen1Floods11 only, and Sen1Floods11 then fine-tuned on `peru-train`.

#### Scenario: Results table
- **WHEN** the final tables are generated
- **THEN** each model row contains test and Peru water IoU (mean ± std) and their difference

### Requirement: Traceable numbers
Every number in the thesis or paper SHALL trace to a results file produced by the pipeline, or for published numbers to a cited table, page or URL.

#### Scenario: Published comparison
- **WHEN** a literature value is quoted next to our results
- **THEN** its source table or page is cited, and differences in splits or protocol are stated

### Requirement: SOTA reproduction check
Before Peru experiments, each foundation model's Sen1Floods11 `valid` and final `test` water IoU SHALL be compared with the published value for the same model and splits (source table and page recorded), and gaps larger than the reported tolerance SHALL be investigated before proceeding.

#### Scenario: Reproduction gap
- **WHEN** our seed mean is more than 2 IoU below the published value for the same model
- **THEN** the gap is logged and phase B/C results for that model are not reported until it is explained or fixed

### Requirement: Improvement claims
A claim that a method beats SOTA SHALL compare against the best reproduced baseline on the same split, and SHALL be made only if the seed mean minus one std exceeds the baseline mean.

#### Scenario: Within-noise result
- **WHEN** a method's mean exceeds the baseline but mean − std does not
- **THEN** it is reported as matching, not beating
