## Purpose

Provide Sen1Floods11 hand-labelled Sentinel-2 chips with the official splits and the correct per-model normalization, so results are comparable with published tables.

## ADDED Requirements

### Requirement: Official splits only
The data pipeline SHALL use the official Sen1Floods11 hand-labelled split files (train 252, valid 89, test 90) and SHALL NOT re-split, subsample or merge them for any reported experiment.

#### Scenario: Split sizes are verified
- **WHEN** the dataset is loaded for any split
- **THEN** the chip count equals the official CSV count (252 / 89 / 90) or loading fails with an error naming the split

#### Scenario: Bolivia split is not used
- **WHEN** an experiment configuration references the `bolivia` split
- **THEN** it is rejected for reported results, since Peru replaces it as the OOD test

### Requirement: Complete and verifiable download
The download step SHALL fetch every S2Hand and LabelHand chip of the official splits and SHALL verify that each file exists and opens.

#### Scenario: Missing chip
- **WHEN** a chip listed in a split CSV is absent or unreadable after download
- **THEN** the step reports the chip ID and exits with a non-zero status

### Requirement: Model-specific normalization
Each model SHALL receive inputs normalized with its official pretraining statistics and band selection (TerraMind: 13-band S2 L1C; Prithvi: its 6 HLS bands; U-Net: statistics computed on `train` only).

#### Scenario: Normalization matches the model
- **WHEN** a training run starts
- **THEN** the run log records the band list, means and stds used, and they match the model's official config

### Requirement: Label convention
Labels SHALL use 1 = water, 0 = not water, −1 = ignore (cloud / no data), and the ignore value SHALL be excluded from losses and metrics.

#### Scenario: Ignore pixels
- **WHEN** a chip contains −1 pixels
- **THEN** those pixels contribute neither to the loss nor to any metric
