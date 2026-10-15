## Purpose

Adapt flood-segmentation models trained on labelled Sen1Floods11 to Peru using only the
unlabelled target pool, with a fixed set of label-free baselines and self-training methods.

## ADDED Requirements

### Requirement: Source checkpoints
Every adaptation run SHALL start from a Sen1Floods11-trained checkpoint of the same model and
seed, selected on Sen1Floods11 `valid` only, and SHALL record that checkpoint's path and
commit.

#### Scenario: Run provenance
- **WHEN** an adaptation run finishes
- **THEN** its run record contains the source checkpoint path, its commit, the seed, the method and its configuration

### Requirement: Label-free index baselines
The system SHALL produce water maps from MNDWI = (B03 − B11)/(B03 + B11) with threshold 0 and
with an Otsu threshold computed on the unlabelled target pool, using no labels.

#### Scenario: Otsu threshold
- **WHEN** the Otsu baseline is computed
- **THEN** the threshold is derived from target-pool pixels only and is stored with the results

### Requirement: Input-alignment baselines
The system SHALL provide input alignment of source images to the target pool (per-band
histogram matching, and low-frequency amplitude transfer per band). Both SHALL be available as
a test-time transform of target images towards the source distribution, applied to the
unchanged source models (paired with source-only, so no training noise), and MAY be run
during supervised source training for every backbone (then with ≥ 3 seeds per transform).
It SHALL also provide re-estimation of batch-normalization statistics on the target pool:
all BN layers for the U-Net, the decoder and neck BN layers for TerraMind.

#### Scenario: Alignment uses target images only
- **WHEN** alignment statistics or amplitude spectra are computed
- **THEN** they come from target-pool images and Sen1Floods11 `train` images, never from dev or test images

### Requirement: Self-training adaptation
The system SHALL provide teacher–student self-training in which:
- the teacher is an exponential moving average of the student;
- pseudo-labels on target images use separate confidence thresholds for water and not-water;
- each step also includes the supervised loss on labelled Sen1Floods11 images.

A masked-consistency variant SHALL train the student on masked target images against the
teacher's pseudo-labels for the unmasked image.

#### Scenario: Pseudo-label thresholds
- **WHEN** a self-training run is configured
- **THEN** the water and not-water thresholds are separate, recorded parameters

#### Scenario: Collapse watch
- **WHEN** the predicted water fraction on the target pool falls below a configured floor during training
- **THEN** the run logs a collapse warning, and the event is recorded in the run record

### Requirement: Comparable compute
All adaptation methods for one backbone SHALL use the same source checkpoints, the same
target pool and a recorded iteration budget, and SHALL log wall-clock time and peak memory.

#### Scenario: Budget recorded
- **WHEN** a method is compared with another on the same backbone
- **THEN** both run records show iterations, wall-clock time and peak memory
