## Purpose

Fine-tune the CNN baseline and two geospatial foundation models reproducibly under one shared training setup that runs on Apple MPS and on rented CUDA GPUs.

## ADDED Requirements

### Requirement: Model set
The study SHALL train U-Net with a ResNet encoder, TerraMind v1-B and Prithvi-EO-2.0 300M, all with the same loss, ignore index, data pipeline and stopping rule, each from a configuration file.

#### Scenario: Shared setup
- **WHEN** two models' configs are compared
- **THEN** they differ only in backbone, neck/decoder, normalization, batch size and learning rate

### Requirement: Multiple seeds
Each model SHALL be trained with at least 3 seeds; the seed SHALL control weight initialization, data order and augmentation.

#### Scenario: Seed reproducibility
- **WHEN** a run is repeated with the same seed, config and commit
- **THEN** the validation water IoU agrees within run-to-run MPS nondeterminism, recorded in the experiment log

### Requirement: Selection on validation only
Hyperparameters and checkpoints SHALL be chosen using the `valid` split only.

#### Scenario: Checkpoint choice
- **WHEN** the final checkpoint of a run is selected
- **THEN** the choice uses only `valid` metrics and no `test` or Peru metric has been computed for that run

### Requirement: Run provenance
Every run SHALL record its full resolved config, seed, git commit, dirty-tree flag, package versions, hardware and wall-clock time.

#### Scenario: Dirty working tree
- **WHEN** a run starts with uncommitted changes
- **THEN** the run records the diff or refuses to start for reported experiments

### Requirement: Local demo before paid compute
Every experiment SHALL complete a local demo run with the same code and config (reduced epochs/data) before it runs on paid compute, and the paid run SHALL have a recorded cost estimate.

#### Scenario: Demo gate
- **WHEN** a run is launched on a rented GPU
- **THEN** the experiment log links a successful local demo of the same commit and config, and the estimated GPU-hours and cost

### Requirement: Device portability
The training and evaluation code SHALL run on `mps` and `cuda` with only configuration changes, and SHALL log the device and precision of every run.

#### Scenario: Same config, different device
- **WHEN** a demo config is switched from `mps` to `cuda`
- **THEN** the run starts without code changes and records the device

### Requirement: Memory safety on MPS
Training SHALL detect when MPS driver memory exceeds the recommended maximum and stop instead of silently swapping.

#### Scenario: Over-memory batch
- **WHEN** allocated MPS memory exceeds `recommended_max_memory`
- **THEN** the run aborts with a message suggesting a smaller batch or gradient accumulation
