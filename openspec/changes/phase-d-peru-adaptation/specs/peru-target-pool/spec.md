## Purpose

Define the unlabelled Peru Sentinel-2 images that adaptation may use, and guarantee that no
Peru label and no held-out chip ever enters adaptation training.

## ADDED Requirements

### Requirement: Pool contents
The target pool SHALL consist of the Sentinel-2 images of the `peru-train` chips of Peru set
v1 (both 2023 dates), read with the same band order and per-model normalization as the
source data. Optional extra unlabelled images MAY be added only if every one lies outside
the dev, test and buffer spatial blocks and outside the `peru-2017-test` area, and is listed
in a versioned pool manifest with checksums.

#### Scenario: Default pool
- **WHEN** an adaptation run starts with the default configuration
- **THEN** it reads exactly the image files listed in `peru-train`, and the run record states the pool name and its number of chips

#### Scenario: Extra image inside a held-out block
- **WHEN** a pool manifest lists an image that intersects a dev, test or buffer block, or the `peru-2017-test` area
- **THEN** the pool builder refuses to write the manifest and names the offending image

### Requirement: No labels during adaptation
Adaptation training SHALL NOT read any Peru label file (UNOSAT or EMSR). Peru labels are
read only by evaluation on `peru-dev`, `peru-test` or `peru-2017-test`.

#### Scenario: Label access attempt
- **WHEN** an adaptation training step tries to load a Peru label raster
- **THEN** the run stops with an error naming the file

### Requirement: Held-out chips excluded
The adaptation pool SHALL NOT contain any `peru-dev`, `peru-test`, buffer or `peru-2017-test`
chip, with or without labels.

#### Scenario: Held-out chip in the pool
- **WHEN** an adaptation run is configured with a pool that contains a dev, test, buffer or 2017 chip
- **THEN** the run refuses to start and lists the chips
