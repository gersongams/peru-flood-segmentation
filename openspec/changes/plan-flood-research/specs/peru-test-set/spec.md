## Purpose

Create a labelled, documented Sentinel-2 flood dataset for Peru (EMSR199, Niño Costero 2017), used both to fine-tune models on Peruvian data (advisor request) and as the only out-of-distribution evaluation.

## ADDED Requirements

### Requirement: Source and scene selection
The Peru test set SHALL derive initial labels from Copernicus EMS EMSR199 flood delineation products for Salinas, Parachique and Piura (Piura Norte optional; Trujillo only if visual review shows standing water), paired with Sentinel-2 imagery of the nearest clear date (7 Apr 2017 unless a closer clear scene is found).

#### Scenario: Scene provenance recorded
- **WHEN** a chip is created
- **THEN** its metadata records the S2 product ID, acquisition date, EMS product code and version, and the date gap between label and image

### Requirement: Sen1Floods11-compatible format
Chips SHALL be 512×512 at 10 m with the same band set as Sen1Floods11 S2Hand and the same label convention (1 / 0 / −1).

#### Scenario: Format check
- **WHEN** the Peru set is loaded through the same pipeline as Sen1Floods11
- **THEN** every chip loads without code changes and passes shape, band and label-value checks

### Requirement: Manual correction of labels
Every chip's rasterized EMS label SHALL be reviewed and corrected by hand against its Sentinel-2 image, and clouds and no-data SHALL be marked −1.

#### Scenario: Unreviewed chip
- **WHEN** a chip has not been marked as reviewed
- **THEN** it is excluded from the evaluation set

### Requirement: Inter-annotator agreement
Both authors SHALL independently label a shared subset of at least 10 chips, and the set's documentation SHALL report IoU and Cohen's kappa between them.

#### Scenario: Agreement reported
- **WHEN** the Peru set is frozen
- **THEN** the data card contains the agreement metrics and the subset chip IDs

### Requirement: Geographic train/dev/test split
The Peru set SHALL be split geographically into `peru-train` (fine-tuning), `peru-dev` (model selection, failure analysis and method design) and `peru-test` (final evaluation only). `peru-test` SHALL be a whole AOI that no other part touches. `peru-train` and `peru-dev` SHALL be separate AOIs, or, if there are too few chips, separate spatial blocks with a buffer of at least one chip width between them.

#### Scenario: Spatial leakage into test
- **WHEN** the split is created
- **THEN** no chip in `peru-test` overlaps spatially with or shares an AOI with any chip in `peru-train` or `peru-dev`

#### Scenario: Training on held-out parts
- **WHEN** a training config lists `peru-dev` or `peru-test` chips as training data
- **THEN** the run refuses to start

### Requirement: Enough chips to train on
The set SHALL target at least 150 reviewed chips with water in `peru-train` and at least 30 in each of `peru-dev` and `peru-test`.

#### Scenario: Too few training chips
- **WHEN** fewer than 150 reviewed water chips are available for `peru-train`
- **THEN** the data card reports the actual count and the Peru fine-tuning results are labelled as low-data

### Requirement: Frozen and documented test set
The Peru set SHALL be frozen (versioned chip list with checksums) before any model is evaluated on it, and SHALL have a data card describing sources, licence, scene dates, cloud fraction per chip, labelling procedure and known limitations.

#### Scenario: Change after evaluation
- **WHEN** a chip or label changes after the first evaluation
- **THEN** the set gets a new version and all Peru results are recomputed and reported against that version
