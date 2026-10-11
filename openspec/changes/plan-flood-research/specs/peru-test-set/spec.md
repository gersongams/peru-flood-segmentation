## Purpose

Create a labelled, documented Sentinel-2 flood test set for Peru (EMSR199, Niño Costero 2017), used as the only out-of-distribution evaluation.

## ADDED Requirements

### Requirement: Source and scene selection
The Peru test set SHALL derive initial labels from Copernicus EMS EMSR199 flood delineation products for Piura (Salinas, Parachique, Piura, Piura Norte) and Trujillo, paired with Sentinel-2 imagery of the nearest clear date (7 Apr 2017 unless a closer clear scene is found).

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

### Requirement: Geographic dev/test split
The Peru set SHALL be split by area of interest (no AOI in both parts) into `peru-dev` (analysis and method design) and `peru-test` (final evaluation only), with each part containing water chips from at least two AOIs where possible.

#### Scenario: Spatial leakage
- **WHEN** the split is created
- **THEN** no chip in `peru-test` overlaps spatially with or shares an AOI with any chip in `peru-dev`

### Requirement: Frozen and documented test set
The Peru set SHALL be frozen (versioned chip list with checksums) before any model is evaluated on it, and SHALL have a data card describing sources, licence, scene dates, cloud fraction per chip, labelling procedure and known limitations.

#### Scenario: Change after evaluation
- **WHEN** a chip or label changes after the first evaluation
- **THEN** the set gets a new version and all Peru results are recomputed and reported against that version
