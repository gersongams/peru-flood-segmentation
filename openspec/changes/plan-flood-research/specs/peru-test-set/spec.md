## Purpose

Build a Peru flood dataset without manual labelling, from expert flood maps already drawn by UNOSAT and Copernicus EMS, used to train models on Peruvian data (advisor request) and to test them. Sources and checks: `docs/peru_label_sources.md`.

## ADDED Requirements

### Requirement: Label sources
Training, development and in-event test labels SHALL come from the UNOSAT Cyclone Yaku 2023 products (Lambayeque + Piura, Sentinel-2 of 2023-03-21 and 2023-04-01), paired with the Sentinel-2 L1C acquisition the labels were drawn on. A cross-event test SHALL come from Copernicus EMS EMSR199 (2017) paired with the nearest clear Sentinel-2 L1C scene (2017-04-07).

#### Scenario: Scene provenance recorded
- **WHEN** a chip is created
- **THEN** its manifest row records the label product and version, the S2 product ID, the acquisition date, and the date gap between label and image (0 for UNOSAT)

### Requirement: Automatic label rules
Each chip label SHALL be derived by fixed rules: 1 = inside the product's water/flood extent; 0 = inside the analysed extent and outside water; −1 = outside the analysed extent, inside the product's cloud layer, or S2 no-data. For EMSR199 (radar labels a week before the S2 image), pixels where the EMS polygon and the S2 water index disagree SHALL be set to −1, and this SHALL be reported as a limitation.

#### Scenario: Rule change
- **WHEN** a label rule changes
- **THEN** all chips are regenerated and the dataset version increases

### Requirement: Sen1Floods11-compatible format
Chips SHALL be 512×512 at 10 m with the same 13-band L1C band set as Sen1Floods11 S2Hand and the same label convention (1 / 0 / −1).

#### Scenario: Format check
- **WHEN** the Peru set is loaded through the same pipeline as Sen1Floods11
- **THEN** every chip loads without code changes and passes shape, band and label-value checks

### Requirement: Label quality check
The dataset build SHALL produce an overlay image per chip (true colour, SWIR false colour, label) and an automatic agreement score between the label and a water index. Chips whose agreement is below a fixed threshold SHALL be listed for an optional human look; no chip is edited by hand.

#### Scenario: Doubtful chips
- **WHEN** the build finishes
- **THEN** a list of the lowest-agreement chips and their overlays is written next to the manifest

### Requirement: Geographic train/dev/test split
The UNOSAT 2023 set SHALL be split by space into `peru-train`, `peru-dev` and `peru-test` (spatial blocks with at least one chip width of buffer between parts). The two 2023 dates cover the same ground, so a location SHALL fall in one part for both dates. EMSR199 2017 SHALL be a separate test set, `peru-2017-test`, never trained on.

#### Scenario: Spatial leakage
- **WHEN** the split is created
- **THEN** no chip in `peru-dev`, `peru-test` or `peru-2017-test` overlaps spatially with a `peru-train` chip of any date

#### Scenario: Training on held-out parts
- **WHEN** a training config lists `peru-dev`, `peru-test` or `peru-2017-test` chips as training data
- **THEN** the run refuses to start

### Requirement: Frozen and documented dataset
The Peru set SHALL be frozen (versioned chip list with checksums) before any model is evaluated on it, and SHALL have a data card describing sources, licences (UNOSAT CC BY-SA), scene dates, label rules, cloud fraction per chip and known limitations (UNOSAT labels are preliminary and not field-validated).

#### Scenario: Change after evaluation
- **WHEN** a chip or label changes after the first evaluation
- **THEN** the set gets a new version and all Peru results are recomputed and reported against that version
