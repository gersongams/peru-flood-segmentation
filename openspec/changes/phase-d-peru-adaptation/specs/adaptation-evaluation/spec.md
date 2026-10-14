## Purpose

Specify how adapted models are selected, compared with baselines and the supervised upper
bound, and finally scored, so that phase-D claims are fair and pre-registered.

## ADDED Requirements

### Requirement: Label-free headline, oracle selection as a secondary row
The headline result of each adaptation method SHALL use hyperparameters and an iteration
budget fixed in the design before any adaptation run, and the final checkpoint, so that no
Peru label influences it. Runs tuned or checkpoint-selected on `peru-dev` labels SHALL be
reported only as a secondary row named "oracle selection", next to the MNDWI threshold tuned
on `peru-dev`, which uses the same label information. `peru-dev` SHALL be scored offline
from saved checkpoints, never inside an adaptation run. `peru-test` and `peru-2017-test`
SHALL be evaluated once per frozen configuration, after all phase-D choices are fixed.

#### Scenario: Headline row
- **WHEN** the phase-D results table is generated
- **THEN** the headline row of each method comes from the pre-registered defaults and the final checkpoint, and any dev-selected run appears only in the "oracle selection" row

#### Scenario: Premature test evaluation
- **WHEN** a test evaluation is requested for an adaptation configuration not marked frozen
- **THEN** the evaluation refuses to run

### Requirement: Three-level comparison
For each backbone, results SHALL report the source-only model, each adaptation method and
baseline, and the Peru-trained model (upper bound), with ≥ 3 seeds (mean ± std), plus the
share of the source→upper-bound gap closed by each method: (IoU_method − IoU_source) /
(IoU_upper − IoU_source). Label-free index baselines (MNDWI > 0, Otsu on the target pool) and
the label-using MNDWI threshold tuned on `peru-dev` SHALL appear in the same table.

#### Scenario: Gap-closed table
- **WHEN** phase-D tables are generated
- **THEN** each method row shows water IoU, precision, recall and gap closed, for the same backbone and seeds

### Requirement: Same-backbone baselines
Every baseline SHALL be run under this protocol on the same backbone and seeds as the methods
it is compared with. Numbers from other papers SHALL NOT appear in the comparison rows.

#### Scenario: Literature number
- **WHEN** a published adaptation result is mentioned
- **THEN** it appears only in the discussion with its source table, not in the comparison table

### Requirement: Metrics
Results SHALL report pooled water IoU, precision and recall, and a boundary-tolerant water
IoU that excludes labelled pixels within 1 px of a label water edge. Results SHALL also
report water IoU per event: the 2023 Yaku event on `peru-test`, and the 2017 event on
`peru-2017-test`.

#### Scenario: Metric set
- **WHEN** an adapted model is evaluated on a Peru split
- **THEN** the output contains plain IoU, tolerant IoU, precision and recall

### Requirement: Label-sensitivity row
The final evaluation SHALL also score `peru-test` without the chips flagged "doubtful" by
`scripts/label_qa.py` (both variants: all blobs, and without near-cloud blobs), as fixed in the
phase-D decision record before any test result was seen.

#### Scenario: Sensitivity scoring
- **WHEN** the final `peru-test` evaluation runs
- **THEN** it writes the full-set score and both doubtful-excluded scores for every model
