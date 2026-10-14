# Domain shift between Sen1Floods11 and Peru (task 6.2)

Date: 2026-10-14. Splits: Sen1Floods11 `train` / `valid`, `peru-train`, `peru-dev`. No test
split used. Method details: `docs/METHODS_NOTES.md`; runs: `docs/EXPERIMENT_LOG.md`.

## 1. Transfer matrix (water IoU)

`scripts/transfer_matrix.py` → `outputs/transfer/matrix.{md,csv}`; every value parsed from a
`terratorch validate` log. Peru-only and Sen1Floods11 → Peru: mean ± std over 3 seeds;
Sen1Floods11-only: seed 0.

| Training | Model | Sen1Floods11 `valid` | `peru-dev` |
|---|---|---|---|
| Sen1Floods11-only | U-Net | **0.8188** | 0.8354 |
| Sen1Floods11-only | TerraMind | **0.8168** | 0.7951 |
| Sen1Floods11-only | Prithvi | **0.8327** | 0.8758 |
| Peru-only | U-Net | 0.6898 ± 0.0065 | **0.9415 ± 0.0019** |
| Peru-only | TerraMind | 0.6559 ± 0.0131 | **0.9265 ± 0.0025** |
| Peru-only | Prithvi | 0.6527 ± 0.0190 | **0.9218 ± 0.0006** |
| Sen1Floods11 → Peru | U-Net | 0.7019 ± 0.0133 | **0.9383 ± 0.0018** |
| Sen1Floods11 → Peru | TerraMind | 0.6911 ± 0.0234 | **0.9282 ± 0.0009** |
| Sen1Floods11 → Peru | Prithvi | 0.6805 ± 0.0056 | **0.9236 ± 0.0006** |

Bold = in-domain (the checkpoint was selected on that split, so these cells are slightly
optimistic). Compare cells **within a column**. Across columns, scores are not comparable:
the two sets differ in labels, water share (`valid` 11.0 %, `peru-dev` 12.4 % of labelled
pixels) and difficulty. Sen1Floods11-only U-Net and Prithvi even score higher on `peru-dev` than on
`valid`, because `peru-dev` has large, easy water bodies.

- **Sen1Floods11 → Peru costs 5–13 points** on `peru-dev`: in-domain 0.92–0.94, transferred
  U-Net −10.6, TerraMind −13.1, Prithvi −4.6 against the Peru-only mean of the same model.
  Prithvi transfers best and TerraMind worst.
- **Peru → Sen1Floods11 costs 13–18 points** on `valid`: U-Net −12.9, TerraMind
  −16.1, Prithvi −18.0 against the Sen1Floods11-only checkpoint of the same model. Peru is
  one event in one region, Sen1Floods11 `train` spans 10 regions (chip-name prefixes: Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA), so a Peru-trained model
  sees a narrower distribution. The two gaps are measured from different in-domain
  baselines, so compare them with care: as a share of the in-domain IoU they are 5–14 %
  on `peru-dev` and 16–22 % on `valid`, so the Peru → Sen1Floods11 direction is still the
  harder one.
- **After Peru-only training the CNN generalizes best**: U-Net 0.690 on `valid` vs
  TerraMind 0.656 and Prithvi 0.653. With Peru labels, the foundation models' pretraining
  does not buy out-of-domain robustness.
- **Fine-tuning forgets Sen1Floods11**: starting from the Sen1Floods11 checkpoints
  (0.819 / 0.817 / 0.833 on `valid`), Peru fine-tuning drops them to 0.70 / 0.69 / 0.68
  (−12 to −15 points), although they stay 1–4 points above Peru-only training from the
  pretrained backbone.

## 2. Input distributions

`scripts/band_stats.py` → `outputs/band_stats/{summary.md,bands.csv,indices.csv}`,
histograms `outputs/band_stats/figs/hist_{water,dry}.png`. Every 37th labelled pixel per chip; shift = Peru
minus Sen1Floods11 train, in Sen1Floods11-train std units of the band (what the U-Net sees
after normalization). Harmonization checked: no leftover +1000 DN offset (Peru water
B11/B12 p5 74 / 43 DN vs Sen1Floods11 130 / 75).

| | peru-train | peru-dev |
|---|---|---|
| Dry land, B02–B04 and B12 (SMD) | +0.79 to +0.92 | +1.38 to +1.68 |
| Dry land, B11 | +0.53 | +0.89 |
| Dry land, B08 / B8A (NIR) | +0.18 / +0.14 | +0.02 / −0.03 |
| Water, all bands, max \|SMD\| | 0.25 (B03) | 0.40 (B8A) |

| Water index, median (p5–p95) | Sen1Floods11 train | peru-train | peru-dev |
|---|---|---|---|
| MNDWI, water | 0.49 (0.03–0.81) | 0.72 (0.15–0.89) | 0.72 (0.28–0.87) |
| NDWI, water | 0.10 (−0.16–0.45) | 0.36 (−0.12–0.63) | 0.42 (0.07–0.60) |
| NDWI, dry | −0.38 (−0.60 to −0.12) | −0.15 (−0.51–0.00) | −0.06 (−0.31–0.07) |

- **The shift is in the dry land, not the water.** Peru's dry land (Sechura desert, bare
  soil) is far brighter in the visible and SWIR bands than Sen1Floods11's land, which is
  mostly vegetated (dry NDWI −0.38); Peru water differs by at most 0.4 std.
- **Peru dry land sits near the low-contrast water range.** Its NDWI is close to 0 (median
  −0.06 on dev vs −0.38 in Sen1Floods11). This is where the Sen1Floods11-only models make
  false positives on Peru (5–7 % of dry pixels with NDWI 0–0.1, Finding 2 in
  `docs/failure_analysis_dev.md`), and where Peru-trained models set a stricter boundary.
- Peru water is clearer by MNDWI (median 0.72 vs 0.49); Sen1Floods11 water includes more
  turbid, mixed and vegetated water.

## What it shows

The main shift between the two datasets is the background, not the water: bright,
unvegetated desert whose spectra overlap low-contrast water. Hypothesis, consistent with
the false-positive pattern but not tested: this is why Sen1Floods11-only models over-predict
water in Peru, and why Peru-trained models (which learned a strict desert/water boundary)
transfer badly to Sen1Floods11's vegetated scenes.
Foundation-model pretraining does not remove the gap in either direction. This motivates
reporting Peru results separately from Sen1Floods11 (RQ on generalization) and not
expecting a single model to serve both without Peru data.

Caveats: Sen1Floods11-only rows are one seed; the fine-tuned seeds share one starting
checkpoint; pixel sampling is not stratified by event; peru-dev's background is brighter
than peru-train's (dry-land SMD +1.4–1.7 vs +0.8–0.9), because dev blocks were drawn only
from blocks with water, so dev is not a random sample of the Peru background.
