# Data card: Peru flood segmentation set, v1

Frozen 2026-10-14. Checksums: `docs/peru_v1_checksums.txt` (pins the manifests and
split lists; the manifests pin every chip by SHA-256). Verify a copy with
`uv run python scripts/freeze_peru.py --check`. All figures below are computed from
the manifests (`data/peru/*/manifest.csv`, `data/peru/unosat2023/splits/split.csv`);
km² = pixel fraction × 5.12 km × 5.12 km per chip.

## Purpose

Train and evaluate flood (surface water) segmentation on Peruvian Sentinel-2 imagery,
with no manual labelling (authors' decision, following a teacher's suggestion: train on Peru only). Labels come from expert flood
maps turned into rasters by fixed rules.

## Imagery

- Sentinel-2 L1C, all 13 bands as DN (reflectance × 10,000), same band order as
  Sen1Floods11 S2Hand; downloaded from the Copernicus Data Space Process API.
- The +1000 DN offset of processing baseline ≥ 04.00 is removed (Sentinel Hub
  `harmonizeValues`), so values match pre-2022 L1C.
- 512 × 512 chips at 10 m, UTM 17S (EPSG:32717), on a fixed 5,120 m grid.
- Label value 1 = water, 0 = not water, −1 = ignore (excluded from loss and metrics).

## Part 1: UNOSAT 2023 (`peru-train`, `peru-dev`, `peru-test`)

- **Event:** Cyclone Yaku, Lambayeque and Piura, March–April 2023.
- **Labels:** UNOSAT FL20230321PER, drawn on Sentinel-2 (CC BY-SA, "preliminary, not
  field-validated"); sources and links in `docs/peru_label_sources.md`.
  - 2023-03-22 image: `WaterExtent_20230321` (flood + existing water).
  - 2023-04-01 image: `FloodExtent_20230401` + existing water of 22 March (no water
    extent layer was published for this date; the layer's attribute says 2023-04-03, a
    day with no S2 pass, and it matches the 1 April image best).
- **Ignore (−1):** outside UNOSAT's analysed extent, inside UNOSAT's cloud layer, S2
  no-data, or thick cloud (B02 > 4500 and B11 < 1.15·B02, dilated 5 px). s2cloudless
  and Sen2Cor SCL are not used: they flag the bright Sechura desert as cloud.
- **Builder:** `scripts/build_peru_unosat.py`.
- **Size:** 1,291 chips at 668 locations (668 on 2023-03-22, 623 on 2023-04-01).
  Water 1,017 km² on 22 March (UNOSAT reports about 1,023 km²) and 1,003 km² on
  1 April. Mean ignore share 14.7 % (22 March) and 22.0 % (1 April).
- **Split:** `scripts/split_peru_unosat.py`. Spatial blocks of 4 × 4 chips (≈ 20 km)
  assigned at random (seed 0) until dev and test each hold ≈ 15 % of the water; a
  location is in the same split on both dates; train chips touching a dev/test chip
  are dropped as a buffer. 0 held-out chips touch train.

| Split | Chips | Locations | Chips with water > 1 % | Water km² | Dry km² | Mean ignore |
|---|---|---|---|---|---|---|
| train | 753 | 392 | 164 | 744 | 15,278 | 18.8 % |
| dev | 174 | 87 | 79 | 489 | 3,442 | 13.8 % |
| test | 172 | 90 | 63 | 366 | 3,144 | 22.2 % |
| buffer (unused) | 192 | 99 | 46 | 421 | 3,796 | 16.2 % |

Water is 4.6 % of labelled pixels in train and 12.4 % in dev: the dev/test blocks
were drawn only from blocks with water, so they are richer in water than train.

## Part 2: EMSR199 2017 (`peru-2017-test`, cross-event)

- **Event:** Niño Costero, March 2017. Copernicus EMS EMSR199, the only EMS flood
  activation in Peru. Delineations drawn on radar of 23–30 March 2017.
- **Image:** Sentinel-2 of 2017-04-07, the nearest clear date.
- **Labels:** 1 = EMS water (crisis + hydrography polygons) and MNDWI > 0;
  0 = outside EMS water and MNDWI ≤ 0; −1 = disagreement, outside the AOI, thick cloud,
  or no-data. The disagreement rule is needed because image and labels are 1–2 weeks
  apart; it removes hard pixels, so scores on this set are optimistic.
- **Builder:** `scripts/build_peru_emsr2017.py`.

| AOI | Chips | Chips with water > 1 % | Water km² | Mean ignore | Mean disagreement | Chips touching 2023 train |
|---|---|---|---|---|---|---|
| Parachique | 72 | 53 | 515 | 17.0 % | 11.6 % | 16 |
| Salinas | 67 | 52 | 679 | 30.2 % | 27.0 % | 22 |
| Piura | 65 | 26 | 78 | 24.2 % | 12.4 % | 65 |

The AOIs overlap the 2023 area. 101 chips (73 with water, 653 km² of water) do not touch
any 2023 `peru-train` location; report these as the main 2017 number and all 204
alongside.

## Use rules

- Train on `peru-train` only; select on `peru-dev` only. `peru-test` and
  `peru-2017-test` are evaluated once, at the end. Enforced by
  `floodseg.peru.check_train_split`, which refuses a train split containing held-out
  chips.
- Each model keeps its official input normalization (model configs).

## Known limitations

- Labels are preliminary expert products, not field-validated, and no person checked
  them chip by chip; the checks done are in `docs/peru_label_sources.md` (date match,
  clouds, offset).
- Thick-cloud masking is a fixed spectral rule; thin cloud and haze remain.
- How UNOSAT made the labels: "automated remote sensing techniques … reviewed and revised as
  necessary by the analyst" (UNOSAT methodology note on HDX). The Lambayeque/Piura water and
  flood polygons were vectorized from a **20 m raster** on the Sentinel-2 grid (87–91 % of
  vertices on a 20 m grid, median edge 20 m; cloud layers from a 10 m raster; analysis extent
  hand-drawn), `scripts/unosat_provenance.py`. Labels are therefore 20 m products rasterized
  to 10 m, so water edges are uncertain by about one 10 m pixel. The extraction method is
  unpublished; the 20 m grid is consistent with a SWIR index (B11/B12 are 20 m), but red-edge
  bands and Sen2Cor's SCL are also 20 m, so this is not proven.
- Possible label errors: 32 UNOSAT chips (2.5 %) have ≥ 5 ha of clear label/MNDWI
  disagreement and 56 more have 1–5 ha, almost all pixels labelled dry that look like water.
  About half of that area is next to thick cloud (possible shadow, haze or surf); without
  it, 15 chips are doubtful. List and rule: `docs/peru_doubtful_chips.md`
  (`scripts/label_qa.py`). The rule misses turbid water, so unflagged chips are not
  guaranteed clean. v1 is unchanged; the list only annotates it.
- `peru-dev` (and `peru-test`) blocks were drawn only from blocks with water, so their
  background is brighter desert than `peru-train`'s (`docs/domain_shift_dev.md`).
- The 1 April 2023 label reuses the 22 March existing-water layer.
- The 2017 labels exclude pixels where EMS and the water index disagree.
- One region and two events; results may not transfer to the Amazon or the highlands.
