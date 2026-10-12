# Peru flood label sources without manual labelling

Checked 2026-10-12. Goal: train and test only on Peru (advisor) with no hand-labelling.

## Found

| Source | Event | Sensor of the labels | Layers | Area | Licence | Notes |
|---|---|---|---|---|---|---|
| **UNOSAT** EQ/FL20230321PER ([HDX 22 Mar](https://data.humdata.org/dataset/3e1cf0c7-5e87-43e6-b04f-2d1c3f982061), [HDX 1 Apr](https://data.humdata.org/dataset/69d4820c-c1bd-4938-9409-877e06c284dc), [HDX Tumbes](https://data.humdata.org/dataset/72f6401a-8b0f-4a9e-b286-2c0ec88e721b)) | Cyclone Yaku, Mar–Apr 2023 | **Sentinel-2** (same image as the label) | AnalysisExtent, CloudObstruction, FloodExtent, WaterExtent | Lambayeque + Piura: analysis 17 929 km², flood ~990 km² per date (2023-03-21, 2023-04-01; water extent also 2023-04-21). Tumbes: 984 km², flood 11–20 km² | CC BY-SA | "Preliminary, not field-validated" (HDX notes). Downloaded to `data/unosat2023/raw/`. |
| **Copernicus EMS EMSR199** | Niño Costero, Mar 2017 | Radar (Cosmo-SkyMed, Radarsat, …), 23–30 Mar 2017 | Flood delineation per AOI | see `docs/peru_dataset_proposal.md` | Copernicus | Only Copernicus EMS flood activation in Peru (activation API list: EMSR199, EMSN038 Lima 2017, EMSN037 risk). |
| **Sentinel-1 GRD** over Piura | 2017 | — | imagery | passes 20, 23, 26 Mar, 1, 4 Apr 2017 (CDSE catalogue) | free | Matches the EMSR199 label dates. |
| Copernicus GFM | 2015– (archive), global | Sentinel-1, automatic | flood extent, reference water, exclusion | — | — | Not checked in detail; automatic labels, fallback only. |
| International Charter 811 (CONIDA) | 2023 | PlanetScope, WorldView-2, S1 | maps | Piura districts | — | Mostly map PDFs; not used. |

Area figures computed with geopandas in UTM from the downloaded shapefiles.

## Recommendation

1. **Train / dev / in-event test on UNOSAT 2023 (Sentinel-2, labels drawn on the same image).** All three models stay optical. Split geographically (e.g. Lambayeque vs Piura blocks); never split by date, because the 21 Mar and 1 Apr layers cover the same ground. Water = WaterExtent; CloudObstruction and outside AnalysisExtent = −1.
2. **Cross-event test on EMSR199 2017** (different year and El Niño type): Sentinel-2 of 7 Apr 2017 with pixels where EMS and the image disagree set to −1, reported as a limitation; optional Sentinel-1 of 26 Mar 2017 for TerraMind.
3. **Optional sanity check:** one person looks at 10–15 overlays (~1 h). Without it, the thesis states that the labels are the UNOSAT / EMS products, unchecked.
