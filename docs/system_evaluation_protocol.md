# System evaluation protocol (HE3): fixed before implementation and measurement

Fixed 2026-10-17, before the system exists. Nothing here may change after the first measurement.
A change made before the first measurement needs a dated entry in METHODS_NOTES. The commit that
adds this file is the evidence of when it was fixed. Hypothesis HE3 and requirements RF1–RF7:
METHODS_NOTES 2026-10-16 "Thesis reframed around a system"; thesis Chapter I, Tabla 1.1.

## 1. Definitions

- **Escena completa:** one Sentinel-2 L1C tile (MGRS 100 km square, about 110 × 110 km; Copernicus
  *S2 Products*, SentiWiki) for one acquisition date. In practice this is the whole
  `S2X_MSIL1C_…_T17Mxx_…` product.
- **Mapa de agua y de incertidumbre:** for every 10 m pixel of the tile:
  - the water probability (seed mean) and the uncertainty (seed std) of the selected model
    (3-seed ensemble);
  - the cloud / no-data mask;
  - all written as GeoTIFF.
- **Descarga:** obtaining the complete L1C product of the tile from Copernicus Data Space Ecosystem
  through its OData product download, **including authentication**, until the archive is on
  local disk.
  - The Process API is not used for the timed path. Its cost in processing units, about
    2 000 per tile-run by our estimate, would not fit a realistic quota for 18 runs; OData
    downloads use no processing units.

## 2. Timing protocol

| Item | Fixed value |
|---|---|
| Hardware | Apple M4 Max, 36 GB unified memory, PyTorch MPS backend. The authors' computer, plugged in, with no other heavy job running (no training or inference in the background) |
| Software | The system at a tagged commit. Record the commit, macOS version and library versions in the run log |
| Model | The model selected under OE1 (frozen checkpoints, sha256 from `configs/frozen.yaml`), as a 3-seed ensemble |
| State at start | **Warm system:** the application server is running and the models are loaded in memory. Image cache **empty for that tile and date**: no copy of the product and no intermediate files. A cold start (process start plus model loading) is measured once per tile and reported separately; it does not count toward HE3 |
| Timer start | The backend receives the request (tile, date) from the application (RF1 done), timestamped in the server log |
| Timer stop | The probability, uncertainty and mask GeoTIFFs of the whole tile are written and closed, and the map is available to the application, timestamped in the server log |
| Included | Authentication, catalog lookup, product download, decompression and band reading, resampling to 10 m, normalization, inference, mosaicking, GeoTIFF writing, tile generation for the map |
| Excluded | Human interaction time; RF5 metrics and RF6 area computation (timed separately and reported, not part of HE3) |
| Scenes | The **6 tiles** that cover UNOSAT's analysed extent, for the acquisition of **2023-03-22**: 17MNP (49.3 % of the extent), 17MNQ (18.7 %), 17MPN (17.8 %), 17MNN (6.7 %), 17MPP (4.9 %), 17MPQ (2.6 %). The percentages are the share of extent sample points per MGRS square, from a 0.02° grid. If a tile has no 2023-03-22 product in the catalog, use the nearest available date and log it |
| Repetitions | **3 per tile** (18 timed runs), cache cleared before each run, at least 10 min between runs of the same tile |
| Network | Record downlink capacity right before each run with macOS `networkQuality -s`. No minimum is imposed; the values are reported |
| Failed runs | A run that fails because of an external service error (HTTP 5xx, timeout, CDSE outage) is logged and repeated, at most 3 times per run. The failures are reported. A failure caused by the system itself is a failure of RF2/RF3 |
| **Decision rule (HE3, time)** | Met if the **median of the 3 runs is ≤ 15 min for every one of the 6 tiles**. All 18 individual times, the maximum and the network values are reported |
| Output | `outputs/system_eval/timing_runs.csv`, with one row per run: tile, date, rep, start, stop, minutes, network Mbps, commit, failures. Plus the server logs |

## 3. Functional requirement tests (HE3, requirements)

Each requirement passes if its test passes. Evidence is a log excerpt and a screenshot, stored in
`outputs/system_eval/rf/`. HE3 (requirements) is met only if **7 of 7** pass.

| RF | Test |
|---|---|
| RF1 | In the app, select the event "Yaku 2023", the date 2023-03-22 and a rectangle inside 17MNP. The request reaches the backend with the right tile, date and bounds (log) |
| RF2 | (a) For a tile and date not in the cache, the product is downloaded from CDSE and processed. (b) For an already processed tile, it loads from the cache without downloading |
| RF3 | The probability GeoTIFF exists for the whole tile, covers the tile footprint, and values lie in [0, 1]. On the `peru-test` chips it overlaps, it agrees with the frozen pass (sigmoid of `outputs/logits_test` peru-trained logits, seed mean): **mean absolute probability difference ≤ 0.02 and IoU of the water masks (p > 0.5) ≥ 0.95**. The tolerance allows for the different input path: raw SAFE bands and their resampling versus the Process API chips |
| RF4 | The uncertainty layer (seed std) and the cloud / no-data layer (project thick-cloud rule; no-data where all bands are 0) are displayed and can be toggled |
| RF5 | Over the 2023 analysed extent, the UNOSAT reference overlays the map, and agreement metrics (IoU, precision, recall) are shown. On chips of `peru-test`, the metrics match `scripts/score_frozen.py` to 3 decimals |
| RF6 | For at least 2 dates, the app shows the flooded area (definition in §4) and the total water area per date on a timeline |
| RF7 | Export of the probability, uncertainty and flood maps as GeoTIFF, and of the flood polygons as GeoJSON. Both files open in QGIS / `rasterio` / `geopandas` with the correct CRS and footprint |

## 4. Definition of "área inundada" (RF6)

**Flooded area at date *t* = A = Área((W_t \ W_0) ∩ V_t ∩ V_0), computed pixel by pixel.**
It is the surface of the pixels that change from not-water to water between a pre-event reference
image and *t*. It is **not** a difference of two total areas, which would net new water against
receding water.
- W_t and W_0 are the water pixels at *t* and on the reference image.
- V_t and V_0 are the valid pixels (no cloud, no data) at *t* and on the reference image.
- Both images are the same MGRS tile on the same grid. (Clarified 2026-10-17 before
  implementation, after review feedback. The definition is unchanged; only the formula is now
  explicit.)
- **Water** means probability > 0.5, i.e. z > 0, the frozen evaluation rule (`configs/frozen.yaml`). Written "≥ 0.5" until 2026-10-17 and corrected for consistency before any measurement.
- **Valid** means not cloud and not no-data, at both *t* and the reference.
- Total water at *t* is reported alongside.
- **The reference image:** the clearest (least thick-cloud) Sentinel-2 acquisition of the same
  tile in the 60 days before the event. The event start is the user's input; for Yaku it is
  2023-03-04 (INDECI, Compendio 2024, Tabla 6).
  - The reference water map is computed once and cached. It is outside the HE3 timer.
- **This is UNOSAT's own logic.** Its FloodExtent is "New Water / Water Increase" relative to
  pre-existing water: WaterExtent 1 023.4 km² = FloodExtent 986.1 km² + pre-existing 37.3 km²
  on 2023-03-22. Source: `scripts/unosat_reconcile.py`, `outputs/unosat_reconcile.txt`.
- **A static permanent-water mask was rejected.** ESA WorldCover 2021 class 80 covers 173.0 km²
  of the analysed extent. 105.9 km² of it lies inside UNOSAT's 2023 *flood* polygons, and only
  22.6 km² inside the pre-existing water. Most likely these are desert lagoons that persisted
  from earlier rain events. Subtracting it would remove about 11 % of the flood UNOSAT mapped.
- **Interpretation caveat:** pixels already wet before the event but expanded by it are not
  counted as flood unless they were dry on the reference image.
- **Wording (WRITING_RULES §11.12):** the app and the thesis say "área inundada" only for this
  quantity. Otherwise they say "agua detectada".
