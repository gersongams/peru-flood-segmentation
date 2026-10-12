"""Build Peru chips from the UNOSAT Cyclone Yaku 2023 flood maps (no manual labels).

For each 512x512 chip on a 10 m UTM 17S grid over UNOSAT's analysed extent
(Lambayeque + Piura), fetch the 13-band Sentinel-2 L1C image the labels were
drawn on from the CDSE Process API and rasterize the label by fixed rules:

    1  inside the water extent of that date
    0  inside the analysed extent, outside water
   -1  outside the analysed extent, inside UNOSAT's cloud layer, S2 no-data,
       or thick cloud (B02 > 4500 and B11 < 1.15 * B02, dilated by CLOUD_BUFFER
       pixels). UNOSAT's cloud layer misses many clouds and water was drawn under
       some of them. s2cloudless (CLM) and Sen2Cor SCL flag the bright Sechura
       desert as ~100 % cloud here, so neither is used.

Label layers per date (UNOSAT FL20230321PER, docs/peru_label_sources.md):
    2023-03-22  WaterExtent_20230321 (flood + existing water)
    2023-04-01  FloodExtent_20230401 + existing water of 2023-03-22
                (no WaterExtent was published for this date)

Writes data/peru/unosat2023/{s2,label}/<date>_<chip>.tif and manifest.csv.

    uv run python scripts/build_peru_unosat.py --limit 6   # local demo
    uv run python scripts/build_peru_unosat.py             # all chips
"""

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peru_cloud_survey import REPO, SH, token  # noqa: E402

RAW = REPO / "data" / "unosat2023" / "raw" / "0401" / "FL0230321PER_SHP"
OUT = REPO / "data" / "peru" / "unosat2023"
EPSG = 32717  # UTM 17S
RES, SIZE = 10, 512
CHIP = RES * SIZE  # metres
MIN_ANALYSED = 0.5  # keep chips at least half inside the analysed extent
CLOUD_BUFFER = 5  # pixels
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

DATES = {
    "2023-03-22": {"label_date": "2023-03-22", "water": ["S2_20230321_WaterExtent_Lambayeque_Piura"],
                   "cloud": "S2_20230321_CloudObstruction_Lambayeque_Piura"},
    # UNOSAT's attribute says 2023-04-03, a day with no S2 pass over the area;
    # the layer agrees best with the 2023-04-01 image.
    "2023-04-01": {"label_date": "2023-04-03?", "water": ["S2_20230401_FloodExtent_Lambayeque_Piura", "existing_water_20230322"],
                   "cloud": "S2_20230401_CloudObstruction_Lambayeque_Piura"},
}
ANALYSED = "S2_20230321_AnalysisExtent_Lambayeque_Piura"

# Same 13 L1C bands as Sen1Floods11 S2Hand, as DN (reflectance x 10000).
# harmonizeValues (Sentinel Hub default) removes the +1000 offset of
# processing baseline >= 04.00, so values match pre-2022 L1C.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: %s, units: "DN"}],
          output: {bands: 14, sampleType: "UINT16"}};
}
function evaluatePixel(s) { return [%s, s.dataMask]; }
""" % (str(BANDS + ["dataMask"]).replace("'", '"'), ", ".join(f"s.{b}" for b in BANDS))


def layer(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(RAW / f"{name}.shp").to_crs(EPSG)


def load_layers() -> tuple[gpd.GeoDataFrame, dict]:
    analysed = layer(ANALYSED)
    w22 = layer("S2_20230321_WaterExtent_Lambayeque_Piura")
    f22 = layer("S2_20230321_FloodExtent_Lambayeque_Piura")
    existing = gpd.GeoDataFrame(geometry=[w22.union_all().difference(f22.union_all())], crs=EPSG)
    per_date = {}
    for date, spec in DATES.items():
        water = [existing if n == "existing_water_20230322" else layer(n) for n in spec["water"]]
        per_date[date] = {"label_date": spec["label_date"], "water": [g for w in water for g in w.geometry],
                          "cloud": list(layer(spec["cloud"]).geometry)}
    return analysed, per_date


def chip_grid(analysed: gpd.GeoDataFrame) -> list[tuple[str, tuple]]:
    area = analysed.union_all()
    minx, miny, maxx, maxy = area.bounds
    x0, y0 = np.floor(minx / CHIP) * CHIP, np.floor(miny / CHIP) * CHIP
    chips = []
    for x in np.arange(x0, maxx, CHIP):
        for y in np.arange(y0, maxy, CHIP):
            b = box(x, y, x + CHIP, y + CHIP)
            if b.intersection(area).area / b.area >= MIN_ANALYSED:
                chips.append((f"E{int(x)}_N{int(y)}", (x, y, x + CHIP, y + CHIP)))
    return chips


def fetch_s2(tok: str, bounds: tuple, date: str) -> np.ndarray:
    body = {
        "input": {
            "bounds": {"bbox": list(bounds), "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{EPSG}"}},
            "data": [{"type": "sentinel-2-l1c", "dataFilter": {
                "timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"}}}],
        },
        "output": {"width": SIZE, "height": SIZE,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    for attempt in range(3):
        r = requests.post(f"{SH}/api/v1/process", json=body,
                          headers={"Authorization": f"Bearer {tok}"}, timeout=180)
        if r.status_code == 429 and attempt < 2:
            import time
            time.sleep(30)
            continue
        r.raise_for_status()
        with rasterio.open(io.BytesIO(r.content)) as src:
            return src.read()
    raise RuntimeError("unreachable")


def thick_cloud(s2: np.ndarray) -> np.ndarray:
    blue, swir = s2[1].astype(float), s2[11].astype(float)
    return (blue > 4500) & (swir < 1.15 * blue)


def agreement(s2: np.ndarray, label: np.ndarray) -> float:
    """IoU between label water and MNDWI > 0 on labelled pixels (quality flag only)."""
    g, s = s2[2].astype(float), s2[11].astype(float)
    wat = (g - s) / (g + s + 1e-6) > 0
    ok = label >= 0
    lab = label == 1
    union = (ok & (lab | wat)).sum()
    return round(float((ok & lab & wat).sum() / union), 4) if union else float("nan")


def make_label(bounds: tuple, analysed_geoms, layers: dict, valid: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    transform = rasterio.transform.from_bounds(*bounds, SIZE, SIZE)

    def burn(geoms):
        return rasterize(((g, 1) for g in geoms), out_shape=(SIZE, SIZE), transform=transform,
                         fill=0, dtype="uint8").astype(bool) if geoms else np.zeros((SIZE, SIZE), bool)

    label = np.zeros((SIZE, SIZE), np.int16)
    label[burn(layers["water"])] = 1
    cloud = binary_dilation(cloud, iterations=CLOUD_BUFFER)
    label[~burn(analysed_geoms) | burn(layers["cloud"]) | ~valid | cloud] = -1
    return label


def write(path: Path, arr: np.ndarray, bounds: tuple, dtype: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "width": SIZE, "height": SIZE, "count": arr.shape[0], "dtype": dtype,
               "crs": f"EPSG:{EPSG}", "transform": rasterio.transform.from_bounds(*bounds, SIZE, SIZE),
               "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N chips with water (demo)")
    args = ap.parse_args()

    analysed, per_date = load_layers()
    analysed_geoms = list(analysed.geometry)
    chips = chip_grid(analysed)
    print(f"{len(chips)} chips on the grid", flush=True)
    if args.limit:
        water_all = gpd.GeoSeries(per_date["2023-03-22"]["water"], crs=EPSG).union_all()
        chips = [c for c in chips if box(*c[1]).intersects(water_all)][: args.limit]

    manifest = OUT / ("manifest_demo.csv" if args.limit else "manifest.csv")
    done = set()
    if manifest.exists():
        with manifest.open() as f:
            done = {(r["date"], r["chip"]) for r in csv.DictReader(f)}
    OUT.mkdir(parents=True, exist_ok=True)
    new = not manifest.exists()
    tok = token()
    with manifest.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["chip", "date", "minx", "miny", "label_source", "label_sensor_date",
                        "water_frac", "dry_frac", "ignore_frac", "mndwi_iou", "s2_sha256", "label_sha256"])
        for i, (cid, bounds) in enumerate(chips):
            for date, layers in per_date.items():
                if (date, cid) in done:
                    continue
                try:
                    s2 = fetch_s2(tok, bounds, date)
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 401:
                        tok = token()
                        s2 = fetch_s2(tok, bounds, date)
                    else:
                        raise
                valid = s2[13].astype(bool)
                label = make_label(bounds, analysed_geoms, layers, valid, thick_cloud(s2))
                if (label == -1).all():
                    continue
                tag = f"{date}_{cid}"
                s2_sha = write(OUT / "s2" / f"{tag}.tif", s2[:13], bounds, "uint16")
                lb_sha = write(OUT / "label" / f"{tag}.tif", label[None], bounds, "int16")
                n = label.size
                w.writerow([cid, date, int(bounds[0]), int(bounds[1]), "UNOSAT FL20230321PER", layers["label_date"],
                            round((label == 1).sum() / n, 4), round((label == 0).sum() / n, 4),
                            round((label == -1).sum() / n, 4), agreement(s2, label), s2_sha, lb_sha])
                f.flush()
            print(f"{i + 1}/{len(chips)} {cid}", flush=True)


if __name__ == "__main__":
    main()
