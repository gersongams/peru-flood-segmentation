"""Build the cross-event Peru test set from Copernicus EMS EMSR199 (2017).

EMS delineations were drawn on radar of 23-30 Mar 2017; the nearest clear
Sentinel-2 image is 2017-04-07 (docs/peru_dataset_proposal.md §3b), so labels
keep only pixels where EMS and the image agree:

    1  inside EMS water (flood + reference water) and MNDWI > 0
    0  outside EMS water and MNDWI <= 0
   -1  disagreement, outside the AOI, thick cloud, or S2 no-data

AOIs: Salinas, Parachique, Piura (Piura Norte is weak, Trujillo has almost no
standing water on 2017-04-07). These overlap the UNOSAT 2023 area, so each
chip records whether it touches a 2023 `peru-train` location.

Writes data/peru/emsr2017/{s2,label}/<chip>.tif and manifest.csv.

    uv run python scripts/build_peru_emsr2017.py --limit 4   # local demo
    uv run python scripts/build_peru_emsr2017.py
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_peru_unosat import CHIP, CLOUD_BUFFER, EPSG, SIZE, fetch_s2, thick_cloud, write  # noqa: E402
from peru_cloud_survey import REPO, token  # noqa: E402

RAW = REPO / "data" / "emsr199" / "raw"
OUT = REPO / "data" / "peru" / "emsr2017"
SPLIT_2023 = REPO / "data" / "peru" / "unosat2023" / "splits" / "split.csv"
DATE = "2017-04-07"
MIN_AOI = 0.5
AOIS = {
    "salinas": "EMSR199_14SALINAS_DELINEATION_OVERVIEW_v1",
    "parachique": "EMSR199_11PARACHIQUE_DELINEATION_OVERVIEW-MONIT01_v2",
    "piura": "EMSR199_10PIURA_DELINEATION_OVERVIEW-MONIT01_v3",
}


def read(product: str, suffix: str) -> gpd.GeoDataFrame | None:
    files = glob.glob(str(RAW / product / f"*{suffix}.shp"))
    return gpd.read_file(files[0]).to_crs(EPSG) if files else None


def train_2023_cells() -> set:
    s = pd.read_csv(SPLIT_2023)
    cells = s.chip[s.split == "train"].str.extract(r"E(\d+)_N(\d+)").astype(int)
    return {(x // CHIP, y // CHIP) for x, y in zip(cells[0], cells[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only N chips with water per AOI (demo)")
    args = ap.parse_args()

    train_cells = train_2023_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / ("manifest_demo.csv" if args.limit else "manifest.csv")
    tok = token()
    seen: set = set()
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip", "aoi", "date", "minx", "miny", "label_source", "label_sensor_dates",
                    "water_frac", "dry_frac", "ignore_frac", "disagree_frac", "touches_2023_train",
                    "s2_sha256", "label_sha256"])
        for name, product in AOIS.items():
            aoi = read(product, "area_of_interest").union_all()
            water = [g for sfx in ("crisis_information_poly", "hydrography_poly")
                     if (d := read(product, sfx)) is not None for g in d.geometry]
            water_u = gpd.GeoSeries(water, crs=EPSG).union_all()
            minx, miny, maxx, maxy = aoi.bounds
            cells = [(x, y) for x in np.arange(np.floor(minx / CHIP) * CHIP, maxx, CHIP)
                     for y in np.arange(np.floor(miny / CHIP) * CHIP, maxy, CHIP)
                     if box(x, y, x + CHIP, y + CHIP).intersection(aoi).area / CHIP**2 >= MIN_AOI]
            if args.limit:
                cells = [c for c in cells if box(c[0], c[1], c[0] + CHIP, c[1] + CHIP).intersects(water_u)][: args.limit]
            for x, y in cells:
                cid = f"E{int(x)}_N{int(y)}"
                if cid in seen:  # AOIs overlap; keep the first
                    continue
                seen.add(cid)
                bounds = (x, y, x + CHIP, y + CHIP)
                try:
                    s2 = fetch_s2(tok, bounds, DATE)
                except requests.HTTPError as e:
                    if e.response is None or e.response.status_code != 401:
                        raise
                    tok = token()
                    s2 = fetch_s2(tok, bounds, DATE)
                T = rasterio.transform.from_bounds(*bounds, SIZE, SIZE)
                ems = rasterize(((g, 1) for g in water), out_shape=(SIZE, SIZE), transform=T, fill=0).astype(bool) \
                    if water else np.zeros((SIZE, SIZE), bool)
                inside = rasterize([(aoi, 1)], out_shape=(SIZE, SIZE), transform=T, fill=0).astype(bool)
                g, s = s2[2].astype(float), s2[11].astype(float)
                wet = (g - s) / (g + s + 1e-6) > 0
                label = np.full((SIZE, SIZE), -1, np.int16)
                label[ems & wet] = 1
                label[~ems & ~wet] = 0
                cloud = binary_dilation(thick_cloud(s2), iterations=CLOUD_BUFFER)
                bad = ~inside | cloud | ~s2[13].astype(bool)
                label[bad] = -1
                if (label == -1).all():
                    continue
                s2_sha = write(OUT / "s2" / f"{cid}.tif", s2[:13], bounds, "uint16")
                lb_sha = write(OUT / "label" / f"{cid}.tif", label[None], bounds, "int16")
                n = label.size
                ix, iy = int(x) // CHIP, int(y) // CHIP
                touches = any((ix + dx, iy + dy) in train_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                w.writerow([cid, name, DATE, int(x), int(y), f"EMSR199 {product}", "2017-03-23..30 (radar)",
                            round((label == 1).sum() / n, 4), round((label == 0).sum() / n, 4),
                            round((label == -1).sum() / n, 4), round(((ems != wet) & ~bad).sum() / n, 4),
                            touches, s2_sha, lb_sha])
                f.flush()
                print(f"{name} {cid}", flush=True)


if __name__ == "__main__":
    main()
