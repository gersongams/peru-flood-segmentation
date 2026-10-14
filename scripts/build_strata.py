"""Per-pixel stratum layers for the failure analysis (task 6.1), on each chip's grid.

For every UNOSAT 2023 chip of the given splits, writes
data/peru/strata/unosat2023/<date>_<chip>.npz with
    landcover   ESA WorldCover 2021 v200 class (uint8; 0 = no data), nearest
    elev        Copernicus DEM GLO-30 elevation in m (int16), bilinear
    slope       slope in degrees from the resampled DEM (uint8)
    existing    pre-event water: UNOSAT WaterExtent_20230321 minus FloodExtent_20230321
                (bool; the same layer for both dates, see build_peru_unosat.py)
    cloud_dist  distance in pixels to the nearest thick-cloud pixel (thick_cloud rule
                of build_peru_unosat.py, before dilation), capped at 255 (uint8)
    ndwi        McFeeters NDWI (B03 - B08) / (B03 + B08) x 100 (int8); separates open
                water (high) from turbid, shallow or vegetated water (low)

Water-body size and distance to the water edge come from the label itself and are
computed in the analysis. Sources: data/aux/worldcover (s3 esa-worldcover v200/2021),
data/aux/cop_dem (s3 copernicus-dem-30m), docs/EXPERIMENT_LOG.md.

    uv run python scripts/build_strata.py dev
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_peru_unosat import EPSG, OUT, RES, SIZE, layer, thick_cloud  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUX = REPO / "data" / "aux"
DEST = REPO / "data" / "peru" / "strata" / "unosat2023"


def warp(paths: list[Path], bounds: tuple, resampling: Resampling, dtype: str) -> np.ndarray:
    """Mosaic the tiles onto the chip grid; later tiles fill earlier gaps."""
    T = rasterio.transform.from_bounds(*bounds, SIZE, SIZE)
    out = np.zeros((SIZE, SIZE), dtype)
    filled = np.zeros((SIZE, SIZE), bool)
    for p in paths:
        with rasterio.open(p) as src, WarpedVRT(src, crs=f"EPSG:{EPSG}", transform=T, width=SIZE,
                                                height=SIZE, resampling=resampling, nodata=0) as vrt:
            a = vrt.read(1)
        new = (a != 0) & ~filled
        out[new] = a[new]
        filled |= new
    return out


def main() -> None:
    splits = sys.argv[1:] or ["dev"]
    m = pd.read_csv(OUT / "manifest.csv").merge(pd.read_csv(OUT / "splits" / "split.csv"), on=["chip", "date"])
    m = m[m.split.isin(splits)]
    w22 = layer("S2_20230321_WaterExtent_Lambayeque_Piura").union_all()
    f22 = layer("S2_20230321_FloodExtent_Lambayeque_Piura").union_all()
    existing = gpd.GeoSeries([w22.difference(f22)], crs=EPSG).explode().tolist()
    wc = sorted((AUX / "worldcover").glob("*.tif"))
    dem = sorted((AUX / "cop_dem").glob("*.tif"))
    DEST.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(m.itertuples()):
        stem = f"{r.date}_{r.chip}"
        bounds = (r.minx, r.miny, r.minx + RES * SIZE, r.miny + RES * SIZE)
        T = rasterio.transform.from_bounds(*bounds, SIZE, SIZE)
        lc = warp(wc, bounds, Resampling.nearest, "uint8")
        z = warp(dem, bounds, Resampling.bilinear, "float32")
        gy, gx = np.gradient(z, RES)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        ex = rasterize(((g, 1) for g in existing), out_shape=(SIZE, SIZE), transform=T, fill=0).astype(bool)
        with rasterio.open(OUT / "s2" / f"{stem}.tif") as src:
            s2 = src.read()
        cloud = thick_cloud(s2)
        g, n = s2[2].astype(float), s2[7].astype(float)
        ndwi = np.clip(np.round(100 * (g - n) / (g + n + 1e-6)), -100, 100).astype(np.int8)
        cd = distance_transform_edt(~cloud) if cloud.any() else np.full((SIZE, SIZE), 255.0)
        np.savez_compressed(DEST / f"{stem}.npz", landcover=lc, elev=np.round(z).astype(np.int16),
                            slope=np.clip(slope, 0, 90).astype(np.uint8), existing=ex,
                            cloud_dist=np.clip(cd, 0, 255).astype(np.uint8), ndwi=ndwi)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(m)}", flush=True)
    print(f"wrote {len(m)} chips to {DEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
