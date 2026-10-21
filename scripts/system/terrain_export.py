"""3D terrain spike: export a zone of Bajo Piura / Sechura for the Three.js view.

Inputs: Copernicus DEM GLO-30 tiles in data/aux/cop_dem (elevation), and the system scenes of
17MNP in outputs/system/scenes (Sentinel-2 RGB preview and water probability per date).
Outputs in app/public/terrain/piura/ (git-ignored, derived data):
  height.bin        float32 little-endian, GRID × GRID, metres, north-up, row 0 = north
  rgb.jpg           Sentinel-2 true colour of the reference date (TEX × TEX)
  water_<date>.png  water probability on valid pixels (0 elsewhere), per date (TEX × TEX); the
                    page draws the p = 0.5 contour, so edges are smooth below the pixel size
  meta.json         bounds (UTM 17S and WGS84), size in metres, grid sizes, dates, height range

    uv run python scripts/system/terrain_export.py
"""

import glob
import json

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

from floodseg.target_pool import REPO

NAME = "piura"
# Bajo Piura / Sechura around Laguna La Niña, UTM 17S (EPSG:32717), 50 × 50 km.
BOUNDS = (515_000, 9_335_000, 565_000, 9_385_000)
EPSG = 32717
GRID = 512   # height samples per side (~98 m)
TEX = 4096   # texture pixels per side (~12 m)
SCENES = {"2023-01-26": "pre-evento", "2023-03-22": "22 mar 2023", "2023-04-01": "1 abr 2023"}
RGB_DATE = "2023-01-26"  # clear pre-event image as the ground texture
OUT = REPO / "app" / "public" / "terrain" / NAME


def warp(src_arrays, src_transform, src_crs, size: int, resampling, dtype="float32", nodata=None):
    w, s, e, n = BOUNDS
    dst = np.zeros((src_arrays.shape[0], size, size), dtype)
    tr = rasterio.transform.from_bounds(w, s, e, n, size, size)
    for i in range(src_arrays.shape[0]):
        reproject(src_arrays[i], dst[i], src_transform=src_transform, src_crs=src_crs, dst_transform=tr,
                  dst_crs=f"EPSG:{EPSG}", resampling=resampling, src_nodata=nodata)
    return dst


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w, s, e, n = BOUNDS

    # Elevation: merge the GLO-30 tiles, warp to the zone grid (bilinear).
    srcs = [rasterio.open(p) for p in glob.glob(str(REPO / "data" / "aux" / "cop_dem" / "*.tif"))]
    dem, dem_tr = merge(srcs)
    h = warp(dem.astype(np.float32), dem_tr, srcs[0].crs, GRID, Resampling.bilinear)[0]
    h = np.maximum(h, 0)  # sea at 0 m
    h.astype("<f4").tofile(OUT / "height.bin")

    # Texture: Sentinel-2 RGB preview of the scene.
    scene = REPO / "outputs" / "system" / "scenes" / f"17MNP_{RGB_DATE}"
    win_bounds = BOUNDS
    with rasterio.open(scene / "rgb.tif") as src:
        win = rasterio.windows.from_bounds(*win_bounds, src.transform)
        rgb = src.read(window=win, out_shape=(3, TEX, TEX), resampling=Resampling.average)
    Image.fromarray(np.moveaxis(rgb, 0, -1)).save(OUT / "rgb.jpg", quality=88)

    # Water per date: probability on valid pixels; the page thresholds it at 0.5 (km² below: p > 0.5).
    dates = []
    for date, label in SCENES.items():
        d = REPO / "outputs" / "system" / "scenes" / f"17MNP_{date}"
        if not (d / "probability.tif").exists():
            continue
        with rasterio.open(d / "probability.tif") as src, rasterio.open(d / "mask.tif") as m:
            win = rasterio.windows.from_bounds(*win_bounds, src.transform)
            p = src.read(1, window=win)
            mk = m.read(1, window=win)
        ok = (p != 65535) & (mk == 0)
        water = ((p > 5000) & ok).astype(np.float32)
        prob = np.where(ok, p / 10000, 0).astype(np.float32)
        img = Image.fromarray(np.round(prob * 255).astype(np.uint8)).resize((TEX, TEX), Image.BILINEAR)
        img.save(OUT / f"water_{date}.png")
        dates.append({"date": date, "label": label, "water_km2": round(float(water.sum()) * 100 / 1e6, 1)})

    t = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(w, s)
    lon1, lat1 = t.transform(e, n)
    meta = {"name": NAME, "title": "Bajo Piura y Sechura", "bounds_utm": BOUNDS, "epsg": EPSG,
            "bounds_wgs84": [lon0, lat0, lon1, lat1], "size_m": [e - w, n - s], "grid": GRID, "tex": TEX,
            "height_min": float(h.min()), "height_max": float(h.max()), "rgb_date": RGB_DATE, "dates": dates,
            "sources": ["Copernicus DEM GLO-30", "Copernicus Sentinel-2 L1C (modified)", "U-Net (Perú), 3 semillas"]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
