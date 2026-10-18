"""Expert reference on a scene's grid (RF5) and the agreement metrics shown in the app.

UNOSAT 2023 is rasterized with the same rule as the Peru labels (scripts/build_peru_unosat.py,
`load_layers` / `make_label`): water = the date's flood polygons plus the pre-existing water of
2023-03-22; ignore = outside UNOSAT's analysed extent or inside its cloud-obstruction layer.
The system's own cloud / no-data mask is applied when the metrics are computed, so only pixels
valid for both are scored. Water in the system map: probability > 0.5.
Codes: 0 dry · 1 water · 255 ignore.
"""

import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from floodseg.system.pipeline import NODATA16, SCALE, _write_cog, scene_dir
from floodseg.target_pool import REPO

UNOSAT = REPO / "data" / "unosat2023" / "raw" / "0401" / "FL0230321PER_SHP"
UNOSAT_DATES = {
    "2023-03-22": {"water": ["S2_20230321_FloodExtent_Lambayeque_Piura", "existing_water_20230322"],
                   "cloud": "S2_20230321_CloudObstruction_Lambayeque_Piura"},
    "2023-04-01": {"water": ["S2_20230401_FloodExtent_Lambayeque_Piura", "existing_water_20230322"],
                   "cloud": "S2_20230401_CloudObstruction_Lambayeque_Piura"},
}
ANALYSED = "S2_20230321_AnalysisExtent_Lambayeque_Piura"
IGNORE = 255
PARTS = {"train": 1, "dev": 2, "test": 3, "buffer": 4}  # 0: outside every chip


def available(date: str) -> bool:
    return date in UNOSAT_DATES


def _layer(name: str, crs) -> list:
    return list(gpd.read_file(UNOSAT / f"{name}.shp").to_crs(crs).geometry)


def reference(scene: str) -> np.ndarray:
    """(H, W) uint8 reference on the scene grid; cached as reference.tif."""
    d = scene_dir(*scene.split("_"))
    path = d / "reference.tif"
    if path.exists():
        with rasterio.open(path) as src:
            return src.read(1)
    meta = json.loads((d / "meta.json").read_text())
    spec = UNOSAT_DATES[meta["date"]]
    with rasterio.open(d / "mask.tif") as src:
        profile, crs, shape, transform = src.profile, src.crs, (src.height, src.width), src.transform

    def burn(geoms) -> np.ndarray:
        return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)

    w22 = gpd.read_file(UNOSAT / "S2_20230321_WaterExtent_Lambayeque_Piura.shp").to_crs(crs).union_all()
    f22 = gpd.read_file(UNOSAT / "S2_20230321_FloodExtent_Lambayeque_Piura.shp").to_crs(crs).union_all()
    water = []
    for name in spec["water"]:
        water += [w22.difference(f22)] if name == "existing_water_20230322" else _layer(name, crs)
    ref = np.zeros(shape, np.uint8)
    ref[burn(water)] = 1
    ref[~burn(_layer(ANALYSED, crs)) | burn(_layer(spec["cloud"], crs))] = IGNORE
    _write_cog(path, ref, profile, IGNORE, "nearest")
    return ref


def partitions(scene: str) -> np.ndarray:
    """(H, W) uint8 Peru-split partition of each pixel for the scene's date (PARTS codes), from the
    chip names in the split file (lower-left corner, 5 120 m)."""
    import pandas as pd
    from floodseg.peru import UNOSAT_SPLIT
    d = scene_dir(*scene.split("_"))
    date = scene.split("_")[1]
    with rasterio.open(d / "mask.tif") as src:
        t, h, w = src.transform, src.height, src.width
    out = np.zeros((h, w), np.uint8)
    split = pd.read_csv(UNOSAT_SPLIT)
    for chip, part in split[split.date == date][["chip", "split"]].itertuples(index=False):
        x0, y0 = (int(v[1:]) for v in chip.split("_"))
        c0, r0 = int(round((x0 - t.c) / t.a)), int(round((t.f - (y0 + 5120)) / -t.e))
        if c0 >= w or r0 >= h or c0 + 512 <= 0 or r0 + 512 <= 0:
            continue
        out[max(r0, 0):r0 + 512, max(c0, 0):c0 + 512] = PARTS[part]
    return out


def metrics(scene: str, bbox_utm: tuple[float, float, float, float] | None = None, part: str | None = None) -> dict:
    """IoU, precision, recall and areas (km²) over pixels valid in both the system mask and the
    reference, optionally inside a UTM bounding box (W, S, E, N) and/or one Peru-split partition."""
    d = scene_dir(*scene.split("_"))
    ref = reference(scene)
    with rasterio.open(d / "probability.tif") as src:
        win = from_bounds(*bbox_utm, src.transform).round_offsets().round_lengths() if bbox_utm else None
        if win is not None:
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        prob = src.read(1, window=win)
    with rasterio.open(d / "mask.tif") as src:
        mask = src.read(1, window=win)
    if win is not None:
        r0, c0 = int(win.row_off), int(win.col_off)
        ref = ref[r0:r0 + prob.shape[0], c0:c0 + prob.shape[1]]
    parts = partitions(scene)
    if win is not None:
        parts = parts[r0:r0 + prob.shape[0], c0:c0 + prob.shape[1]]
    ok = (mask == 0) & (ref != IGNORE) & (prob != NODATA16)
    share = {k: float(((parts == v) & ok).sum() / max(ok.sum(), 1)) for k, v in PARTS.items()}
    if part:
        ok &= parts == PARTS[part]
    pred = (prob > SCALE // 2) & ok
    true = (ref == 1) & ok
    tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
    km2 = 100 / 1e6
    return {
        "pixels": int(ok.sum()),
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp_km2": tp * km2, "fp_km2": fp * km2, "fn_km2": fn * km2,
        "reference_water_km2": int(true.sum()) * km2,
        "partition_share": share,  # share of the scored pixels in each split partition (before `part`)
    }
