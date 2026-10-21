"""Expert reference on a scene's grid (RF5) and the agreement metrics shown in the app.

UNOSAT 2023 is rasterized with the same rule as the Peru labels (scripts/build_peru_unosat.py,
`load_layers` / `make_label`): water = the date's flood polygons plus the pre-existing water of
2023-03-22; ignore = outside UNOSAT's analysed extent or inside its cloud-obstruction layer.
The system's own cloud / no-data mask is applied when the metrics are computed, so only pixels
valid for both are scored. Water in the system map: probability > 0.5.
Copernicus EMS 2017 (EMSR199) is rasterized with the rule of the 2017 test labels
(scripts/build_peru_emsr2017.py): EMS was drawn on radar of 23-30 Mar, so only pixels where EMS
and the 2017-04-07 image agree are scored: water = inside EMS water (crisis + hydrography
polygons of the Salinas, Parachique and Piura AOIs) and MNDWI > 0; dry = outside EMS water and
MNDWI <= 0; ignore = disagreement or outside the AOIs. MNDWI comes from B03 and B11 of the
scene's own L1C stack, so building it needs the product archive (pipeline --keep-archive);
the result is cached as reference.tif.
Codes: 0 dry · 1 water · 255 ignore.
"""

import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from floodseg.system.pipeline import DOWNLOADS, NODATA16, SCALE, _write_cog, scene_dir
from floodseg.target_pool import REPO

UNOSAT = REPO / "data" / "unosat2023" / "raw" / "0401" / "FL0230321PER_SHP"
UNOSAT_DATES = {
    "2023-03-22": {"water": ["S2_20230321_FloodExtent_Lambayeque_Piura", "existing_water_20230322"],
                   "cloud": "S2_20230321_CloudObstruction_Lambayeque_Piura"},
    "2023-04-01": {"water": ["S2_20230401_FloodExtent_Lambayeque_Piura", "existing_water_20230322"],
                   "cloud": "S2_20230401_CloudObstruction_Lambayeque_Piura"},
}
ANALYSED = "S2_20230321_AnalysisExtent_Lambayeque_Piura"
EMS = REPO / "data" / "emsr199" / "raw"
EMS_DATE = "2017-04-07"
EMS_AOIS = ["EMSR199_14SALINAS_DELINEATION_OVERVIEW_v1", "EMSR199_11PARACHIQUE_DELINEATION_OVERVIEW-MONIT01_v2",
            "EMSR199_10PIURA_DELINEATION_OVERVIEW-MONIT01_v3"]
EMS_MANIFEST = REPO / "data" / "peru" / "emsr2017" / "manifest.csv"
IGNORE = 255
# 0: outside every chip. test_near_train: 2017 test chips that touch a 2023 peru-train location
# (excluded from the main 2017 number, subset no_2023_train_overlap of scripts/score_frozen.py)
PARTS = {"train": 1, "dev": 2, "test": 3, "buffer": 4, "test_near_train": 5}


def available(date: str) -> bool:
    return date in UNOSAT_DATES or date == EMS_DATE


def source(date: str) -> str:
    return "Copernicus EMS EMSR199" if date == EMS_DATE else "UNOSAT FL20230321PER"


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
    with rasterio.open(d / "mask.tif") as src:
        profile, crs, shape, transform = src.profile, src.crs, (src.height, src.width), src.transform

    def burn(geoms) -> np.ndarray:
        return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)

    if meta["date"] == EMS_DATE:
        ref = _ems(meta, burn, crs)
        _write_cog(path, ref, profile, IGNORE, "nearest")
        return ref
    spec = UNOSAT_DATES[meta["date"]]

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


def _ems(meta: dict, burn, crs) -> np.ndarray:
    from floodseg.system.scene import read_stack
    archive = DOWNLOADS / f"{meta['product']}.zip"
    if not archive.exists():
        raise FileNotFoundError(f"{archive.name} not found: the 2017 reference needs the product archive "
                                f"(python -m floodseg.system.pipeline {meta['tile']} {meta['date']} --keep-archive)")
    stack, _ = read_stack(archive)
    g, sw = stack[2].astype(np.float32), stack[11].astype(np.float32)
    del stack
    wet = (g - sw) / (g + sw + 1e-6) > 0

    def layer(product: str, suffix: str) -> list:
        files = list((EMS / product).glob(f"*{suffix}.shp"))
        return list(gpd.read_file(files[0]).to_crs(crs).geometry) if files else []

    ems = burn([g_ for p in EMS_AOIS for sfx in ("crisis_information_poly", "hydrography_poly") for g_ in layer(p, sfx)])
    inside = burn([g_ for p in EMS_AOIS for g_ in layer(p, "area_of_interest")])
    ref = np.full(ems.shape, IGNORE, np.uint8)
    ref[ems & wet] = 1
    ref[~ems & ~wet] = 0
    ref[~inside] = IGNORE
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
    if date == EMS_DATE:  # every 2017 chip is test
        m = pd.read_csv(EMS_MANIFEST)
        rows = zip(m.chip, np.where(m.touches_2023_train, "test_near_train", "test"))
    else:
        split = pd.read_csv(UNOSAT_SPLIT)
        rows = split[split.date == date][["chip", "split"]].itertuples(index=False)
    for chip, part in rows:
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


def frozen_chip_metrics(group: str = "peru/unet", split: str = "test") -> dict:
    """RF5 check (protocol §3): the system's metric code on the Peru chips of one split, with
    the frozen pass's logits and the chip labels, pooled like scripts/score_frozen.py (per seed:
    water if z > 0, labelled pixels only, counts pooled over all chips; then mean ± std over the
    seeds). Independent counting code, so a match checks the system's metric computation."""
    import pandas as pd
    regime, model = group.split("/")
    if split == "2017-test":  # main 2017 subset: chips that touch no 2023 peru-train location
        root = REPO / "outputs" / "logits_2017" / regime / model
        m = pd.read_csv(EMS_MANIFEST)
        stems = list(m.chip[~m.touches_2023_train])
        labels = EMS_MANIFEST.parent / "label"
    else:
        root = REPO / "outputs" / ("logits_test" if split == "test" else "frozen_check/logits_dev") / regime / model
        u = REPO / "data" / "peru" / "unosat2023"
        stems = (u / "splits" / f"{split}.txt").read_text().split()
        labels = u / "label"
    per_seed = {}
    for seed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tp = fp = fn = 0
        for st in stems:
            with rasterio.open(labels / f"{st}.tif") as src:
                y = src.read(1)
            pred = np.load(seed_dir / f"{st}.npy") > 0
            lab = y != -1
            water = y == 1
            tp += int((pred & water & lab).sum())
            fp += int((pred & ~water & lab).sum())
            fn += int((~pred & water & lab).sum())
        per_seed[seed_dir.name] = {"iou": tp / (tp + fp + fn), "precision": tp / (tp + fp), "recall": tp / (tp + fn)}
    df = pd.DataFrame(per_seed).T
    return {"group": group, "split": f"peru-{split}", "chips": len(stems), "seeds": per_seed,
            **{f"{k}_mean": float(df[k].mean()) for k in df}, **{f"{k}_std": float(df[k].std()) for k in df}}
