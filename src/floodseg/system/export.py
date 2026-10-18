"""Export of a scene (RF7): GeoTIFFs of probability, uncertainty, flood and mask, and the flood
polygons as GeoJSON (WGS84), zipped with a metadatos.json.

The flood map is the RF6 quantity (new water vs the pre-event scene, valid in both). Without a
pre-event scene, the export offers the water map instead and says so in the metadata.
"""

import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds, transform as win_transform
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from floodseg.system import preevent
from floodseg.system.pipeline import NODATA16, SCALE, scene_dir

LAYERS = ("probability", "uncertainty", "flood", "mask", "polygons")


def _window(src, bbox_utm):
    if bbox_utm is None:
        return Window(0, 0, src.width, src.height)
    w = from_bounds(*bbox_utm, src.transform).round_offsets().round_lengths()
    return w.intersection(Window(0, 0, src.width, src.height))


def _tif(arr: np.ndarray, profile: dict, transform, nodata, tags: dict) -> bytes:
    p = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype=arr.dtype.name,
             crs=profile["crs"], transform=transform, nodata=nodata, compress="DEFLATE", predictor=2, tiled=True)
    with rasterio.MemoryFile() as mf:
        with mf.open(**p) as dst:
            dst.write(arr, 1)
            dst.update_tags(**tags)
        return mf.read()


def build(scene: str, layers: list[str], ref_scene: str | None, bbox_utm=None) -> tuple[bytes, dict]:
    d = scene_dir(*scene.split("_"))
    meta = json.loads((d / "meta.json").read_text())
    with rasterio.open(d / "probability.tif") as src:
        win = _window(src, bbox_utm)
        profile, tr = src.profile, win_transform(win, src.transform)
        prob = src.read(1, window=win)
    with rasterio.open(d / "mask.tif") as src:
        mask = src.read(1, window=win)
    if ref_scene:
        fl = preevent.flood_mask(scene, ref_scene, window=win)
        flood = (fl == 1).astype(np.uint8)
        flood_kind = "área inundada: agua nueva frente a la imagen pre-evento, válida en ambas fechas"
    else:
        flood = ((prob > SCALE // 2) & (prob != NODATA16) & (mask == 0)).astype(np.uint8)
        flood_kind = "agua detectada (sin imagen pre-evento: no es área inundada)"
    flood[mask != 0] = 255
    stem = f"{scene}"
    files, buf = {}, io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if "probability" in layers:
            z.writestr(f"{stem}_probabilidad.tif", _tif(prob, profile, tr, NODATA16, {"scale_factor": 1 / SCALE, "meaning": "water probability, seed mean"}))
            files["probabilidad"] = "uint16, valor × 1e-4 = probabilidad; 65535 = sin dato"
        if "uncertainty" in layers:
            with rasterio.open(d / "uncertainty.tif") as src:
                unc = src.read(1, window=win)
            z.writestr(f"{stem}_incertidumbre.tif", _tif(unc, profile, tr, NODATA16, {"scale_factor": 1 / SCALE, "meaning": "seed standard deviation"}))
            files["incertidumbre"] = "uint16, valor × 1e-4 = desviación estándar entre semillas; 65535 = sin dato"
        if "flood" in layers:
            z.writestr(f"{stem}_inundacion.tif", _tif(flood, profile, tr, 255, {"meaning": flood_kind}))
            files["inundacion"] = f"uint8, 1 = {flood_kind}; 0 = no; 255 = nube o sin dato"
        if "mask" in layers:
            z.writestr(f"{stem}_nubes.tif", _tif(mask, profile, tr, None, {"meaning": "0 valid, 1 thick cloud, 2 no data"}))
            files["nubes"] = "uint8, 0 = válido, 1 = nube espesa, 2 = sin dato"
        if "polygons" in layers:
            to_wgs = Transformer.from_crs(profile["crs"], "EPSG:4326", always_xy=True).transform
            feats = []
            for geom, v in shapes(flood, mask=flood == 1, transform=tr, connectivity=8):
                g = shp_transform(to_wgs, shape(geom))
                feats.append({"type": "Feature", "properties": {"area_km2": round(shape(geom).area / 1e6, 6)}, "geometry": mapping(g)})
            z.writestr(f"{stem}_inundacion.geojson", json.dumps({"type": "FeatureCollection", "name": f"{stem}_inundacion", "features": feats}))
            files["poligonos"] = f"GeoJSON WGS 84 (EPSG:4326), {len(feats)} polígonos; {flood_kind}"
        info = {
            "escena": scene, "producto": meta["product"], "fecha": meta["date"], "tesela": meta["tile"],
            "imagen_pre_evento": ref_scene.split("_")[1] if ref_scene else None,
            "modelo": meta["model_group"], "pesos": meta["checkpoints"], "regla_agua": meta["water_rule"],
            "crs_rasteres": str(profile["crs"]), "extension": "vista" if bbox_utm else "tesela completa",
            "archivos": files, "version_sistema": meta["git_commit"],
            "exportado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "atribucion": "Contiene datos modificados de Copernicus Sentinel (2023). Prototipo de investigación.",
        }
        z.writestr("metadatos.json", json.dumps(info, indent=2, ensure_ascii=False))
    return buf.getvalue(), info
