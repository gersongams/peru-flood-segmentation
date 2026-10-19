"""Superseded (2026-10-18): the paused chip-based viewer; the system now processes whole tiles
(src/floodseg/system/). Kept for the record. The "viewer data only / no thesis result" rule
below was replaced on 2026-10-16 (METHODS_NOTES "Thesis reframed around a system").

Flood viewer, Stage 1: mosaic the per-chip viewer outputs and write PMTiles per date and layer.

Viewer data only (docs/METHODS_NOTES.md 2026-10-16 "Flood viewer"); the areas written to
stats.json are viewer figures, not thesis results.

Input: outputs/viewer/chips/<event>/<date>/{rgb,label,peru_unet,trhm_unet}/<chip>.tif (UTM 17S).
Each date is reprojected onto the Web Mercator z14 tile grid (≈ 9.5 m pixels here) and cut into
256 px tiles; lower zooms (MINZOOM–13) are 2× averages (data tiles: nearest). Output:
outputs/viewer/tiles/<event>_<date>_<layer>.pmtiles and outputs/viewer/tiles/stats.json,
outputs/viewer/tiles/footprint_<event>_<date>.geojson.

Layers:
  rgb          Sentinel-2 true colour (JPEG)
  water_peru   AI water probability, Peru-trained U-Net ensemble (PNG, blue ramp, p ≥ 0.25)
  water_trhm   same, U-Net trained without Peru labels (training-time histogram matching)
  uncert       seed-ensemble std of the Peru-trained model (PNG, std ≥ 0.05)
  reference    UNOSAT 2023 / EMS 2017 water (PNG)
  errors       Peru-trained model vs reference: false water (red), missed water (amber)
  clouds       thick cloud / no data (PNG, grey)
  data         packed values for the cursor readout (lossless PNG, z ≥ 11):
               R = p_peru·255, G = std·510, B = p_trhm·255,
               A = 255 − code, code: 0 dry ref · 1 water ref · 2 ignore · 3 cloud · 4 no data;
               fully transparent (A = 0) outside the analysed chips

    uv run python scripts/viewer/build_tiles.py
"""

import io
import json
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[2]
CHIPS = REPO / "outputs" / "viewer" / "chips"
OUT = REPO / "outputs" / "viewer" / "tiles"
MAXZOOM, MINZOOM = 14, 8
TS = 256
MERC = CRS.from_epsg(3857)
HALF = 20037508.342789244


def grid(bounds_merc):
    """z14 tile range and the transform of the mosaic aligned to it."""
    w, s, e, n = bounds_merc
    lw, ls = Transformer.from_crs(3857, 4326, always_xy=True).transform(w, s)
    le, ln = Transformer.from_crs(3857, 4326, always_xy=True).transform(e, n)
    tl = mercantile.tile(lw, ln, MAXZOOM)
    br = mercantile.tile(le, ls, MAXZOOM)
    size = 2 * HALF / 2 ** MAXZOOM / TS
    x0 = -HALF + tl.x * TS * size
    y0 = HALF - tl.y * TS * size
    nx, ny = br.x - tl.x + 1, br.y - tl.y + 1
    return tl.x, tl.y, nx, ny, from_origin(x0, y0, size, size)


def mosaic(files, bands, tr, shape, resampling, dtype=np.uint8, fill=0):
    out = np.full((bands, *shape), fill, dtype)
    for f in files:
        with rasterio.open(f) as src:
            for b in range(bands):
                tmp = np.full(shape, fill, dtype)
                reproject(src.read(b + 1), tmp, src_transform=src.transform, src_crs=src.crs, dst_transform=tr,
                          dst_crs=MERC, resampling=resampling, src_nodata=None, dst_nodata=None, init_dest_nodata=False)
                win = rasterio.windows.from_bounds(*transform_bounds(src.crs, MERC, *src.bounds), transform=tr).round_offsets().round_lengths()
                r0, c0 = max(win.row_off, 0), max(win.col_off, 0)
                r1, c1 = min(win.row_off + win.height, shape[0]), min(win.col_off + win.width, shape[1])
                out[b, r0:r1, c0:c1] = tmp[r0:r1, c0:c1]
    return out


def ramp(v, stops):
    """v in [0,1] → RGB by linear interpolation over [(pos, (r,g,b)), ...]."""
    pos = np.array([s[0] for s in stops]); cols = np.array([s[1] for s in stops], np.float32)
    return np.stack([np.interp(v, pos, cols[:, k]) for k in range(3)], -1)


def downsample(rgba: np.ndarray) -> np.ndarray:
    """2× average with alpha premultiplication; rgba: (H, W, 4) uint8 → (H/2, W/2, 4)."""
    h, w = rgba.shape[0] // 2 * 2, rgba.shape[1] // 2 * 2
    a = rgba[:h, :w].astype(np.float32)
    al = a[..., 3:4] / 255
    pre = a[..., :3] * al
    pre = pre.reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))
    al2 = al.reshape(h // 2, 2, w // 2, 2, 1).mean((1, 3))
    rgb = np.where(al2 > 0, pre / np.maximum(al2, 1e-6), 0)
    return np.concatenate([rgb, al2 * 255], -1).round().clip(0, 255).astype(np.uint8)


def write_pmtiles(path, levels, x0, y0, fmt, meta, minzoom=MINZOOM):
    """levels: {zoom: (H, W, 4) uint8 array aligned to the z grid at tile origin (x0>>dz, y0>>dz)}."""
    entries = []
    for z, arr in levels.items():
        if z < minzoom:
            continue
        dz = MAXZOOM - z
        tx0, ty0 = x0 >> dz, y0 >> dz
        # offset of the array origin inside the first tile at this zoom
        offx = (x0 * TS >> dz) - tx0 * TS
        offy = (y0 * TS >> dz) - ty0 * TS
        H, W = arr.shape[:2]
        pad = np.zeros((offy + H + TS, offx + W + TS, 4), np.uint8)
        pad[offy:offy + H, offx:offx + W] = arr
        for ty in range((offy + H + TS - 1) // TS):
            for tx in range((offx + W + TS - 1) // TS):
                t = pad[ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS]
                if t.shape[:2] != (TS, TS) or not t[..., 3].any():
                    continue
                buf = io.BytesIO()
                if fmt == "jpg":
                    Image.fromarray(t[..., :3]).save(buf, "JPEG", quality=82)
                else:
                    Image.fromarray(t, "RGBA").save(buf, "PNG", optimize=True)
                entries.append((zxy_to_tileid(z, tx0 + tx, ty0 + ty), buf.getvalue()))
    entries.sort()
    with open(path, "wb") as f:
        w = Writer(f)
        for tid, data in entries:
            w.write_tile(tid, data)
        lon0, lat0, lon1, lat1 = meta["bounds"]
        w.finalize({
            "tile_type": TileType.JPEG if fmt == "jpg" else TileType.PNG,
            "tile_compression": Compression.NONE,
            "min_zoom": minzoom, "max_zoom": MAXZOOM,
            "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(lat0 * 1e7),
            "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(lat1 * 1e7),
            "center_zoom": 10, "center_lon_e7": int((lon0 + lon1) / 2 * 1e7), "center_lat_e7": int((lat0 + lat1) / 2 * 1e7),
        }, {"name": meta["name"], "attribution": meta["attribution"]})
    return len(entries)


def pyramid(rgba, nearest=False):
    levels = {MAXZOOM: rgba}
    cur = rgba
    for z in range(MAXZOOM - 1, MINZOOM - 1, -1):
        cur = cur[::2, ::2] if nearest else downsample(cur)
        levels[z] = cur
    return levels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    attribution = "Contains modified Copernicus Sentinel data; reference: UNOSAT (CC BY-SA) / Copernicus EMS"
    for ev_dir in sorted(p for p in CHIPS.iterdir() if p.is_dir()):
        for date_dir in sorted(p for p in ev_dir.iterdir() if p.is_dir()):
            ev, date = ev_dir.name, date_dir.name
            chips = sorted(p.stem for p in (date_dir / "rgb").glob("*.tif"))
            if not chips or not all((date_dir / m / f"{c}.tif").exists() for m in ("peru_unet", "trhm_unet") for c in chips):
                print(f"skip {ev} {date}: incomplete"); continue
            with rasterio.open(date_dir / "rgb" / f"{chips[0]}.tif") as src:
                crs = src.crs
            boxes = []
            for c in chips:
                with rasterio.open(date_dir / "rgb" / f"{c}.tif") as src:
                    boxes.append(box(*src.bounds))
            fp = unary_union(boxes)
            to_merc = Transformer.from_crs(crs, 3857, always_xy=True).transform
            to_ll = Transformer.from_crs(crs, 4326, always_xy=True).transform
            x0, y0, nx, ny, tr = grid(shp_transform(to_merc, fp).bounds)
            shape = (ny * TS, nx * TS)
            files = lambda m: [date_dir / m / f"{c}.tif" for c in chips]
            print(f"{ev} {date}: {len(chips)} chips, mosaic {shape}", flush=True)

            rgb = mosaic(files("rgb"), 3, tr, shape, Resampling.bilinear)
            pe = mosaic(files("peru_unet"), 3, tr, shape, Resampling.bilinear)
            pe[2] = mosaic(files("peru_unet"), 3, tr, shape, Resampling.nearest)[2]
            th = mosaic(files("trhm_unet"), 1, tr, shape, Resampling.bilinear)[0]
            lab = mosaic(files("label"), 1, tr, shape, Resampling.nearest, fill=254)[0]  # 254 = outside chips
            inside = lab != 254
            p, sd, msk = pe[0] / 255, pe[1] / 510, pe[2]
            valid = inside & (msk == 0)

            def rgba(color, alpha):
                return np.concatenate([color.round().clip(0, 255).astype(np.uint8),
                                       (alpha * 255).round().clip(0, 255).astype(np.uint8)[..., None]], -1)

            blue = [(0, (150, 210, 245)), (0.5, (60, 140, 220)), (1, (14, 60, 160))]
            layers = {}
            layers["rgb"] = (np.concatenate([rgb.transpose(1, 2, 0), (inside * 255).astype(np.uint8)[..., None]], -1), "jpg", False)
            for key, prob in (("water_peru", p), ("water_trhm", th / 255)):
                a = np.where(valid, np.clip((prob - 0.25) / 0.5, 0, 1) * 0.85, 0)
                layers[key] = (rgba(ramp(prob, blue), a), "png", False)
            a = np.where(valid, np.clip((sd - 0.05) / 0.25, 0, 1) * 0.9, 0)
            layers["uncert"] = (rgba(ramp(np.clip(sd / 0.3, 0, 1), [(0, (253, 200, 120)), (0.5, (230, 110, 60)), (1, (150, 30, 90))]), a), "png", False)
            ref = (lab == 1) & inside
            layers["reference"] = (rgba(np.broadcast_to(np.array([0, 170, 160], np.float32), (*shape, 3)), ref * 0.6), "png", False)
            pw, rv = p >= 0.5, valid & (lab <= 1)
            fpos, fneg = rv & pw & (lab == 0), rv & ~pw & (lab == 1)
            col = np.zeros((*shape, 3), np.float32); col[fpos] = (228, 70, 50); col[fneg] = (245, 170, 20)
            layers["errors"] = (rgba(col, (fpos | fneg) * 0.85), "png", False)
            cl = inside & (msk > 0)
            layers["clouds"] = (rgba(np.broadcast_to(np.array([235, 235, 235], np.float32), (*shape, 3)), cl * 0.55), "png", False)
            code = np.where(msk == 2, 4, np.where(msk == 1, 3, np.where(lab == 255, 2, np.where(lab == 1, 1, 0))))
            data = np.stack([pe[0], pe[1], th, np.where(inside, 255 - code, 0)], -1).astype(np.uint8)
            layers["data"] = (data, "png", True)

            ll = transform_bounds(crs, "EPSG:4326", *fp.bounds)
            n_tiles = {}
            for name, (arr, fmt, nearest) in layers.items():
                path = OUT / f"{ev}_{date}_{name}.pmtiles"
                n_tiles[name] = write_pmtiles(path, pyramid(arr, nearest), x0, y0, fmt,
                                              {"name": f"{ev} {date} {name}", "bounds": ll, "attribution": attribution},
                                              minzoom=11 if name == "data" else MINZOOM)
                print(f"  {name}: {n_tiles[name]} tiles, {path.stat().st_size / 1e6:.1f} MB", flush=True)

            # Areas on the native UTM chips (10 m pixels), not on the Mercator mosaic.
            px_km2 = 1e-4
            a_peru = a_trhm = a_ref = a_cloud = a_valid = 0
            for c in chips:
                with rasterio.open(date_dir / "peru_unet" / f"{c}.tif") as s:
                    q = s.read()
                with rasterio.open(date_dir / "trhm_unet" / f"{c}.tif") as s:
                    t = s.read(1)
                with rasterio.open(date_dir / "label" / f"{c}.tif") as s:
                    y = s.read(1)
                v = q[2] == 0
                a_valid += v.sum(); a_cloud += (q[2] == 1).sum()
                a_peru += (v & (q[0] >= 128)).sum(); a_trhm += (v & (t >= 128)).sum(); a_ref += (y == 1).sum()
            stats[f"{ev}_{date}"] = dict(
                event=ev, date=date, chips=len(chips), bounds=ll,
                area_km2=dict(analysed=round(len(chips) * 26.2144, 1), valid=round(a_valid * px_km2, 1), cloud=round(a_cloud * px_km2, 1),
                              water_peru_unet=round(a_peru * px_km2, 1), water_trhm_unet=round(a_trhm * px_km2, 1),
                              water_reference=round(a_ref * px_km2, 1)),
                tiles=n_tiles)
            geo = {"type": "Feature", "properties": {"event": ev, "date": date},
                   "geometry": mapping(shp_transform(to_ll, fp.simplify(50)))}
            (OUT / f"footprint_{ev}_{date}.geojson").write_text(json.dumps(geo))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v["area_km2"] for k, v in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
