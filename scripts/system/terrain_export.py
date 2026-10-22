"""3D terrain spike: export a zone of Bajo Piura / Sechura for the Three.js view.

Inputs: Copernicus DEM GLO-30 tiles in data/aux/cop_dem (elevation), and the system scenes of
17MNP in outputs/system/scenes (Sentinel-2 RGB preview and water probability per date).
Outputs in app/public/terrain/piura/ (git-ignored, derived data):
  height.bin        float32 little-endian, GRID × GRID, metres, north-up, row 0 = north
  rgb.jpg           Sentinel-2 true colour of the reference date (TEX × TEX)
  water_<date>.png  per analysed date (TEX × TEX): R = water probability on valid pixels (0
                    elsewhere; the page draws the p = 0.5 contour, so edges are smooth below the
                    pixel size), G = 255 where the system has no result (cloud or no data)
  meta.json         bounds (UTM 17S and WGS84), size in metres, grid sizes, height range, and per
                    analysed date the zone statistics below
  weather.json      daily precipitation (ERA5, zone centre) and river discharge (GloFAS, Río Piura
                    cell nearest Piura city), Jan–May of each event year, via Open-Meteo

Zone statistics per analysed date (water: p > 0.5; valid: system mask 0):
  water_km2      water on valid pixels; nodata_share: share of the zone without a result
  flooded_km2    RF6 flooded area against the event's pre-event scene, Área((W_t \ W_0) ∩ V_t ∩ V_0)
                 (2023 only: no sourced 2017 event start yet, so no 2017 pre-event scene)
  comparable     nodata_share <= COMPARABLE: totals of cloudier dates cover too little of the zone
  spread         between consecutive comparable dates of the same event, on pixels valid at both
                 dates and at the pre-event date: newly flooded km² (flooded at t, not at t−1),
                 receded km², net change and net km² per day; `observed_share` is that common
                 valid share of the zone. Cloud gaps make each step a lower bound on change.
  reference      agreement with the expert reference inside the zone (floodseg.system.reference)

    uv run python scripts/system/terrain_export.py
"""

import glob
import json

import numpy as np
import rasterio
import requests
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
TILE = "17MNP"
EVENTS = {"2017": {"pre": None}, "2023": {"pre": "2023-01-26"}}  # pre-event: preevent_17MNP_2023-03-04.json
RGB_DATE = "2023-01-26"  # clear pre-event image as the default ground texture
MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "set", "oct", "nov", "dic"]
CENTRE = (-5.79, -80.64)       # zone centre, for ERA5 precipitation
COMPARABLE = 0.2  # max share of the zone without a result for a date to enter totals and spread
DISCHARGE = (-5.175, -80.675)  # GloFAS cell of the Río Piura nearest Piura city (largest flow nearby)
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

    # Water per analysed date: probability on valid pixels (R) and no-result flag (G).
    from floodseg.system import reference
    km2 = 100 / 1e6
    dates = []
    for year, ev in EVENTS.items():
        days = sorted(d.name.split("_")[1] for d in (REPO / "outputs" / "system" / "scenes").glob(f"{TILE}_{year}-*")
                      if (d / "probability.tif").exists())
        if ev["pre"] and ev["pre"] not in days:
            days.insert(0, ev["pre"])
        grids, prev = {}, None
        for date in days:
            d = REPO / "outputs" / "system" / "scenes" / f"{TILE}_{date}"
            with rasterio.open(d / "probability.tif") as src, rasterio.open(d / "mask.tif") as m:
                win = rasterio.windows.from_bounds(*win_bounds, src.transform)
                p = src.read(1, window=win)
                mk = m.read(1, window=win)
            ok = (p != 65535) & (mk == 0)
            wat = (p > 5000) & ok
            grids[date] = (wat, ok)
            prob = np.where(ok, p / 10000, 0).astype(np.float32)
            rgb = np.zeros((*p.shape, 3), np.uint8)
            rgb[..., 0] = np.round(prob * 255)
            rgb[..., 1] = np.where(ok, 0, 255)
            Image.fromarray(rgb).resize((TEX, TEX), Image.BILINEAR).save(OUT / f"water_{date}.png")
            y, mo, da = date.split("-")
            row = {"date": date, "event": year, "label": f"{int(da)} {MONTHS[int(mo) - 1]} {y}",
                   "is_pre": date == ev["pre"], "water_km2": round(float(wat.sum()) * km2, 1),
                   "nodata_share": round(float(1 - ok.mean()), 3), "comparable": bool(1 - ok.mean() <= COMPARABLE)}
            if ev["pre"] and date != ev["pre"]:
                w0, v0 = grids[ev["pre"]]
                row["flooded_km2"] = round(float((wat & ~w0 & ok & v0).sum()) * km2, 1)
                row["flood_observed_share"] = round(float((ok & v0).mean()), 3)
                if prev is not None and row["comparable"]:
                    wp, vp = grids[prev]
                    common = ok & vp & v0
                    f_t, f_p = wat & ~w0 & common, wp & ~w0 & common
                    n_days = (np.datetime64(date) - np.datetime64(prev)).astype(int)
                    new, gone = float((f_t & ~f_p).sum()) * km2, float((f_p & ~f_t).sum()) * km2
                    row["spread"] = {"from": prev, "days": int(n_days), "new_km2": round(new, 1),
                                     "receded_km2": round(gone, 1), "net_km2": round(new - gone, 1),
                                     "net_km2_per_day": round(float((new - gone) / n_days), 1),
                                     "observed_share": round(float(common.mean()), 3)}
                if row["comparable"]:
                    prev = date
            if reference.available(date):
                mtr = reference.metrics(f"{TILE}_{date}", BOUNDS)
                row["reference"] = {"source": reference.source(date), "pixels": mtr["pixels"],
                                    **{k: round(mtr[k], 3) for k in ("iou", "precision", "recall")}}
            dates.append(row)
            print(row, flush=True)

    weather = {}
    for year in EVENTS:
        q = f"start_date={year}-01-01&end_date={year}-05-31"
        u_rain = (f"https://archive-api.open-meteo.com/v1/archive?latitude={CENTRE[0]}&longitude={CENTRE[1]}&{q}"
                  f"&daily=precipitation_sum&timezone=America%2FLima")
        u_q = f"https://flood-api.open-meteo.com/v1/flood?latitude={DISCHARGE[0]}&longitude={DISCHARGE[1]}&{q}&daily=river_discharge"
        rain, dis = requests.get(u_rain, timeout=60).json(), requests.get(u_q, timeout=60).json()
        weather[year] = {"days": rain["daily"]["time"], "precip_mm": rain["daily"]["precipitation_sum"],
                         "discharge_m3s": dis["daily"]["river_discharge"]}
    (OUT / "weather.json").write_text(json.dumps({
        "precip": {"source": "ERA5 reanalysis via Open-Meteo Historical Weather API", "point": CENTRE,
                   "note": "suma diaria en el centro de la zona, reanálisis de ~25 km, no es un pluviómetro"},
        "discharge": {"source": "GloFAS v4 via Open-Meteo Flood API", "point": DISCHARGE,
                      "note": "caudal diario modelado en la celda del río Piura más cercana a la ciudad de Piura, aguas arriba de la zona"},
        "retrieved": str(np.datetime64("today")), "years": weather}))

    t = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(w, s)
    lon1, lat1 = t.transform(e, n)
    meta = {"name": NAME, "title": "Bajo Piura y Sechura", "bounds_utm": BOUNDS, "epsg": EPSG,
            "bounds_wgs84": [lon0, lat0, lon1, lat1], "size_m": [e - w, n - s], "grid": GRID, "tex": TEX,
            "height_min": float(h.min()), "height_max": float(h.max()), "rgb_date": RGB_DATE, "dates": dates,
            "pre_event": {y: e["pre"] for y, e in EVENTS.items()},
            "sources": ["Copernicus DEM GLO-30", "Copernicus Sentinel-2 L1C (modified)", "U-Net (Perú), 3 semillas"]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
