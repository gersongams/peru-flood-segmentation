"""Backend of the flood-analysis web app (FastAPI).

The models are loaded once at start-up (the "warm system" of the HE3 protocol). A job request
(RF1) is timestamped on arrival; the job ends when the GeoTIFFs are written and the map tiles can
be served. Both timestamps go to outputs/system/server.log and outputs/system/jobs.jsonl.
Jobs run one at a time.

    uv run uvicorn floodseg.system.server:app --port 8000
"""

import io
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from pyproj import Transformer
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader

from floodseg.system import cdse, export, pipeline, preevent, reference
from floodseg.system.infer import Ensemble
from floodseg.target_pool import REPO

LOG_DIR = REPO / "outputs" / "system"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("floodseg.system")
log.setLevel(logging.INFO)
_h = logging.FileHandler(LOG_DIR / "server.log")
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_h)

# Events the app offers (RF1). Tiles: MGRS squares covering the reference extent.
EVENTS = {
    "yaku2023": {"name": "Ciclón Yaku 2023", "start": "2023-03-04", "range": ["2023-03-01", "2023-04-15"],
                 "tiles": ["17MNP", "17MNQ", "17MPN", "17MNN", "17MPP", "17MPQ"], "reference": "UNOSAT"},
    "nino2017": {"name": "Niño Costero 2017", "start": "2017-03-15", "range": ["2017-03-15", "2017-04-15"],
                 "tiles": ["17MNP", "17MNQ"], "reference": "Copernicus EMS"},
}

app = FastAPI(title="Inundaciones Perú")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {"ensemble": None}
_jobs: dict[str, dict] = {}
_run_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@app.on_event("startup")
def _load_models() -> None:
    t = time.perf_counter()
    _state["ensemble"] = Ensemble("peru/unet")
    _state["cold_start_s"] = round(time.perf_counter() - t, 2)
    log.info("models loaded: peru/unet in %.2f s", _state["cold_start_s"])


# ---------------------------------------------------------------- jobs (RF1, RF2, RF3)

class JobRequest(BaseModel):
    event: str
    tile: str
    date: str
    bounds: list[float] | None = None  # view rectangle [W, S, E, N], WGS84 (logged; the whole tile is processed)


def _run_job(job: dict) -> None:
    def progress(stage: str, frac: float, detail: str) -> None:
        s = job["stages"][stage]
        if s["status"] == "pending":
            s["status"], s["started"] = "running", time.perf_counter()
        s["fraction"], s["detail"] = round(frac, 4), detail
        if frac >= 1.0:
            s["status"], s["seconds"] = "done", round(time.perf_counter() - s["started"], 2)

    with _run_lock:
        job["status"] = "running"
        try:
            meta = pipeline.run(job["tile"], job["date"], _state["ensemble"], progress)
            job["status"], job["scene"] = "done", f"{job['tile']}_{job['date']}"
            job["product"] = meta["product"]
        except Exception as e:  # reported to the app; external errors are retried by the operator
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
            log.exception("job %s failed", job["id"])
        job["stop"] = _now()
        job["seconds"] = round(time.perf_counter() - job["t0"], 2)
        log.info("job %s %s stop=%s seconds=%s", job["id"], job["status"], job["stop"], job["seconds"])
        with open(LOG_DIR / "jobs.jsonl", "a") as f:
            f.write(json.dumps({k: v for k, v in job.items() if k != "t0"}) + "\n")


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    t0, received = time.perf_counter(), _now()
    if req.event not in EVENTS:
        raise HTTPException(400, f"unknown event {req.event}")
    jid = uuid.uuid4().hex[:12]
    cached = pipeline.is_cached(req.tile, req.date)
    job = {"id": jid, "event": req.event, "tile": req.tile, "date": req.date, "bounds": req.bounds,
           "received": received, "t0": t0, "cached": cached, "status": "queued", "error": None,
           "stages": {s: {"status": "pending", "fraction": 0.0, "detail": "", "seconds": None} for s in pipeline.STAGES}}
    _jobs[jid] = job
    log.info("job %s received tile=%s date=%s bounds=%s cached=%s", jid, req.tile, req.date, req.bounds, cached)
    if cached:  # RF2b: load from the cache, no download
        job.update(status="done", scene=f"{req.tile}_{req.date}", stop=_now(),
                   seconds=round(time.perf_counter() - t0, 3))
        for s in job["stages"].values():
            s.update(status="cached", fraction=1.0)
        log.info("job %s served from cache", jid)
    else:
        threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return _public(job)


def _public(job: dict) -> dict:
    out = {k: v for k, v in job.items() if k != "t0"}
    if job["status"] in ("queued", "running"):
        out["elapsed"] = round(time.perf_counter() - job["t0"], 1)
    out["stages"] = {k: {f: v for f, v in s.items() if f != "started"} for k, s in job["stages"].items()}
    return out


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    if jid not in _jobs:
        raise HTTPException(404)
    return _public(_jobs[jid])


# ---------------------------------------------------------------- catalog and scenes

@app.get("/api/events")
def events() -> dict:
    return EVENTS


@app.get("/api/status")
def status() -> dict:
    return {"models_loaded": _state["ensemble"] is not None, "model": "peru/unet",
            "cold_start_s": _state.get("cold_start_s"), "busy": _run_lock.locked()}


COLS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
ROWS = "ABCDEFGHJKLMNPQRSTUV"


def utm_bounds(tile: str) -> tuple[tuple[float, float, float, float], int]:
    """UTM bounds (W, S, E, N) and EPSG of an S2 tile (MGRS square, 109.8 km, upper-left corner
    20 m west of the square's west edge and at its north edge). Southern-hemisphere zones (Peru)."""
    zone, sq = int(tile[:2]), tile[3:]
    col_set = (zone - 1) % 3  # 0: A–H, 1: J–R, 2: S–Z
    e = (COLS.index(sq[0]) - 8 * col_set + 1) * 100_000
    row = ROWS.index(sq[1]) - (5 if zone % 2 == 0 else 0)
    n = (row % 20) * 100_000
    while n < 8_900_000:  # Peru: northings 8.0–10.0 Mm in UTM south; MGRS rows repeat every 2 000 km
        n += 2_000_000
    x0, y1 = e - 20, n + 100_000
    return (x0, y1 - 109_800, x0 + 109_800, y1), 32700 + zone


def footprint(tile: str) -> dict:
    """WGS84 polygon of an S2 tile."""
    (x0, y0, x1, y1), epsg = utm_bounds(tile)
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    ring = [t.transform(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
    return {"type": "Feature", "properties": {"tile": tile}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


@app.get("/api/footprints")
def footprints(tiles: str) -> dict:
    return {"type": "FeatureCollection", "features": [footprint(t) for t in tiles.split(",")]}


@app.get("/api/acquisitions")
def acquisitions(tile: str, start: str, end: str) -> list[dict]:
    items = cdse.acquisitions(tile, start, end)
    for it in items:
        it["cached"] = pipeline.is_cached(tile, it["date"])
    return items


@app.get("/api/scenes")
def scenes() -> list[dict]:
    out = []
    for d in sorted(pipeline.SCENES.glob("*/meta.json")):
        m = json.loads(d.read_text())
        out.append({"scene": d.parent.name, **{k: m[k] for k in ("tile", "date", "product", "pixels", "total_s")},
                    "bounds": _bounds_wgs84(d.parent.name)})
    return out


@lru_cache(maxsize=64)
def _bounds_wgs84(scene: str) -> list[float]:
    with Reader(str(pipeline.SCENES / scene / "probability.tif")) as r:
        return list(r.get_geographic_bounds("EPSG:4326"))


# ---------------------------------------------------------------- map tiles

LAYERS = {"rgb": "rgb.tif", "probability": "probability.tif", "uncertainty": "uncertainty.tif", "mask": "mask.tif"}
_EMPTY = None


def _empty_png() -> bytes:
    global _EMPTY
    if _EMPTY is None:
        b = io.BytesIO()
        Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(b, "PNG")
        _EMPTY = b.getvalue()
    return _EMPTY


def _ramp(v: np.ndarray, stops: list[tuple[float, tuple]]) -> np.ndarray:
    """Piecewise-linear RGBA colour ramp; v in [0, 1]."""
    xs = [s[0] for s in stops]
    return np.stack([np.interp(v, xs, [s[1][k] for s in stops]) for k in range(4)], -1).astype(np.uint8)


WATER = [(0.0, (150, 210, 245, 150)), (0.5, (60, 140, 220, 220)), (1.0, (14, 60, 160, 240))]
UNCERT = [(0.0, (253, 200, 120, 0)), (0.1, (253, 200, 120, 170)), (0.5, (150, 30, 90, 235))]


def _colorize(layer: str, data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = data.shape[1:]
    if layer == "rgb":
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., :3] = np.moveaxis(data[:3], 0, -1)
        rgba[..., 3] = np.where(valid & (data[:3].sum(0) > 0), 255, 0)
        return rgba
    v = data[0].astype(np.float32)
    if layer == "probability":
        p = v / pipeline.SCALE
        rgba = _ramp((p - 0.5) / 0.5, WATER)
        rgba[(p <= 0.5) | ~valid] = 0
        return rgba
    if layer == "uncertainty":
        s = v / pipeline.SCALE
        rgba = _ramp(s, UNCERT)
        rgba[~valid] = 0
        return rgba
    if layer == "mask":
        rgba = np.zeros((h, w, 4), np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        hatch = ((yy + xx) % 6) < 2
        cloud, nod = v == 1, v == 2
        rgba[cloud] = (241, 238, 232, 200)
        rgba[cloud & hatch] = (200, 194, 184, 220)
        rgba[nod] = (120, 112, 100, 120)
        return rgba
    raise HTTPException(404, f"unknown layer {layer}")


_ref_lock = threading.Lock()


def _ensure_reference(scene: str) -> None:
    with _ref_lock:
        reference.reference(scene)


def _png(rgba: np.ndarray) -> Response:
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG", optimize=False, compress_level=3)
    return Response(b.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


def _read_tile(path, x: int, y: int, z: int, resampling: str = "nearest"):
    try:
        with Reader(str(path)) as r:
            return r.tile(x, y, z, tilesize=256, resampling_method=resampling)
    except TileOutsideBounds:
        return None


@app.get("/api/tiles/{scene}/{layer}/{z}/{x}/{y}.png")
def tile(scene: str, layer: str, z: int, x: int, y: int) -> Response:
    d = pipeline.SCENES / scene
    if layer == "flood":
        return flood_tile(scene, z, x, y)
    if layer in ("reference", "errors"):
        if not reference.available(scene.split("_")[1]):
            return Response(_empty_png(), media_type="image/png")
        _ensure_reference(scene)
        ref = _read_tile(d / "reference.tif", x, y, z)
        if ref is None:
            return Response(_empty_png(), media_type="image/png")
        rv = ref.data[0]
        rgba = np.zeros((256, 256, 4), np.uint8)
        if layer == "reference":
            rgba[rv == 1] = (0, 164, 154, 150)
            return _png(rgba)
        pr, mk = _read_tile(d / "probability.tif", x, y, z), _read_tile(d / "mask.tif", x, y, z)
        ok = (rv != reference.IGNORE) & (mk.data[0] == 0) & (pr.data[0] != pipeline.NODATA16)
        pred, true = pr.data[0] > pipeline.SCALE // 2, rv == 1
        rgba[ok & pred & true] = (37, 99, 184, 230)
        rgba[ok & pred & ~true] = (212, 85, 60, 240)
        rgba[ok & ~pred & true] = (233, 162, 27, 240)
        return _png(rgba)
    if layer not in LAYERS:
        raise HTTPException(404, f"unknown layer {layer}")
    path = pipeline.SCENES / scene / LAYERS[layer]
    if not path.exists():
        raise HTTPException(404, f"no scene {scene}")
    try:
        with Reader(str(path)) as r:
            img = r.tile(x, y, z, tilesize=256, resampling_method="nearest" if layer == "mask" else "bilinear")
    except TileOutsideBounds:
        return Response(_empty_png(), media_type="image/png")
    data = img.data
    valid = np.asarray(img.mask) > 0
    if layer in ("probability", "uncertainty"):
        valid &= data[0] != pipeline.NODATA16
    return _png(_colorize(layer, data, valid))


# ---------------------------------------------------------------- reference metrics (RF5)

@app.get("/api/metrics/{scene}")
def scene_metrics(scene: str, bbox: str | None = None, part: str | None = None) -> dict:
    """Agreement with the expert reference. bbox: W,S,E,N in WGS84 (the map view)."""
    date = scene.split("_")[1]
    if not reference.available(date):
        return {"available": False}
    meta = json.loads((pipeline.SCENES / scene / "meta.json").read_text())
    box = None
    if bbox:
        w, s_, e, n = (float(v) for v in bbox.split(","))
        t = _to_utm(meta["crs"])
        xs, ys = zip(*(t.transform(a, b) for a, b in [(w, s_), (e, s_), (e, n), (w, n)]))
        box = (min(xs), min(ys), max(xs), max(ys))
    _ensure_reference(scene)
    return {"available": True, "reference": "UNOSAT FL20230321PER", **reference.metrics(scene, box, part)}


# ---------------------------------------------------------------- pixel readout

@lru_cache(maxsize=8)
def _to_utm(crs: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


@app.get("/api/pixel/{scene}")
def pixel(scene: str, lon: float, lat: float) -> dict:
    d = pipeline.SCENES / scene
    meta = json.loads((d / "meta.json").read_text())
    x, y = _to_utm(meta["crs"]).transform(lon, lat)
    out = {"lon": lon, "lat": lat}
    import rasterio
    for name in ("probability", "uncertainty", "mask"):
        with rasterio.open(d / f"{name}.tif") as src:
            l, b, r, t = src.bounds
            if not (l <= x < r and b <= y < t):
                return {**out, "inside": False}
            v = int(next(src.sample([(x, y)]))[0])
        out[name] = None if (name != "mask" and v == pipeline.NODATA16) else (v / pipeline.SCALE if name != "mask" else v)
    out["inside"] = True
    out["state"] = {0: "valid", 1: "cloud", 2: "nodata"}[out["mask"]]
    return out


# ---------------------------------------------------------------- flooded area (RF6)

class PreeventRequest(BaseModel):
    event: str
    tile: str


def _event_of(date: str) -> str | None:
    return next((k for k, e in EVENTS.items() if e["range"][0] <= date <= e["range"][1]), None)


@app.post("/api/preevent")
def create_preevent(req: PreeventRequest) -> dict:
    """Choose (once) and process the pre-event reference scene of a tile. Not part of the HE3 timer."""
    ev = EVENTS[req.event]
    jid = uuid.uuid4().hex[:12]
    stages = {"select": {"status": "pending", "fraction": 0.0, "detail": "", "seconds": None}}
    stages.update({s: {"status": "pending", "fraction": 0.0, "detail": "", "seconds": None} for s in pipeline.STAGES})
    job = {"id": jid, "kind": "preevent", "event": req.event, "tile": req.tile, "date": None, "bounds": None,
           "received": _now(), "t0": time.perf_counter(), "cached": False, "status": "queued", "error": None,
           "stages": stages}
    _jobs[jid] = job
    log.info("preevent job %s received tile=%s event=%s", jid, req.tile, req.event)

    def work() -> None:
        def progress(stage: str, frac: float, detail: str) -> None:
            st = job["stages"][stage]
            if st["status"] == "pending":
                st["status"], st["started"] = "running", time.perf_counter()
            st["fraction"], st["detail"] = round(frac, 4), detail
            if frac >= 1.0:
                st["status"], st["seconds"] = "done", round(time.perf_counter() - st["started"], 2)

        with _run_lock:
            job["status"] = "running"
            try:
                rec = preevent.load_record(req.tile, ev["start"])
                progress("select", 0.0, "evaluando nubosidad de las fechas candidatas")
                if rec is None:
                    b, epsg = utm_bounds(req.tile)
                    rec = preevent.choose(req.tile, ev["start"], b, epsg)
                    preevent.record_path(req.tile, ev["start"]).write_text(json.dumps(rec, indent=2))
                progress("select", 1.0, rec["chosen"])
                job["date"] = rec["chosen"]
                if pipeline.is_cached(req.tile, rec["chosen"]):
                    job["cached"] = True
                    for k in pipeline.STAGES:
                        job["stages"][k].update(status="cached", fraction=1.0)
                else:
                    pipeline.run(req.tile, rec["chosen"], _state["ensemble"], progress)
                job["status"], job["scene"] = "done", f"{req.tile}_{rec['chosen']}"
            except Exception as e:
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
                log.exception("preevent job %s failed", jid)
            job["stop"] = _now()
            job["seconds"] = round(time.perf_counter() - job["t0"], 2)
            log.info("preevent job %s %s chosen=%s seconds=%s", jid, job["status"], job["date"], job["seconds"])

    threading.Thread(target=work, daemon=True).start()
    return _public(job)


def _ref_scene(scene: str) -> tuple[str | None, dict | None]:
    tile, date = scene.split("_")
    ev = _event_of(date)
    if ev is None:
        return None, None
    rec = preevent.load_record(tile, EVENTS[ev]["start"])
    if rec is None or not pipeline.is_cached(tile, rec["chosen"]):
        return None, rec
    return f"{tile}_{rec['chosen']}", rec


@lru_cache(maxsize=64)
def _flood_cached(scene: str, ref: str) -> dict:
    return preevent.flood(scene, ref)


@app.get("/api/flood/{scene}")
def flood(scene: str) -> dict:
    ref, rec = _ref_scene(scene)
    if ref is None:
        return {"available": False, "record": rec}
    if ref == scene:
        return {"available": True, "is_reference": True, "reference_scene": ref, "record": rec}
    return {"available": True, "is_reference": False, "record": rec, **_flood_cached(scene, ref)}


def flood_tile(scene: str, z: int, x: int, y: int) -> Response:
    ref, _ = _ref_scene(scene)
    if ref is None or ref == scene:
        return Response(_empty_png(), media_type="image/png")
    d_t, d_0 = pipeline.SCENES / scene, pipeline.SCENES / ref
    reads = [_read_tile(d / f, x, y, z) for d in (d_t, d_0) for f in ("probability.tif", "mask.tif")]
    if any(r is None for r in reads):
        return Response(_empty_png(), media_type="image/png")
    p_t, m_t, p_0, m_0 = (r.data[0] for r in reads)
    both = (m_t == 0) & (m_0 == 0) & (p_t != pipeline.NODATA16) & (p_0 != pipeline.NODATA16)
    w_t, w_0 = p_t > pipeline.SCALE // 2, p_0 > pipeline.SCALE // 2
    rgba = np.zeros((256, 256, 4), np.uint8)
    rgba[w_t & ~w_0 & both] = (29, 79, 154, 235)
    rgba[w_t & w_0 & both] = (127, 176, 224, 200)
    return _png(rgba)


# ---------------------------------------------------------------- export (RF7)

@app.get("/api/export/{scene}")
def export_scene(scene: str, layers: str = ",".join(export.LAYERS), bbox: str | None = None) -> Response:
    if not pipeline.is_cached(*scene.split("_")):
        raise HTTPException(404, f"no scene {scene}")
    ref, _ = _ref_scene(scene)
    if ref == scene:
        ref = None
    box = None
    if bbox:
        meta = json.loads((pipeline.SCENES / scene / "meta.json").read_text())
        w, s_, e, n = (float(v) for v in bbox.split(","))
        t = _to_utm(meta["crs"])
        xs, ys = zip(*(t.transform(a, b) for a, b in [(w, s_), (e, s_), (e, n), (w, n)]))
        box = (min(xs), min(ys), max(xs), max(ys))
    t0 = time.perf_counter()
    data, info = export.build(scene, [x for x in layers.split(",") if x in export.LAYERS], ref, box)
    log.info("export %s layers=%s bbox=%s bytes=%d seconds=%.1f", scene, layers, bbox, len(data), time.perf_counter() - t0)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{scene}.zip"'})
