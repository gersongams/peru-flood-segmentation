"""Pre-event reference image for the flooded area (RF6) and the flooded-area computation.

Reference image (docs/system_evaluation_protocol.md §4): the clearest Sentinel-2 acquisition of
the same tile in the 60 days before the event start, "clearest" meaning the smallest share of
thick cloud under the project rule (B02 > 4500 and B11 < 1.15·B02) among valid pixels. Candidates
are ranked on 200 m previews from the Process API (B02, B11, dataMask; harmonized DN), which cost
few processing units; only the chosen product is downloaded in full (OData) and processed like any
scene. Ties go to the date closest to the event. Computed once per tile and event, outside the
HE3 timer.

Flooded area at t: A = Área((W_t \\ W_0) ∩ V_t ∩ V_0), water = probability > 0.5, valid = mask 0.
"""

import io
import json
from datetime import date as Date, timedelta

import numpy as np
import rasterio
import requests

from floodseg.system import cdse
from floodseg.system.pipeline import NODATA16, SCALE, SCENES, scene_dir

SH_PROCESS = "https://sh.dataspace.copernicus.eu/api/v1/process"
PREVIEW_M = 200
WINDOW_DAYS = 60
EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B11", "dataMask"], units: "DN"}], output: {bands: 3, sampleType: "UINT16"}};
}
function evaluatePixel(s) { return [s.B02, s.B11, s.dataMask]; }
"""


def thick_cloud_share(tok: str, utm_bounds: tuple, epsg: int, day: str) -> tuple[float, float]:
    """(thick-cloud share of valid pixels, valid share) on a 200 m preview of one date."""
    w = int(round((utm_bounds[2] - utm_bounds[0]) / PREVIEW_M))
    body = {
        "input": {"bounds": {"bbox": list(utm_bounds), "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                  "data": [{"type": "sentinel-2-l1c", "dataFilter": {"timeRange": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"}}}]},
        "output": {"width": w, "height": w, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    r = requests.post(SH_PROCESS, json=body, headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as src:
        b02, b11, dm = src.read().astype(np.float32)
    valid = dm > 0
    thick = (b02 > 4500) & (b11 < 1.15 * b02) & valid
    return (float(thick.sum() / valid.sum()) if valid.any() else 1.0), float(valid.mean())


def choose(tile: str, event_start: str, utm_bounds: tuple, epsg: int) -> dict:
    """Rank the candidate dates and return the chosen one with the full ranking."""
    start = Date.fromisoformat(event_start)
    first, last = (start - timedelta(days=WINDOW_DAYS)).isoformat(), (start - timedelta(days=1)).isoformat()
    tok = cdse.token()
    seen, ranking = set(), []
    for p in cdse.acquisitions(tile, first, last):
        if p["date"] in seen:
            continue
        seen.add(p["date"])
        share, valid = thick_cloud_share(tok, utm_bounds, epsg, p["date"])
        ranking.append({"date": p["date"], "thick_cloud": round(share, 4), "valid": round(valid, 4),
                        "catalog_cloud_cover": p["cloud_cover"]})
    if not ranking:
        raise LookupError(f"no acquisition of {tile} between {first} and {last}")
    # Least thick cloud; a mostly empty swath edge is not "clear", so require half the tile valid.
    pool = [r for r in ranking if r["valid"] >= 0.5] or ranking
    best = min(pool, key=lambda r: (r["thick_cloud"], -Date.fromisoformat(r["date"]).toordinal()))
    return {"tile": tile, "event_start": event_start, "window": [first, last], "chosen": best["date"],
            "rule": "min thick-cloud share on 200 m previews, valid >= 0.5, tie -> latest", "ranking": ranking}


def record_path(tile: str, event_start: str):
    return SCENES / f"preevent_{tile}_{event_start}.json"


def flood(scene: str, ref_scene: str) -> dict:
    """Flooded area (new water vs the pre-event scene) and total water of `scene`, km²."""
    def read(s: str):
        d = scene_dir(*s.split("_"))
        with rasterio.open(d / "probability.tif") as a, rasterio.open(d / "mask.tif") as m:
            return a.read(1), m.read(1)
    p_t, m_t = read(scene)
    p_0, m_0 = read(ref_scene)
    v_t, v_0 = (m_t == 0) & (p_t != NODATA16), (m_0 == 0) & (p_0 != NODATA16)
    w_t, w_0 = (p_t > SCALE // 2) & v_t, (p_0 > SCALE // 2) & v_0
    km2 = 100 / 1e6
    both = v_t & v_0
    return {"scene": scene, "reference_scene": ref_scene,
            "flooded_km2": int((w_t & ~w_0 & both).sum()) * km2,
            "water_km2": int(w_t.sum()) * km2,
            "water_ref_km2": int(w_0.sum()) * km2,
            "valid_both_share": float(both.mean()), "valid_share": float(v_t.mean())}


def flood_mask(scene: str, ref_scene: str, window=None) -> np.ndarray:
    """uint8: 1 flooded (new water), 2 water also before the event, 0 otherwise / not valid."""
    def read(s: str):
        d = scene_dir(*s.split("_"))
        with rasterio.open(d / "probability.tif") as a, rasterio.open(d / "mask.tif") as m:
            return a.read(1, window=window), m.read(1, window=window)
    p_t, m_t = read(scene)
    p_0, m_0 = read(ref_scene)
    both = (m_t == 0) & (m_0 == 0) & (p_t != NODATA16) & (p_0 != NODATA16)
    w_t, w_0 = p_t > SCALE // 2, p_0 > SCALE // 2
    out = np.zeros(p_t.shape, np.uint8)
    out[w_t & ~w_0 & both] = 1
    out[w_t & w_0 & both] = 2
    return out


def load_record(tile: str, event_start: str) -> dict | None:
    p = record_path(tile, event_start)
    return json.loads(p.read_text()) if p.exists() else None
