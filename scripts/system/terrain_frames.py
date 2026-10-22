"""Photo timeline for the 3D view: one true-colour frame of the zone per Sentinel-2 acquisition.

For every L1C acquisition of tile 17MNP between January and May of each flood year (2017 Niño
Costero, 2023 Yaku), requests the zone of scripts/system/terrain_export.py from the Sentinel Hub
Process API (CDSE): B04, B03, B02, the s2cloudless cloud probability CLP and dataMask, TEX_FRAME
px per side. Saves the RGB with the stretch of the system's rgb layer (DN / 3000, gamma 0.8).

Cloud for display: CLP > 0.8, dilated 2 px (~50 m). The binary CLM flagged bright desert sand
(28–33 % of clear-looking January–February frames) and 27 % of the AI water of 2023-03-22; CLP >
0.8 drops the sand to ~4 % but still flags ~21 % of that water, so it is a display rule only.

Composite per event year ("median of the last 3 clear observations"): each pixel keeps its last 3
observations that are valid and not cloud; the composite shows the one of median brightness (sum
of R, G, B), or the latest when it has fewer than 3, or its first valid observation when it has
none. A last-clear-pixel composite was tried first: s2cloudless misses cloud edges and all cloud
shadows, so it was mottled, and stricter CLP thresholds (0.5, 0.3) left white patches instead. The
median removes a missed cloud (bright) or shadow (dark) seen once, at the cost of a lag: a change
shows once it is seen in 2 of the last 3 clear observations (on 2023-03-22 the lagoon looks smaller
than in that day's frame). Each composite records `fresh_share` (zone share observed clear on that
date) and the median and 90th-percentile age in days of the latest clear observation per pixel.
The page shows the composite by default, labelled with its lag, and the raw frame on request.
No metric uses these images.
→ app/public/terrain/piura/frames/<date>.jpg (raw), composite/<date>.jpg, frames.json

    uv run python scripts/system/terrain_frames.py
"""

import io
import json
from datetime import datetime, timezone

import numpy as np
import rasterio
import requests
from PIL import Image
from scipy.ndimage import binary_dilation

from floodseg.system import cdse

import terrain_export as tx

TILE = "17MNP"
YEARS = {"2017": ("2017-01-01", "2017-05-31"), "2023": ("2023-01-01", "2023-05-31")}
TEX_FRAME = 2048
CLEAR = 0.2  # raw frame listed as clear: cloud share <= CLEAR and valid >= 0.9
CLP_CLOUD = 204  # CLP is 0–255; 204 ≈ probability 0.8
SH_PROCESS = "https://sh.dataspace.copernicus.eu/api/v1/process"
EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B03", "B04", "CLP", "dataMask"], units: ["DN", "DN", "DN", "DN", "DN"]}],
          output: {bands: 5, sampleType: "UINT16"}};
}
function evaluatePixel(s) { return [s.B04, s.B03, s.B02, s.CLP, s.dataMask]; }
"""
OUT = tx.OUT / "frames"
CACHE = tx.REPO / "data" / "system" / "frames_cache"  # raw responses, outside the served public/ folder


def fetch(tok: str, day: str) -> np.ndarray:
    body = {
        "input": {"bounds": {"bbox": list(tx.BOUNDS), "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{tx.EPSG}"}},
                  "data": [{"type": "sentinel-2-l1c", "dataFilter": {"timeRange": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"}}}]},
        "output": {"width": TEX_FRAME, "height": TEX_FRAME, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    r = requests.post(SH_PROCESS, json=body, headers={"Authorization": f"Bearer {tok}"}, timeout=180)
    r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as src:
        return src.read()


def frame(day: str, tok: list[str]) -> np.ndarray:
    """(5, n, n): R, G, B (DN), CLP, dataMask. Cached on disk (CACHE/<date>.npz) so a rerun (or a
    failed run) does not request the same date again. Retries timeouts and 5xx, refreshes the token on 401."""
    cache = CACHE / f"{day}.npz"
    if cache.exists():
        return np.load(cache)["a"]
    for attempt in range(4):
        try:
            a = fetch(tok[0], day)
            break
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 401:
                tok[0] = cdse.token()
            elif code < 500 or attempt == 3:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
    np.savez_compressed(cache, a=a)
    return a


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (tx.OUT / "composite").mkdir(exist_ok=True)
    tok = [cdse.token()]
    frames = []
    for year, (start, end) in YEARS.items():
        days = sorted({p["date"] for p in cdse.acquisitions(TILE, start, end)})
        n = TEX_FRAME
        buf = np.zeros((3, n, n, 3), np.uint8)  # last 3 clear observations per pixel, newest first
        count = np.zeros((n, n), np.uint8)
        first = np.zeros((n, n, 3), np.uint8)  # first valid observation, for pixels never seen clear
        seen_any = np.zeros((n, n), bool)
        seen = np.full((n, n), np.datetime64("NaT"), "datetime64[D]")
        for day in days:
            a = frame(day, tok)
            valid = a[4] > 0
            cl = binary_dilation(a[3] > CLP_CLOUD, iterations=2) & valid
            cloud = float(cl.sum() / max(valid.sum(), 1))
            vshare = float(valid.mean())
            rgb = np.stack([np.rint(np.clip(a[k].astype(np.float32) / 3000, 0, 1) ** 0.8 * 255) for k in range(3)], -1).astype(np.uint8)
            rgb[~valid] = 0
            Image.fromarray(rgb).save(OUT / f"{day}.jpg", quality=85)
            fresh = valid & ~cl
            new = valid & ~seen_any
            first[new] = rgb[new]
            seen_any |= valid
            buf[2][fresh] = buf[1][fresh]
            buf[1][fresh] = buf[0][fresh]
            buf[0][fresh] = rgb[fresh]
            count[fresh] = np.minimum(count[fresh] + 1, 3)
            seen[fresh] = np.datetime64(day)
            mid = np.argsort(buf.astype(np.int16).sum(-1), axis=0)[1]
            med = np.take_along_axis(buf, mid[None, ..., None], 0)[0]
            comp = np.where((count >= 3)[..., None], med, np.where((count >= 1)[..., None], buf[0], first))
            Image.fromarray(comp).save(tx.OUT / "composite" / f"{day}.jpg", quality=85)
            age = (np.datetime64(day) - seen[count > 0]).astype(int)
            clear = cloud <= CLEAR and vshare >= 0.9
            frames.append({"date": day, "year": year, "cloud": round(cloud, 3), "valid": round(vshare, 3), "clear": clear,
                           "fresh_share": round(float(fresh.mean()), 3),
                           "composite_clear_share": round(float((count > 0).mean()), 3),
                           "composite_median_age_days": int(np.median(age)) if age.size else None,
                           "composite_p90_age_days": int(np.percentile(age, 90)) if age.size else None})
            print(f"{day} cloud {cloud:.1%} valid {vshare:.1%} fresh {fresh.mean():.1%}{' clear' if clear else ''}", flush=True)
    (tx.OUT / "frames.json").write_text(json.dumps({
        "tile": TILE, "bounds_utm": list(tx.BOUNDS), "epsg": tx.EPSG, "size_px": TEX_FRAME,
        "cloud_rule": f"s2cloudless CLP > 0.8, dilated 2 px, share of valid pixels; clear if <= {CLEAR} and valid >= 0.9",
        "composite": "per year and pixel, median-brightness of the last 3 clear observations (latest if fewer; first valid if none)",
        "stretch": "DN / 3000 clipped, gamma 0.8 (as the system rgb layer)",
        "source": "Copernicus Sentinel-2 L1C via Sentinel Hub Process API (CDSE)",
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frames": frames}, indent=1))


if __name__ == "__main__":
    main()
