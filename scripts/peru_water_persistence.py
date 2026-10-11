"""Rough check: share of EMS flood polygons that still look like water on a given S2 date.

MNDWI = (B03 - B11) / (B03 + B11) > 0 on cloud-free pixels, at 20 m. An indicator
for planning label correction, not a label. Writes data/peru/water_persistence.csv.
"""

import csv
import io
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.features import geometry_mask

sys.path.insert(0, str(Path(__file__).parent))
from peru_cloud_survey import AOIS, REPO, SH, layers, token  # noqa: E402

DATE = sys.argv[1] if len(sys.argv) > 1 else "2017-04-07"
RES = 20
EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: ["B03", "B11", "CLM", "dataMask"], output: {bands: 3, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  let mndwi = (s.B03 - s.B11) / (s.B03 + s.B11 + 1e-6);
  return [mndwi, s.CLM, s.dataMask];
}
"""


def main() -> None:
    tok = token()
    rows = []
    for name, product in AOIS.items():
        aoi, flood = layers(product)
        minx, miny, maxx, maxy = aoi.total_bounds
        w, h = int((maxx - minx) / RES), int((maxy - miny) / RES)
        body = {
            "input": {
                "bounds": {"bbox": [minx, miny, maxx, maxy],
                           "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{aoi.crs.to_epsg()}"}},
                "data": [{"type": "sentinel-2-l1c", "dataFilter": {
                    "timeRange": {"from": f"{DATE}T00:00:00Z", "to": f"{DATE}T23:59:59Z"}}}],
            },
            "output": {"width": w, "height": h,
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": EVALSCRIPT,
        }
        r = requests.post(f"{SH}/api/v1/process", json=body,
                          headers={"Authorization": f"Bearer {tok}"}, timeout=300)
        r.raise_for_status()
        with rasterio.open(io.BytesIO(r.content)) as src:
            mndwi, clm, valid = src.read()
        t = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
        in_aoi = ~geometry_mask(aoi.geometry, (h, w), t)
        in_flood = ~geometry_mask(flood.geometry, (h, w), t) & in_aoi
        clear = (valid > 0) & (clm == 0)
        water = clear & (mndwi > 0)
        fc = (clear & in_flood).sum()
        rows.append({
            "aoi": name, "date": DATE,
            "ems_flood_km2": round(in_flood.sum() * RES**2 / 1e6, 1),
            "clear_flood_km2": round(fc * RES**2 / 1e6, 1),
            "water_in_clear_flood_pct": round(100 * (water & in_flood).sum() / max(fc, 1), 1),
            "water_outside_ems_km2": round((water & in_aoi & ~in_flood).sum() * RES**2 / 1e6, 1),
        })
        print(rows[-1])
    out = REPO / "data" / "peru" / f"water_persistence_{DATE}.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
