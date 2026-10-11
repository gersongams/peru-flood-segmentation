"""Clear-sky survey of Sentinel-2 L1C over the EMSR199 Piura/Trujillo AOIs.

For every S2 L1C acquisition 2017-03-15..2017-04-30 over each AOI, fetch the
s2cloudless cloud mask (CLM) and data mask from the Sentinel Hub Process API on
CDSE at 60 m, and report the clear fraction over the AOI and over the EMS flood
polygons. Writes data/peru/cloud_survey.csv.

    uv run python scripts/peru_cloud_survey.py
"""

import csv
import glob
import io
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import geometry_mask

REPO = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "pyproject.toml").exists() else Path.cwd()
RAW = REPO / "data" / "emsr199" / "raw"
OUT = REPO / "data" / "peru" / "cloud_survey.csv"
SH = "https://sh.dataspace.copernicus.eu"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
START, END = "2017-03-15T00:00:00Z", "2017-04-30T23:59:59Z"
RES = 60  # metres; enough for cloud statistics

# Latest version of each AOI's delineation product.
AOIS = {
    "piura": "EMSR199_10PIURA_DELINEATION_OVERVIEW-MONIT01_v3",
    "parachique": "EMSR199_11PARACHIQUE_DELINEATION_OVERVIEW-MONIT01_v2",
    "piura_north": "EMSR199_12PIURANORTH_DELINEATION_OVERVIEW-MONIT01_v1",
    "trujillo": "EMSR199_13TRUJILLO_DELINEATION_OVERVIEW-MONIT01_v1",
    "salinas": "EMSR199_14SALINAS_DELINEATION_OVERVIEW_v1",
}

EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: ["CLM", "dataMask"], output: {bands: 2, sampleType: "UINT8"}};
}
function evaluatePixel(s) { return [s.CLM, s.dataMask]; }
"""


def env(name: str) -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return os.environ[name]


def token() -> str:
    r = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": env("CDSE_CLIENT_ID"),
        "client_secret": env("CDSE_CLIENT_SECRET"),
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def layers(product: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    folder = RAW / product
    aoi = gpd.read_file(glob.glob(str(folder / "*area_of_interest.shp"))[0])
    flood = gpd.read_file(glob.glob(str(folder / "*crisis_information_poly.shp"))[0])
    return aoi, flood.to_crs(aoi.crs)


def acquisitions(tok: str, bbox_wgs84: list[float]) -> list[dict]:
    """Distinct acquisition dates with their tiles and tile-level cloud cover."""
    body = {"collections": ["sentinel-2-l1c"], "bbox": bbox_wgs84,
            "datetime": f"{START}/{END}", "limit": 100}
    r = requests.post(f"{SH}/api/v1/catalog/1.0.0/search", json=body,
                      headers={"Authorization": f"Bearer {tok}"}, timeout=60)
    r.raise_for_status()
    by_date: dict[str, dict] = {}
    for f in r.json()["features"]:
        d = f["properties"]["datetime"][:10]
        e = by_date.setdefault(d, {"date": d, "tiles": set(), "tile_cloud": []})
        e["tiles"].add(f["id"].split("_")[5] if f["id"].count("_") >= 5 else f["id"])
        e["tile_cloud"].append(f["properties"].get("eo:cloud_cover"))
    return sorted(by_date.values(), key=lambda e: e["date"])


def cloud_mask(tok: str, bounds, crs_epsg: int, date: str, width: int, height: int) -> np.ndarray:
    body = {
        "input": {
            "bounds": {"bbox": list(bounds), "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{crs_epsg}"}},
            "data": [{"type": "sentinel-2-l1c", "dataFilter": {
                "timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"}}}],
        },
        "output": {"width": width, "height": height,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    r = requests.post(f"{SH}/api/v1/process", json=body,
                      headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as src:
        return src.read()  # (2, H, W): CLM, dataMask


def main() -> None:
    tok = token()
    rows = []
    for name, product in AOIS.items():
        aoi, flood = layers(product)
        epsg = aoi.crs.to_epsg()
        minx, miny, maxx, maxy = aoi.total_bounds
        w, h = int((maxx - minx) / RES), int((maxy - miny) / RES)
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
        in_aoi = ~geometry_mask(aoi.geometry, (h, w), transform)
        in_flood = ~geometry_mask(flood.geometry, (h, w), transform) & in_aoi
        bbox = list(aoi.to_crs(4326).total_bounds)
        for acq in acquisitions(tok, bbox):
            clm, valid = cloud_mask(tok, (minx, miny, maxx, maxy), epsg, acq["date"], w, h)
            seen = valid.astype(bool)
            clear = seen & (clm == 0)
            covered = (seen & in_aoi).sum() / in_aoi.sum()
            rows.append({
                "aoi": name, "date": acq["date"], "tiles": " ".join(sorted(acq["tiles"])),
                "tile_cloud_pct": " ".join(f"{c:.0f}" for c in acq["tile_cloud"] if c is not None),
                "aoi_covered_pct": round(100 * covered, 1),
                "aoi_clear_pct": round(100 * (clear & in_aoi).sum() / in_aoi.sum(), 1),
                "flood_clear_pct": round(100 * (clear & in_flood).sum() / max(in_flood.sum(), 1), 1),
            })
            print(rows[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
