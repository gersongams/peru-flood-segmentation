"""Reconcile UNOSAT area figures for 2023-03-22 and test a static permanent-water mask for RF6.

1. UNOSAT EQ/FL20230321PER layers (data/unosat2023/raw/0322): area of FloodExtent ("New Water /
   Water Increase"), WaterExtent (flood + pre-existing water), their difference, measured on the
   geometry in UTM 17S (EPSG:32717); compared with the project's rasterized labels (manifest
   water_frac × 26.2144 km² per 512² chip at 10 m).
2. ESA WorldCover 2021 v200 class 80 ("Permanent water bodies") inside UNOSAT's analysis extent,
   pre-existing water and flood extent (km² from pixel counts at the AOI's mean latitude). Tests
   whether a static mask could separate flood from permanent water for RF6.

    uv run python scripts/unosat_reconcile.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "unosat2023" / "raw" / "0322" / "EQ20230321PER_SHP"
WC = sorted((REPO / "data" / "aux" / "worldcover").glob("ESA_WorldCover_10m_2021_v200_*_Map.tif"))


def layer(name: str, crs: int):
    return gpd.read_file(RAW / f"S2_20230321_{name}_Lambayeque_Piura.shp").to_crs(crs).union_all()


def main() -> None:
    ae, fl, wa = (layer(k, 32717) for k in ("AnalysisExtent", "FloodExtent", "WaterExtent"))
    km2 = lambda g: g.area / 1e6
    print(f"AnalysisExtent {km2(ae):.1f} km² | FloodExtent {km2(fl):.1f} | WaterExtent {km2(wa):.1f} | "
          f"pre-existing (Water − Flood) {km2(wa.difference(fl)):.1f} | Flood outside Water {km2(fl.difference(wa)):.2f}")
    m = pd.read_csv(REPO / "data" / "peru" / "unosat2023" / "manifest.csv")
    d = m[m.date == "2023-03-22"]
    print(f"project rasterized water 2023-03-22: {(d.water_frac * 26.2144).sum():.1f} km² over {len(d)} chips")

    ae4, fl4, wa4 = (layer(k, 4326) for k in ("AnalysisExtent", "FloodExtent", "WaterExtent"))
    pre4 = wa4.difference(fl4)
    lat = np.deg2rad((ae4.bounds[1] + ae4.bounds[3]) / 2)
    tot = dict(analysis=0.0, preexisting=0.0, flood=0.0)
    for f in WC:
        with rasterio.open(f) as src:
            px = abs(src.transform.a) * 111320 * np.cos(lat) * abs(src.transform.e) * 110574 / 1e6
            for k, geom in (("analysis", ae4), ("preexisting", pre4), ("flood", fl4)):
                try:
                    a, _ = mask(src, [geom], crop=True, nodata=0)
                except ValueError:  # geometry outside this WorldCover tile
                    continue
                tot[k] += float((a[0] == 80).sum() * px)
    print("WorldCover 2021 class 80 (km²): " + ", ".join(f"in {k} {v:.1f}" for k, v in tot.items()))


if __name__ == "__main__":
    main()
