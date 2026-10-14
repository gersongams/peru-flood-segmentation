"""How were the UNOSAT 2023 Peru polygons made? Geometry evidence (task 6.6 open question).

For each UNOSAT shapefile in data/unosat2023/raw/, reprojected to UTM 17S (EPSG:32717, the
Sentinel-2 grid of the area): share of vertices on a 10 m and on a 20 m grid (±0.5 m),
share of axis-aligned edges, and median edge length. Polygons vectorized from a raster have
staircase edges on the raster grid; hand-digitized polygons do not.

UNOSAT's own methodology note (HDX dataset metadata, `methodology_other` of both Lambayeque
/Piura datasets, read via the HDX API 2026-10-14): "For flood extractions, landcover mapping
and similar analyses, a variety of automated remote sensing techniques are used to extract
the relevant information which is then reviewed and revised as necessary by the analyst."

Writes outputs/label_qa/unosat_geometry.csv.

    uv run python scripts/unosat_provenance.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "unosat2023" / "raw"


def on_grid(c: np.ndarray, step: float) -> float:
    return float(np.mean((np.mod(c[:, 0] + 0.5, step) < 1) & (np.mod(c[:, 1] + 0.5, step) < 1)))


def main() -> None:
    rows, seen = [], set()
    for f in sorted(RAW.rglob("*.shp")):
        if f.name in seen:  # the same layers ship in several zips
            continue
        seen.add(f.name)
        g = gpd.read_file(f).to_crs(32717)
        rings = [np.asarray(p.exterior.coords) for p in g.geometry.explode(index_parts=False) if p.geom_type == "Polygon"]
        c = np.concatenate(rings)
        d = np.concatenate([np.diff(r, axis=0) for r in rings])
        length = np.hypot(d[:, 0], d[:, 1])
        d, length = d[length > 0], length[length > 0]
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90
        rows.append(dict(layer=f.stem, vertices=len(c), on_10m=on_grid(c, 10), on_20m=on_grid(c, 20),
                         axis_aligned=float(((ang < 0.5) | (ang > 89.5)).mean()), median_edge_m=float(np.median(length))))
    df = pd.DataFrame(rows)
    out = REPO / "outputs" / "label_qa"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "unosat_geometry.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
