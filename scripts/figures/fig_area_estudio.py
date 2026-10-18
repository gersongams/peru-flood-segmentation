"""Study-area map: Peru chips (UNOSAT 2023 and Copernicus EMS 2017) on the north coast of Peru.

Inputs
  data/peru/unosat2023/manifest.csv, splits/split.csv, s2/<date>_<chip>.tif   2023 chips and splits
  data/peru/emsr2017/manifest.csv, s2/<chip>.tif                             2017 test chips (all 204)
  data/aux/cop_dem/*.tif                                                      Copernicus DEM GLO-30 (hillshade)
  data/aux/naturalearth/*.zip   Natural Earth 10m: countries, admin-1, populated places, ocean
                                (downloaded from https://naciscdn.org/naturalearth/10m/ if missing)
Output
  thesis/E_IMAGENES/fig_area_estudio.pdf (+ PNG preview in outputs/figures_preview/)

Background: true-colour Sentinel-2 (L1C, bands B04/B03/B02, DN / 10 000) from the project's own chips,
read at 1/8 resolution (80 m). One date per chip location: the date with the smallest fraction of
bright (B02 > 0.30, cloud proxy) or empty pixels. Locations that only exist in 2017 use the 2017 image.
Outside the chips: Copernicus DEM hillshade on a neutral land colour, Natural Earth ocean.
Counts in the legend: unique chip locations and chip-dates, computed from split.csv / manifest.csv.

Run: uv run python scripts/figures/fig_area_estudio.py
"""
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from matplotlib.patches import Patch, Rectangle
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = style.ROOT
U23 = ROOT / "data/peru/unosat2023"
E17 = ROOT / "data/peru/emsr2017"
NE = ROOT / "data/aux/naturalearth"
DEM = ROOT / "data/aux/cop_dem"
CRS = "EPSG:32717"  # WGS 84 / UTM 17S, the chips' CRS
CHIP = 5120  # m: 512 px x 10 m
DS = 8  # downsampling factor for the mosaic (80 m pixels)
PAD = 6000  # m of map around the chips
INSET = 350  # m: split outlines are drawn this far inside their region
LL_BOX = box(-82.0, -8.0, -78.5, -4.0)  # lon/lat window for clipping Natural Earth before reprojection

NE_FILES = {
    "countries": "cultural/ne_10m_admin_0_countries",
    "admin1": "cultural/ne_10m_admin_1_states_provinces",
    "places": "cultural/ne_10m_populated_places",
    "ocean": "physical/ne_10m_ocean",
}
# Towns labelled on the map (all taken from Natural Earth populated places, ADM0NAME == Peru).
TOWNS = ["Piura", "Chiclayo", "Sechura", "Chulucanas", "Olmos"]
SEA = "#DCE4EA"
LAND = "#EEEDE8"
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]


def natural_earth(key: str) -> gpd.GeoDataFrame:
    path = NE / f"{Path(NE_FILES[key]).name}.zip"
    if not path.exists():
        NE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://naciscdn.org/naturalearth/10m/{NE_FILES[key]}.zip", path)
    return gpd.read_file(f"zip://{path}")


def draw_lines(ax, geoms, **kw) -> None:
    """Plot the boundary lines of shapely geometries (geopandas cannot take dash tuples)."""
    for g in geoms:
        b = g.boundary if g.geom_type.endswith("Polygon") else g
        for line in getattr(b, "geoms", [b]):
            x, y = line.xy
            ax.plot(x, y, **kw)


def chip_tables():
    split = pd.read_csv(U23 / "splits/split.csv")
    man = pd.read_csv(U23 / "manifest.csv").merge(split, on=["chip", "date"], validate="1:1")
    assert man.groupby("chip").split.nunique().max() == 1  # split is by location
    m17 = pd.read_csv(E17 / "manifest.csv")
    m17["split"] = "2017"
    return man, m17


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        assert src.crs.to_string() == CRS and src.width * src.res[0] == CHIP
        n = src.width // DS
        return src.read([4, 3, 2], out_shape=(3, n, n), resampling=Resampling.average).astype(np.float32) / 1e4


def bad_fraction(rgb: np.ndarray) -> float:
    return float(((rgb[2] > 0.30) | (rgb.sum(0) == 0)).mean())


def mosaic(man: pd.DataFrame, m17: pd.DataFrame, x0: float, y1: float, w: int, h: int):
    """RGB canvas (h, w, 3) at CHIP / (512 / DS) m; NaN where no chip."""
    res = CHIP / (512 // DS)
    canvas = np.full((h, w, 3), np.nan, np.float32)
    jobs, used = [], []
    for chip, g in man.groupby("chip"):
        imgs = [read_rgb(U23 / f"s2/{d}_{chip}.tif") for d in g.date]
        best = int(np.argmin([bad_fraction(im) for im in imgs]))
        used.append(g.date.iat[best])
        jobs.append((g.minx.iat[0], g.miny.iat[0], imgs[best]))
    for r in m17[~m17.chip.isin(man.chip)].itertuples():
        jobs.append((r.minx, r.miny, read_rgb(E17 / f"s2/{r.chip}.tif")))
        used.append(r.date)
    print("mosaic dates (locations):", pd.Series(used).value_counts().to_dict())
    for minx, miny, rgb in jobs:
        c, r = int(round((minx - x0) / res)), int(round((y1 - (miny + CHIP)) / res))
        n = rgb.shape[1]
        tile = np.moveaxis(rgb, 0, -1)
        tile[tile.sum(-1) == 0] = np.nan
        canvas[r:r + n, c:c + n] = np.where(np.isnan(tile), canvas[r:r + n, c:c + n], tile)
    # One joint, gentle stretch for all three bands (per-band stretches look neon).
    lo, hi = np.nanpercentile(canvas, [0.5, 98])
    out = np.clip((canvas - lo) / (hi - lo), 0, 1) ** 0.85
    alpha = (~np.isnan(canvas).any(-1)).astype(np.float32)
    return np.dstack([np.nan_to_num(out), alpha])


def hillshade(x0, y1, w, h, res=100.0):
    srcs = [rasterio.open(p) for p in sorted(DEM.glob("*.tif"))]
    dem, tr = merge(srcs, res=srcs[0].res[0] * 3, resampling=Resampling.average)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    dst = np.full((int(h), int(w)), np.nan, np.float32)
    reproject(dem[0].astype(np.float32), dst, src_transform=tr, src_crs=crs, src_nodata=np.nan,
              dst_transform=from_origin(x0, y1, res, res), dst_crs=CRS, resampling=Resampling.bilinear)
    flat = np.isnan(dst)
    dst[flat] = 0.0  # outside the DEM tiles: flat terrain (coastal plain), so no seam in the shading
    hs = LightSource(azdeg=315, altdeg=45).hillshade(dst, vert_exag=2.0, dx=res, dy=res)
    return hs


def main() -> None:
    style.apply()
    man, m17 = chip_tables()
    allxy = pd.concat([man[["minx", "miny"]], m17[["minx", "miny"]]])
    x0, x1 = allxy.minx.min() - PAD, allxy.minx.max() + CHIP + PAD
    y0, y1 = allxy.miny.min() - PAD, allxy.miny.max() + CHIP + PAD
    extent = (x0, x1, y0, y1)
    box_main = box(x0, y0, x1, y1)

    countries = natural_earth("countries")
    peru = countries[countries.ADM0_A3 == "PER"]
    admin1 = natural_earth("admin1")
    admin1 = admin1[admin1.adm0_a3 == "PER"]
    ocean = natural_earth("ocean")
    places = natural_earth("places")
    places = places[(places.ADM0NAME == "Peru") & places.NAME.isin(TOWNS)].to_crs(CRS)
    assert sorted(places.NAME) == sorted(TOWNS), places.NAME.tolist()

    # Figure layout: main map on the left, inset and legend on the right.
    W = style.TEXT_WIDTH
    H = 13.0 * style.CM
    fig = plt.figure(figsize=(W, H))
    map_h = 0.93
    map_w = map_h * H * (x1 - x0) / (y1 - y0) / W
    ax = fig.add_axes([0.075, 0.055, map_w, map_h])
    ax.set_facecolor(SEA)

    # Hillshade over a neutral land colour, clipped to land by drawing the ocean on top.
    res_hs = 100.0
    hs = hillshade(x0, y1, (x1 - x0) / res_hs, (y1 - y0) / res_hs, res_hs)
    land = np.array([int(LAND[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    shade = land[None, None, :] * (0.80 + 0.20 * hs[..., None]) + 0.02
    ax.imshow(np.clip(shade, 0, 1), extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    ocean.clip(LL_BOX).to_crs(CRS).plot(ax=ax, color=SEA, lw=0, zorder=1)

    # Sentinel-2 mosaic.
    res_m = CHIP / (512 // DS)
    rgba = mosaic(man, m17, x0, y1, int(round((x1 - x0) / res_m)), int(round((y1 - y0) / res_m)))
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest", zorder=2)

    # Department boundaries and coast (Natural Earth, reprojected to UTM 17S).
    adm = admin1.clip(LL_BOX).to_crs(CRS)
    draw_lines(ax, adm.geometry, color="#555555", lw=0.5, ls=(0, (4, 2)), zorder=3)
    draw_lines(ax, peru.clip(LL_BOX).to_crs(CRS).geometry, color="#8C8C8C", lw=0.4, zorder=3)

    # Chip footprints, dissolved per split (unique locations).
    order = ["buffer", "train", "dev", "test"]
    colour = dict(style.SPLIT, buffer="#9A9A9A")
    loc = man.drop_duplicates("chip")
    for s in order:
        g = loc[loc.split == s]
        geom = unary_union([box(x, y, x + CHIP, y + CHIP) for x, y in zip(g.minx, g.miny)])
        # Outlines drawn slightly inside each region so that neighbouring splits do not overprint.
        draw_lines(ax, [geom.buffer(-INSET, join_style="mitre")], color=colour[s], lw=1.0, zorder=5)
    geom17 = unary_union([box(x, y, x + CHIP, y + CHIP) for x, y in zip(m17.minx, m17.miny)])
    draw_lines(ax, [geom17.buffer(-3 * INSET, join_style="mitre")], color=style.SPLIT["2017"], lw=1.0,
               ls=(0, (2.5, 1.5)), zorder=6)

    # Place names.
    offsets = {"Piura": (4, 3, "left"), "Chiclayo": (4, 3, "left"), "Sechura": (-4, 0, "right"),
               "Chulucanas": (4, -3, "left"), "Olmos": (4, 0, "left")}
    for r in places.itertuples():
        dx, dy, ha = offsets[r.NAME]
        ax.plot(r.geometry.x, r.geometry.y, "o", ms=2.6, mfc="white", mec=style.INK, mew=0.6, zorder=8)
        ax.annotate(r.NAME, (r.geometry.x, r.geometry.y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7, color=style.INK, zorder=8, path_effects=HALO)
    # Department names (Natural Earth admin-1), in small caps-like style to tell them from towns.
    dep_xy = {"Piura": (633000, 9372000), "Lambayeque": (633000, 9328000)}
    for name, (x, y) in dep_xy.items():
        assert name in admin1.name.values
        ax.text(x, y, name.upper(), fontsize=6.5, color="#555555", ha="center", va="center", zorder=7,
                path_effects=HALO)
    ax.text(530000, 9268000, "Océano\nPacífico", fontsize=7, color="#6A8193",
            ha="center", va="center", fontstyle="italic", zorder=7)

    # Axes in UTM km.
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    xt = np.arange(np.ceil(x0 / 50000) * 50000, x1, 50000)
    yt = np.arange(np.ceil(y0 / 50000) * 50000, y1, 50000)
    ax.set_xticks(xt, [f"{v / 1000:.0f}" for v in xt])
    ax.set_yticks(yt, [f"{v / 1000:.0f}" for v in yt], rotation=90, va="center")
    ax.set_xlabel("Este (km), WGS 84 / UTM 17S", labelpad=2)
    ax.set_ylabel("Norte (km)", labelpad=2)

    # Scale bar (50 km, two segments) and north arrow.
    sx, sy = x0 + 7000, y0 + 8000
    for i, c in enumerate(["#333333", "white", "#333333", "white"]):
        ax.add_patch(Rectangle((sx + i * 12500, sy), 12500, 1600, fc=c, ec="#333333", lw=0.5, zorder=9))
    for d in (0, 25, 50):
        ax.text(sx + d * 1000, sy + 3200, f"{d}", ha="center", va="bottom", fontsize=6.5, zorder=9)
    ax.text(sx + 50000 + 3000, sy + 800, "km", ha="left", va="center", fontsize=6.5, zorder=9)
    nx, ny = x1 - 10000, y1 - 24000
    ax.annotate("", xy=(nx, ny + 14000), xytext=(nx, ny), zorder=9,
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5", lw=0.8, color="#333333"))
    ax.text(nx, ny + 16000, "N", ha="center", va="bottom", fontsize=7.5, fontweight="bold", zorder=9)

    # Right column: inset with Peru and the main extent.
    left = 0.075 + map_w + 0.035
    rw = 0.985 - left
    iax = fig.add_axes([left, 0.47, rw, 0.51])
    sam = countries[countries.CONTINENT == "South America"]
    lim = (-82.5, -68.0, -18.8, 0.5)
    sam.plot(ax=iax, color="#F2F2F0", edgecolor="#B5B5B5", lw=0.3)
    peru.plot(ax=iax, color="#D9D9D4", edgecolor="#6E6E6E", lw=0.5)
    ext_ll = gpd.GeoSeries([box_main], crs=CRS).to_crs("EPSG:4326")
    ext_ll.boundary.plot(ax=iax, color=style.INK, lw=0.9)
    iax.set_xlim(lim[0], lim[1])
    iax.set_ylim(lim[2], lim[3])
    iax.set_aspect(1 / np.cos(np.deg2rad(9)))
    iax.set_facecolor(SEA)
    iax.set_xticks([])
    iax.set_yticks([])
    iax.set_xlabel("")
    iax.set_ylabel("")
    for side in iax.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.5)
    iax.text(-75.3, -9.5, "PERÚ", fontsize=7, ha="center", color="#444444")
    iax.text(-81.8, -14, "Océano\nPacífico", fontsize=6.5, ha="left", color="#6A8193", fontstyle="italic")

    # Legend with counts (locations and chip-dates) computed from the files.
    n_loc = loc.split.value_counts()
    n_cd = man.split.value_counts()
    handles, labels = [], []
    for s in ["train", "dev", "test"]:
        handles.append(Patch(fc="none", ec=colour[s], lw=1.0))
        labels.append(f"{style.SPLIT_NAME[s]}: {n_cd[s]} / {n_loc[s]}")
    handles.append(Patch(fc="none", ec=colour["buffer"], lw=1.0))
    labels.append(f"Amortiguamiento (sin uso):\n{n_cd['buffer']} / {n_loc['buffer']}")
    handles.append(Patch(fc="none", ec=style.SPLIT["2017"], lw=1.0, ls=(0, (2.5, 1.5))))
    labels.append(f"{style.SPLIT_NAME['2017']}: {len(m17)} / {m17.chip.nunique()}")
    lax = fig.add_axes([left, 0.035, rw, 0.40])
    lax.axis("off")
    lax.text(0, 1.0, "Recortes de Sentinel-2", fontsize=7.5, fontweight="bold", va="top")
    lax.text(0, 0.9, "recortes / ubicaciones", fontsize=6.5, color=style.GREY, va="top")
    leg = lax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, 0.82), fontsize=6.8,
                     handlelength=1.6, handleheight=1.0, labelspacing=0.6, borderaxespad=0, borderpad=0)
    lax.text(0, 0.255, "UNOSAT 2023: 2023-03-22 y 2023-04-01", fontsize=6.5, color=style.GREY, va="top")
    lax.text(0, 0.185, "Copernicus EMS 2017: 2017-04-07", fontsize=6.5, color=style.GREY, va="top")
    sym = [plt.Line2D([], [], color="#555555", lw=0.5, ls=(0, (4, 2))),
           plt.Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec=style.INK, mew=0.6)]
    lax.legend(sym, ["Límite departamental", "Ciudad"], loc="upper left", bbox_to_anchor=(0, 0.09),
               fontsize=6.8, handlelength=1.6, borderaxespad=0, borderpad=0)
    lax.add_artist(leg)

    print(style.save(fig, "fig_area_estudio"))
    print("locations:", n_loc.to_dict(), "chip-dates:", n_cd.to_dict(),
          "2017:", m17.chip.nunique(), len(m17), "2017 also in 2023:", m17.chip.isin(man.chip).sum())


if __name__ == "__main__":
    main()
