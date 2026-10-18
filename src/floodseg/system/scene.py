"""Read a whole S2 L1C SAFE archive into the 13-band, 10 m DN stack the models were trained on.

The training chips came from the Sentinel Hub Process API (scripts/build_peru_unosat.py):
- `harmonizeValues` (default) removes the RADIO_ADD_OFFSET (−1000 DN) of processing baseline
  ≥ 04.00, so values match pre-2022 L1C. Here the offset is read per band from MTD_MSIL1C.xml
  and applied the same way: DN + offset, clipped at 0, no-data (DN 0) kept at 0.
- Upsampling default NEAREST. The 20 m and 60 m grids of a tile share the 10 m grid's origin,
  so nearest upsampling is pixel replication (np.repeat).
"""

import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
CLOUD_BUFFER = 5  # px, as scripts/build_peru_unosat.py and scripts/viewer/infer_all.py


def band_paths(archive: Path) -> dict[str, str]:
    """GDAL /vsizip/ path of each band's JP2 inside the product zip."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    out = {}
    for b in BANDS:
        hits = [n for n in names if re.search(rf"/IMG_DATA/[^/]*_{b}\.jp2$", n)]
        if len(hits) != 1:
            raise ValueError(f"{archive.name}: {len(hits)} files for band {b}")
        out[b] = f"/vsizip/{archive}/{hits[0]}"
    return out


def radiometric_offsets(archive: Path) -> np.ndarray:
    """RADIO_ADD_OFFSET per band (0 for baselines before 04.00, which have none)."""
    with zipfile.ZipFile(archive) as z:
        mtd = next(n for n in z.namelist() if n.endswith("MTD_MSIL1C.xml"))
        xml = z.read(mtd).decode()
    off = np.zeros(len(BANDS), np.int32)
    for bid, val in re.findall(r'<RADIO_ADD_OFFSET band_id="(\d+)">(-?\d+)</RADIO_ADD_OFFSET>', xml):
        off[int(bid)] = int(val)
    return off


def read_stack(archive: Path, threads: int = 13) -> tuple[np.ndarray, dict]:
    """(13, H, W) uint16 harmonized DN at 10 m, plus the rasterio profile of the 10 m grid."""
    paths = band_paths(archive)
    with rasterio.open(paths["B02"]) as ref:
        h, w = ref.height, ref.width
        profile = dict(ref.profile, driver="GTiff", count=1)
    offsets = radiometric_offsets(archive)

    def load(i: int) -> tuple[int, np.ndarray]:
        with rasterio.Env(GDAL_NUM_THREADS="2"), rasterio.open(paths[BANDS[i]]) as src:
            a = src.read(1)
        f = h // a.shape[0]
        if f > 1:
            a = np.repeat(np.repeat(a, f, axis=0), f, axis=1)
        a = a[:h, :w]
        if offsets[i]:
            v = a.astype(np.int32) + offsets[i]
            a = np.where(a == 0, 0, np.clip(v, 0, 65535)).astype(np.uint16)
        return i, a

    stack = np.empty((len(BANDS), h, w), np.uint16)
    with ThreadPoolExecutor(threads) as ex:
        for i, a in ex.map(load, range(len(BANDS))):
            stack[i] = a
    return stack, profile


def cloud_nodata_mask(stack: np.ndarray) -> np.ndarray:
    """0 valid · 1 thick cloud (project rule, dilated 5 px) · 2 no data (all bands 0)."""
    nodata = ~(stack != 0).any(0)
    blue, swir = stack[1], stack[11].astype(np.float32)
    thick = binary_dilation((blue > 4500) & (swir < 1.15 * blue), iterations=CLOUD_BUFFER)
    return np.where(nodata, 2, np.where(thick, 1, 0)).astype(np.uint8)
