"""Full-scene processing: one S2 L1C tile and date → water probability, uncertainty and mask.

Stages (each timed): auth → catalog → download → read (JP2 decode, offset, 10 m) → mask →
inference → write (COG GeoTIFFs with overviews, which the map tiles are cut from).
Outputs in outputs/system/scenes/<tile>_<date>/:
  probability.tif  uint16, water probability (seed mean) × 10 000, nodata 65535
  uncertainty.tif  uint16, seed std × 10 000, nodata 65535
  mask.tif         uint8, 0 valid · 1 thick cloud · 2 no data
  rgb.tif          uint8 ×3, B04/B03/B02 preview (0–3000 DN, gamma 0.8)
  meta.json        product, offsets, model checkpoints, commit, stage times
The product archive is deleted after processing; the outputs are the cache (RF2b).

    uv run python -m floodseg.system.pipeline 17MNP 2023-03-22 [--keep-archive] [--reuse-archive]
"""

import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from floodseg.system import cdse
from floodseg.system.infer import Ensemble
from floodseg.system.scene import cloud_nodata_mask, radiometric_offsets, read_stack
from floodseg.target_pool import REPO

SCENES = REPO / "outputs" / "system" / "scenes"
DOWNLOADS = REPO / "data" / "system" / "downloads"
SCALE = 10_000
NODATA16 = 65535
STAGES = ["auth", "catalog", "download", "read", "mask", "inference", "write"]  # timed path; "compare" is optional

Progress = Callable[[str, float, str], None]  # (stage, fraction 0–1, detail)


def scene_dir(tile: str, date: str) -> Path:
    return SCENES / f"{tile}_{date}"


def is_cached(tile: str, date: str) -> bool:
    return (scene_dir(tile, date) / "meta.json").exists()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO)
    d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO)
    return r.stdout.strip() + ("-dirty" if d.stdout.strip() else "")


def _write_cog(path: Path, arr: np.ndarray, profile: dict, nodata, resampling: str) -> None:
    arr = arr if arr.ndim == 3 else arr[None]
    p = dict(driver="COG", width=profile["width"], height=profile["height"], count=arr.shape[0],
             dtype=arr.dtype.name, crs=profile["crs"], transform=profile["transform"], nodata=nodata,
             compress="DEFLATE", predictor=2, blocksize=512, overview_resampling=resampling,
             num_threads="ALL_CPUS")
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **p) as dst:
        dst.write(arr)
    tmp.rename(path)


def _quantize(v: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    q = np.clip(np.rint(v * SCALE), 0, SCALE).astype(np.uint16)
    q[invalid] = NODATA16
    return q


def _rgb(stack: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    out = np.empty((3, *stack.shape[1:]), np.uint8)
    for k, b in enumerate((3, 2, 1)):
        v = np.clip(stack[b].astype(np.float32) / 3000, 0, 1) ** 0.8
        out[k] = np.rint(v * 255)
        out[k][nodata] = 0
    return out


def run(tile: str, date: str, ensemble: Ensemble, progress: Progress | None = None,
        keep_archive: bool = False, reuse_archive: bool = False, compare: Ensemble | None = None) -> dict:
    """Process one tile and date end to end. Returns the meta dict (also written to meta.json)."""
    say = progress or (lambda s, f, d="": None)
    out = scene_dir(tile, date)
    times, stamps = {}, {"start": _now()}
    t0 = time.perf_counter()

    def mark(stage: str, since: float) -> float:
        now = time.perf_counter()
        times[stage] = round(now - since, 2)
        say(stage, 1.0, f"{times[stage]:.1f} s")
        return now

    say("auth", 0.0, "")
    tok = cdse.token()
    t = mark("auth", t0)

    say("catalog", 0.0, "")
    products = cdse.find_l1c(tile, date)
    if not products:
        raise LookupError(f"no L1C product for tile {tile} on {date}")
    product = products[0]
    t = mark("catalog", t)

    archive = DOWNLOADS / f"{product['Name']}.zip"
    say("download", 0.0, product["Name"])
    reused = reuse_archive and archive.exists()  # development only (CLI flag); the app always downloads
    if not reused:
        archive.unlink(missing_ok=True)
        cdse.download(product, archive, tok, lambda d, n: say("download", d / n if n else 0.0, f"{d >> 20} / {n >> 20} MB"))
    t = mark("download", t)

    say("read", 0.0, "13 bandas")
    stack, profile = read_stack(archive)
    offsets = radiometric_offsets(archive)
    t = mark("read", t)

    say("mask", 0.0, "")
    mask = cloud_nodata_mask(stack)
    nodata = mask == 2
    t = mark("mask", t)

    say("inference", 0.0, "")
    prob, std = ensemble.predict_tile(stack, nodata, profile["transform"], progress=lambda d, n: say("inference", d / n, f"{d} / {n} ventanas"))
    t = mark("inference", t)

    say("write", 0.0, "")
    out.mkdir(parents=True, exist_ok=True)
    _write_cog(out / "probability.tif", _quantize(prob, nodata), profile, NODATA16, "average")
    _write_cog(out / "uncertainty.tif", _quantize(std, nodata), profile, NODATA16, "average")
    _write_cog(out / "mask.tif", mask, profile, None, "nearest")
    _write_cog(out / "rgb.tif", _rgb(stack, nodata), profile, None, "average")
    t = mark("write", t)
    stamps["stop"] = _now()  # HE3 timer stop: the selected model's GeoTIFFs are written
    total_s = round(time.perf_counter() - t0, 2)

    comparison = None
    if compare is not None:  # optional second model, after the timer (protocol: only the selected model is timed)
        say("compare", 0.0, compare.group_id)
        p2, _ = compare.predict_tile(stack, nodata, profile["transform"], progress=lambda d, n: say("compare", d / n, f"{d} / {n} ventanas"))
        _write_cog(out / "probability_compare.tif", _quantize(p2, nodata), profile, NODATA16, "average")
        t = mark("compare", t)
        comparison = {"model_group": compare.group_id, "checkpoints": compare.checkpoints, "seconds": times["compare"],
                      "water_valid": int(((p2 > 0.5) & (mask == 0)).sum())}

    valid = mask == 0
    meta = {
        "tile": tile, "date": date, "product": product["Name"], "product_id": product["Id"],
        "product_bytes": product["ContentLength"], "archive_reused": reused, "radio_add_offset": offsets.tolist(),
        "model_group": ensemble.group_id, "checkpoints": ensemble.checkpoints,
        "crs": str(profile["crs"]), "transform": list(profile["transform"])[:6],
        "shape": [profile["height"], profile["width"]], "scale": SCALE, "nodata": NODATA16,
        "water_rule": "probability > 0.5",
        "pixels": {"valid": int(valid.sum()), "cloud": int((mask == 1).sum()), "nodata": int(nodata.sum()),
                   "water_valid": int(((prob > 0.5) & valid).sum())},
        "times_s": times, "total_s": total_s, "timestamps": stamps, "comparison": comparison,
        "git_commit": _commit(), "device": ensemble.device, "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if not keep_archive:
        archive.unlink(missing_ok=True)
    return meta


def clear(tile: str, date: str) -> None:
    """Empty the cache of one tile and date (outputs and any archive), as before a timed run."""
    shutil.rmtree(scene_dir(tile, date), ignore_errors=True)
    for f in DOWNLOADS.glob(f"*_T{tile}_*"):
        if f"_{date.replace('-', '')}T" in f.name:
            f.unlink()


if __name__ == "__main__":
    tile, date = sys.argv[1], sys.argv[2]
    keep = "--keep-archive" in sys.argv
    reuse = "--reuse-archive" in sys.argv
    ens = Ensemble("peru/unet")
    last = {}

    def show(stage: str, frac: float, detail: str) -> None:
        if frac in (0.0, 1.0) or frac - last.get(stage, 0) >= 0.1:
            last[stage] = frac
            print(f"[{datetime.now():%H:%M:%S}] {stage:<10} {frac:6.1%} {detail}", flush=True)

    m = run(tile, date, ens, show, keep_archive=keep, reuse_archive=reuse)
    print(json.dumps({k: m[k] for k in ("product", "times_s", "total_s", "pixels")}, indent=2))
