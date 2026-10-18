"""Flood viewer, Stage 1: run frozen model ensembles on every chip of the existing events.

Viewer data only: no thesis result comes from here (docs/METHODS_NOTES.md 2026-10-16 "Flood
viewer"). It reuses the frozen checkpoints (sha256 checked against configs/frozen.yaml) and the
same input pipeline as the evaluation (floodseg.target_pool.normalize); it does not touch the
one-look outputs of scripts/evaluate_frozen.py.

Per chip and model group (3 seeds), written to outputs/viewer/chips/<event>/<date>/<model>/<chip>.tif
(UTM 17S, uint8, 3 bands):
  1  water probability, mean of the seeds' sigmoid(z)       × 255
  2  uncertainty, std of the seeds' probabilities             × 510 (std ≤ 0.5), clipped
  3  mask: 0 valid · 1 thick cloud (project rule, dilated 5 px) · 2 no data
Once per chip and date: rgb/<chip>.tif (B04/B03/B02, fixed stretch 0–3000 DN, gamma 0.8) and
label/<chip>.tif (0 dry · 1 water · 255 ignore; UNOSAT 2023 or EMS 2017 reference).

    uv run python scripts/viewer/infer_all.py [--limit N]
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from scipy.ndimage import binary_dilation
from terratorch.tasks import SemanticSegmentationTask

from floodseg.target_pool import REPO, model_input_spec, normalize, sha256

OUT = REPO / "outputs" / "viewer" / "chips"
GROUPS = {"peru_unet": "peru/unet", "trhm_unet": "trhm/unet"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 16
CLOUD_BUFFER = 5  # px, as scripts/build_peru_unosat.py


def chips() -> list[dict]:
    """(event, date, stem, image path, label path) for every chip on disk."""
    out = []
    u = REPO / "data" / "peru" / "unosat2023"
    for _, r in pd.read_csv(u / "manifest.csv").iterrows():
        stem = f"{r.date}_{r.chip}"
        out.append(dict(event="yaku2023", date=r.date, chip=r.chip, img=u / "s2" / f"{stem}.tif", lab=u / "label" / f"{stem}.tif"))
    e = REPO / "data" / "peru" / "emsr2017"
    for _, r in pd.read_csv(e / "manifest.csv").iterrows():
        out.append(dict(event="nino2017", date=r.date, chip=r.chip, img=e / "s2" / f"{r.chip}.tif", lab=e / "label" / f"{r.chip}.tif"))
    return out


def cloud_mask(x: np.ndarray) -> np.ndarray:
    nodata = ~(x != 0).any(0)
    thick = binary_dilation((x[1] > 4500) & (x[11] < 1.15 * x[1]), iterations=CLOUD_BUFFER)
    return np.where(nodata, 2, np.where(thick, 1, 0)).astype(np.uint8)


def rgb(x: np.ndarray) -> np.ndarray:
    v = np.clip(x[[3, 2, 1]].astype(np.float32) / 3000, 0, 1) ** 0.8
    return (v * 255).round().astype(np.uint8)


def write(path: Path, arr: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = dict(profile, count=arr.shape[0], dtype="uint8", nodata=None, compress="deflate", predictor=2)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    frozen = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())
    groups = {g["id"]: g for g in frozen["groups"]}
    items = chips()[: a.limit] if a.limit else chips()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()

    # Per-date RGB, label and cloud mask (model-independent).
    for it in items:
        base = OUT / it["event"] / it["date"]
        if (base / "rgb" / f"{it['chip']}.tif").exists():
            continue
        with rasterio.open(it["img"]) as src:
            x, prof = src.read(), src.profile
        with rasterio.open(it["lab"]) as src:
            y = src.read(1)
        write(base / "rgb" / f"{it['chip']}.tif", rgb(x), prof)
        write(base / "label" / f"{it['chip']}.tif", np.where(y < 0, 255, y).astype(np.uint8)[None], prof)
    print(f"rgb/label: {len(items)} chips", flush=True)

    for name, gid in GROUPS.items():
        g = groups[gid]
        spec = model_input_spec(REPO / "configs" / f"{g['model']}.yaml")
        nets = []
        for seed, r in g["runs"].items():
            assert sha256(REPO / r["checkpoint"]) == r["sha256"], f"{gid} {seed}: checkpoint changed since the freeze"
            nets.append(SemanticSegmentationTask.load_from_checkpoint(REPO / r["checkpoint"], map_location="cpu").model.to(DEVICE).eval())
        todo = [it for it in items if not (OUT / it["event"] / it["date"] / name / f"{it['chip']}.tif").exists()]
        for i in range(0, len(todo), BATCH):
            part = todo[i:i + BATCH]
            raws, profs = [], []
            for it in part:
                with rasterio.open(it["img"]) as src:
                    raws.append(src.read()); profs.append(src.profile)
            xb = torch.stack([normalize(r, spec) for r in raws]).to(DEVICE)
            with torch.no_grad():
                p = torch.stack([torch.sigmoid((o := n(xb).output.float())[:, 1] - o[:, 0]) for n in nets]).cpu().numpy()
            mean, std = p.mean(0), p.std(0)
            for k, it in enumerate(part):
                out = np.stack([(mean[k] * 255).round(), np.clip(std[k] * 510, 0, 255).round(), cloud_mask(raws[k])]).astype(np.uint8)
                write(OUT / it["event"] / it["date"] / name / f"{it['chip']}.tif", out, profs[k])
            if i % (BATCH * 10) == 0:
                print(f"{name}: {i + len(part)}/{len(todo)}", flush=True)
        del nets
        if DEVICE == "mps":
            torch.mps.empty_cache()
        (OUT / f"run_{name}.json").write_text(json.dumps(dict(
            group=gid, checkpoints={s: r["checkpoint"] for s, r in g["runs"].items()},
            sha256={s: r["sha256"] for s, r in g["runs"].items()}, chips=len(items), device=DEVICE, git_commit=commit), indent=2))
        print(f"{name}: done", flush=True)


if __name__ == "__main__":
    main()
