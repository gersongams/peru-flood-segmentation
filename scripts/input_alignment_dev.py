"""Test-time input alignment on `peru-dev` (phase-D task 3.4, part 1; design amendment
2026-10-15 in docs/METHODS_NOTES.md).

The unchanged phase-D source models (U-Net, TerraMind; 3 seeds each) predict on dev images
transformed towards Sen1Floods11:
- `hm`:  per-band histogram matching, global map fitted from the unlabelled pool CDF to the
         Sen1Floods11 `train` image CDF (floodseg.input_alignment.hist_match);
- `fda`: per band, low-frequency amplitude (β = 0.01) replaced by the mean amplitude of the
         Sen1Floods11 `train` images.
Fitted parameters (pool and source quantiles, source mean amplitude) are computed from images
only and cached in outputs/phaseD/alignment/. Logits go to
outputs/logits_dev/<hm|fda>/<model>/seed<s>/ with run.json; scoring is offline
(scripts/phaseD_dev_table.py).

    uv run python scripts/input_alignment_dev.py [model ...]   # default: all models
"""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from terratorch.tasks import SemanticSegmentationTask

from floodseg.input_alignment import fda, hist_match, mean_amplitude, quantiles, sample_pixels
from floodseg.sen1floods11 import CHIP_DIR, official_ids
from floodseg.target_pool import PERU_IMAGES, REPO, forbid_peru_labels, model_input_spec, normalize, read_manifest

SOURCES = {"unet": "s1f11_src_unet", "terramind_v1_base": "s1f11_src_terramind_lr5e-5",
           "prithvi_eo_v2_300": "s1f11_src_prithvi"}
BATCHES = {"unet": 16, "terramind_v1_base": 8, "prithvi_eo_v2_300": 4}
CACHE = REPO / "outputs" / "phaseD" / "alignment"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BETA = 0.01


def fit() -> dict:
    f = CACHE / "params.npz"
    if f.exists():
        return dict(np.load(f))
    CACHE.mkdir(parents=True, exist_ok=True)
    s1f = [CHIP_DIR / "S2Hand" / f"{i}_S2Hand.tif" for i in official_ids("train")]
    with forbid_peru_labels():
        q_pool = quantiles(sample_pixels(read_manifest("peru-train")))
    p = dict(q_pool=q_pool, q_s1f=quantiles(sample_pixels(s1f)), amp_s1f=mean_amplitude(s1f))
    np.savez(f, **p)
    return p


def main(*models: str) -> None:
    p = fit()
    transforms = {"hm": lambda x: hist_match(x, p["q_pool"], p["q_s1f"]),
                  "fda": lambda x: fda(x, p["amp_s1f"], BETA)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    stems = (REPO / "data" / "peru" / "unosat2023" / "splits" / "dev.txt").read_text().split()
    for model, folder in SOURCES.items():
        if models and model not in models:
            continue
        spec = model_input_spec(REPO / "configs" / f"{model}.yaml")
        for seed in (0, 1, 2):
            ckpt = Path(glob.glob(str(REPO / "outputs" / folder / model / "lightning_logs" / f"seed{seed}" / "checkpoints" / "best-*.ckpt"))[0])
            task = SemanticSegmentationTask.load_from_checkpoint(ckpt, map_location="cpu").to(DEVICE).eval()
            for name, tf in transforms.items():
                out = REPO / "outputs" / "logits_dev" / name / model / f"seed{seed}"
                out.mkdir(parents=True, exist_ok=True)
                b = BATCHES[model]
                with torch.no_grad():
                    for i in range(0, len(stems), b):
                        part = stems[i:i + b]
                        xs = []
                        for s in part:
                            with rasterio.open(PERU_IMAGES / f"{s}.tif") as src:
                                xs.append(normalize(tf(src.read()), spec))
                        o = task.model(torch.stack(xs).to(DEVICE)).output.float()
                        for s, z in zip(part, (o[:, 1] - o[:, 0]).cpu().numpy().astype(np.float32)):
                            np.save(out / f"{s}.npy", z)
                rec = dict(method=name, model=model, source_ckpt=str(ckpt.relative_to(REPO)), beta=BETA if name == "fda" else None,
                           params=str((CACHE / "params.npz").relative_to(REPO)), git_commit=commit, git_dirty=dirty,
                           device=DEVICE, dev_files=len(stems))
                (out / "run.json").write_text(json.dumps(rec, indent=2))
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
