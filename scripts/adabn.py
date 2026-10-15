"""AdaBN (phase-D task 3.3): re-estimate BatchNorm statistics on the unlabelled Peru target
pool, then save `peru-dev` logits. U-Net: all 46 BN layers. TerraMind: its encoder has no BN;
the 9 BN layers of its UNet decoder and neck (design amendment 2026-10-15).

AdaBN (Li et al. 2016): keep all weights of the source model; replace every BatchNorm layer's
running mean/variance by statistics of the target domain. Here: all 46 BatchNorm2d layers of
the U-Net (encoder and decoder) are reset and re-estimated as an exact cumulative average
(momentum=None) over one pass of the 753 `configs/pools/peru-train.csv` images (batch 16; TerraMind 8), no
augmentation, no gradients, under the Peru-label guard. Then eval mode, and the water logit
z = logit_water − logit_other of every `peru-dev` chip is saved as SaveValPredictions does
(`outputs/logits_dev/adabn/<model>/seed<s>/<stem>.npy`) with a run record `run.json`.
Dev scoring happens afterwards in scripts/phaseD_dev_table.py (offline, as the spec requires).

    uv run python scripts/adabn.py [unet|terramind_v1_base]
"""

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from terratorch.tasks import SemanticSegmentationTask
from torch.utils.data import DataLoader

from floodseg.target_pool import (PERU_IMAGES, REPO, TargetPoolDataset, forbid_peru_labels, load_image,
                                  model_input_spec, read_manifest)

SOURCES = {"unet": "s1f11_src_unet", "terramind_v1_base": "s1f11_src_terramind_lr5e-5"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCHES = {"unet": 16, "terramind_v1_base": 8}  # TerraMind fits batch 8 on the Mac


def adapt_bn(model: torch.nn.Module, loader) -> int:
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the whole pass
        m.train()
    with torch.no_grad():
        for b in loader:
            model(b["image"].to(DEVICE))
    model.eval()
    return len(bns)


def main(model: str = "unet") -> None:
    src_dir = REPO / "outputs" / SOURCES[model] / model / "lightning_logs"
    out_dir = REPO / "outputs" / "logits_dev" / "adabn" / model
    spec = model_input_spec(REPO / "configs" / f"{model}.yaml")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    batch = BATCHES[model]
    stems = (REPO / "data" / "peru" / "unosat2023" / "splits" / "dev.txt").read_text().split()
    for seed in (0, 1, 2):
        ckpt = Path(glob.glob(str(src_dir / f"seed{seed}" / "checkpoints" / "best-*.ckpt"))[0])
        task = SemanticSegmentationTask.load_from_checkpoint(ckpt, map_location="cpu").to(DEVICE)
        t0 = time.time()
        with forbid_peru_labels():
            paths = read_manifest("peru-train")
            loader = DataLoader(TargetPoolDataset(paths, spec), batch_size=batch, shuffle=False, num_workers=0)
            n_bn = adapt_bn(task.model, loader)
        adapt_s = time.time() - t0
        out = out_dir / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for i in range(0, len(stems), batch):
                part = stems[i:i + batch]
                x = torch.stack([load_image(PERU_IMAGES / f"{s}.tif", spec) for s in part]).to(DEVICE)
                o = task.model(x).output.float()
                for s, z in zip(part, (o[:, 1] - o[:, 0]).cpu().numpy().astype(np.float32)):
                    np.save(out / f"{s}.npy", z)
        rec = dict(method="AdaBN", model=model, source_ckpt=str(ckpt.relative_to(REPO)), pool="peru-train", pool_images=len(paths),
                   bn_layers=n_bn, batch=batch, device=DEVICE, adapt_seconds=round(adapt_s, 1),
                   git_commit=commit, git_dirty=dirty, dev_files=len(stems))
        (out / "run.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
