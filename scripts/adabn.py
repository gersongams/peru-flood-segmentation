"""AdaBN for the U-Net (phase-D task 3.3): re-estimate BatchNorm statistics on the unlabelled
Peru target pool, then save `peru-dev` logits.

AdaBN (Li et al. 2016): keep all weights of the source model; replace every BatchNorm layer's
running mean/variance by statistics of the target domain. Here: all 46 BatchNorm2d layers of
the U-Net (encoder and decoder) are reset and re-estimated as an exact cumulative average
(momentum=None) over one pass of the 753 `configs/pools/peru-train.csv` images, batch 16, no
augmentation, no gradients, under the Peru-label guard. Then eval mode, and the water logit
z = logit_water − logit_other of every `peru-dev` chip is saved as SaveValPredictions does
(`outputs/logits_dev/adabn/unet/seed<s>/<stem>.npy`) with a run record `run.json`.
Dev scoring happens afterwards in scripts/phaseD_dev_table.py (offline, as the spec requires).

    uv run python scripts/adabn.py
"""

import glob
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from terratorch.tasks import SemanticSegmentationTask
from torch.utils.data import DataLoader

from floodseg.target_pool import (PERU_IMAGES, REPO, TargetPoolDataset, forbid_peru_labels, load_image,
                                  model_input_spec, read_manifest)

SRC = REPO / "outputs" / "s1f11_src_unet" / "unet" / "lightning_logs"
OUT = REPO / "outputs" / "logits_dev" / "adabn" / "unet"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 16


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


def main() -> None:
    spec = model_input_spec(REPO / "configs" / "unet.yaml")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    stems = (REPO / "data" / "peru" / "unosat2023" / "splits" / "dev.txt").read_text().split()
    for seed in (0, 1, 2):
        ckpt = Path(glob.glob(str(SRC / f"seed{seed}" / "checkpoints" / "best-*.ckpt"))[0])
        task = SemanticSegmentationTask.load_from_checkpoint(ckpt, map_location="cpu").to(DEVICE)
        t0 = time.time()
        with forbid_peru_labels():
            paths = read_manifest("peru-train")
            loader = DataLoader(TargetPoolDataset(paths, spec), batch_size=BATCH, shuffle=False, num_workers=0)
            n_bn = adapt_bn(task.model, loader)
        adapt_s = time.time() - t0
        out = OUT / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for i in range(0, len(stems), BATCH):
                part = stems[i:i + BATCH]
                x = torch.stack([load_image(PERU_IMAGES / f"{s}.tif", spec) for s in part]).to(DEVICE)
                o = task.model(x).output.float()
                for s, z in zip(part, (o[:, 1] - o[:, 0]).cpu().numpy().astype(np.float32)):
                    np.save(out / f"{s}.npy", z)
        rec = dict(method="AdaBN", source_ckpt=str(ckpt.relative_to(REPO)), pool="peru-train", pool_images=len(paths),
                   bn_layers=n_bn, batch=BATCH, device=DEVICE, adapt_seconds=round(adapt_s, 1),
                   git_commit=commit, git_dirty=dirty, dev_files=len(stems))
        (out / "run.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
