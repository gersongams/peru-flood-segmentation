"""Shared offline evaluation of adapted models on `peru-dev` (phase D).

`adapt_bn`: AdaBN pass (reset BN statistics, exact cumulative average over one pass of the
pool, no gradients, other modules in eval mode). `predict_dev`: save the water logit
z = logit_water − logit_other of every dev chip, as floodseg.callbacks.SaveValPredictions does.
Used by scripts/adabn.py and scripts/eval_adapted.py.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floodseg.target_pool import PERU_IMAGES, REPO, TargetPoolDataset, forbid_peru_labels, load_image, read_manifest

DEV_SPLIT = REPO / "data" / "peru" / "unosat2023" / "splits" / "dev.txt"


def adapt_bn(model: torch.nn.Module, spec: dict, device: str, batch: int, pool: str = "peru-train") -> tuple[int, int]:
    """Re-estimate every BatchNorm layer of `model` on the pool. Returns (BN layers, pool images)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the whole pass
        m.train()
    with forbid_peru_labels():
        paths = read_manifest(pool)
        with torch.no_grad():
            for b in DataLoader(TargetPoolDataset(paths, spec), batch_size=batch, shuffle=False, num_workers=0):
                model(b["image"].to(device))
    model.eval()
    return len(bns), len(paths)


def predict_dev(model: torch.nn.Module, spec: dict, device: str, batch: int, out: Path) -> int:
    stems = DEV_SPLIT.read_text().split()
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(stems), batch):
            part = stems[i:i + batch]
            x = torch.stack([load_image(PERU_IMAGES / f"{s}.tif", spec) for s in part]).to(device)
            o = model(x).output.float()
            for s, z in zip(part, (o[:, 1] - o[:, 0]).cpu().numpy().astype(np.float32)):
                np.save(out / f"{s}.npy", z)
    return len(stems)
