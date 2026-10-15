"""Run ON the Mac. Stage the phase-D source checkpoints for upload, weights only.

Copies each source run's run.json and a slim best-*.ckpt (state_dict, hyper_parameters and
Lightning version; no optimizer state) to outputs/phaseD/upload/ with the repo layout, and
checks that the slim file loads and gives the same weights.

    uv run python scripts/cloud/stage_phaseD.py
"""

import glob
import shutil
from pathlib import Path

import torch
from terratorch.tasks import SemanticSegmentationTask

REPO = Path(__file__).resolve().parents[2]
SOURCES = {"unet": "s1f11_src_unet", "terramind_v1_base": "s1f11_src_terramind_lr5e-5"}
UP = REPO / "outputs" / "phaseD" / "upload"

for model, folder in SOURCES.items():
    for seed in (0, 1, 2):
        src = REPO / "outputs" / folder / model / "lightning_logs" / f"seed{seed}"
        ckpt = Path(glob.glob(str(src / "checkpoints" / "best-*.ckpt"))[0])
        dst = UP / "outputs" / folder / model / "lightning_logs" / f"seed{seed}"
        (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "run.json", dst / "run.json")
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        slim = {k: c[k] for k in ("state_dict", "hyper_parameters", "pytorch-lightning_version")}
        torch.save(slim, dst / "checkpoints" / ckpt.name)
        t = SemanticSegmentationTask.load_from_checkpoint(dst / "checkpoints" / ckpt.name, map_location="cpu")
        assert all(torch.equal(v, c["state_dict"][k]) for k, v in t.state_dict().items()), ckpt
        print(f"{model} seed{seed}: {ckpt.stat().st_size / 1e6:.0f} MB -> {(dst / 'checkpoints' / ckpt.name).stat().st_size / 1e6:.0f} MB, loads, weights equal")
