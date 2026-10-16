"""Sen1Floods11 `valid` logits of the phase-D source models (3 seeds, U-Net + TerraMind), for
fitting a label-free (with respect to Peru) temperature in scripts/phaseD_analysis.py.

Each official `valid` chip (89) is read with floodseg.target_pool.load_image (the config's bands,
means/stds, NaN → no_data_replace), predicted by the source checkpoint (`best-*.ckpt`, chosen on
`valid`), and the water logit z = logit_water − logit_other saved as float32 to
outputs/logits_valid/s1f11_src/<model>/seed<s>/<id>_S2Hand.npy with run.json.
Check: pooled water IoU from these logits vs the best `val/IoU_water`-type value in the run's
metrics.csv (printed side by side).

    uv run python scripts/phaseD_valid_logits.py [model ...]   # default: all models
"""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from terratorch.tasks import SemanticSegmentationTask

from floodseg.metrics import counts, scores
from floodseg.sen1floods11 import CHIP_DIR, official_ids
from floodseg.target_pool import REPO, load_image, model_input_spec

SOURCES = {"unet": "s1f11_src_unet", "terramind_v1_base": "s1f11_src_terramind_lr5e-5",
           "prithvi_eo_v2_300": "s1f11_src_prithvi"}
BATCHES = {"unet": 8, "terramind_v1_base": 8, "prithvi_eo_v2_300": 4}  # Prithvi-300M fits batch 4 on the Mac
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main(*models: str) -> None:
    ids = official_ids("valid")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    for model, folder in SOURCES.items():
        if models and model not in models:
            continue
        spec = model_input_spec(REPO / "configs" / f"{model}.yaml")
        for seed in (0, 1, 2):
            run = REPO / "outputs" / folder / model / "lightning_logs" / f"seed{seed}"
            ckpt = Path(glob.glob(str(run / "checkpoints" / "best-*.ckpt"))[0])
            net = SemanticSegmentationTask.load_from_checkpoint(ckpt, map_location="cpu").model.to(DEVICE).eval()
            out = REPO / "outputs" / "logits_valid" / "s1f11_src" / model / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            c = np.zeros((2, 3), np.int64)
            with torch.no_grad():
                for i in range(0, len(ids), BATCHES[model]):
                    part = ids[i:i + BATCHES[model]]
                    x = torch.stack([load_image(CHIP_DIR / "S2Hand" / f"{s}_S2Hand.tif", spec) for s in part]).to(DEVICE)
                    o = net(x).output.float()
                    for s, z in zip(part, (o[:, 1] - o[:, 0]).cpu().numpy().astype(np.float32)):
                        np.save(out / f"{s}_S2Hand.npy", z)
                        with rasterio.open(CHIP_DIR / "LabelHand" / f"{s}_LabelHand.tif") as src:
                            c += counts(z > 0, src.read(1))
            m = pd.read_csv(run / "metrics.csv")
            col = [k for k in m.columns if k.startswith("val") and "IoU" in k and ("water" in k.lower() or "flood" in k.lower())]
            logged = float(m[col[0]].max()) if col else float("nan")
            rec = dict(model=model, seed=seed, source_ckpt=str(ckpt.relative_to(REPO)), split="valid", chips=len(ids),
                       pooled_water_iou=scores(c)["iou"], logged_best_val_column=col[0] if col else None,
                       logged_best_val=logged, git_commit=commit, device=DEVICE)
            (out / "run.json").write_text(json.dumps(rec, indent=2))
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
