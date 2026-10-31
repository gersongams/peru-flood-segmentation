"""Input-band importance of the Peru-trained models by permutation (course deliverable "Avance
sem 7"; exploratory, dev-only: reads `peru-dev` images and labels, no test split).

Permutation importance for an image model: band b (or a group of bands) of every chip is replaced
by the same band of another chip, drawn by a random permutation of the 174 `peru-dev` chips.
This keeps the band's values and spatial texture but breaks its link with the other bands and the
label. The importance is the drop in pooled water IoU (label != −1, water if z > 0, i.e.
p > 0.5) against the unpermuted input. Bands are permuted on raw DN, before each model's
official normalization. Groups (correlated bands share importance, so groups are also permuted
together): visible = coastal aerosol, blue, green, red; red edge = RE1–RE3; NIR = broad and
narrow NIR; SWIR = SWIR1, SWIR2; atmospheric = water vapour, cirrus. Only the bands a model reads
are permuted for it.

Models: the frozen `peru/<model>` groups of configs/frozen.yaml (trained on `peru-train`),
seed 0, 2 permutations per band or group (the same permutations for every model).

    uv run python scripts/variable_importance_dev.py [--seeds seed0 seed1 seed2] [--repeats 2]
→ outputs/variable_importance/{runs,importance}.csv, run.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from terratorch.tasks import SemanticSegmentationTask

from floodseg.normalization import S2_L1C_BANDS
from floodseg.provenance import git_state
from floodseg.target_pool import REPO, model_input_spec, normalize, sha256

UNOSAT = REPO / "data" / "peru" / "unosat2023"
OUT = REPO / "outputs" / "variable_importance"
MODELS = ["unet", "terramind_v1_base", "prithvi_eo_v2_300"]
BATCHES = {"unet": 16, "terramind_v1_base": 8, "prithvi_eo_v2_300": 4}
GROUPS = {
    "visible": ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED"],
    "red_edge": ["RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "nir": ["NIR_BROAD", "NIR_NARROW"],
    "swir": ["SWIR_1", "SWIR_2"],
    "atmospheric": ["WATER_VAPOR", "CIRRUS"],
}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_dev() -> tuple[list[str], np.ndarray, np.ndarray]:
    stems = (UNOSAT / "splits" / "dev.txt").read_text().split()
    xs, ys = [], []
    for s in stems:
        with rasterio.open(UNOSAT / "s2" / f"{s}.tif") as src:
            xs.append(src.read().astype(np.float32))
        with rasterio.open(UNOSAT / "label" / f"{s}.tif") as src:
            ys.append(src.read(1).astype(np.int8))
    return stems, np.stack(xs), np.stack(ys)


def iou(model: torch.nn.Module, spec: dict, x: np.ndarray, y: np.ndarray, batch: int) -> float:
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            inp = torch.stack([normalize(r, spec) for r in x[i:i + batch]]).to(DEVICE)
            o = model(inp).output.float()
            pred = ((o[:, 1] - o[:, 0]) > 0).cpu().numpy()
            yy = y[i:i + batch]
            ok, w = yy != -1, yy == 1
            tp += int((pred & w & ok).sum())
            fp += int((pred & ~w & ok).sum())
            fn += int((~pred & w & ok).sum())
    return tp / (tp + fp + fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=["seed0"])
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {g["id"]: g for g in yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())["groups"]}
    stems, x, y = load_dev()
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(stems)) for _ in range(a.repeats)]
    rows = []
    t_all = time.time()
    for m in MODELS:
        spec = model_input_spec(REPO / "configs" / f"{m}.yaml")
        used = spec["output_bands"]
        units = [(b, "banda", [b]) for b in used]
        units += [(g, "grupo", [b for b in bs if b in used]) for g, bs in GROUPS.items()]
        units = [u for u in units if u[2]]
        for seed in a.seeds:
            r = frozen[f"peru/{m}"]["runs"][seed]
            assert sha256(REPO / r["checkpoint"]) == r["sha256"], f"{m} {seed}: checkpoint changed"
            task = SemanticSegmentationTask.load_from_checkpoint(REPO / r["checkpoint"], map_location="cpu").to(DEVICE)
            model = task.model.eval()
            t0 = time.time()
            base = iou(model, spec, x, y, BATCHES[m])
            rows.append(dict(model=m, seed=seed, unit="ninguna", kind="base", repeat=-1, iou=base))
            print(f"{m} {seed} base IoU {base:.4f} ({time.time() - t0:.0f} s)", flush=True)
            for name, kind, bands in units:
                idx = [S2_L1C_BANDS.index(b) for b in bands]
                for k, p in enumerate(perms):
                    xp = x.copy()
                    xp[:, idx] = x[p][:, idx]
                    v = iou(model, spec, xp, y, BATCHES[m])
                    rows.append(dict(model=m, seed=seed, unit=name, kind=kind, repeat=k, iou=v))
                print(f"  {name}: {base - np.mean([q['iou'] for q in rows[-len(perms):]]):+.4f}", flush=True)
            del task, model
            if DEVICE == "mps":
                torch.mps.empty_cache()

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    base = runs[runs.kind == "base"].set_index(["model", "seed"]).iou
    perm = runs[runs.kind != "base"].copy()
    perm["drop"] = base.reindex(pd.MultiIndex.from_frame(perm[["model", "seed"]])).to_numpy() - perm.iou
    imp = perm.groupby(["model", "unit", "kind"]).drop.agg(["mean", "std", "count"]).reset_index()
    imp.rename(columns={"mean": "iou_drop", "std": "iou_drop_std", "count": "n"}).to_csv(OUT / "importance.csv", index=False)
    (OUT / "run.json").write_text(json.dumps(dict(
        split="peru-dev", chips=len(stems), seeds=a.seeds, repeats=a.repeats, perm_seed=0, device=DEVICE,
        seconds=round(time.time() - t_all),
        git={k: v for k, v in git_state().items() if k in ('commit', 'branch', 'dirty')}), indent=2))


if __name__ == "__main__":
    main()
