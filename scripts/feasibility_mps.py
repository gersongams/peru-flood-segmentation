"""Compute-feasibility test: fine-tune a geospatial foundation model on Sen1Floods11 via TerraTorch.

Measures seconds/iteration (forward + backward + optimizer step) and peak memory for a short
training run on a given device (default: Apple MPS), falling back through batch sizes on OOM.

The model is built through TerraTorch's SemanticSegmentationTask + EncoderDecoderFactory (the same
path a Lightning training run uses); the loss is computed with the task's own loss handler. Chips
are preloaded into RAM so that disk / GeoTIFF decoding time is NOT part of the step timing.

Usage (from repo root):
    uv run python scripts/feasibility_mps.py --backbone terramind_v1_base --batch-sizes 8,4,2
    uv run python scripts/feasibility_mps.py --backbone prithvi_eo_v2_300 --steps 30
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/feasibility_mps.py ...   # CPU fallback for missing ops

Each run prints one line prefixed with `RESULT ` followed by JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

from terratorch.datamodules.sen1floods11 import MEANS, STDS
from terratorch.datasets import Sen1Floods11NonGeo
from terratorch.tasks import SemanticSegmentationTask

HLS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

# Model configs mirror the published fine-tuning configs:
#  - TerraMind: IBM/terramind configs/terramind_v1_base_sen1floods11.yaml (S2L1C only here)
#  - Prithvi-EO-2.0: terrastackai/terratorch examples/segmentation/segmentation_sen1floods11.yaml
#    (its `aux_loss: {aux_head: 1.0}` is omitted: that config defines no aux decoder)
#  - prithvi_eo_v2_300_upernet: self-chosen lighter variant (no pyramid neck, UperNet at 1/16 res.),
#    kept to reproduce the MPS adaptive-pool issue documented in docs/feasibility.md
MODEL_CONFIGS: dict[str, dict] = {
    "terramind_v1_base": {
        "bands": list(Sen1Floods11NonGeo.all_band_names),  # S2Hand is Sentinel-2 L1C, 13 bands
        "model_args": {
            "backbone": "terramind_v1_base",
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L1C"],
            "necks": [
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            "head_dropout": 0.1,
            "num_classes": 2,
        },
    },
    "prithvi_eo_v2_300": {
        "bands": HLS_BANDS,
        "model_args": {
            "backbone": "prithvi_eo_v2_300",
            "backbone_pretrained": True,
            "backbone_bands": HLS_BANDS,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            "head_channel_list": [256],
            "head_dropout": 0.1,
            "num_classes": 2,
        },
    },
    "prithvi_eo_v2_300_upernet": {
        "bands": HLS_BANDS,
        "model_args": {
            "backbone": "prithvi_eo_v2_300",
            "backbone_pretrained": True,
            "backbone_bands": HLS_BANDS,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
            ],
            "decoder": "UperNetDecoder",
            "decoder_channels": 256,
            # Default pool_scales (1, 2, 3, 6) crash on MPS: adaptive_avg_pool2d needs the 32x32
            # feature map (512 px / patch 16) to be divisible by the output size, and 32 % 3 != 0.
            # PYTORCH_ENABLE_MPS_FALLBACK=1 does not help (it is a RuntimeError, not a missing op).
            "decoder_pool_scales": (1, 2, 4, 8),
            "head_dropout": 0.1,
            "num_classes": 2,
        },
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(MODEL_CONFIGS), default="terramind_v1_base")
    p.add_argument("--data-root", default="data/sen1floods11")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--batch-sizes", default="8,4,2", help="tried in order; next one used on OOM")
    p.add_argument("--steps", type=int, default=50, help="timed training steps (after warmup)")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps (kernel compilation etc.)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--img-size", type=int, default=512, help="center crop size (chips are 512x512)")
    p.add_argument("--train-split-size", type=int, default=None,
                   help="chips in the full train split for epoch extrapolation (default: count official CSV)")
    p.add_argument("--max-chips", type=int, default=64, help="chips preloaded into RAM")
    return p.parse_args()


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def device_mem_bytes(device: str) -> int:
    if device == "mps":
        return torch.mps.driver_allocated_memory()
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    return 0


def free_device(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def count_official_train_split(data_root: Path) -> int:
    csv = data_root / "v1.1/splits/flood_handlabeled/flood_train_data.csv"
    with open(csv) as f:
        return sum(1 for line in f if line.strip())


def load_chips(data_root: Path, bands: list[str], img_size: int, max_chips: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the (subset) train split into RAM, normalized with the datamodule's per-band stats."""
    ds = Sen1Floods11NonGeo(data_root=str(data_root), split="train", bands=bands)
    n = min(len(ds), max_chips)
    if n == 0:
        sys.exit(f"No training chips found under {data_root}")
    imgs, masks = [], []
    for i in range(n):
        s = ds[i]
        imgs.append(s["image"])
        masks.append(s["mask"])
    x = torch.stack(imgs).float()
    y = torch.stack(masks).long()
    mean = torch.tensor([MEANS[b] for b in bands]).view(1, -1, 1, 1)
    std = torch.tensor([STDS[b] for b in bands]).view(1, -1, 1, 1)
    x = (x - mean) / std
    if img_size < x.shape[-1]:
        o = (x.shape[-1] - img_size) // 2
        x = x[..., o:o + img_size, o:o + img_size]
        y = y[..., o:o + img_size, o:o + img_size]
    return x, y


def build_task(backbone: str) -> SemanticSegmentationTask:
    cfg = MODEL_CONFIGS[backbone]
    return SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=cfg["model_args"],
        loss="dice",
        ignore_index=-1,
        lr=2e-5,
        freeze_backbone=False,
        freeze_decoder=False,
    )


def run(task, x_all, y_all, bs, args) -> dict:
    device = args.device
    model = task.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    n = x_all.shape[0]
    step_times, peak = [], 0
    total = args.warmup + args.steps
    losses = []
    for step in range(total):
        idx = torch.randint(0, n, (bs,))
        x = x_all[idx].to(device, non_blocking=True)
        y = y_all[idx].to(device, non_blocking=True)
        sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
            loss = task.train_loss_handler.compute_loss(out, y, task.criterion, task.aux_loss)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(device)
        dt = time.perf_counter() - t0
        peak = max(peak, device_mem_bytes(device))
        if device == "mps" and peak > torch.mps.recommended_max_memory():
            # MPS rarely raises OOM; it spills into swap and slows down >100x. Treat as OOM.
            raise torch.OutOfMemoryError(
                f"out of memory: MPS driver memory {peak / 2**30:.1f} GiB exceeds "
                f"recommended_max_memory {torch.mps.recommended_max_memory() / 2**30:.1f} GiB")
        losses.append(loss.item())
        if step >= args.warmup:
            step_times.append(dt)
        print(f"  step {step:3d} bs={bs} {dt:.3f}s loss={losses[-1]:.4f} mem={peak / 2**30:.2f}GiB", flush=True)
    return {
        "step_times": step_times,
        "peak_device_bytes": peak,
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    data_root = Path(args.data_root)
    cfg = MODEL_CONFIGS[args.backbone]
    if args.device == "mps" and not torch.backends.mps.is_available():
        sys.exit("MPS not available")

    train_split_size = args.train_split_size or count_official_train_split(data_root)
    t = time.perf_counter()
    x_all, y_all = load_chips(data_root, cfg["bands"], args.img_size, args.max_chips)
    load_s = time.perf_counter() - t
    print(f"Loaded {x_all.shape[0]} chips {tuple(x_all.shape[1:])} in {load_s:.1f}s", flush=True)

    t = time.perf_counter()
    task = build_task(args.backbone)
    build_s = time.perf_counter() - t
    n_params = sum(p.numel() for p in task.parameters())
    print(f"Built {args.backbone}: {n_params / 1e6:.1f}M params in {build_s:.1f}s", flush=True)

    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        free_device(args.device)
        try:
            r = run(task, x_all, y_all, bs, args)
        except (RuntimeError, torch.OutOfMemoryError) as e:
            msg = str(e).splitlines()[0][:300]
            print(f"RESULT {json.dumps({'backbone': args.backbone, 'batch_size': bs, 'ok': False, 'error': msg})}",
                  flush=True)
            if "out of memory" in str(e).lower() or "MPS backend out of memory" in str(e):
                task.zero_grad(set_to_none=True)
                continue
            raise
        med = statistics.median(r["step_times"])
        steps_per_epoch = train_split_size // bs  # drop_last=True in the datamodule
        result = {
            "backbone": args.backbone,
            "ok": True,
            "device": args.device,
            "precision": args.precision,
            "img_size": args.img_size,
            "batch_size": bs,
            "params_M": round(n_params / 1e6, 1),
            "timed_steps": len(r["step_times"]),
            "s_per_iter_median": round(med, 3),
            "s_per_iter_mean": round(statistics.mean(r["step_times"]), 3),
            "s_per_iter_min": round(min(r["step_times"]), 3),
            "s_per_iter_max": round(max(r["step_times"]), 3),
            "imgs_per_s": round(bs / med, 2),
            "peak_mps_driver_GiB": round(r["peak_device_bytes"] / 2**30, 2),
            "peak_rss_GiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),  # bytes on macOS
            "train_split_size": train_split_size,
            "steps_per_epoch": steps_per_epoch,
            "epoch_train_min_extrapolated": round(steps_per_epoch * med / 60, 2),
            "loss_first_last": [round(r["first_loss"], 4), round(r["last_loss"], 4)],
            "torch": torch.__version__,
            "macos": platform.mac_ver()[0],
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
            "chip_load_s": round(load_s, 1),
            "model_build_s": round(build_s, 1),
        }
        print("RESULT " + json.dumps(result), flush=True)
        break


if __name__ == "__main__":
    main()
