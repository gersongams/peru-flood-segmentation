"""Run phase-D self-training from a Sen1Floods11 source checkpoint (tasks 4.1–4.3, 4.6, 4.8).

    uv run python scripts/self_train.py --model unet --seed 0 [--variant st|mic] [--iters 4000] [--demo]

Defaults are the pre-registered configuration (design.md decision 9, METHODS_NOTES 2026-10-15).
The whole fit runs inside the Peru-label guard. Every 1,000 iterations the teacher weights (at
the end also the student's) are saved to <out>/checkpoints/iter<N>.pt; <out>/run.json records the source
checkpoint and its commit, the config, wall clock, peak memory and collapse events.
Evaluation is offline: scripts/eval_adapted.py (AdaBN pass on the pool, then peru-dev logits).
`--demo`: 20 iterations, batch 2 + 2, checkpoint every 10 (local check of losses, mixing, EMA).
"""

import argparse
import glob
import importlib
import json
import subprocess
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.utilities import CombinedLoader
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
from torch.utils.data import DataLoader

from floodseg.self_training import SelfTraining
from floodseg.target_pool import (D4, REPO, TargetPoolDataset, forbid_peru_labels, guard_worker_init, model_input_spec,
                                  read_manifest)

SOURCES = {"unet": ("s1f11_src_unet", 3e-4, 8), "terramind_v1_base": ("s1f11_src_terramind_lr5e-5", 5e-5, 4)}
MASKING = {"patch": 64, "ratio": 0.7}


def instantiate(spec):
    mod, cls = spec["class_path"].rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)(**spec.get("init_args", {}))


class SaveEvery(pl.Callback):
    """Every `every` iterations save the teacher (the evaluated model); at the end also the student."""

    def __init__(self, out: Path, every: int):
        self.out, self.every = out, every

    def on_train_batch_end(self, trainer, module, *_):
        step = trainer.global_step
        final = step == trainer.max_steps
        if step % self.every == 0 or final:
            self.out.mkdir(parents=True, exist_ok=True)
            state = {"teacher": module.teacher.state_dict(), "iter": step}
            if final:
                state["student"] = module.student.state_dict()
            torch.save(state, self.out / f"iter{step}.pt")


def peak_memory_mb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20  # current driver allocation (no peak API on MPS)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=SOURCES, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["st", "mic"], default="st")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--q_end", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()
    folder, src_lr, half_batch = SOURCES[a.model]
    iters, every = (20, 10) if a.demo else (a.iters, 1000)
    if a.demo:
        half_batch = 2
    tag = f"{a.variant}" + ("" if (a.alpha, a.q_end) == (0.99, 0.5) else f"_a{a.alpha}_q{a.q_end}") + ("_demo" if a.demo else "")
    out = Path(a.out) if a.out else REPO / "outputs" / "phaseD" / "self_train" / tag / a.model / f"seed{a.seed}"
    pl.seed_everything(a.seed, workers=True)

    src_dir = REPO / "outputs" / folder / a.model / "lightning_logs" / f"seed{a.seed}"
    ckpt = Path(glob.glob(str(src_dir / "checkpoints" / "best-*.ckpt"))[0])
    src_commit = json.loads((src_dir / "run.json").read_text())["git"]["commit"]
    task = SemanticSegmentationTask.load_from_checkpoint(ckpt, map_location="cpu")

    cfg = yaml.safe_load((REPO / "configs" / f"{a.model}.yaml").read_text())["data"]["init_args"]
    cfg = {**cfg, "batch_size": half_batch, "num_workers": a.num_workers,
           "train_transform": [instantiate(t) for t in cfg.get("train_transform", [])]}
    dm = GenericNonGeoSegmentationDataModule(**cfg)
    dm.setup("fit")
    spec = model_input_spec(REPO / "configs" / f"{a.model}.yaml")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    module = SelfTraining(task, dm.aug, lr=0.1 * src_lr, iters=iters, alpha=a.alpha, q_end=a.q_end,
                          masking=MASKING if a.variant == "mic" else None)
    trainer = pl.Trainer(max_steps=iters, accelerator="auto", devices=1,
                         precision="16-mixed" if device == "cuda" else "32-true",
                         logger=pl.loggers.CSVLogger(out, name="", version=""), enable_checkpointing=False,
                         callbacks=[SaveEvery(out / "checkpoints", every)], log_every_n_steps=10 if not a.demo else 1,
                         enable_progress_bar=False)
    t0 = time.time()
    with forbid_peru_labels():
        pool = read_manifest("peru-train")
        kw = dict(batch_size=half_batch, shuffle=True, drop_last=True, num_workers=a.num_workers,
                  worker_init_fn=guard_worker_init, persistent_workers=a.num_workers > 0)
        loaders = CombinedLoader({"src": DataLoader(dm.train_dataset, **kw),
                                  "tgt": DataLoader(TargetPoolDataset(pool, spec, transform=D4()), **kw)},
                                 mode="max_size_cycle")
        trainer.fit(module, train_dataloaders=loaders)
    rec = dict(method=f"self-training ({a.variant})", model=a.model, seed=a.seed, source_ckpt=str(ckpt.relative_to(REPO)),
               source_commit=src_commit, pool="peru-train", pool_images=len(pool), config=module.cfg,
               half_batch=half_batch, precision=trainer.precision, device=device, iterations=trainer.global_step,
               wall_clock_s=round(time.time() - t0, 1), peak_memory_mb=peak_memory_mb(device),
               collapse_events=module.collapse_events, label_guard="forbid_peru_labels + guard_worker_init",
               git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip(),
               git_dirty=bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip()),
               evaluated_model="teacher (after AdaBN on the pool)")
    (out / "run.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
