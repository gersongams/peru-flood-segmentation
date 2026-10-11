"""Record what produced a run: config, seed, git state, versions, hardware, timing.

Every training run writes `run.json` (and `git.diff` when the tree is dirty) into
the logger's run directory, so any number in the thesis can be traced back.
"""

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import lightning.pytorch as pl
import torch

PACKAGES = ["torch", "lightning", "terratorch", "torchgeo", "timm", "numpy"]
REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_state() -> dict:
    """Commit hash, branch and uncommitted diff (empty string when clean)."""
    diff = _git("diff", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(diff or untracked),
        "diff": diff,
        "untracked": untracked.splitlines(),
    }


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine()}
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        info["mps_recommended_max_gib"] = torch.mps.recommended_max_memory() / 2**30
    return info


class RunProvenance(pl.Callback):
    """Write `run.json` at the start of fit and update it with timing at the end.

    A full run refuses to start on a dirty working tree, so reported numbers always
    map to a commit. A demo run (`limit_train_batches` < 1, see
    configs/trainer/demo.yaml) may be dirty; its diff is saved as `git.diff`.
    """

    def __init__(self, require_clean: bool = True) -> None:
        self.require_clean = require_clean
        self._start = 0.0
        self._path: Path | None = None
        self._record: dict = {}

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if stage != "fit" or not trainer.is_global_zero:
            return
        git = git_state()
        demo = trainer.limit_train_batches != 1.0
        if git["dirty"] and self.require_clean and not demo:
            raise RuntimeError(
                "Working tree is dirty. Commit first, or add -c configs/trainer/demo.yaml "
                "for a demo run."
            )
        run_dir = Path(trainer.log_dir or trainer.default_root_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if git["dirty"]:
            (run_dir / "git.diff").write_text(git["diff"])
        self._record = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "seed": _seed_from_env(),
            "demo": demo,
            "device": str(trainer.strategy.root_device),
            "precision": str(trainer.precision),
            "git": {k: v for k, v in git.items() if k != "diff"},
            "versions": package_versions(),
            "hardware": hardware(),
            "hparams": _jsonable(dict(pl_module.hparams)),
            "normalization": _normalization(trainer.datamodule),
        }
        self._path = run_dir / "run.json"
        self._write()
        self._start = time.perf_counter()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._path is None:
            return
        self._record["wall_clock_s"] = round(time.perf_counter() - self._start, 1)
        self._record["epochs_completed"] = trainer.current_epoch
        self._record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self._write()

    def _write(self) -> None:
        assert self._path is not None
        self._path.write_text(json.dumps(self._record, indent=2, default=str))


def _normalization(datamodule) -> dict:
    """Bands, means and stds a TerraTorch datamodule actually feeds the model."""
    aug = getattr(datamodule, "aug", None)

    def as_list(v):
        return v.flatten().tolist() if hasattr(v, "flatten") else v

    return _jsonable({
        "bands": [str(getattr(b, "value", b)) for b in getattr(datamodule, "output_bands", None) or []],
        "means": as_list(getattr(aug, "means", None)),
        "stds": as_list(getattr(aug, "stds", None)),
    })


def _seed_from_env() -> int | None:
    import os

    seed = os.environ.get("PL_GLOBAL_SEED")
    return int(seed) if seed is not None else None


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))
