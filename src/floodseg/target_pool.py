"""Unlabelled Peru target pool for phase-D adaptation (spec `peru-target-pool`).

The default pool is the Sentinel-2 images of the `peru-train` chips (both 2023 dates). The
pool only knows image paths: `TargetPoolDataset` returns images, never masks. Two guards:
- `check_pool` refuses held-out chips (`peru-dev`, `peru-test`, buffer, `peru-2017-test`)
  and chips that are not in the Peru v1 split at all (extra imagery needs its own spatial
  check first, task 2.4);
- `forbid_peru_labels` makes any `rasterio.open` of a Peru label file raise during
  adaptation training.
Images are read as the model's dataset does (`dataset_bands` → `output_bands`, float32,
NaN → 0) and normalized with the model config's means/stds, so they match what the model
saw in source training (tests/test_target_pool.py compares with TerraTorch's datamodule).
"""

import contextlib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from torch.utils.data import Dataset

from floodseg.peru import UNOSAT_SPLIT, held_out_stems

REPO = Path(__file__).resolve().parents[2]
PERU_IMAGES = REPO / "data" / "peru" / "unosat2023" / "s2"
POOL_DIR = REPO / "configs" / "pools"
# Every location that holds Peru labels (rasters and the raw vector maps).
PERU_LABEL_ROOTS = [
    REPO / "data" / "peru" / "unosat2023" / "label",
    REPO / "data" / "peru" / "emsr2017" / "label",
    REPO / "data" / "unosat2023",
    REPO / "data" / "emsr199",
]


def default_pool_stems(unosat_split: Path = UNOSAT_SPLIT) -> list[str]:
    """`<date>_<chip>` stems of all `peru-train` chips (753 in Peru set v1)."""
    s = pd.read_csv(unosat_split)
    s = s[s.split == "train"]
    return sorted(f"{d}_{c}" for c, d in zip(s.chip, s.date))


def check_pool(stems: list[str], unosat_split: Path = UNOSAT_SPLIT, **paths) -> None:
    """Raise if the pool holds a held-out chip or a chip outside the Peru v1 split."""
    held = sorted(set(stems) & held_out_stems(unosat_split=unosat_split, **paths))
    if held:
        raise RuntimeError(f"Target pool contains {len(held)} held-out Peru chips: {held[:10]}")
    s = pd.read_csv(unosat_split)
    unknown = sorted(set(stems) - {f"{d}_{c}" for c, d in zip(s.chip, s.date)})
    if unknown:
        raise RuntimeError(
            f"Target pool contains {len(unknown)} chips not in the Peru v1 split: {unknown[:10]}; "
            "extra imagery needs the spatial check of task 2.4 first."
        )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(name: str = "peru-train", verify: bool = True, image_dir: Path = PERU_IMAGES) -> list[Path]:
    """Image paths of a versioned pool manifest (`configs/pools/<name>.csv`), guarded and,
    with `verify`, checked against the stored sha256."""
    m = pd.read_csv(POOL_DIR / f"{name}.csv")
    check_pool(list(m.stem))
    paths = [image_dir / f"{s}.tif" for s in m.stem]
    if verify:
        bad = [p.name for p, h in zip(paths, m.sha256) if sha256(p) != h]
        if bad:
            raise RuntimeError(f"Pool {name}: {len(bad)} images changed since the manifest: {bad[:5]}")
    return paths


def model_input_spec(config: str | Path) -> dict:
    """`dataset_bands`, `output_bands`, `means`, `stds`, `no_data_replace` of a model config."""
    a = yaml.safe_load(Path(config).read_text())["data"]["init_args"]
    return {k: a.get(k) for k in ("dataset_bands", "output_bands", "means", "stds", "no_data_replace")}


class TargetPoolDataset(Dataset):
    """Unlabelled pool images, normalized for one model. Items: {"image": float32 (C, H, W)}."""

    def __init__(self, paths: list[Path], spec: dict, transform=None):
        check_pool([Path(p).stem for p in paths])
        self.paths = [Path(p) for p in paths]
        self.idx = [spec["dataset_bands"].index(b) for b in spec["output_bands"]]
        self.mean = np.asarray(spec["means"], np.float32)[:, None, None]
        self.std = np.asarray(spec["stds"], np.float32)[:, None, None]
        self.no_data = spec["no_data_replace"] if spec["no_data_replace"] is not None else 0
        self.transform = transform  # e.g. student augmentation, applied after normalization

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        x = _read(self.paths[i], self.idx, self.mean, self.std, self.no_data)
        return {"image": self.transform(x) if self.transform else x}


class D4:
    """Random dihedral transform (flips and 90° rotations) of a (C, H, W) tensor; picklable,
    so it works in spawned DataLoader workers."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(0, 4, (1,)))
        x = torch.rot90(x, k, dims=(1, 2))
        return torch.flip(x, dims=(2,)) if torch.rand(1) < 0.5 else x


def _read(path, idx, mean, std, no_data) -> torch.Tensor:
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)[idx]
    return torch.from_numpy((np.nan_to_num(x, nan=no_data) - mean) / std)


def normalize(raw: np.ndarray, spec: dict) -> torch.Tensor:
    """A raw all-band image (dataset_bands, H, W) → normalized model input, as the pool reads it."""
    idx = [spec["dataset_bands"].index(b) for b in spec["output_bands"]]
    mean = np.asarray(spec["means"], np.float32)[:, None, None]
    std = np.asarray(spec["stds"], np.float32)[:, None, None]
    no_data = spec["no_data_replace"] if spec["no_data_replace"] is not None else 0
    return torch.from_numpy((np.nan_to_num(raw.astype(np.float32)[idx], nan=no_data) - mean) / std)


def load_image(path: Path, spec: dict) -> torch.Tensor:
    """One image normalized for a model, as in the pool (for evaluation scripts: no pool
    guard, so it also reads dev/test images)."""
    with rasterio.open(path) as src:
        return normalize(src.read(), spec)


def _is_label(path) -> Path | None:
    try:
        p = Path(path).resolve()
    except TypeError:  # file objects, MemoryFile, ...
        return None
    return p if any(p.is_relative_to(r) for r in PERU_LABEL_ROOTS) else None


def _guarded(original):
    def guarded(fp, *args, **kwargs):
        if (p := _is_label(fp)) is not None:
            raise RuntimeError(f"Peru label read during adaptation: {p}")
        return original(fp, *args, **kwargs)

    guarded._peru_guard = True
    return guarded


@contextlib.contextmanager
def forbid_peru_labels():
    """Within the block, `rasterio.open` of any Peru label file raises (spec: the run stops
    with an error naming the file). rioxarray and TerraTorch call `rasterio.open`, so this
    covers them. DataLoader workers started by spawn (macOS default) do not inherit the
    patch: pass `guard_worker_init` as their `worker_init_fn`."""
    original = rasterio.open
    rasterio.open = _guarded(original)
    try:
        yield
    finally:
        rasterio.open = original


def guard_worker_init(worker_id: int) -> None:
    """`worker_init_fn` that installs the label guard for the life of a DataLoader worker."""
    if not getattr(rasterio.open, "_peru_guard", False):
        rasterio.open = _guarded(rasterio.open)
