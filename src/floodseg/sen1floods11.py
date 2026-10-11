"""Sen1Floods11 official hand-labelled splits: read, write for TerraTorch, verify.

The official split CSVs are the only source of truth (never re-split). TerraTorch's
`Sen1Floods11NonGeo` reads `flood_{train,valid,test}_data.txt` lists, which
`write_split_lists` generates from the CSVs.
"""

import csv
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data" / "sen1floods11"
SPLIT_DIR = DATA_ROOT / "v1.1" / "splits" / "flood_handlabeled"
CHIP_DIR = DATA_ROOT / "v1.1" / "data" / "flood_events" / "HandLabeled"

# Counted from the official split CSVs (docs/feasibility.md).
OFFICIAL_SIZES = {"train": 252, "valid": 89, "test": 90}
# The advisor dropped Bolivia as the OOD test; Peru replaces it.
EXCLUDED_SPLITS = {"bolivia"}
S2_BANDS = 13
LABEL_VALUES = {-1, 0, 1}


def official_ids(split: str, split_dir: Path = SPLIT_DIR) -> list[str]:
    """Chip ids (e.g. `Ghana_103272`) of an official split, checked against its size."""
    if split in EXCLUDED_SPLITS:
        raise ValueError(f"Split '{split}' is excluded from reported results; use Peru.")
    if split not in OFFICIAL_SIZES:
        raise ValueError(f"Unknown split '{split}'; expected one of {sorted(OFFICIAL_SIZES)}.")
    with (split_dir / f"flood_{split}_data.csv").open() as f:
        ids = [row[0].removesuffix("_S1Hand.tif") for row in csv.reader(f) if row]
    if len(ids) != OFFICIAL_SIZES[split] or len(set(ids)) != len(ids):
        raise ValueError(
            f"Split '{split}' has {len(ids)} ids ({len(set(ids))} unique), "
            f"expected {OFFICIAL_SIZES[split]}."
        )
    return ids


def write_split_lists(split_dir: Path = SPLIT_DIR) -> None:
    """Write the `.txt` lists TerraTorch reads, one chip id per line."""
    for split in OFFICIAL_SIZES:
        ids = official_ids(split, split_dir)
        (split_dir / f"flood_{split}_data.txt").write_text("\n".join(ids) + "\n")


def check_chip(chip: str, chip_dir: Path = CHIP_DIR) -> list[str]:
    """Problems with one chip's S2Hand and LabelHand files; empty list if fine."""
    problems = []
    s2 = chip_dir / "S2Hand" / f"{chip}_S2Hand.tif"
    label = chip_dir / "LabelHand" / f"{chip}_LabelHand.tif"
    try:
        with rasterio.open(s2) as src:
            if src.count != S2_BANDS:
                problems.append(f"S2Hand has {src.count} bands, expected {S2_BANDS}")
            s2_shape = src.shape
            src.read(1)
    except Exception as e:  # missing or corrupt file
        return [f"S2Hand unreadable: {e}"]
    try:
        with rasterio.open(label) as src:
            values = set(np.unique(src.read(1)).tolist())
            if src.shape != s2_shape:
                problems.append(f"LabelHand shape {src.shape} != S2Hand {s2_shape}")
    except Exception as e:
        return problems + [f"LabelHand unreadable: {e}"]
    if not values <= LABEL_VALUES:
        problems.append(f"LabelHand values {sorted(values - LABEL_VALUES)} outside {{-1, 0, 1}}")
    return problems
