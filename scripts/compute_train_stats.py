"""Per-band mean and std of Sen1Floods11 S2Hand over the official train split (raw DN).

Used to normalize the U-Net baseline, which has no pretraining statistics.
NaN / no-data pixels are skipped. Writes configs/stats/sen1floods11_train_s2.json.

    uv run python scripts/compute_train_stats.py
"""

import json

import numpy as np
import rasterio

from floodseg.sen1floods11 import CHIP_DIR, REPO, official_ids

OUT = REPO / "configs" / "stats" / "sen1floods11_train_s2.json"


def main() -> None:
    total = total_sq = count = 0
    for chip in official_ids("train"):
        with rasterio.open(CHIP_DIR / "S2Hand" / f"{chip}_S2Hand.tif") as src:
            x = src.read().astype(np.float64).reshape(src.count, -1)
        valid = np.isfinite(x).all(axis=0)
        x = x[:, valid]
        total = total + x.sum(axis=1)
        total_sq = total_sq + (x**2).sum(axis=1)
        count += x.shape[1]
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "Sen1Floods11 v1.1 S2Hand, official train split (252 chips), raw DN",
        "pixels": int(count),
        "means": [round(v, 3) for v in mean.tolist()],
        "stds": [round(v, 3) for v in std.tolist()],
    }, indent=2) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
