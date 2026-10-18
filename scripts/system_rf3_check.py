"""RF3 parity check: full-tile probability vs the frozen pass on the peru-test chips it covers.

docs/system_evaluation_protocol.md §3, RF3: on the `peru-test` chips inside the tile, the
system's probability must agree with the frozen pass (mean over the 3 seeds of sigmoid(z) from
outputs/logits_test/peru/unet): mean absolute probability difference ≤ 0.02 and IoU of the water
masks (p > 0.5) ≥ 0.95. Pixels: every chip pixel that is not no-data in the tile output.

    uv run python scripts/system_rf3_check.py 17MNP 2023-03-22
"""

import json
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from floodseg.peru import UNOSAT_SPLIT
from floodseg.system.pipeline import NODATA16, SCALE, scene_dir
from floodseg.target_pool import REPO

LOGITS = REPO / "outputs" / "logits_test" / "peru" / "unet"
CHIP = 5120  # m (512 px × 10 m), lower-left corner in the chip name
MAX_MAD, MIN_IOU = 0.02, 0.95


def main(tile: str, date: str) -> None:
    out = scene_dir(tile, date)
    split = pd.read_csv(UNOSAT_SPLIT)
    chips = split[(split.split == "test") & (split.date == date)].chip.tolist()
    rows, a_all, b_all = [], [], []
    with rasterio.open(out / "probability.tif") as src:
        l, b, r, t = src.bounds
        for c in chips:
            x0, y0 = (int(v[1:]) for v in c.split("_"))
            if not (x0 >= l and y0 >= b and x0 + CHIP <= r and y0 + CHIP <= t):
                continue
            q = src.read(1, window=from_bounds(x0, y0, x0 + CHIP, y0 + CHIP, src.transform))
            if q.shape != (512, 512):
                continue
            z = np.stack([np.load(LOGITS / s / f"{date}_{c}.npy") for s in ("seed0", "seed1", "seed2")])
            ref = (1 / (1 + np.exp(-z))).mean(0)
            ok = q != NODATA16
            sys_p = q[ok].astype(np.float32) / SCALE
            ref_p = ref[ok]
            a, bb = sys_p > 0.5, ref_p > 0.5
            union = (a | bb).sum()
            rows.append(dict(chip=c, pixels=int(ok.sum()), mad=float(np.abs(sys_p - ref_p).mean()),
                             iou=float((a & bb).sum() / union) if union else float("nan"),
                             water_sys=int(a.sum()), water_ref=int(bb.sum())))
            a_all.append(sys_p); b_all.append(ref_p)
    df = pd.DataFrame(rows)
    sp, rp = np.concatenate(a_all), np.concatenate(b_all)
    A, B = sp > 0.5, rp > 0.5
    summary = dict(tile=tile, date=date, chips=len(df), pixels=int(len(sp)),
                   mad=float(np.abs(sp - rp).mean()), iou=float((A & B).sum() / (A | B).sum()),
                   thresholds=dict(max_mad=MAX_MAD, min_iou=MIN_IOU))
    summary["pass"] = bool(summary["mad"] <= MAX_MAD and summary["iou"] >= MIN_IOU)
    dest = REPO / "outputs" / "system_eval" / "rf"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / f"rf3_{tile}_{date}_chips.csv", index=False)
    (dest / f"rf3_{tile}_{date}.json").write_text(json.dumps(summary, indent=2))
    print(df.describe().loc[["mean", "min", "max"], ["mad", "iou"]].to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
