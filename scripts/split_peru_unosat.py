"""Spatial-block split of the UNOSAT 2023 Peru chips into train / dev / test.

Chips are grouped into BLOCK x BLOCK chip blocks (~20 km). Blocks are assigned
at random (fixed seed) until dev and test each hold ~15 % of the water pixels;
the rest is train. A location is in one split for both dates. Train chips that
touch a dev/test chip (8-neighbourhood) are dropped as a buffer.

Writes data/peru/unosat2023/splits/{train,dev,test}.txt (chip file stems) and
split.csv (chip, split).

    uv run python scripts/split_peru_unosat.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "peru" / "unosat2023"
CHIP = 5120  # metres
BLOCK = 4  # chips per block side
SEED = 0
SHARE = {"test": 0.15, "dev": 0.15}  # of water pixels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    args = ap.parse_args()
    m = pd.read_csv(ROOT / "manifest.csv")
    loc = m.groupby("chip").agg(minx=("minx", "first"), miny=("miny", "first"), water=("water_frac", "sum")).reset_index()
    loc["ix"], loc["iy"] = loc.minx // CHIP, loc.miny // CHIP
    loc["block"] = (loc.ix // args.block).astype(str) + "_" + (loc.iy // args.block).astype(str)

    blocks = loc.groupby("block").water.sum()
    blocks = blocks[blocks > 0]  # only blocks with water go to dev/test
    order = np.random.default_rng(args.seed).permutation(blocks.index.to_numpy())
    total = loc.water.sum()
    assign, filled = {}, {k: 0.0 for k in SHARE}
    for b in order:
        for k, share in SHARE.items():
            if filled[k] < share * total:
                assign[b] = k
                filled[k] += blocks[b]
                break
    loc["split"] = loc.block.map(assign).fillna("train")

    held = {(x, y) for x, y, s in zip(loc.ix, loc.iy, loc.split) if s != "train"}
    near = [any((x + dx, y + dy) in held for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            for x, y in zip(loc.ix, loc.iy)]
    loc.loc[(loc.split == "train") & np.array(near), "split"] = "buffer"

    m = m.merge(loc[["chip", "split"]], on="chip")
    summary = m.groupby("split").agg(chips=("chip", "size"), locations=("chip", "nunique"),
                                     water_chips=("water_frac", lambda w: int((w > 0.01).sum())),
                                     water_km2=("water_frac", lambda w: round(w.sum() * CHIP**2 / 1e6)))
    print(summary)
    if args.dry_run:
        return
    out = ROOT / "splits"
    out.mkdir(exist_ok=True)
    m[["chip", "date", "split"]].to_csv(out / "split.csv", index=False)
    for s in ("train", "dev", "test"):
        stems = sorted(f"{d}_{c}" for c, d in zip(m.chip[m.split == s], m.date[m.split == s]))
        (out / f"{s}.txt").write_text("\n".join(stems) + "\n")


if __name__ == "__main__":
    main()
