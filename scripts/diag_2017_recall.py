"""Post-hoc diagnostic on `peru-2017-test` (declared in docs/METHODS_NOTES.md 2026-10-16 "One-time
test evaluation" before it was run; changes no reported number).

Per frozen row (seed means): recall on label-water pixels binned by MNDWI (0–0.1, 0.1–0.3, > 0.3),
and the share of ignored (−1) pixels with MNDWI > 0 that the row predicts as water. Subsets: the
101 chips not touching 2023 train, and all 204. MNDWI = (B03 − B11)/(B03 + B11) as in
scripts/index_baseline.py; water if z > 0.

    uv run python scripts/diag_2017_recall.py
→ outputs/phaseD/test/diag_2017_recall.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

REPO = Path(__file__).resolve().parents[1]
E = REPO / "data" / "peru" / "emsr2017"
L = REPO / "outputs" / "logits_2017"
BINS = [("0–0.1", 0.0, 0.1), ("0.1–0.3", 0.1, 0.3), (">0.3", 0.3, np.inf)]


def main() -> None:
    m = pd.read_csv(E / "manifest.csv")
    groups = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())["groups"]
    groups = [g for g in groups if (L / g["regime"] / g["model"]).exists()]
    keys = [b[0] for b in BINS] + ["ignored_mndwi>0"]
    acc = {}  # (subset, group, seed, key) -> [hit, n]
    for chip, out in zip(m.chip, ~m.touches_2023_train):
        with rasterio.open(E / "s2" / f"{chip}.tif") as src:
            x = src.read().astype(np.float64)
        v = (x[2] - x[11]) / (x[2] + x[11] + 1e-6)
        with rasterio.open(E / "label" / f"{chip}.tif") as src:
            y = src.read(1)
        masks = {k: (y == 1) & (v > lo) & (v <= hi) for k, lo, hi in BINS}
        masks["ignored_mndwi>0"] = (y == -1) & (v > 0)
        subs = ["all"] + (["no_2023_train_overlap"] if out else [])
        for g in groups:
            for seed in g["runs"]:
                p = np.load(L / g["regime"] / g["model"] / seed / f"{chip}.npy") > 0
                for k, mk in masks.items():
                    hit, n = int((p & mk).sum()), int(mk.sum())
                    for s in subs:
                        a = acc.setdefault((s, g["id"], seed, k), [0, 0])
                        a[0] += hit
                        a[1] += n
    rows = [dict(subset=s, group=g, seed=sd, key=k, rate=h / n if n else np.nan, pixels=n) for (s, g, sd, k), (h, n) in acc.items()]
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "outputs" / "phaseD" / "test" / "diag_2017_recall.csv", index=False)
    t = df.groupby(["subset", "group", "key"]).rate.mean().unstack("key")[keys]
    px = df.groupby(["subset", "key"]).pixels.first().unstack("key")[keys]
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print("\npixels per bin:\n", px.to_string())


if __name__ == "__main__":
    main()
