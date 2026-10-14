"""Size of the float16-probability saturation bias (2026-10-14 correction).

Reads the old float16 probabilities (outputs/preds_dev/peru/prithvi_eo_v2_300/seed*) and
compares error-detection AUROC as stored with AUROC after clipping both tails
symmetrically at the largest float16 below 1 (0.99951). See docs/METHODS_NOTES.md.

    uv run python scripts/float16_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uncertainty_dev import auroc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
P = REPO / "outputs" / "preds_dev" / "peru" / "prithvi_eo_v2_300"
LABEL = REPO / "data" / "peru" / "unosat2023" / "label"


def h(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main() -> None:
    stems = sorted(p.stem for p in (P / "seed0").glob("*.npy"))
    ys, ps = [], {k: [] for k in (0, 1, 2)}
    for st in stems:
        with rasterio.open(LABEL / f"{st}.tif") as src:
            ys.append(src.read(1).ravel())
        for k in ps:
            ps[k].append(np.load(P / f"seed{k}" / f"{st}.npy").ravel())
    y = np.concatenate(ys)
    ok = y != -1
    y = y[ok]
    raw = {k: np.concatenate(v)[ok] for k, v in ps.items()}
    q = float(np.nextafter(np.float16(1), np.float16(0)))
    p0 = raw[0].astype(np.float32)
    print(f"seed 0: share p >= {q:.5f}: {(p0 >= q).mean():.3f}; share p <= {1 - q:.5f}: {(p0 <= 1 - q).mean():.3f}")
    for name, clip in (("as stored", False), ("symmetric clip", True)):
        pp = {k: np.clip(v.astype(np.float32), 1 - q, q) if clip else v.astype(np.float32) for k, v in raw.items()}
        p = pp[0]
        a1 = auroc(-np.maximum(p, 1 - p), (p > 0.5) != (y == 1))
        pm = np.mean(list(pp.values()), axis=0)
        a2 = auroc(h(pm), (pm > 0.5) != (y == 1))
        print(f"{name}: seed-0 AUROC {a1:.3f}, ensemble (entropy) AUROC {a2:.3f}")


if __name__ == "__main__":
    main()
