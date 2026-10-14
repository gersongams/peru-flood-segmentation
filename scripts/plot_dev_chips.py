"""Figures for the failure analysis: SWIR false colour, label and predictions of chips.

    uv run python scripts/plot_dev_chips.py 2023-04-01_E522240_N9349120 ...
    uv run python scripts/plot_dev_chips.py --top 10      # chips with most Peru-only errors

Writes outputs/analysis_dev/figs/<chip>.png. False colour = SWIR2 / NIR / red
(B12, B8A, B04): water is dark blue/black, wet soil dark, desert bright.
Prediction panels: white = water, red = false positive, blue = missed water,
grey = ignore (-1).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "peru" / "unosat2023"
PRED = REPO / "outputs" / "logits_dev"
A = REPO / "outputs" / "analysis_dev"
PANELS = [("peru", "unet"), ("peru", "terramind_v1_base"), ("peru", "prithvi_eo_v2_300"), ("s1f11", "prithvi_eo_v2_300")]
NAMES = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}


def false_colour(s2: np.ndarray) -> np.ndarray:
    rgb = np.stack([s2[12], s2[8], s2[3]], -1).astype(float)  # B12, B8A, B04
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def error_map(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    img = np.zeros(y.shape + (3,))
    img[(pred == 1) & (y == 1)] = 1
    img[(pred == 1) & (y == 0)] = (0.9, 0.1, 0.1)
    img[(pred == 0) & (y == 1)] = (0.1, 0.4, 1.0)
    img[y == -1] = 0.5
    return img


def plot(stem: str) -> Path:
    with rasterio.open(DATA / "s2" / f"{stem}.tif") as src:
        s2 = src.read()
    with rasterio.open(DATA / "label" / f"{stem}.tif") as src:
        y = src.read(1)
    lab = np.where(y == -1, 0.5, y.astype(float))
    fig, ax = plt.subplots(1, 2 + len(PANELS), figsize=(3.2 * (2 + len(PANELS)), 3.4))
    ax[0].imshow(false_colour(s2)); ax[0].set_title("S2 B12/B8A/B04")
    ax[1].imshow(lab, cmap="gray", vmin=0, vmax=1); ax[1].set_title("UNOSAT label")
    for a, (regime, model) in zip(ax[2:], PANELS):
        p = np.load(PRED / regime / model / "seed0" / f"{stem}.npy") > 0  # water logit
        a.imshow(error_map(p, y)); a.set_title(f"{'S1F11-only' if regime == 's1f11' else 'Peru'} {NAMES[model]}")
    for a in ax:
        a.axis("off")
    fig.suptitle(stem, fontsize=9)
    fig.tight_layout()
    out = A / "figs" / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chips", nargs="*")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()
    chips = list(args.chips)
    if args.top:
        c = pd.read_csv(A / "chips.csv")
        e = c[c.regime == "peru"].groupby("chip")[["fp", "fn"]].sum().sum(axis=1)
        chips += list(e.sort_values(ascending=False).index[: args.top])
    for st in chips:
        print(plot(st).relative_to(REPO))


if __name__ == "__main__":
    main()
