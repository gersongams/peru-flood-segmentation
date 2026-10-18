"""Qualitative comparison of the frozen configurations on `peru-test` (UNOSAT 2023).

Inputs
- Imagery data/peru/unosat2023/s2/<stem>.tif (13 L1C bands, DN; RGB = B04/B03/B02 = bands 4/3/2).
- Labels data/peru/unosat2023/label/<stem>.tif (1 water, 0 not water, -1 no label / ignored).
- Test chips data/peru/unosat2023/splits/test.txt (172 chips) and outputs/label_qa/chips.csv (flags,
  only printed).
- Water logits z = logit_water - logit_other written once by scripts/evaluate_frozen.py,
  outputs/logits_test/<regime>/<model>/seed{0,1,2}/<stem>.npy (float32, 512 x 512).

Prediction = 3-seed ensemble: mean of the seed probabilities sigmoid(z), as the logit
log p - log(1 - p) of scripts/uncertainty_dev.ens_logit; water where that logit > 0 (p > 0.5),
the same threshold as scripts/score_frozen.py (z > 0 per seed).

Columns: RGB | UNOSAT label | U-Net source (s1f11_src) | U-Net + training-time histogram matching
(trhm, frozen) | TerraMind source | TerraMind + training-time FDA (trfda, frozen) | Prithvi source
(frozen; no adaptation) | U-Net trained on Peru (peru, upper reference).

Chip selection rule (fixed before rendering; the doubtful-label chips are kept, matching the
`full` test table):
1. candidates = peru-test chips whose labelled water is >= 5 % of their labelled pixels (37);
2. per candidate, delta = ensemble IoU(U-Net trhm) - ensemble IoU(U-Net s1f11_src) on the chip;
3. sort by delta and take, for the target quantiles 0.125 / 0.375 / 0.625 / 0.875, the candidate
   at the nearest rank round(q * (n - 1)); if its location is already chosen, take the
   next-nearest rank (ties: lower rank first). Location = the E.._N.. part of the stem (each
   location is in the test split on both dates), so no place is shown twice. Distinct 4 x 4
   split blocks were tried and rejected: the high-delta chips lie in few blocks, so the 0.875 pick
   fell back to rank 8 (negative delta), defeating the quantile design. Rows c and d are
   neighbouring chips of one block.
Rows are shown in increasing delta, so the figure covers a failure of adaptation, two moderate
cases and a large gain.

Tiles show the full 512 x 512 chip (5.12 km). Per-tile IoU (corner) = tp / (tp + fp + fn) on the
labelled pixels of that chip for the ensemble prediction (floodseg.metrics.counts, plain);
it is a per-chip value, not the pooled mean +/- std of the test table. A self-check reproduces
the seed-0 counts of outputs/phaseD/test/peru-test_chip_counts.csv exactly.

    uv run python scripts/figures/fig_comparacion_modelos.py
-> thesis/E_IMAGENES/fig_comparacion_modelos.pdf, outputs/figures_preview/fig_comparacion_modelos.png,
   outputs/figures_preview/fig_comparacion_modelos_chips.csv (selection table)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import style  # noqa: E402
from uncertainty_dev import ens_logit  # noqa: E402

from floodseg.metrics import counts  # noqa: E402

REPO = style.ROOT
U = REPO / "data" / "peru" / "unosat2023"
LOGITS = REPO / "outputs" / "logits_test"
SEEDS = ("seed0", "seed1", "seed2")
MIN_WATER = 0.05
QUANTILES = (0.125, 0.375, 0.625, 0.875)
COLS = [  # (regime, model, header)
    ("s1f11_src", "unet", "U-Net\norigen"),
    ("trhm", "unet", "U-Net + igualación\nde histogramas"),
    ("s1f11_src", "terramind_v1_base", "TerraMind\norigen"),
    ("trfda", "terramind_v1_base", "TerraMind\n+ FDA"),
    ("s1f11_src", "prithvi_eo_v2_300", "Prithvi\norigen"),
    ("peru", "unet", "U-Net entrenada\nen el Perú"),
]
SEL_A, SEL_B = ("trhm", "unet"), ("s1f11_src", "unet")


def label(stem: str) -> np.ndarray:
    with rasterio.open(U / "label" / f"{stem}.tif") as src:
        return src.read(1)


def ensemble_water(regime: str, model: str, stem: str) -> np.ndarray:
    zs = [np.load(LOGITS / regime / model / s / f"{stem}.npy").astype(np.float64) for s in SEEDS]
    return ens_logit(zs) > 0


def iou(c: np.ndarray) -> float:
    tp, fp, fn = c
    return tp / (tp + fp + fn) if tp + fp + fn else float("nan")


def select() -> pd.DataFrame:
    stems = (U / "splits" / "test.txt").read_text().split()
    rows = []
    for s in stems:
        y = label(s)
        ok = y != -1
        wf = (y == 1).sum() / max(ok.sum(), 1)
        if wf < MIN_WATER:
            continue
        ia, ib = (iou(counts(ensemble_water(*g, s), y)[0]) for g in (SEL_A, SEL_B))
        rows.append(dict(stem=s, block=s.split("_", 1)[1], water_frac=wf, labelled_frac=ok.mean(),
                         iou_trhm=ia, iou_src=ib, delta=ia - ib))
    d = pd.DataFrame(rows).sort_values(["delta", "stem"]).reset_index(drop=True)
    d["rank"] = d.index
    chosen, blocks = [], set()
    for q in QUANTILES:
        target = round(q * (len(d) - 1))
        for r in sorted(d.index, key=lambda i: (abs(i - target), i)):
            if d.loc[r, "block"] not in blocks:
                chosen.append(r)
                blocks.add(d.loc[r, "block"])
                break
    out = d.loc[chosen].assign(quantile=QUANTILES)
    qa = pd.read_csv(REPO / "outputs" / "label_qa" / "chips.csv").set_index("chip")
    out["qa_flag"] = qa.loc[out.stem, "flag"].values
    print(f"{len(d)} candidates; chosen:\n{out.to_string(index=False)}")
    return out


def self_check(stems: list[str]) -> None:
    cc = pd.read_csv(REPO / "outputs" / "phaseD" / "test" / "peru-test_chip_counts.csv")
    cc = cc[cc.seed == "seed0"].set_index(["group", "chip"])
    for s in stems:
        y = label(s)
        for regime, model, _ in COLS:
            z = np.load(LOGITS / regime / model / "seed0" / f"{s}.npy")
            ref = cc.loc[(f"{regime}/{model}", s), ["tp", "fp", "fn"]].to_numpy()
            assert (counts(z > 0, y)[0] == ref).all(), (s, regime, model)
    print("self-check: seed-0 counts match peru-test_chip_counts.csv")


def rgb(stem: str) -> np.ndarray:
    with rasterio.open(U / "s2" / f"{stem}.tif") as src:
        x = src.read([4, 3, 2]).astype(np.float64)  # B04, B03, B02 (1-based band indexes)
    return np.moveaxis(x, 0, -1)


def stretch(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0, 1) ** 0.85


def error_rgb(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    img = np.ones(y.shape + (3,))
    img[pred & (y == 1)] = to_rgb(style.TP)
    img[pred & (y == 0)] = to_rgb(style.FP)
    img[~pred & (y == 1)] = to_rgb(style.FN)
    img[y == -1] = to_rgb(style.IGNORE)
    return img


def label_rgb(y: np.ndarray) -> np.ndarray:
    img = np.ones(y.shape + (3,))
    img[y == 1] = to_rgb(style.TP)
    img[y == -1] = to_rgb(style.IGNORE)
    return img


def tile(ax, img, iou_val=None, **kw):
    im = ax.imshow(img, interpolation="none", **kw)  # native pixels in the PDF
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("#9A9A9A")
    if iou_val is not None:
        ax.text(0.04, 0.04, f"{iou_val:.3f}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.6, color=style.INK,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))
    return im


def main() -> None:
    style.apply()
    sel = select()
    stems = list(sel.stem)
    self_check(stems)
    images = [rgb(s) for s in stems]
    # One linear stretch for all rows: 1st-99th percentile of the RGB values of the labelled pixels
    # of the shown chips (ignored pixels include thick cloud, which would darken every row).
    lo, hi = np.percentile(np.concatenate([i[label(s) != -1].ravel() for s, i in zip(stems, images)]), [1, 99])
    print(f"RGB stretch DN {lo:.0f}-{hi:.0f}")

    ncol, nrow = 2 + len(COLS), len(stems)
    left, gap = 0.035, 0.004  # figure fractions: row-label strip, gutter
    W = style.TEXT_WIDTH
    tw = (1 - left - gap * (ncol - 1)) * W / ncol  # tile width in inches
    top_in, bot_in = 0.30, 0.20
    H = top_in + bot_in + nrow * tw + (nrow - 1) * gap * W
    fig = plt.figure(figsize=(W, H))
    headers = ["Imagen\n(RGB)", "Etiqueta\n(UNOSAT)"] + [c[2] for c in COLS]
    table = []
    for r, (s, img) in enumerate(zip(stems, images)):
        y = label(s)
        y0 = 1 - (top_in + r * (tw + gap * W) + tw) / H
        axs = [fig.add_axes([left + c * (tw / W + gap), y0, tw / W, tw / H]) for c in range(ncol)]
        tile(axs[0], stretch(img, lo, hi))
        tile(axs[1], label_rgb(y))
        for ax, (regime, model, _) in zip(axs[2:], COLS):
            p = ensemble_water(regime, model, s)
            c = counts(p, y)[0]
            tile(ax, error_rgb(p, y), iou(c))
            table.append(dict(stem=s, group=f"{regime}/{model}", tp=c[0], fp=c[1], fn=c[2], iou=iou(c)))
        date, loc = s.split("_", 1)
        axs[0].text(-0.07, 0.5, f"{'abcd'[r]}) {date}\n{loc.replace('_', ' ')}", transform=axs[0].transAxes,
                    rotation=90, ha="right", va="center", fontsize=5.6, color=style.INK, linespacing=1.25)
        if r == 0:
            for ax, h in zip(axs, headers):
                ax.set_title(h, fontsize=6.0, pad=2.5, color=style.INK, linespacing=1.1)
    legend = [Patch(fc=style.TP, ec="none", label="Agua (acierto)"),
              Patch(fc=style.FP, ec="none", label="Falso positivo"),
              Patch(fc=style.FN, ec="none", label="Omisión"),
              Patch(fc="white", ec="#9A9A9A", lw=0.4, label="No agua (acierto)"),
              Patch(fc=style.IGNORE, ec="none", label="Sin etiqueta (ignorado)")]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=6.2, handlelength=1.0, handleheight=0.9,
               handletextpad=0.4, columnspacing=1.3, bbox_to_anchor=(0.5 + left / 2, 0.0), borderaxespad=0.1)
    path = style.save(fig, "fig_comparacion_modelos")
    sel.to_csv(style.PREVIEW / "fig_comparacion_modelos_chips.csv", index=False)
    pd.DataFrame(table).to_csv(style.PREVIEW / "fig_comparacion_modelos_iou.csv", index=False)
    print(path)


if __name__ == "__main__":
    main()
