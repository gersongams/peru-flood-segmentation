"""Probability, uncertainty and errors of the frozen U-Net + training-time histogram matching (trhm)
on `peru-test` (UNOSAT 2023).

Inputs: as scripts/figures/fig_comparacion_modelos.py (imagery, labels with -1 = ignored,
test.txt, and the water logits z = logit_water - logit_other of
outputs/logits_test/trhm/unet/seed{0,1,2}/<stem>.npy written once by scripts/evaluate_frozen.py).

Per pixel, with p_s = sigmoid(z_s) of seed s = 0, 1, 2 (raw logits, no temperature scaling):
- probability of water  p = (p_0 + p_1 + p_2) / 3 (3-seed ensemble mean);
- uncertainty = mutual information (epistemic part, seed disagreement), in bits:
  MI = H(p) - (H(p_0) + H(p_1) + H(p_2)) / 3, with H(q) = -q log2 q - (1 - q) log2 (1 - q);
  0 when the seeds agree, at most 1 bit. (Predictive entropy H(p) was not used: it is a
  monotone function of |p - 0.5| and would repeat the probability panel.) Computed in log space
  from the logits, as in scripts/uncertainty_dev.py (there in nats; here divided by ln 2);
- error map: water where p > 0.5 (ensemble logit > 0), against the label; ignored pixels grey.

Chip selection rule (fixed before rendering): candidates = peru-test chips whose labelled water
is >= 5 % of their labelled pixels (37). Sort by the per-chip ensemble IoU of U-Net trhm and
take, for target quantiles 0.1 / 0.5 / 0.9, the chip at the nearest rank round(q * (n - 1));
if its location (E.._N.., present on both dates) is already chosen, the next-nearest rank
(ties: lower rank first). Full 512 x 512 chips. The IoU printed in each error tile is the
per-chip tp / (tp + fp + fn) of the ensemble on the labelled pixels (floodseg.metrics.counts).

    uv run python scripts/figures/fig_incertidumbre.py
-> thesis/E_IMAGENES/fig_incertidumbre.pdf, outputs/figures_preview/fig_incertidumbre.png and
   fig_incertidumbre_chips.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import style  # noqa: E402
from fig_comparacion_modelos import (LOGITS, MIN_WATER, SEEDS, U, error_rgb, iou, label,  # noqa: E402
                                     label_rgb, rgb, stretch, tile)
from uncertainty_dev import ens_logit  # noqa: E402

from floodseg.metrics import counts  # noqa: E402

REGIME, MODEL = "trhm", "unet"
QUANTILES = (0.1, 0.5, 0.9)
MI_MAX = 0.5  # colour-scale limit (bits), shared by all rows; values above are drawn at the top colour


def entropy_bits(z: np.ndarray) -> np.ndarray:
    """Binary entropy of sigmoid(z) in bits, stable for any z."""
    lp, lq = -np.logaddexp(0, -z), -np.logaddexp(0, z)  # log p, log(1 - p)
    return -(np.exp(lp) * lp + np.exp(lq) * lq) / np.log(2)


def ensemble(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ensemble logit, mean probability, mutual information in bits)."""
    zs = [np.load(LOGITS / REGIME / MODEL / s / f"{stem}.npy").astype(np.float64) for s in SEEDS]
    ze = ens_logit(zs)
    p = np.mean([np.exp(-np.logaddexp(0, -z)) for z in zs], axis=0)
    mi = entropy_bits(ze) - np.mean([entropy_bits(z) for z in zs], axis=0)
    return ze, p, np.clip(mi, 0, None)  # clip only float round-off (MI >= 0 by Jensen)


def select() -> pd.DataFrame:
    rows = []
    for s in (U / "splits" / "test.txt").read_text().split():
        y = label(s)
        ok = y != -1
        if (y == 1).sum() / max(ok.sum(), 1) < MIN_WATER:
            continue
        rows.append(dict(stem=s, loc=s.split("_", 1)[1], iou=iou(counts(ensemble(s)[0] > 0, y)[0])))
    d = pd.DataFrame(rows).sort_values(["iou", "stem"]).reset_index(drop=True)
    chosen, locs = [], set()
    for q in QUANTILES:
        target = round(q * (len(d) - 1))
        for r in sorted(d.index, key=lambda i: (abs(i - target), i)):
            if d.loc[r, "loc"] not in locs:
                chosen.append(r)
                locs.add(d.loc[r, "loc"])
                break
    out = d.loc[chosen].assign(rank=chosen, quantile=QUANTILES)
    print(f"{len(d)} candidates; chosen:\n{out.to_string(index=False)}")
    return out


def main() -> None:
    style.apply()
    sel = select()
    stems = list(sel.stem)
    images = [rgb(s) for s in stems]
    lo, hi = np.percentile(np.concatenate([i[label(s) != -1].ravel() for s, i in zip(stems, images)]), [1, 99])

    ncol, nrow = 5, len(stems)
    left, gap = 0.035, 0.006
    W = style.TEXT_WIDTH
    tw = (1 - left - gap * (ncol - 1)) * W / ncol
    top_in, bot_in = 0.30, 0.50
    H = top_in + bot_in + nrow * tw + (nrow - 1) * gap * W
    fig = plt.figure(figsize=(W, H))
    headers = ["Imagen (RGB)", "Etiqueta (UNOSAT)", "Probabilidad de agua", "Incertidumbre (IM)", "Errores"]
    cols_x = [left + c * (tw / W + gap) for c in range(ncol)]
    stats = []
    for r, (s, img) in enumerate(zip(stems, images)):
        y = label(s)
        ze, p, mi = ensemble(s)
        c = counts(ze > 0, y)[0]
        y0 = 1 - (top_in + r * (tw + gap * W) + tw) / H
        axs = [fig.add_axes([x, y0, tw / W, tw / H]) for x in cols_x]
        tile(axs[0], stretch(img, lo, hi))
        tile(axs[1], label_rgb(y))
        im_p = tile(axs[2], p, cmap="cividis", vmin=0, vmax=1)
        im_u = tile(axs[3], mi, cmap="magma", vmin=0, vmax=MI_MAX)
        tile(axs[4], error_rgb(ze > 0, y), iou(c))
        stats.append(dict(stem=s, tp=c[0], fp=c[1], fn=c[2], iou=iou(c), mi_mean=float(mi.mean()),
                          mi_p99=float(np.percentile(mi, 99)), mi_max=float(mi.max()),
                          share_above_vmax=float((mi > MI_MAX).mean())))
        date, loc = s.split("_", 1)
        axs[0].text(-0.05, 0.5, f"{'abc'[r]}) {date}\n{loc.replace('_', ' ')}", transform=axs[0].transAxes,
                    rotation=90, ha="right", va="center", fontsize=6.2, color=style.INK, linespacing=1.25)
        if r == 0:
            for ax, h in zip(axs, headers):
                ax.set_title(h, fontsize=6.8, pad=3, color=style.INK)
    # slim colour bars under the probability and uncertainty columns, legend under the rest
    cb_h, cb_y = 0.07 / H, (bot_in - 0.17) / H
    for x, im, lab, ticks in [(cols_x[2], im_p, "Probabilidad media de agua", [0, 0.25, 0.5, 0.75, 1]),
                              (cols_x[3], im_u, "Información mutua (bits)", [0, 0.1, 0.2, 0.3, 0.4, 0.5])]:
        cax = fig.add_axes([x + 0.06 * tw / W, cb_y, 0.88 * tw / W, cb_h])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks,
                          extend="max" if im is im_u else "neither", extendfrac=0.05)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, length=2, width=0.4, pad=1.5)
        cb.ax.set_xticklabels([f"{t:g}" for t in ticks])
        cb.set_label(lab, fontsize=6.2, labelpad=2)
    kw = dict(loc="upper left", fontsize=6.2, handlelength=1.0, handleheight=0.9, handletextpad=0.4,
              labelspacing=0.3, borderaxespad=0, borderpad=0)
    y_leg = (bot_in - 0.06) / H
    fig.legend(handles=[Patch(fc=style.TP, ec="none", label="Agua"),
                        Patch(fc="white", ec="#9A9A9A", lw=0.4, label="No agua"),
                        Patch(fc=style.IGNORE, ec="none", label="Sin etiqueta (ignorado)")],
               bbox_to_anchor=(cols_x[1] + 0.06 * tw / W, y_leg), **kw)
    fig.legend(handles=[Patch(fc=style.TP, ec="none", label="Agua (acierto)"),
                        Patch(fc=style.FP, ec="none", label="Falso positivo"),
                        Patch(fc=style.FN, ec="none", label="Omisión")],
               bbox_to_anchor=(cols_x[4] + 0.06 * tw / W, y_leg), **kw)
    path = style.save(fig, "fig_incertidumbre")
    sel.merge(pd.DataFrame(stats), on="stem", suffixes=("_select", "")).to_csv(
        style.PREVIEW / "fig_incertidumbre_chips.csv", index=False)
    print(pd.DataFrame(stats).to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
