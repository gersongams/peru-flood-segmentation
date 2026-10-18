"""fig_validacion_prueba: water IoU on validation, 2023 test and 2017 test for all 11 rows.

Cleveland dot plot, one row per configuration grouped by backbone. Open circle: peru-dev
(validation, 174 chips); filled circle: peru-test 2023 (172 chips); diamond: peru-2017-test without
2023-train overlap (101 chips). A thin line joins validation and 2023 test (the drop); the 2017
diamond shows the reversal (source-only models score highest in 2017). Bold labels: the three
frozen primary rows.

Validation values come from the frozen-path check (peru-dev_check_groups.csv), which holds the
recomputed AdaBN U-Net value 0.8993 used in the thesis tables.

Inputs:
  outputs/phaseD/test/peru-dev_check_groups.csv    subset=full; iou_mean, role
  outputs/phaseD/test/peru-test_groups.csv         subset=full; iou_mean
  outputs/phaseD/test/peru-2017-test_groups.csv    subset=no_2023_train_overlap; iou_mean
Outputs:
  thesis/E_IMAGENES/fig_validacion_prueba.pdf, outputs/figures_preview/fig_validacion_prueba.png

    uv run python scripts/figures/fig_validacion_prueba.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import rows  # noqa: E402
import style  # noqa: E402

style.apply()

BLOCKS = {
    "unet": ["s1f11_src/unet", "adabn/unet", "trhm/unet", "peru/unet"],
    "terramind_v1_base": ["s1f11_src/terramind_v1_base", "hm/terramind_v1_base", "trhm/terramind_v1_base",
                          "trfda/terramind_v1_base", "peru/terramind_v1_base"],
    "prithvi_eo_v2_300": ["s1f11_src/prithvi_eo_v2_300", "peru/prithvi_eo_v2_300"],
}
G = [g for b in BLOCKS.values() for g in b]
assert sorted(G) == sorted(rows.TABLE_ROWS)
devdf = rows.pick(rows.load("peru-dev_check_groups.csv", subset="full"), G)
dev = devdf["iou_mean"]
t23 = rows.pick(rows.load("peru-test_groups.csv", subset="full"), G)["iou_mean"]
t17 = rows.pick(rows.load("peru-2017-test_groups.csv", subset="no_2023_train_overlap"), G)["iou_mean"]

fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 9.0 * style.CM))
fig.subplots_adjust(left=0.30, right=0.985, top=0.90, bottom=0.11)
ax.patch.set_visible(False)

DY, GAP = 0.55, 0.75
y = 0.0
yt, ytl, bold, headers = [], [], [], []
for model, groups in BLOCKS.items():
    col = style.MODEL[model]
    headers.append((y + 0.05, style.MODEL_NAME[model]))
    y -= 0.62
    for g in groups:
        ax.plot([0.5, 1.0], [y, y], color=style.LIGHT, lw=0.4, zorder=0)
        ax.plot([dev[g], t23[g]], [y, y], color=col, lw=1.0, alpha=0.55, solid_capstyle="butt", zorder=2)
        ax.plot(dev[g], y, "o", ms=4.6, mfc="none", mec=col, mew=0.9, zorder=5)
        ax.plot(t23[g], y, "o", ms=4.8, mfc=col, mec="white", mew=0.5, zorder=4)
        ax.plot(t17[g], y, "D", ms=4.0, mfc=style.INK, mec="white", mew=0.5, zorder=4)
        yt.append(y)
        ytl.append(rows.method_name(g))
        bold.append(devdf.loc[g, "role"] == "primary")
        y -= DY
    y -= GAP - DY

ax.set_yticks(yt, ytl)
for lab, b in zip(ax.get_yticklabels(), bold):
    lab.set_fontweight("bold" if b else "normal")
ax.tick_params(axis="y", length=0, labelsize=7, labelcolor=style.INK, pad=4)
ax.spines["left"].set_visible(False)
ax.set_xlim(0.55, 0.96)
ax.set_ylim(y + GAP - 0.4, headers[0][0] + 0.25)
ax.set_xticks(np.arange(0.55, 0.951, 0.05))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_xlabel("IoU de agua")

fig.canvas.draw()
for yh, name in headers:
    _, yf = fig.transFigure.inverted().transform(ax.transData.transform((0.6, yh)))
    fig.text(0.005, yf, name, ha="left", va="center", fontsize=8, fontweight="bold", color=style.INK)
    fig.add_artist(plt.Line2D([0.005, 0.985], [yf - 0.022, yf - 0.022], color=style.LIGHT, lw=0.6, zorder=0))

# key
items = [("validación peruana", dict(marker="o", mfc="white", mec=style.INK, ms=4.4)),
         ("prueba de 2023", dict(marker="o", mfc=style.INK, mec=style.INK, ms=4.4)),
         ("prueba de 2017 (101 recortes)", dict(marker="D", mfc=style.INK, mec=style.INK,
                                                                ms=3.8))]
kx, ky = ax.get_position().x0, 0.965
for txt, mk in items:
    fig.add_artist(plt.Line2D([kx + 0.006], [ky], ls="none", mew=0.8, **mk))
    t = fig.text(kx + 0.018, ky, txt, ha="left", va="center", fontsize=6.8, color=style.INK)
    fig.canvas.draw()
    kx += 0.018 + t.get_window_extent().transformed(fig.transFigure.inverted()).width + 0.035
style.save(fig, "fig_validacion_prueba")
