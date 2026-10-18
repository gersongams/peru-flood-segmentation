"""fig_prueba_ic: paired differences of water IoU with 95 % chip-bootstrap CIs (forest plot).

Rows are read from bootstrap_diffs.csv in the file's a - b orientation (no sign flips). Groups:
  Perú 2023 (peru-test, full): adapted - source (U-Net trHM, TerraMind trFDA);
    frozen - Peru-trained (U-Net trHM, Prithvi source; the file has no TerraMind trFDA - Perú row);
    source - source (U-Net - Prithvi, U-Net - TerraMind).
  Sen1Floods11 test (test, full): U-Net - Prithvi, U-Net - TerraMind (source models).
The p_a_not_better column is not shown.

Inputs:
  outputs/phaseD/test/bootstrap_diffs.csv   columns a, b, diff, ci_low, ci_high
Outputs:
  thesis/E_IMAGENES/fig_prueba_ic.pdf, outputs/figures_preview/fig_prueba_ic.png

    uv run python scripts/figures/fig_prueba_ic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import rows  # noqa: E402
import style  # noqa: E402

style.apply()

MINUS = "−"
# (split, subset, header, [(a, b, label, colour key or None)])
GROUPS = [
    ("peru-test", "full", "Adaptada frente al modelo de origen · prueba peruana de 2023", [
        ("trhm/unet", "s1f11_src/unet", f"U-Net: igualación de histogramas (entren.) {MINUS} origen", "unet"),
        ("trfda/terramind_v1_base", "s1f11_src/terramind_v1_base", f"TerraMind: FDA (entren.) {MINUS} origen",
         "terramind_v1_base"),
    ]),
    ("peru-test", "full", "Configuración congelada frente al modelo entrenado en el Perú · prueba peruana de 2023", [
        ("trhm/unet", "peru/unet", f"U-Net: igualación de histogramas (entren.) {MINUS} entrenada en el Perú",
         "unet"),
        ("s1f11_src/prithvi_eo_v2_300", "peru/prithvi_eo_v2_300", f"Prithvi: origen {MINUS} entrenado en el Perú",
         "prithvi_eo_v2_300"),
    ]),
    ("peru-test", "full", "Modelos de origen · prueba peruana de 2023", [
        ("s1f11_src/unet", "s1f11_src/prithvi_eo_v2_300", f"U-Net, origen {MINUS} Prithvi, origen", None),
        ("s1f11_src/unet", "s1f11_src/terramind_v1_base", f"U-Net, origen {MINUS} TerraMind, origen", None),
    ]),
    ("test", "full", "Modelos de origen · prueba de Sen1Floods11", [
        ("s1f11_src/unet", "s1f11_src/prithvi_eo_v2_300", f"U-Net {MINUS} Prithvi", None),
        ("s1f11_src/unet", "s1f11_src/terramind_v1_base", f"U-Net {MINUS} TerraMind", None),
    ]),
]

diffs = rows.load("bootstrap_diffs.csv")

fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 8.2 * style.CM))
fig.subplots_adjust(left=0.47, right=0.795, top=0.985, bottom=0.12)
ax.patch.set_visible(False)

y = 0.0
yt, ytl, headers = [], [], []
for split, subset, header, items in GROUPS:
    d = diffs[(diffs["split"] == split) & (diffs["subset"] == subset)]
    headers.append((y, header))
    y -= 0.78
    y_first = y
    for a, b, label, key in items:
        r = rows.diff_row(d, a, b)
        col = style.MODEL[key] if key else style.INK
        ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
        for e in (r.ci_low, r.ci_high):
            ax.plot([e, e], [y - 0.13, y + 0.13], color=col, lw=0.9, zorder=2)
        ax.plot(r["diff"], y, "o", ms=4.6, mfc=col, mec="white", mew=0.6, zorder=3)
        txt = f"{r['diff']:+.3f}  [{r.ci_low:+.3f}; {r.ci_high:+.3f}]".replace("-", MINUS)
        ax.text(1.03, y, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8,
                color=style.INK, family="Arial")
        yt.append(y)
        ytl.append(label)
        y -= 0.62
    ax.plot([0, 0], [y + 0.62 - 0.38, y_first + 0.38], color=style.GREY, lw=0.6, zorder=1)
    y -= 0.22

ax.set_yticks(yt, ytl)
ax.tick_params(axis="y", length=0, labelsize=6.8, labelcolor=style.INK, pad=4)
ax.spines["left"].set_visible(False)
ax.set_ylim(y - 0.15, headers[0][0] + 0.35)
ax.set_xlim(-0.25, 0.33)
ax.set_xticks(np.arange(-0.2, 0.31, 0.1))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}".replace("-", MINUS) if v else "0"))
ax.set_xlabel(f"Diferencia pareada de IoU de agua (a {MINUS} b) e IC 95 %")

fig.canvas.draw()
for yh, name in headers:
    _, yf = fig.transFigure.inverted().transform(ax.transData.transform((0, yh)))
    fig.text(0.005, yf, name, ha="left", va="center", fontsize=7.3, fontweight="bold", color=style.INK)
_, ytop = fig.transFigure.inverted().transform(ax.transData.transform((0, headers[0][0])))
fig.text(0.995, ytop, "diferencia [IC 95 %]", ha="right", va="center", fontsize=6.5, color=style.GREY)
# direction hints under the axis ends
for x, txt, ha in [(-0.245, f"← a menor que b", "left"), (0.325, "a mayor que b →", "right")]:
    ax.text(x, ax.get_ylim()[0] + 0.05, txt, ha=ha, va="bottom", fontsize=6.3, color=style.GREY)
style.save(fig, "fig_prueba_ic")
