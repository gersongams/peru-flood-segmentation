"""fig_brecha: transfer gap and adaptation on the 2023 Peru test, one block per model.

For each backbone, water IoU on peru-test (full, 172 chips) of the source-only model (origen),
the frozen adapted row (U-Net trHM, TerraMind trFDA; Prithvi has none) and the Peru-trained model,
each with its 95 % chip-bootstrap CI of the seed mean. The light band spans origen -> Perú (the
gap); the label gives the share of the gap closed by the adapted row.

Inputs:
  outputs/phaseD/test/peru-test_groups.csv   subset=full; iou_mean, gap_closed
  outputs/phaseD/test/bootstrap_groups.csv   split=peru-test, subset=full; ci_low, ci_high
Outputs:
  thesis/E_IMAGENES/fig_brecha.pdf, outputs/figures_preview/fig_brecha.png

    uv run python scripts/figures/fig_brecha.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import rows  # noqa: E402
import style  # noqa: E402

style.apply()

CHAINS = {
    "unet": ["s1f11_src/unet", "trhm/unet", "peru/unet"],
    "terramind_v1_base": ["s1f11_src/terramind_v1_base", "trfda/terramind_v1_base", "peru/terramind_v1_base"],
    "prithvi_eo_v2_300": ["s1f11_src/prithvi_eo_v2_300", None, "peru/prithvi_eo_v2_300"],
}
STAGE = ["origen (sin adaptar)", "adaptada sin etiquetas peruanas", "entrenada en el Perú"]

groups = rows.load("peru-test_groups.csv", subset="full")
boot = rows.load("bootstrap_groups.csv", split="peru-test", subset="full")
wanted = [g for c in CHAINS.values() for g in c if g]
g = rows.pick(groups, wanted)
b = rows.pick(boot, wanted)
assert np.allclose(g["iou_mean"], b["iou"]), "bootstrap point estimate differs from the group mean"

fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 7.4 * style.CM))
fig.subplots_adjust(left=0.27, right=0.91, top=0.90, bottom=0.13)

DY = 0.62          # spacing of the three sub-rows inside a block
BLOCK = 3 * DY + 0.75
y0 = 0.0
yt, ytl, headers = [], [], []
for model, chain in CHAINS.items():
    col = style.MODEL[model]
    ys = [y0 - i * DY for i in range(3)]
    src, peru = g.loc[chain[0], "iou_mean"], g.loc[chain[2], "iou_mean"]
    # the gap: light band from origen to Perú over the whole block
    ax.fill_betweenx([ys[0] + 0.28, ys[2] - 0.28], src, peru, color=col, alpha=0.07, lw=0, zorder=0)
    for x in (src, peru):
        ax.plot([x, x], [ys[0] + 0.28, ys[2] - 0.28], color=col, lw=0.5, alpha=0.45, ls=(0, (2, 2)), zorder=1)
    pts = []
    for i, grp in enumerate(chain):
        y = ys[i]
        yt.append(y)
        if grp is None:
            ytl.append("sin configuración adaptada")
            continue
        label = STAGE[i]
        if i == 1:
            label = rows.method_name(grp)
        if i == 2:
            label = rows.method_name(grp)
        ytl.append(label)
        lo, hi, x = b.loc[grp, "ci_low"], b.loc[grp, "ci_high"], g.loc[grp, "iou_mean"]
        ax.plot([lo, hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.07, y + 0.07], color=col, lw=0.9, zorder=2)
        face = "white" if i == 0 else col
        marker = "s" if i == 2 else "o"
        ax.plot(x, y, marker=marker, ms=5.2 if marker == "o" else 4.6, mfc=face, mec=col, mew=1.0,
                ls="none", zorder=4)
        note = f"{x:.3f}"
        ax.text(hi + 0.007, y, note, va="center", ha="left", fontsize=6.5, color=style.INK)
        if i == 1:
            gc = g.loc[grp, "gap_closed"]
            ax.text(1.035, y, f"{100 * gc:.0f} %", transform=ax.get_yaxis_transform(), va="center",
                    ha="right", fontsize=7, color=style.INK)
    # model name, left of the block
    headers.append((ys[0] + 0.40, style.MODEL_NAME[model]))
    y0 -= BLOCK

ax.patch.set_visible(False)
ax.text(1.035, 0.62, "brecha\ncerrada", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        fontsize=6.5, color=style.GREY, linespacing=1.0)
ax.set_yticks(yt, ytl)
ax.tick_params(axis="y", length=0, labelsize=7, labelcolor=style.INK, pad=4)
ax.spines["left"].set_visible(False)
ax.set_xlim(0.44, 0.93)
ax.set_ylim(y0 + BLOCK - 2 * DY - 0.45, 0.62)
ax.set_xticks(np.arange(0.45, 0.931, 0.05))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
# block headers and rules, from the figure's left edge to the axes' right edge
fig.canvas.draw()
for yh, name in headers:
    xf, yf = fig.transFigure.inverted().transform(ax.transData.transform((ax.get_xlim()[0], yh)))
    fig.text(0.005, yf + 0.004, name, ha="left", va="bottom", fontsize=8, fontweight="bold", color=style.INK)
    fig.add_artist(plt.Line2D([0.005, 0.985], [yf, yf], color=style.LIGHT, lw=0.6, zorder=0))
# key for the marker shapes, left-aligned with the plotting area
kx, ky = ax.get_position().x0 + 0.01, 0.965
for txt, mk, face in [("origen", "o", "white"), ("adaptada", "o", style.INK),
                      ("entrenado en el Perú", "s", style.INK)]:
    fig.add_artist(plt.Line2D([kx], [ky], marker=mk, ms=4.2, mfc=face, mec=style.INK, mew=0.9, ls="none"))
    t = fig.text(kx + 0.012, ky, txt, ha="left", va="center", fontsize=6.5, color=style.INK)
    fig.canvas.draw()
    kx += 0.012 + t.get_window_extent().transformed(fig.transFigure.inverted()).width + 0.035
ax.set_xlabel("IoU de agua en la prueba peruana de 2023 (172 recortes)")
style.save(fig, "fig_brecha")
