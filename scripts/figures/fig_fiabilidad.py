"""fig_fiabilidad: (a) reliability diagram of the Peru-trained models on peru-dev; (b) ECE of the
frozen rows on the 2023 test, single model vs temperature vs ensemble + temperature.

(a) Confidence = max(p, 1 - p), 15 equal-width bins on [0.5, 1]; the 3 seeds are pooled ("single").
    Accuracy is plotted against the mean confidence of each bin (column conf, not the bin centre).
    The strip below gives the labelled pixels per bin (log scale): almost all pixels fall in the
    top bin, so the low-confidence bins, although far below the diagonal, weigh little in the ECE.
(b) ECE in % over all labelled pixels and within 10 px of a label water edge ("cerca del borde"),
    for the rows of thesis table 3.8. Methods: single (mean over 3 seeds), temp_s1f (temperature
    fitted on Sen1Floods11 validation, mean over seeds), ens+temp_s1f (3-seed ensemble, then that
    temperature). The Peru-trained U-Net has no Sen1Floods11 temperature: temp_dev and
    ens+temp_dev (fitted on Peru validation, reference only), marked with an asterisk.

Inputs:
  outputs/analysis_dev/reliability_bins.csv   split=peru-dev, regime=peru, method=single, region=all
  outputs/phaseD/test/calibration.csv         split=peru-test, subset=full; ece_all_mean, ece_near_mean
Outputs:
  thesis/E_IMAGENES/fig_fiabilidad.pdf, outputs/figures_preview/fig_fiabilidad.png
Prints the share of pixels in the top confidence bin per model (for the caption).

    uv run python scripts/figures/fig_fiabilidad.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import rows  # noqa: E402
import style  # noqa: E402

style.apply()
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})

bins = pd.read_csv(rows.ROOT / "outputs" / "analysis_dev" / "reliability_bins.csv")
bins = bins[(bins.split == "peru-dev") & (bins.regime == "peru") & (bins.method == "single") & (bins.region == "all")]
cal = rows.load("calibration.csv", split="peru-test", subset="full")

CAL_ROWS = ["trhm/unet", "trfda/terramind_v1_base", "s1f11_src/prithvi_eo_v2_300", "adabn/unet", "peru/unet"]
METHODS = [("single", "individual", "o", "white"), ("temp_s1f", "temperatura", "o", None),
           ("ens+temp_s1f", "ensamble y temperatura", "D", None)]

fig = plt.figure(figsize=(style.TEXT_WIDTH, 8.4 * style.CM))
gs = GridSpec(2, 1, figure=fig, height_ratios=[4.2, 1.0], left=0.07, right=0.375, top=0.90, bottom=0.12,
              hspace=0.10)
ax = fig.add_subplot(gs[0, 0])
axn = fig.add_subplot(gs[1, 0], sharex=ax)

# ---------- (a) reliability ----------
ax.plot([0.5, 1], [0.5, 1], color=style.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
for m in rows.MODELS:
    d = bins[bins.model == m].sort_values("bin")
    assert len(d) == 15, m
    col = style.MODEL[m]
    ax.plot(d.conf, d.acc, color=col, lw=1.0, marker="o", ms=2.8, mfc=col, mec="white", mew=0.4, zorder=3)
    axn.step(np.r_[d.lo, 1.0], np.r_[d.n, d.n.iloc[-1]], where="post", color=col, lw=0.8)
    top = d.n.iloc[-1] / d.n.sum()
    print(f"{m}: top bin [{d.lo.iloc[-1]:.4f}, 1] holds {100 * top:.2f} % of {d.n.sum():,} pixels")
# direct labels: placed relative to each curve (x position and offset are layout only)
for m, xl, off, va in [("terramind_v1_base", 0.64, 0.03, "bottom"), ("unet", 0.83, 0.016, "bottom"),
                       ("prithvi_eo_v2_300", 0.80, -0.018, "top")]:
    d = bins[bins.model == m].sort_values("bin")
    yl = np.interp(xl, d.conf, d.acc)
    ax.text(xl, yl + off, style.MODEL_NAME[m], color=style.MODEL[m], fontsize=6.8, ha="center", va=va,
            rotation=0)
fig.canvas.draw()
p0, p1 = ax.transData.transform([(0.6, 0.6), (0.9, 0.9)])
ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
ax.text(0.80, 0.81, "calibración perfecta", rotation=ang, rotation_mode="anchor", fontsize=6.3,
        color=style.GREY, ha="center", va="bottom")
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.45, 1.0)
ax.set_ylabel("Exactitud")
ax.tick_params(labelbottom=False)
ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
axn.set_yscale("log")
axn.set_ylim(5e3, 5e8)
axn.set_yticks([1e4, 1e6, 1e8], [r"$10^{4}$", r"$10^{6}$", r"$10^{8}$"])
axn.minorticks_off()
axn.set_ylabel("Píxeles", fontsize=7)
axn.set_xlabel("Confianza, máx(p, 1 − p)")
axn.set_xticks(np.arange(0.5, 1.01, 0.1))
axn.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

# ---------- (b) ECE dot plots ----------
right = GridSpec(1, 2, figure=fig, left=0.71, right=0.985, top=0.79, bottom=0.12, wspace=0.14)
axa, axb = fig.add_subplot(right[0, 0]), fig.add_subplot(right[0, 1], sharey=None)
ys = -np.arange(len(CAL_ROWS)) * 1.0
for axe, col_name, xmax, title in [(axa, "ece_all_mean", 3.5, "todos los píxeles"),
                                   (axb, "ece_near_mean", 25, "cerca del borde")]:
    for y, g in zip(ys, CAL_ROWS):
        _, model = rows.split_group(g)
        mcol = style.MODEL[model]
        meths = [mm if g != "peru/unet" else mm.replace("s1f", "dev") for mm, *_ in METHODS]
        vals = [100 * rows.pick(cal[cal.method == mm], [g])[col_name].iloc[0] for mm in meths]
        axe.plot([min(vals), max(vals)], [y, y], color=mcol, lw=0.8, alpha=0.5, zorder=1)
        for v, (_, _, mk, face) in zip(vals, METHODS):
            axe.plot(v, y, marker=mk, ms=4.4 if mk == "o" else 3.8, mfc=face or mcol, mec=mcol, mew=0.8,
                     ls="none", zorder=3)
    axe.set_xlim(0, xmax)
    axe.set_ylim(ys[-1] - 0.6, 0.6)
    axe.spines["left"].set_visible(False)
    axe.tick_params(axis="y", length=0)
    axe.set_title(title, fontsize=7, color=style.INK, pad=3)
    axe.set_xlabel("ECE (%)")
    for y in ys:
        axe.axhline(y, color=style.LIGHT, lw=0.4, zorder=0)
axa.set_yticks(ys, [rows.row_name(g) + ("*" if g == "peru/unet" else "") for g in CAL_ROWS])
axa.tick_params(axis="y", labelsize=6.8, labelcolor=style.INK)
axb.set_yticks([])
axa.set_xticks([0, 1, 2, 3])
axb.set_xticks([0, 10, 20])
TY = 0.955
fig.text(0.0, TY, "a) Entrenados en el Perú · validación peruana", fontsize=7.8, color=style.INK,
         ha="left", va="center")
fig.canvas.draw()
xb = min(t.get_window_extent().transformed(fig.transFigure.inverted()).x0 for t in axa.get_yticklabels())
fig.text(xb, TY, "b) Configuraciones congeladas · prueba de 2023", fontsize=7.8, color=style.INK,
         ha="left", va="center")
kx, ky = xb, 0.885
for (_, txt, mk, face) in METHODS:
    fig.add_artist(plt.Line2D([kx + 0.006], [ky], marker=mk, ms=4.0 if mk == "o" else 3.5,
                              mfc=face or style.INK, mec=style.INK, mew=0.8, ls="none"))
    t = fig.text(kx + 0.018, ky, txt, ha="left", va="center", fontsize=6.6, color=style.INK)
    fig.canvas.draw()
    kx += 0.018 + t.get_window_extent().transformed(fig.transFigure.inverted()).width + 0.03
style.save(fig, "fig_fiabilidad")
