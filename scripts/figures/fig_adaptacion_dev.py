"""fig_adaptacion_dev: water IoU on peru-dev by adaptation method, one panel per model.

Per-seed values (small dots) and mean ± standard deviation over the three seeds (marker and bar)
for the source-only model, each adaptation method and the Peru-trained model. The light band spans
source-only -> Peru-trained (the gap). Test-time FDA (IoU about 0.01) is left out of the axis range
and named in the caption of the report.

Input:
  outputs/phaseD/dev_table_runs.csv   regime, model, seed, iou (peru-dev, 174 chips)
Outputs:
  parcial/figuras/fig_adaptacion_dev.pdf, outputs/figures_preview/fig_adaptacion_dev.png

    uv run python scripts/figures/fig_adaptacion_dev.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import style  # noqa: E402

style.apply()

ROWS = [
    ("s1f11_src", "Origen, sin adaptar"),
    ("adabn", "AdaBN"),
    ("hm", "Igualación de histogramas (inferencia)"),
    ("trhm", "Igualación de histogramas (entrenamiento)"),
    ("trfda", "FDA (entrenamiento)"),
    ("st", "Autoentrenamiento"),
    ("peru", "Entrenado en el Perú"),
]
# Frozen primary configuration of each model (configs/frozen.yaml).
PRIMARY = {"unet": "trhm", "terramind_v1_base": "trfda", "prithvi_eo_v2_300": "s1f11_src"}

runs = pd.read_csv(style.ROOT / "outputs" / "phaseD" / "dev_table_runs.csv")
runs = runs[runs.regime.isin([r for r, _ in ROWS])]
stats = runs.groupby(["model", "regime"]).iou.agg(["mean", "std", "count"])
assert (stats["count"] == 3).all(), "every plotted row needs three seeds"

models = list(style.MODEL)
fig, axs = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 6.6 * style.CM), sharey=True)
fig.subplots_adjust(left=0.30, right=0.99, top=0.90, bottom=0.16, wspace=0.10)
ys = list(range(len(ROWS)))[::-1]

for ax, model in zip(axs, models):
    col = style.MODEL[model]
    src = stats.loc[(model, "s1f11_src"), "mean"]
    peru = stats.loc[(model, "peru"), "mean"]
    ax.axvspan(src, peru, color=col, alpha=0.08, lw=0, zorder=0)
    for y, (regime, _) in zip(ys, ROWS):
        if (model, regime) not in stats.index:
            ax.text(0.70, y, "no evaluado", ha="center", va="center", fontsize=6.5, color=style.GREY)
            continue
        mean, std = stats.loc[(model, regime), ["mean", "std"]]
        seeds = runs[(runs.model == model) & (runs.regime == regime)].iou
        ax.plot(seeds, [y] * len(seeds), ".", color=col, ms=3, alpha=0.45, zorder=2)
        marker = "s" if regime == "peru" else "o"
        face = "white" if regime == "s1f11_src" else col
        ax.errorbar(mean, y, xerr=std, fmt=marker, color=col, mfc=face, ms=4.5, lw=1.0, capsize=2, zorder=3)
        bold = "bold" if PRIMARY[model] == regime else "normal"
        ax.text(mean, y + 0.36, f"{mean:.3f}", ha="center", va="center", fontsize=6.5, color=style.INK,
                fontweight=bold)
    ax.set_title(style.MODEL_NAME[model], color=col, fontweight="bold")
    ax.set_xlim(0.44, 0.98)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(-0.6, len(ROWS) - 0.3)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)

axs[0].set_yticks(ys)
axs[0].set_yticklabels([name for _, name in ROWS])
axs[1].set_xlabel("IoU de agua en la validación peruana (174 recortes)")

out = style.ROOT / "parcial" / "figuras" / "fig_adaptacion_dev.pdf"
fig.savefig(out)
style.PREVIEW.mkdir(parents=True, exist_ok=True)
fig.savefig(style.PREVIEW / "fig_adaptacion_dev.png")
print(out)
print(stats.round(4).to_string())
