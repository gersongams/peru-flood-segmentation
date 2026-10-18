"""Shared figure style for the thesis (Arial like the text, colour-blind-safe palette, A4 text width).

Every figure script imports this module so that all figures share fonts, sizes and colours.
Figures are saved as PDF (vector) into thesis/E_IMAGENES/ at their final printed size, so text in
them prints at 7-9 pt and is never scaled in LaTeX (use width=\\textwidth only for full-width ones).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis" / "E_IMAGENES"
PREVIEW = ROOT / "outputs" / "figures_preview"  # git-ignored PNG previews

CM = 1 / 2.54
TEXT_WIDTH = 15.5 * CM  # A4, margins 3 cm left / 2.5 cm right (Preambulo.tex)

# Paul Tol "bright" / "muted" (colour-blind safe).
MODEL = {"unet": "#4477AA", "terramind_v1_base": "#EE6677", "prithvi_eo_v2_300": "#228833"}
MODEL_NAME = {"unet": "U-Net", "terramind_v1_base": "TerraMind v1-B", "prithvi_eo_v2_300": "Prithvi-EO-2.0"}
SPLIT = {"train": "#4477AA", "dev": "#CCBB44", "test": "#EE6677", "2017": "#AA3377"}
SPLIT_NAME = {"train": "Entrenamiento", "dev": "Validación", "test": "Prueba", "2017": "Prueba 2017"}
# Error maps: correct water, false positive (commission), false negative (omission), ignored.
TP, FP, FN, IGNORE = "#1A4B8C", "#F2A33A", "#D6457A", "#E4E4E4"  # grey levels 0.27 / 0.68 / 0.46 / 0.89
GREY = "#6E6E6E"
LIGHT = "#E6E6E6"
INK = "#222222"


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": LIGHT,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.formatter.use_locale": False,
    })


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW / f"{name}.png", dpi=200)
    plt.close(fig)
    return path
