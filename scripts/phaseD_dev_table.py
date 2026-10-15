"""Phase-D comparison table on `peru-dev` (tasks 3.1, 3.2; later rows from sections 3–4).

Rows, all scored with floodseg.metrics (plain and boundary-tolerant water IoU, precision,
recall; pooled over the 174 dev chips):
- Label-free index baselines: MNDWI > 0, and MNDWI > t_Otsu with t_Otsu = Otsu threshold of
  the MNDWI of the unlabelled target pool (`configs/pools/peru-train.csv`, every 50th pixel
  with any non-zero band, read under the Peru-label guard).
- Oracle row: MNDWI > t_dev, with t_dev the threshold in [−0.30, 0.60] (step 0.01) that
  maximizes plain IoU on peru-dev itself (uses dev labels; optimistic by construction).
- Models, from saved float32 logits (water if z > 0), one row per run, then mean ± std per
  regime and backbone:
  `s1f11_src` = phase-D Sen1Floods11 source models (3 seeds, source-only),
  `peru` = Peru-trained models (upper bound),
  `s1f11` = phase-C seed-0 Sen1Floods11 models (reference only),
  `s1f11_src_lr2e-5` = superseded TerraMind source runs at LR 2e-5 (diagnostic only; not used),
  `adabn` = U-Net source models after AdaBN (scripts/adabn.py).
- Gap closed (adaptation rows only) vs source-only and the Peru-trained upper bound of the same
  backbone: (IoU − IoU_source) / (IoU_upper − IoU_source).
MNDWI = (B03 − B11)/(B03 + B11), as in scripts/index_baseline.py.

Writes outputs/phaseD/dev_table_runs.csv and outputs/phaseD/dev_table.md.

    uv run python scripts/phaseD_dev_table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.filters import threshold_otsu

from floodseg.metrics import counts, scores
from floodseg.target_pool import forbid_peru_labels, read_manifest

REPO = Path(__file__).resolve().parents[1]
PERU = REPO / "data" / "peru" / "unosat2023"
LOGITS = REPO / "outputs" / "logits_dev"
OUT = REPO / "outputs" / "phaseD"
REGIMES = {"s1f11_src": "Sen1Floods11 source-only (phase D)", "peru": "Peru-trained (upper bound)",
           "s1f11": "Sen1Floods11 seed 0 (phase C, Mac, config LRs; reference)",
           "s1f11_src_lr2e-5": "Sen1Floods11 TerraMind LR 2e-5 (superseded, diagnostic)",
           "adabn": "AdaBN on the pool (source-only + BN statistics)"}
NOT_METHODS = {"s1f11_src", "peru", "s1f11", "s1f11_src_lr2e-5"}  # no gap-closed value
MODELS = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}
T_SWEEP = np.round(np.arange(-0.30, 0.601, 0.01), 2)


def mndwi(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (x[2] - x[11]) / (x[2] + x[11] + 1e-6)


def otsu_on_pool() -> tuple[float, int]:
    v = []
    with forbid_peru_labels():
        paths = read_manifest("peru-train")
        for p in paths:
            with rasterio.open(p) as src:
                x = src.read()
            v.append(mndwi(x)[(x != 0).any(0)][::50])
    return float(threshold_otsu(np.concatenate(v))), len(paths)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_otsu, n_pool = otsu_on_pool()
    print(f"Otsu MNDWI threshold on the unlabelled pool ({n_pool} images): {t_otsu:.4f}", flush=True)
    stems = (PERU / "splits" / "dev.txt").read_text().split()
    runs = {d: np.zeros((2, 3), np.int64) for r in REGIMES for d in sorted((LOGITS / r).glob("*/seed*"))}
    idx = {t: np.zeros((2, 3), np.int64) for t in [0.0, round(t_otsu, 4), *T_SWEEP]}
    for st in stems:
        with rasterio.open(PERU / "s2" / f"{st}.tif") as src:
            v = mndwi(src.read())
        with rasterio.open(PERU / "label" / f"{st}.tif") as src:
            y = src.read(1)
        for t in idx:
            idx[t] += counts(v > t, y)
        for d in runs:
            runs[d] += counts(np.load(d / f"{st}.npy") > 0, y)
    for d in runs:
        n = len(list(d.glob("*.npy")))
        assert n == len(stems), f"{d}: {n} logit files, expected {len(stems)}"

    rows = [dict(regime=d.parts[-3], model=d.parts[-2], seed=d.parts[-1], **scores(c)) for d, c in runs.items()]
    t_dev = max(T_SWEEP, key=lambda t: scores(idx[t])["iou"])
    for name, t in (("MNDWI > 0", 0.0), (f"MNDWI > {t_otsu:.4f} (Otsu on pool)", round(t_otsu, 4)),
                    (f"MNDWI > {t_dev:+.2f} (tuned on dev: oracle)", t_dev)):
        rows.append(dict(regime="index", model=name, seed="-", **scores(idx[t])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dev_table_runs.csv", index=False)

    cols = ["iou", "iou_tol", "precision", "recall"]
    g = df[df.regime != "index"].groupby(["regime", "model"])[cols].agg(["mean", "std"])
    n = df[df.regime != "index"].groupby(["regime", "model"]).size()
    lines = ["| Row | Backbone | seeds | water IoU | tolerant IoU (excl. ≤ 1 px) | precision | recall | gap closed |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df[df.regime == "index"].iterrows():
        lines.append(f"| {r.model} | — | — | {r.iou:.4f} | {r.iou_tol:.4f} | {r.precision:.4f} | {r.recall:.4f} | — |")
    for reg, rn in REGIMES.items():
        for m, mn in MODELS.items():
            if (reg, m) not in g.index:
                continue
            cells = [f"{g.loc[(reg, m), (c, 'mean')]:.4f}" + (f" ± {g.loc[(reg, m), (c, 'std')]:.4f}" if n[(reg, m)] > 1 else "")
                     for c in cols]
            gap = "—"
            if reg not in NOT_METHODS and ("s1f11_src", m) in g.index and ("peru", m) in g.index:
                src, up = g.loc[("s1f11_src", m), ("iou", "mean")], g.loc[("peru", m), ("iou", "mean")]
                gap = f"{(g.loc[(reg, m), ('iou', 'mean')] - src) / (up - src):.0%}"
            lines.append(f"| {rn} | {mn} | {n[(reg, m)]} | " + " | ".join(cells) + f" | {gap} |")
    text = (f"Pool: {n_pool} unlabelled `peru-train` images; Otsu MNDWI threshold {t_otsu:.4f}. "
            f"Dev-tuned threshold: {t_dev:+.2f}.\n\n" + "\n".join(lines) + "\n")
    (OUT / "dev_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
