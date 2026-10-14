"""Index-threshold baselines: water IoU of MNDWI > t and NDWI > t (no learning).

MNDWI = (B03 − B11)/(B03 + B11) (as in Portalés-Julià et al. 2025, RSE 328:114882, and our
label QA); NDWI = (B03 − B08)/(B03 + B08). Pooled water IoU over labelled pixels, plus
precision and recall, on `peru-dev` and Sen1Floods11 `valid`, for thresholds 0 (the
standard choice, fixed a priori) and a sweep (the sweep is optimistic: it picks t on the
evaluated split itself, so it is an upper bound for any single global threshold). A label-free
threshold is also evaluated on `peru-dev`: Otsu on the MNDWI of the unlabelled `peru-train`
images (every 50th valid pixel; labels not read).

Writes outputs/analysis_dev/index_baseline.{csv,md}.

    uv run python scripts/index_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

REPO = Path(__file__).resolve().parents[1]
S1F = REPO / "data" / "sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled"
PERU = REPO / "data" / "peru" / "unosat2023"
OUT = REPO / "outputs" / "analysis_dev"
TS = list(np.round(np.arange(-0.3, 0.61, 0.05), 2))


def pairs(name: str) -> list[tuple[Path, Path]]:
    if name == "peru-dev":
        stems = [l.strip() for l in (PERU / "splits" / "dev.txt").read_text().split() if l.strip()]
        return [(PERU / "s2" / f"{s}.tif", PERU / "label" / f"{s}.tif") for s in stems]
    ids = [l.strip().split(",")[0].removesuffix("_S1Hand.tif")
           for l in (REPO / "data" / "sen1floods11" / "v1.1" / "splits" / "flood_handlabeled" / "flood_valid_data.txt").read_text().splitlines() if l.strip()]
    return [(S1F / "S2Hand" / f"{i}_S2Hand.tif", S1F / "LabelHand" / f"{i}_LabelHand.tif") for i in ids]


def otsu_peru_train() -> float:
    from skimage.filters import threshold_otsu

    v = []
    for s in (PERU / "splits" / "train.txt").read_text().split():
        with rasterio.open(PERU / "s2" / f"{s}.tif") as src:
            x = src.read().astype(np.float64)
        ok = (x != 0).any(0)
        v.append(((x[2] - x[11]) / (x[2] + x[11] + 1e-6))[ok][::50])
    return float(threshold_otsu(np.concatenate(v)))


def main() -> None:
    t_otsu = otsu_peru_train()
    print(f"Otsu MNDWI threshold on unlabelled peru-train: {t_otsu:.4f}")
    TS.append(round(t_otsu, 4))
    rows = []
    for name in ("peru-dev", "Sen1Floods11 valid"):
        acc = {(ix, t): np.zeros(3) for ix in ("MNDWI", "NDWI") for t in TS}
        for img, lab in pairs(name):
            with rasterio.open(img) as src:
                x = src.read().astype(np.float64)
            with rasterio.open(lab) as src:
                y = src.read(1)
            ok = (y >= 0) & (x != 0).any(0)
            w = (y == 1)[ok]
            for ix, (a, b) in (("MNDWI", (2, 11)), ("NDWI", (2, 7))):
                v = ((x[a] - x[b]) / (x[a] + x[b] + 1e-6))[ok]
                for t in TS:
                    p = v > t
                    acc[(ix, t)] += ((p & w).sum(), (p & ~w).sum(), (~p & w).sum())
        for (ix, t), (tp, fp, fn) in acc.items():
            rows.append(dict(split=name, index=ix, t=t, iou=tp / (tp + fp + fn), precision=tp / (tp + fp), recall=tp / (tp + fn)))
        print(f"{name}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "index_baseline.csv", index=False)
    lines = ["| split | index | IoU at t = 0 (precision / recall) | best t on the same split | IoU at best t |", "|---|---|---|---|---|"]
    for (sp, ix), g in df.groupby(["split", "index"], sort=False):
        z = g[g.t == 0].iloc[0]
        b = g.loc[g.iou.idxmax()]
        lines.append(f"| {sp} | {ix} | {z.iou:.4f} ({z.precision:.3f} / {z.recall:.3f}) | {b.t:+.2f} | {b.iou:.4f} |")
    o = df[(df.split == "peru-dev") & (df["index"] == "MNDWI") & (df.t == round(t_otsu, 4))].iloc[0]
    lines.append(f"\nLabel-free: MNDWI > {t_otsu:.4f} (Otsu on unlabelled peru-train) on peru-dev: IoU {o.iou:.4f} "
                 f"(precision {o.precision:.3f}, recall {o.recall:.3f}).")
    (OUT / "index_baseline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
