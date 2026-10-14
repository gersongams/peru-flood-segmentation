"""Edge-tolerant water IoU on peru-dev (input to the phase-D decision, task 6.5).

Recomputes water IoU from the float32 logits (outputs/logits_dev/<regime>/<model>/seed<s>)
after excluding labelled pixels within k px of a label water edge, k = 0 (plain IoU), 1, 2,
3. Edge pixels and distances are defined as in scripts/analyze_dev.py ("edge (≤1 px)" =
distance < 1.5): k px excludes distance < k + 0.5. Ignore (−1) counts as not water when the
edge is found, as in the failure analysis. If the U-Net's lead over the foundation models
vanishes at small k, their edge loss is within label imprecision; if it stays, it is a
real loss of spatial detail.

Writes outputs/analysis_dev/edge_tolerance.{csv,md}.

    uv run python scripts/edge_tolerance.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion, distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "outputs" / "logits_dev"
LABEL = REPO / "data" / "peru" / "unosat2023" / "label"
OUT = REPO / "outputs" / "analysis_dev"
KS = (0, 1, 2, 3)
MODELS = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}
REGIMES = {"peru": "Peru-only", "s1f11_ft": "Sen1Floods11 → Peru"}


def main() -> None:
    runs = [d for r in REGIMES for d in sorted((PRED / r).glob("*/seed*"))]
    stems = sorted(p.stem for p in runs[0].glob("*.npy"))
    acc = {(d, k): np.zeros(3) for d in runs for k in KS}  # tp, fp, fn
    for st in stems:
        with rasterio.open(LABEL / f"{st}.tif") as src:
            y = src.read(1)
        w = y == 1
        edge = (w ^ binary_erosion(w)) | (~w ^ binary_erosion(~w))
        dist = distance_transform_edt(~edge)
        ok = y != -1
        for d in runs:
            pred = np.load(d / f"{st}.npy") > 0
            for k in KS:
                m = ok & (dist >= k + 0.5) if k else ok
                p, t = pred[m], w[m]
                acc[(d, k)] += ((p & t).sum(), (p & ~t).sum(), (~p & t).sum())
    rows = [dict(regime=d.parts[-3], model=d.parts[-2], seed=d.parts[-1], k=k, iou=tp / (tp + fp + fn))
            for (d, k), (tp, fp, fn) in acc.items()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "edge_tolerance.csv", index=False)
    g = df.groupby(["regime", "model", "k"]).iou.agg(["mean", "std"])
    lines = ["| Training | Model | " + " | ".join("plain IoU" if k == 0 else f"excl. ≤ {k} px" for k in KS) + " |",
             "|---|---|" + "---|" * len(KS)]
    for r, rn in REGIMES.items():
        for m, mn in MODELS.items():
            lines.append(f"| {rn} | {mn} | " + " | ".join(f"{g.loc[(r, m, k), 'mean']:.4f} ± {g.loc[(r, m, k), 'std']:.4f}" for k in KS) + " |")
        for m in ("terramind_v1_base", "prithvi_eo_v2_300"):
            gap = [g.loc[(r, "unet", k), "mean"] - g.loc[(r, m, k), "mean"] for k in KS]
            lines.append(f"| {rn} | U-Net − {MODELS[m]} | " + " | ".join(f"{100 * x:+.2f} pts" for x in gap) + " |")
    (OUT / "edge_tolerance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
