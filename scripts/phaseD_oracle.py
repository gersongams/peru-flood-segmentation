"""Oracle selection for self-training on `peru-dev` (phase-D task 4.7; spec: reported only as a
secondary "oracle selection" row, next to the dev-tuned MNDWI threshold).

For each backbone, the candidates are all seed-0 `st` runs (the default α = 0.99, q 20 → 50 %
and the grid α ∈ {0.99, 0.999} × final q ∈ {20, 50, 80} %) at every saved checkpoint
(iterations 1,000–3,000 from the packed predictions in <run>/dev_pred/, 4,000 from the final
logits in outputs/logits_dev/<tag>/). The (configuration, iteration) with the best plain dev
water IoU is the oracle choice. Uses dev labels by construction.

Writes outputs/phaseD/oracle_selection.{csv,md}.

    uv run python scripts/phaseD_oracle.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from floodseg.metrics import counts, scores

REPO = Path(__file__).resolve().parents[1]
PERU = REPO / "data" / "peru" / "unosat2023"
ST = REPO / "outputs" / "phaseD" / "self_train"
OUT = REPO / "outputs" / "phaseD"


def config(tag: str) -> tuple[float, float]:
    m = re.fullmatch(r"st(?:_a([\d.]+)_q([\d.]+))?", tag)
    return (float(m.group(1)), float(m.group(2))) if m.group(1) else (0.99, 0.5)


def main() -> None:
    stems = (PERU / "splits" / "dev.txt").read_text().split()
    cands = {}
    for run in sorted(ST.glob("st*/*/seed0")):
        tag, model = run.parts[-3], run.parts[-2]
        if not re.fullmatch(r"st(_a[\d.]+_q[\d.]+)?", tag):
            continue
        for f in sorted((run / "dev_pred").glob("iter*.npz")):
            cands[(model, tag, int(f.stem[4:]))] = ("packed", np.load(f))
        final = REPO / "outputs" / "logits_dev" / tag / model / "seed0"
        if final.exists():
            cands[(model, tag, 4000)] = ("logits", final)
    acc = {k: np.zeros((2, 3), np.int64) for k in cands}
    for st in stems:
        with rasterio.open(PERU / "label" / f"{st}.tif") as src:
            y = src.read(1)
        for k, (kind, v) in cands.items():
            pred = (np.unpackbits(v[st])[: y.size].reshape(y.shape).astype(bool) if kind == "packed"
                    else np.load(v / f"{st}.npy") > 0)
            acc[k] += counts(pred, y)
    rows = [dict(model=m, tag=t, alpha=config(t)[0], q_end=config(t)[1], iter=i, **scores(c)) for (m, t, i), c in acc.items()]
    df = pd.DataFrame(rows).sort_values(["model", "alpha", "q_end", "iter"])
    df.to_csv(OUT / "oracle_selection.csv", index=False)
    lines = ["| Backbone | α | final q | iter 1000 | 2000 | 3000 | 4000 |", "|---|---|---|---|---|---|---|"]
    for (m, a, q), g in df.groupby(["model", "alpha", "q_end"]):
        it = dict(zip(g["iter"], g.iou))
        lines.append(f"| {m} | {a} | {q:.0%} | " + " | ".join(f"{it[i]:.4f}" if i in it else "—" for i in (1000, 2000, 3000, 4000)) + " |")
    best = df.loc[df.groupby("model").iou.idxmax()]
    lines.append("\nOracle choice (best dev IoU over configurations and checkpoints):")
    for _, r in best.iterrows():
        lines.append(f"- {r.model}: α = {r.alpha}, final q = {r.q_end:.0%}, iteration {r['iter']}: IoU {r.iou:.4f} "
                     f"(tolerant {r.iou_tol:.4f}, precision {r.precision:.4f}, recall {r.recall:.4f})")
    (OUT / "oracle_selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
