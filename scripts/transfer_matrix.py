"""Training regime × evaluation set water IoU (task 6.2, part 2).

Rows: Sen1Floods11-only (seed 0), Peru-only (3 seeds), Sen1Floods11 → Peru (3 seeds).
Columns: Sen1Floods11 `valid`, `peru-dev`. Every number is parsed from a
`terratorch validate` log (`val/IoU_Flood`):
  Sen1Floods11-only on valid / dev   outputs/logits_valid/log_s1f11_<m>_seed0.txt,
                                     outputs/logits_dev/log_s1f11_<m>_seed0.txt
  Peru-only, fine-tuned on dev       outputs/logits_dev/log_{peru,s1f11_ft}_<m>_seed<s>.txt
  Peru-only, fine-tuned on valid     outputs/eval_valid/{peru,s1f11_ft}_<m>_seed<s>.log
In-domain cells are optimistic: those checkpoints were selected on that split.

Writes outputs/transfer/matrix.{csv,md}.

    uv run python scripts/transfer_matrix.py
"""

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
O = REPO / "outputs"
MODELS = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}
REGIMES = {"s1f11": "Sen1Floods11-only", "peru": "Peru-only", "s1f11_ft": "Sen1Floods11 → Peru"}


def iou(log: Path) -> float:
    m = re.search(r"val/IoU_Flood\s*│\s*([0-9.]+)", log.read_text())
    if not m:
        raise ValueError(f"no val/IoU_Flood in {log}")
    return float(m.group(1))


def main() -> None:
    rows = []
    for m in MODELS:
        rows.append(dict(regime="s1f11", model=m, seed=0, eval="valid", iou=iou(O / "logits_valid" / f"log_s1f11_{m}_seed0.txt")))
        rows.append(dict(regime="s1f11", model=m, seed=0, eval="peru-dev", iou=iou(O / "logits_dev" / f"log_s1f11_{m}_seed0.txt")))
        for reg in ("peru", "s1f11_ft"):
            for s in range(3):
                rows.append(dict(regime=reg, model=m, seed=s, eval="peru-dev", iou=iou(O / "logits_dev" / f"log_{reg}_{m}_seed{s}.txt")))
                rows.append(dict(regime=reg, model=m, seed=s, eval="valid", iou=iou(O / "eval_valid" / f"{reg}_{m}_seed{s}.log")))
    df = pd.DataFrame(rows)
    out = O / "transfer"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "matrix.csv", index=False)

    def cell(r, m, e):
        v = df[(df.regime == r) & (df.model == m) & (df["eval"] == e)].iou
        return f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f} ± {v.std():.4f}"

    lines = ["| Training | Model | Sen1Floods11 `valid` | `peru-dev` |", "|---|---|---|---|"]
    for r, rn in REGIMES.items():
        for m, mn in MODELS.items():
            lines.append(f"| {rn} | {mn} | {cell(r, m, 'valid')} | {cell(r, m, 'peru-dev')} |")
    (out / "matrix.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
