"""RF5 check: the system's metrics on the peru-test chips match scripts/score_frozen.py to 3 decimals.

Compares floodseg.system.reference.frozen_chip_metrics (the system's own counting code) with the
thesis evaluator's output outputs/phaseD/test/peru-test_groups.csv (subset `full`).
→ outputs/system_eval/rf/rf5_chips.json

    uv run python scripts/system_rf5_check.py
"""

import json

import pandas as pd

from floodseg.system.reference import frozen_chip_metrics
from floodseg.target_pool import REPO

GROUP = "peru/unet"


def main() -> None:
    sysm = frozen_chip_metrics(GROUP, "test")
    g = pd.read_csv(REPO / "outputs" / "phaseD" / "test" / "peru-test_groups.csv")
    ref = g[(g.subset == "full") & (g.group == GROUP)].iloc[0]
    rows = {k: {"system": round(sysm[f"{k}_mean"], 6), "score_frozen": round(float(ref[f"{k}_mean"]), 6),
                "match_3dp": f"{sysm[f'{k}_mean']:.3f}" == f"{ref[f'{k}_mean']:.3f}"}
            for k in ("iou", "precision", "recall")}
    out = {"group": GROUP, "chips": sysm["chips"], "metrics": rows, "pass": all(r["match_3dp"] for r in rows.values())}
    dest = REPO / "outputs" / "system_eval" / "rf"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "rf5_chips.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
