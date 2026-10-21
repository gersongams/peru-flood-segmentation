"""RF5 check: the system's metrics on the Peru test chips match scripts/score_frozen.py to 3 decimals.

Compares floodseg.system.reference.frozen_chip_metrics (the system's own counting code) with the
thesis evaluator's output in outputs/phaseD/test/: peru-test_groups.csv (subset `full`) and
peru-2017-test_groups.csv (subset `no_2023_train_overlap`, the main 2017 number).
→ outputs/system_eval/rf/rf5_chips.json

    uv run python scripts/system_rf5_check.py
"""

import json

import pandas as pd

from floodseg.system.reference import frozen_chip_metrics
from floodseg.target_pool import REPO

GROUP = "peru/unet"
SETS = {"test": ("peru-test_groups.csv", "full"), "2017-test": ("peru-2017-test_groups.csv", "no_2023_train_overlap")}


def check(split: str) -> dict:
    csv_name, subset = SETS[split]
    sysm = frozen_chip_metrics(GROUP, split)
    g = pd.read_csv(REPO / "outputs" / "phaseD" / "test" / csv_name)
    ref = g[(g.subset == subset) & (g.group == GROUP)].iloc[0]
    rows = {k: {"system": round(sysm[f"{k}_mean"], 6), "score_frozen": round(float(ref[f"{k}_mean"]), 6),
                "match_3dp": f"{sysm[f'{k}_mean']:.3f}" == f"{ref[f'{k}_mean']:.3f}"}
            for k in ("iou", "precision", "recall")}
    return {"split": sysm["split"], "subset": subset, "chips": sysm["chips"], "metrics": rows,
            "pass": all(r["match_3dp"] for r in rows.values())}


def main() -> None:
    sets = {split: check(split) for split in SETS}
    out = {"group": GROUP, "sets": sets, "pass": all(v["pass"] for v in sets.values())}
    dest = REPO / "outputs" / "system_eval" / "rf"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "rf5_chips.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
