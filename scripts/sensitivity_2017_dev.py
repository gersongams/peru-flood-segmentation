"""2017 test without the chips near 2023 `dev` locations (post hoc sensitivity, 2026-10-17).

Configuration selection used labelled 2023 `dev`; 57 of the 101 main 2017 chips share a 5,120 m
cell with a 2023 dev location and 77 share or touch one (scripts/overlap_2017_2023.py). This
recomputes the key 2017 paired differences on the main subset minus (a) chips in the same cell as
a dev location and (b) chips in or adjacent to one. Same bootstrap as scripts/bootstrap_frozen.py
(chip resampling, paired, seed-mean pooled IoU, 2,000 replicates, seed 0). No model is re-run.

    uv run python scripts/sensitivity_2017_dev.py → outputs/phaseD/test/sensitivity_2017_dev.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
T = REPO / "outputs" / "phaseD" / "test"
CHIP = 5120
DIFFS = [("trhm/unet", "s1f11_src/unet"), ("trfda/terramind_v1_base", "s1f11_src/terramind_v1_base"),
         ("s1f11_src/unet", "s1f11_src/prithvi_eo_v2_300"), ("s1f11_src/unet", "s1f11_src/terramind_v1_base")]


def cell(chip):
    e, n = chip.split("_")[-2:]
    return int(e[1:]) // CHIP, int(n[1:]) // CHIP


def main() -> None:
    m = pd.read_csv(REPO / "data" / "peru" / "emsr2017" / "manifest.csv")
    main_chips = set(m.chip[~m.touches_2023_train])
    sp = pd.read_csv(REPO / "data" / "peru" / "unosat2023" / "splits" / "split.csv")
    dev = {cell(c) for c in sp.chip[sp.split == "dev"]}
    near = lambda c: any((cell(c)[0] + dx, cell(c)[1] + dy) in dev for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    subsets = {"main (101)": main_chips,
               "main minus same cell as dev": {c for c in main_chips if cell(c) not in dev},
               "main minus same or adjacent to dev": {c for c in main_chips if not near(c)}}
    cc = pd.read_csv(T / "peru-2017-test_chip_counts.csv")
    rows = []
    for name, keep in subsets.items():
        rng = np.random.default_rng(0)
        part = cc[cc.chip.isin(keep)]
        chips = sorted(part.chip.unique())
        idx = {c: i for i, c in enumerate(chips)}
        arr = {}
        for (g, s), p in part.groupby(["group", "seed"]):
            a = np.zeros((len(chips), 3))
            a[[idx[c] for c in p.chip]] = p[["tp", "fp", "fn"]].to_numpy()
            arr.setdefault(g, []).append(a)
        arr = {g: np.stack(v) for g, v in arr.items()}
        w = rng.multinomial(len(chips), np.full(len(chips), 1 / len(chips)), size=2000)
        boot, pt = {}, {}
        for g, a in arr.items():
            tot = np.einsum("bc,sck->bsk", w, a)
            boot[g] = (tot[..., 0] / tot.sum(-1)).mean(1)
            full = a.sum(1)
            pt[g] = (full[:, 0] / full.sum(-1)).mean()
        for a_, b_ in DIFFS:
            lo, hi = np.percentile(boot[a_] - boot[b_], [2.5, 97.5])
            rows.append(dict(subset=name, chips=len(chips), a=a_, b=b_, diff=pt[a_] - pt[b_], ci_low=lo, ci_high=hi))
    out = pd.DataFrame(rows)
    out.to_csv(T / "sensitivity_2017_dev.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
