"""Spatial overlap of the 2017 test chips with the 2023 splits (post hoc audit, 2026-10-17).

The 2017 main subset (101 chips) excludes chips that touch a 2023 `train` location (same rule as
scripts/build_peru_emsr2017.py: same 5,120 m cell or one of its 8 neighbours). Configuration
selection used labelled 2023 `dev`, so this audit counts, for those 101 chips, how many touch a
2023 dev, test or buffer location, and how many share the exact cell.

    uv run python scripts/overlap_2017_2023.py → outputs/phaseD/analysis/overlap_2017_2023.csv
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CHIP = 5120


def cell(chip: str) -> tuple[int, int]:
    e, n = chip.split("_")
    return int(e[1:]) // CHIP, int(n[1:]) // CHIP


def main() -> None:
    m = pd.read_csv(REPO / "data" / "peru" / "emsr2017" / "manifest.csv")
    sp = pd.read_csv(REPO / "data" / "peru" / "unosat2023" / "splits" / "split.csv")
    cells = {s: {cell(c) for c in g.chip} for s, g in sp.groupby("split")}
    rows = []
    for subset, mm in (("no_2023_train_overlap", m[~m.touches_2023_train]), ("all", m)):
        for s, cs in sorted(cells.items()):
            same = sum(cell(c) in cs for c in mm.chip)
            touch = sum(any((x + dx, y + dy) in cs for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                        for x, y in map(cell, mm.chip))
            rows.append(dict(subset=subset, chips=len(mm), split_2023=s, same_cell=same, same_or_adjacent=touch))
    out = pd.DataFrame(rows)
    p = REPO / "outputs" / "phaseD" / "analysis" / "overlap_2017_2023.csv"
    out.to_csv(p, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
