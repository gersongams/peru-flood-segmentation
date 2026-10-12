"""Peru dataset guard: training may only use `peru-train` chips.

`peru-dev`, `peru-test` (UNOSAT 2023) and `peru-2017-test` (EMSR199) are held
out. `check_train_split` raises if a training split list names any of them.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
UNOSAT_SPLIT = REPO / "data" / "peru" / "unosat2023" / "splits" / "split.csv"
EMSR2017_LABELS = REPO / "data" / "peru" / "emsr2017" / "label"


def held_out_stems(unosat_split: Path = UNOSAT_SPLIT, emsr_labels: Path = EMSR2017_LABELS) -> set[str]:
    stems: set[str] = set()
    if unosat_split.exists():
        s = pd.read_csv(unosat_split)
        stems |= {f"{d}_{c}" for c, d, sp in zip(s.chip, s.date, s.split) if sp != "train"}
    if emsr_labels.exists():
        stems |= {p.stem for p in emsr_labels.glob("*.tif")}
    return stems


def check_train_split(train_split: str | Path | None, **paths) -> None:
    """Raise if the training split list contains a held-out Peru chip."""
    if train_split is None:
        return
    names = {line.strip() for line in Path(train_split).read_text().splitlines() if line.strip()}
    leaked = names & held_out_stems(**paths)
    if leaked:
        raise RuntimeError(
            f"Training split {train_split} contains {len(leaked)} held-out Peru chips "
            f"(e.g. {sorted(leaked)[:3]}); only peru-train may be trained on."
        )
