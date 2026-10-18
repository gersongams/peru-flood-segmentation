"""Direct bootstrap CI for HE1b: difference of IoU drops, Sen1Floods11 `test` → Peru (post hoc, 2026-10-17).

D = (IoU_F,Peru − IoU_F,S1F) − (IoU_U,Peru − IoU_U,S1F), F = foundation model, U = U-Net, all
source-only (Sen1Floods11-trained) models. D > 0: the foundation model drops less than the U-Net.
Requested by a chapter review; the 2026-10-17 draft of §3.3 used two within-domain paired CIs
instead. No model is re-run; it reads the per-chip counts of scripts/score_frozen.py.

One replicate resamples the 90 Sen1Floods11 `test` chips and, independently, the Peru chips
(both with replacement); within each domain the same chips are used for all models (paired). Per
model and domain, the pooled water IoU of each seed on the resampled chips, averaged over the 3
seeds (as in scripts/bootstrap_frozen.py). 2,000 replicates, seed 0, percentile intervals. Seed
variation and spatial dependence between chips are not modelled.

    uv run python scripts/bootstrap_drop_diff.py → outputs/phaseD/test/bootstrap_drop_diff.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
T = REPO / "outputs" / "phaseD" / "test"
N_BOOT = 2000
MODELS = ["unet", "terramind_v1_base", "prithvi_eo_v2_300"]
PERU = {"peru-test/full": ("peru-test", None),
        "peru-2017-test/no_2023_train_overlap": ("peru-2017-test", "no_overlap")}


def counts(split: str, keep: set | None) -> tuple[dict, int]:
    cc = pd.read_csv(T / f"{split}_chip_counts.csv")
    cc = cc[cc.group.isin([f"s1f11_src/{m}" for m in MODELS])]
    if keep is not None:
        cc = cc[cc.chip.isin(keep)]
    chips = sorted(cc.chip.unique())
    idx = {c: i for i, c in enumerate(chips)}
    arr = {}
    for (g, s), p in cc.groupby(["group", "seed"]):
        a = np.zeros((len(chips), 3))
        a[[idx[c] for c in p.chip]] = p[["tp", "fp", "fn"]].to_numpy()
        arr.setdefault(g.split("/")[1], []).append(a)
    return {m: np.stack(v) for m, v in arr.items()}, len(chips)


def boot_iou(arr: dict, n: int, rng) -> tuple[dict, dict]:
    w = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)
    b, pt = {}, {}
    for m, a in arr.items():
        tot = np.einsum("bc,sck->bsk", w, a)
        b[m] = (tot[..., 0] / tot.sum(-1)).mean(1)
        full = a.sum(1)
        pt[m] = (full[:, 0] / full.sum(-1)).mean()
    return b, pt


def main() -> None:
    m17 = pd.read_csv(REPO / "data" / "peru" / "emsr2017" / "manifest.csv")
    no_overlap = set(m17.chip[~m17.touches_2023_train])
    rows = []
    for name, (split, sub) in PERU.items():
        rng = np.random.default_rng(0)
        s_arr, ns = counts("test", None)
        p_arr, npp = counts(split, no_overlap if sub else None)
        bs, ps = boot_iou(s_arr, ns, rng)
        bp, pp = boot_iou(p_arr, npp, rng)
        for f in MODELS[1:]:
            d = (bp[f] - bs[f]) - (bp["unet"] - bs["unet"])
            point = (pp[f] - ps[f]) - (pp["unet"] - ps["unet"])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(peru=name, chips_s1f=ns, chips_peru=npp, model=f, drop_model=ps[f] - pp[f],
                             drop_unet=ps["unet"] - pp["unet"], D=point, ci_low=lo, ci_high=hi))
    out = pd.DataFrame(rows)
    out.to_csv(T / "bootstrap_drop_diff.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
