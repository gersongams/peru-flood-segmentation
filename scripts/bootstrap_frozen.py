"""Chip-bootstrap 95 % CIs for the frozen test results (plan-flood-research 7.4; phase-D 6.3).

Reads the per-chip counts of scripts/score_frozen.py. One bootstrap replicate resamples the
chips of the subset with replacement (the same chips for every group, so differences are
paired); per group, the pooled water IoU of each seed is computed on the resampled chips and
averaged over the 3 seeds (the seed-mean the tables report). 2,000 replicates, seed 0;
percentile intervals. Seed variation is not resampled: the CI covers chip sampling only.

Sen1Floods11 `test` and the section-7 diffs were added 2026-10-16 after the section-7 pass (the
replicates of the earlier rows are unchanged: chip weights are drawn per subset before any group).
Paired differences (A − B) for the comparisons the write-up makes; "P(A ≤ B)" is the share of
replicates in which A does not beat B.

    uv run python scripts/bootstrap_frozen.py
→ outputs/phaseD/test/bootstrap_{groups,diffs}.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
T = REPO / "outputs" / "phaseD" / "test"
N_BOOT = 2000
SUBSETS = {"peru-test": ["full", "excl_doubtful"], "peru-2017-test": ["no_2023_train_overlap", "all"], "test": ["full"]}
DIFFS = [("trhm/unet", "peru/unet"), ("trhm/unet", "s1f11_src/prithvi_eo_v2_300"), ("trhm/unet", "adabn/unet"),
         ("trhm/unet", "s1f11_src/unet"), ("trhm/terramind_v1_base", "trfda/terramind_v1_base"),
         ("trfda/terramind_v1_base", "s1f11_src/terramind_v1_base"), ("s1f11_src/prithvi_eo_v2_300", "peru/prithvi_eo_v2_300")]
# section-7 pass (added 2026-10-16): Sen1Floods11 pretraining before Peru fine-tuning; in-domain ranking on `test`
DIFFS += [(f"s1f11_ft/{m}", f"peru/{m}") for m in ("unet", "terramind_v1_base", "prithvi_eo_v2_300")]
DIFFS += [("s1f11_src/unet", "s1f11_src/prithvi_eo_v2_300"), ("s1f11_src/unet", "s1f11_src/terramind_v1_base")]


def subset_chips(split: str, sub: str, chips: list[str]) -> list[str]:
    if split == "test":
        return chips
    if split == "peru-2017-test":
        m = pd.read_csv(REPO / "data" / "peru" / "emsr2017" / "manifest.csv")
        return chips if sub == "all" else sorted(set(m.chip[~m.touches_2023_train]))
    if sub == "full":
        return chips
    qa = pd.read_csv(REPO / "outputs" / "label_qa" / "chips.csv").set_index("chip")
    return [c for c in chips if qa.loc[c, "flag"] != "doubtful"]


def main() -> None:
    rng = np.random.default_rng(0)
    g_rows, d_rows = [], []
    for split, subs in SUBSETS.items():
        cc = pd.read_csv(T / f"{split}_chip_counts.csv")
        chips_all = sorted(cc.chip.unique())
        for sub in subs:
            chips = subset_chips(split, sub, chips_all)
            idx = {c: i for i, c in enumerate(chips)}
            part = cc[cc.chip.isin(idx)]
            arr = {}  # group -> (seeds, chips, 3) counts tp, fp, fn
            for (g, s), p in part.groupby(["group", "seed"]):
                a = np.zeros((len(chips), 3))
                a[[idx[c] for c in p.chip]] = p[["tp", "fp", "fn"]].to_numpy()
                arr.setdefault(g, []).append(a)
            arr = {g: np.stack(v) for g, v in arr.items()}
            w = rng.multinomial(len(chips), np.full(len(chips), 1 / len(chips)), size=N_BOOT)  # chip weights
            boot = {}
            for g, a in arr.items():
                tot = np.einsum("bc,sck->bsk", w, a)
                boot[g] = (tot[..., 0] / tot.sum(-1)).mean(1)
                full = a.sum(1)
                point = (full[:, 0] / full.sum(-1)).mean()
                lo, hi = np.percentile(boot[g], [2.5, 97.5])
                g_rows.append(dict(split=split, subset=sub, chips=len(chips), group=g, iou=point, ci_low=lo, ci_high=hi))
            for a_, b_ in DIFFS:
                if a_ in boot and b_ in boot:
                    d = boot[a_] - boot[b_]
                    pa = next(r["iou"] for r in g_rows if r["split"] == split and r["subset"] == sub and r["group"] == a_)
                    pb = next(r["iou"] for r in g_rows if r["split"] == split and r["subset"] == sub and r["group"] == b_)
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    d_rows.append(dict(split=split, subset=sub, a=a_, b=b_, diff=pa - pb, ci_low=lo, ci_high=hi,
                                       p_a_not_better=float((d <= 0).mean())))
    g, d = pd.DataFrame(g_rows), pd.DataFrame(d_rows)
    g.to_csv(T / "bootstrap_groups.csv", index=False)
    d.to_csv(T / "bootstrap_diffs.csv", index=False)
    pd.set_option("display.width", 200)
    print(g.round(4).to_string(index=False))
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
