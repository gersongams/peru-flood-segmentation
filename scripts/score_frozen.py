"""Score the logits of scripts/evaluate_frozen.py (phase-D task 6.2; spec `adaptation-evaluation`).

Per run and chip: plain and boundary-tolerant [tp, fp, fn] (floodseg.metrics.counts, water if
z > 0), kept for bootstrap CIs. Pooled per subset, then mean ± std over the 3 seeds per group,
and gap closed = (IoU − IoU_source) / (IoU_Peru − IoU_source) against the same backbone's
`s1f11_src` and `peru` groups on the same subset; undefined ("n/a") when IoU_Peru ≤ IoU_source
(no gap to close; happens on peru-2017-test, fixed 2026-10-16 after the first scoring printed
ratios with a negative denominator).

Subsets:
- peru-dev / peru-test: `full`; `excl_doubtful` (chips whose label-QA flag over all blobs is
  "doubtful" removed) and `excl_doubtful_nocloud` (flag without near-cloud blobs), from
  outputs/label_qa/chips.csv (task 4.4; frozen sha256 in configs/frozen.yaml);
- peru-2017-test: `no_2023_train_overlap` (the 101 chips that touch no 2023 `peru-train`
  location: the main 2017 number) and `all` (204).

`--compare` (dev only) also checks the logits against the dev-table logits in
outputs/logits_dev (max |Δz|, pixels whose sign differs) and the per-run IoU against
outputs/phaseD/dev_table_runs.csv.

    uv run python scripts/score_frozen.py peru-dev --logits_root outputs/frozen_check/logits_dev --compare
    uv run python scripts/score_frozen.py peru-test
    uv run python scripts/score_frozen.py peru-2017-test
→ outputs/phaseD/test/<split>_{chip_counts,runs,groups}.csv and <split>.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

from floodseg.metrics import counts, scores

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "configs" / "frozen.yaml"
OUT = REPO / "outputs" / "phaseD" / "test"
ROOTS = {"peru-dev": REPO / "outputs" / "frozen_check" / "logits_dev",
         "peru-test": REPO / "outputs" / "logits_test", "peru-2017-test": REPO / "outputs" / "logits_2017"}
MODELS = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}
ROW_ORDER = ["primary", "secondary", "reference"]


def chips_and_subsets(split: str) -> tuple[list[str], dict[str, Path], dict[str, set]]:
    if split == "peru-2017-test":
        m = pd.read_csv(REPO / "data" / "peru" / "emsr2017" / "manifest.csv")
        stems = list(m.chip)
        labels = {s: REPO / "data" / "peru" / "emsr2017" / "label" / f"{s}.tif" for s in stems}
        return stems, labels, {"no_2023_train_overlap": set(m.chip[~m.touches_2023_train]), "all": set(stems)}
    u = REPO / "data" / "peru" / "unosat2023"
    stems = (u / "splits" / ("dev.txt" if split == "peru-dev" else "test.txt")).read_text().split()
    qa = pd.read_csv(REPO / "outputs" / "label_qa" / "chips.csv").set_index("chip").loc[stems]
    return stems, {s: u / "label" / f"{s}.tif" for s in stems}, {
        "full": set(stems),
        "excl_doubtful": set(qa.index[qa.flag != "doubtful"]),
        "excl_doubtful_nocloud": set(qa.index[qa.flag_no_cloud != "doubtful"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=sorted(ROOTS))
    ap.add_argument("--logits_root", type=Path)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    root = a.logits_root or ROOTS[a.split]
    groups = [g for g in yaml.safe_load(FROZEN.read_text())["groups"] if (root / g["regime"] / g["model"]).exists()]
    stems, labels, subsets = chips_and_subsets(a.split)

    rows, cmp = [], []
    for s in stems:
        with rasterio.open(labels[s]) as src:
            y = src.read(1)
        for g in groups:
            for seed in g["runs"]:
                d = root / g["regime"] / g["model"] / seed
                z = np.load(d / f"{s}.npy")
                c = counts(z > 0, y)
                rows.append(dict(group=g["id"], seed=seed, chip=s, **{f"{k}{t}": c[i, j] for i, t in enumerate(("", "_tol"))
                                                                       for j, k in enumerate(("tp", "fp", "fn"))}))
                if a.compare:
                    z0 = np.load(REPO / "outputs" / "logits_dev" / g["regime"] / g["model"] / seed / f"{s}.npy")
                    cmp.append(dict(group=g["id"], seed=seed, max_abs=float(np.abs(z - z0).max()),
                                    sign_diff=int(((z > 0) != (z0 > 0)).sum())))
    cc = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.split + ("_check" if a.logits_root else "")
    cc.to_csv(OUT / f"{tag}_chip_counts.csv", index=False)

    runs = []
    for sub, keep in subsets.items():
        part = cc[cc.chip.isin(keep)]
        for (gid, seed), p in part.groupby(["group", "seed"]):
            c = np.array([[p.tp.sum(), p.fp.sum(), p.fn.sum()], [p.tp_tol.sum(), p.fp_tol.sum(), p.fn_tol.sum()]])
            runs.append(dict(subset=sub, chips=len(keep), group=gid, seed=seed, **scores(c)))
    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / f"{tag}_runs.csv", index=False)

    cols = ["iou", "iou_tol", "precision", "recall"]
    gs = runs.groupby(["subset", "group"])[cols].agg(["mean", "std"])
    meta = {g["id"]: g for g in groups}
    out, lines = [], [f"# Frozen evaluation on `{a.split}`\n"]
    for sub, keep in subsets.items():
        lines += [f"\n## {sub} ({len(keep)} chips)\n", "| Role | Backbone | Row | water IoU | tolerant IoU | precision | recall | gap closed |",
                  "|---|---|---|---|---|---|---|---|"]
        for role in ROW_ORDER:
            for gid, g in meta.items():
                if g["role"] != role or (sub, gid) not in gs.index:
                    continue
                r = gs.loc[(sub, gid)]
                src, up = (sub, f"s1f11_src/{g['model']}"), (sub, f"peru/{g['model']}")
                gap = np.nan
                undefined = False
                if g["regime"] not in ("s1f11_src", "peru", "s1f11_ft") and src in gs.index and up in gs.index:
                    den = gs.loc[up, ("iou", "mean")] - gs.loc[src, ("iou", "mean")]
                    undefined = den <= 0  # Peru-trained not above source-only: no gap to close
                    gap = np.nan if undefined else (r[("iou", "mean")] - gs.loc[src, ("iou", "mean")]) / den
                out.append(dict(subset=sub, group=gid, role=role, **{f"{c}_{s}": r[(c, s)] for c in cols for s in ("mean", "std")}, gap_closed=gap))
                cells = " | ".join(f"{r[(c, 'mean')]:.4f} ± {r[(c, 'std')]:.4f}" for c in cols)
                lines.append(f"| {role} | {MODELS[g['model']]} | {g['regime']} | {cells} | {'n/a (upper ≤ source)' if undefined else '—' if np.isnan(gap) else f'{gap:.0%}'} |")
    pd.DataFrame(out).to_csv(OUT / f"{tag}_groups.csv", index=False)
    text = "\n".join(lines) + "\n"
    if a.compare:
        cm = pd.DataFrame(cmp).groupby(["group", "seed"]).agg(max_abs=("max_abs", "max"), sign_diff=("sign_diff", "sum"))
        ref = pd.read_csv(OUT.parent / "dev_table_runs.csv")
        ref["group"] = ref.regime + "/" + ref.model
        full = runs[runs.subset == "full"].merge(ref[["group", "seed", "iou"]], on=["group", "seed"], suffixes=("", "_dev_table"))
        full["iou_diff"] = full.iou - full.iou_dev_table
        cm = cm.join(full.set_index(["group", "seed"])[["iou", "iou_dev_table", "iou_diff"]])
        cm.to_csv(OUT / f"{tag}_compare.csv")
        text += "\n## Reproduction of the dev-table logits\n\n```\n" + cm.to_string(float_format="%.3g") + "\n```\n"
    (OUT / f"{tag}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
