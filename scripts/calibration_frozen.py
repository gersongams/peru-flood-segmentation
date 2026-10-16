"""Calibration of the frozen phase-D rows on the held-out sets (method fixed before running:
docs/METHODS_NOTES.md 2026-10-16 "Test calibration of the frozen phase-D rows").

Reads only the logits of the one-time pass (outputs/logits_test, outputs/logits_2017), the frozen
Sen1Floods11-`valid` temperatures (sha256 checked against configs/frozen.yaml) and, for the
Peru-labelled reference `temp_dev`, the stored `peru-dev` logits and labels. Nothing is fitted on
test. Same ECE / AUROC / pixel-set definitions as scripts/phaseD_analysis.py on dev.

    uv run python scripts/calibration_frozen.py
→ outputs/phaseD/test/calibration_runs.csv, calibration.csv, calibration.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_frozen import chips_and_subsets  # noqa: E402
from uncertainty_dev import auroc, ece, ens_logit, fit_t, iou, near_water  # noqa: E402

from floodseg.target_pool import sha256

REPO = Path(__file__).resolve().parents[1]
FROZEN = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())
OUT = REPO / "outputs" / "phaseD" / "test"
ROOTS = {"peru-test": REPO / "outputs" / "logits_test", "peru-2017-test": REPO / "outputs" / "logits_2017"}
SUBSETS = {"peru-test": ["full"], "peru-2017-test": ["no_2023_train_overlap", "all"]}
T_SOURCE = {"s1f11_src": "s1f11_src", "adabn": "s1f11_src", "hm": "s1f11_src", "trhm": "trhm", "trfda": "trfda"}
SEEDS = ["seed0", "seed1", "seed2"]


def load(d: Path, stems: list[str]) -> np.ndarray:
    return np.concatenate([np.load(d / f"{s}.npy").ravel() for s in stems]).astype(np.float64)


def dev_reference() -> tuple[list[str], np.ndarray, np.ndarray]:
    stems = (REPO / "data" / "peru" / "unosat2023" / "splits" / "dev.txt").read_text().split()
    y = np.concatenate([rasterio.open(REPO / "data" / "peru" / "unosat2023" / "label" / f"{s}.tif").read(1).ravel() for s in stems])
    ok = y != -1
    return stems, ok, y[ok][::7]


def main() -> None:
    art = FROZEN["artifacts"]["temperatures_phaseD"]
    assert sha256(REPO / art["path"]) == art["sha256"], "frozen temperatures changed"
    t_valid = json.loads((REPO / art["path"]).read_text())
    groups = [g for g in FROZEN["groups"] if g["pass"] == "phaseD"]
    dev_stems, dev_ok, dev_y = dev_reference()
    t_dev = {}
    for g in groups:
        zs = [load(REPO / "outputs" / "logits_dev" / g["regime"] / g["model"] / s, dev_stems)[dev_ok][::7] for s in SEEDS]
        t_dev[g["id"]] = {s: fit_t(z, dev_y) for s, z in zip(SEEDS, zs)} | {"ensemble": fit_t(ens_logit(zs), dev_y)}
    (OUT / "calibration_temperatures_dev.json").write_text(json.dumps(t_dev, indent=2) + "\n")
    print("T fitted on peru-dev:", json.dumps(t_dev), flush=True)

    rows = []
    for split, root in ROOTS.items():
        stems, labels, subsets = chips_and_subsets(split)
        y2d = [rasterio.open(labels[s]).read(1) for s in stems]
        y = np.concatenate([a.ravel() for a in y2d])
        ok = y != -1
        near = np.concatenate([near_water(a).ravel() for a in y2d])[ok]
        chip = np.repeat(np.arange(len(stems)), y2d[0].size)[ok]
        y = y[ok]
        every = np.zeros(len(y), bool)
        every[::7] = True
        in_sub = {sub: np.isin(chip, [i for i, s in enumerate(stems) if s in subsets[sub]]) for sub in SUBSETS[split]}
        del y2d
        for g in groups:
            zs = [load(root / g["regime"] / g["model"] / s, stems)[ok] for s in SEEDS]
            src = T_SOURCE.get(g["regime"])
            variants = []
            for s, z in zip(SEEDS, zs):
                variants.append(("single", s, z, True))
                if src:
                    variants.append(("temp_s1f", s, z / t_valid[src][g["model"]][s], False))
                variants.append(("temp_dev", s, z / t_dev[g["id"]][s], False))
            zm = ens_logit(zs)
            variants.append(("ensemble", "ens", zm, True))
            if src:
                variants.append(("ens+temp_s1f", "ens", zm / t_valid[src][g["model"]]["ensemble"], False))
            variants.append(("ens+temp_dev", "ens", zm / t_dev[g["id"]]["ensemble"], False))
            for method, s, z, with_auroc in variants:
                for sub, m_sub in in_sub.items():
                    row = dict(split=split, subset=sub, group=g["id"], role=g["role"], method=method, seed=s)
                    for tag, m in (("all", every & m_sub), ("near", near & m_sub)):
                        row[f"pixels_{tag}"] = int(m.sum())
                        row[f"iou_{tag}"] = iou(z[m], y[m])
                        row[f"ece_{tag}"] = ece(z[m], y[m])
                        row[f"auroc_{tag}"] = auroc(-np.abs(z[m]), (z[m] > 0) != (y[m] == 1)) if with_auroc else np.nan
                    rows.append(row)
            print(f"{split} {g['id']}: done", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "calibration_runs.csv", index=False)
    cols = [f"{k}_{t}" for k in ("ece", "auroc", "iou") for t in ("all", "near")]
    c = r.groupby(["split", "subset", "group", "role", "method"], sort=False)[cols].agg(["mean", "std"])
    c.columns = [f"{a}_{b}" for a, b in c.columns]
    c = c.reset_index()
    c.to_csv(OUT / "calibration.csv", index=False)
    lines = ["# Calibration of the frozen phase-D rows on the test sets\n",
             "ECE in %, AUROC of −|z| for errors; mean ± std over seeds (ensemble rows: one value). "
             "Method: METHODS_NOTES 2026-10-16 \"Test calibration of the frozen phase-D rows\".\n"]
    for (split, sub), part in c.groupby(["split", "subset"], sort=False):
        lines += [f"\n## {split} / {sub}\n", "| Row | Role | Method | ECE all | ECE near | AUROC all | AUROC near |", "|---|---|---|---|---|---|---|"]
        for _, x in part.iterrows():
            def f(k, s=100, d=2):
                if np.isnan(x[f"{k}_mean"]):
                    return "—"
                return f"{s * x[f'{k}_mean']:.{d}f}" + ("" if np.isnan(x[f"{k}_std"]) else f" ± {s * x[f'{k}_std']:.{d}f}")
            lines.append(f"| {x.group} | {x.role} | {x.method} | {f('ece_all')} | {f('ece_near')} | {f('auroc_all', 1, 3)} | {f('auroc_near', 1, 3)} |")
    (OUT / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
