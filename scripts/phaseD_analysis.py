"""Phase-D tasks 5.2 and 5.3 on `peru-dev`: where the best label-free methods still fail, and
whether their uncertainty is calibrated (pre-registered plan: docs/METHODS_NOTES.md 2026-10-15).

Regimes per backbone (3 seeds each, float32 water logits in outputs/logits_dev/<regime>/...):
`s1f11_src` (source-only), `adabn`, `hm` (test-time histogram matching), `peru` (Peru-trained
upper bound). The best label-free method is AdaBN for U-Net and HM for TerraMind; both
alignment methods are reported for both backbones.

1. Strata (task 5.2): the phase-C strata (scripts/analyze_dev.strata_for: distance to the label
   water edge, NDWI of all pixels and of water pixels, water-body size, water type, land cover,
   distance to thick cloud, date). Per run, TP/FP/FN/TN over all labelled dev pixels per stratum
   value → IoU, recall, false-positive rate; then mean ± std over seeds.
   → outputs/phaseD/analysis/strata_runs.csv, strata.csv
2. Calibration (task 5.3): ECE (15 bins) and error-detection AUROC of −|z|, on every 7th
   labelled pixel ("all") and on all labelled pixels ≤ 10 px from a label water edge ("near"),
   as scripts/uncertainty_dev.py. Methods:
     single      raw logits, per seed
     temp_s1f    z / T_valid, T fitted by NLL on all labelled Sen1Floods11 `valid` pixels of the
                 same source model and seed (outputs/logits_valid/s1f11_src/; no Peru label).
                 For `peru` (no Sen1Floods11 model) this method is skipped.
     temp_dev    2-fold-by-chip temperature fitted on dev labels (as phase C). Uses Peru
                 labels: the reference for what calibration could reach, not a label-free method.
     ensemble    mean probability of the 3 seeds; AUROC of its −|z|
     ens+temp_s1f  ensemble, T fitted on the `valid` ensemble of the 3 source seeds
   AUROC is reported for single and ensemble only (a single global T is monotone, so it does
   not change AUROC; 2-fold T mixes two temperatures).
   → outputs/phaseD/analysis/calibration_runs.csv, calibration.csv, temperatures.json

    uv run python scripts/phaseD_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_dev import strata_for  # noqa: E402
from uncertainty_dev import auroc, ece, ens_logit, fit_t, iou, near_water  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "outputs" / "logits_dev"
VALID = REPO / "outputs" / "logits_valid" / "s1f11_src"
LABEL_DEV = REPO / "data" / "peru" / "unosat2023" / "label"
LABEL_VALID = REPO / "data" / "sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
OUT = REPO / "outputs" / "phaseD" / "analysis"
REGIMES = ["s1f11_src", "adabn", "hm", "peru"]
MODELS = ["unet", "terramind_v1_base"]
SEEDS = ["seed0", "seed1", "seed2"]
STRATA = ["distance to water edge", "water NDWI", "NDWI (all pixels)", "water body size", "water type",
          "landcover", "distance to thick cloud", "date"]


def load(d: Path, stems: list[str]) -> np.ndarray:
    return np.concatenate([np.load(d / f"{s}.npy").ravel() for s in stems]).astype(np.float64)


def valid_temperatures() -> dict:
    stems = sorted(p.stem for p in (VALID / "unet" / "seed0").glob("*.npy"))
    assert len(stems) == 89, len(stems)
    y = np.concatenate([rasterio.open(LABEL_VALID / f"{s.replace('_S2Hand', '_LabelHand')}.tif").read(1).ravel()
                        for s in stems])
    ok = y != -1
    t = {}
    for m in MODELS:
        zs = [load(VALID / m / s, stems)[ok] for s in SEEDS]
        t[m] = {s: fit_t(z, y[ok]) for s, z in zip(SEEDS, zs)}
        t[m]["ensemble"] = fit_t(ens_logit(zs), y[ok])
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = (REPO / "data" / "peru" / "unosat2023" / "splits" / "dev.txt").read_text().split()
    runs = {(r, m): [DEV / r / m / s for s in SEEDS] for r in REGIMES for m in MODELS}
    for ds in runs.values():
        for d in ds:
            assert len(list(d.glob("*.npy"))) == len(stems), d

    y2d, per_chip = [], []
    for st in stems:
        with rasterio.open(LABEL_DEV / f"{st}.tif") as src:
            y = src.read(1)
        y2d.append(y)
        per_chip.append(strata_for(st, y))
    y = np.concatenate([a.ravel() for a in y2d])
    ok = y != -1
    near = np.concatenate([near_water(a).ravel() for a in y2d])[ok]
    chip = np.repeat(np.arange(len(stems)), y2d[0].size)[ok]
    cats = {s: sorted({v for c in per_chip for v in np.unique(c[s])}) for s in STRATA}
    codes = {s: np.concatenate([np.searchsorted(cats[s], c[s]).ravel() for c in per_chip])[ok] for s in STRATA}
    del per_chip
    y = y[ok]
    true = y == 1
    every = np.zeros(len(y), bool)
    every[::7] = True
    fold = np.random.default_rng(0).permutation(len(stems)) % 2  # same split as uncertainty_dev.py
    f = fold[chip]

    def temp_dev(z):
        zt = np.empty_like(z)
        for k in (0, 1):
            zt[f == k] = z[f == k] / fit_t(z[f != k][::7], y[f != k][::7])
        return zt

    t_valid = valid_temperatures()
    (OUT / "temperatures.json").write_text(json.dumps(t_valid, indent=2) + "\n")
    print("T from Sen1Floods11 valid:", json.dumps(t_valid), flush=True)

    srows, crows = [], []
    for (regime, model), dirs in runs.items():
        zs = [load(d, stems)[ok] for d in dirs]
        for seed, z in zip(SEEDS, zs):
            pred = z > 0
            cm = 2 * (~pred) + (~true)  # 0 TP, 1 FP, 2 FN, 3 TN
            for s in STRATA:
                a = np.bincount(codes[s] * 4 + cm, minlength=4 * len(cats[s])).reshape(-1, 4)
                for v, (tp, fp, fn, tn) in zip(cats[s], a):
                    if tp + fp + fn + tn:
                        srows.append(dict(regime=regime, model=model, seed=seed, stratum=s, value=v, pixels=int(tp + fp + fn + tn),
                                          water_px=int(tp + fn), iou=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                                          recall=tp / (tp + fn) if tp + fn else np.nan, fpr=fp / (fp + tn) if fp + tn else np.nan))
        variants = []  # (method, seed, logits, uncertainty-for-AUROC or None)
        for seed, z in zip(SEEDS, zs):
            variants.append(("single", seed, z, -np.abs(z)))
            if regime != "peru":
                variants.append(("temp_s1f", seed, z / t_valid[model][seed], None))
            variants.append(("temp_dev", seed, temp_dev(z), None))
        zm = ens_logit(zs)
        variants.append(("ensemble", "ens", zm, -np.abs(zm)))
        if regime != "peru":
            variants.append(("ens+temp_s1f", "ens", zm / t_valid[model]["ensemble"], None))
        variants.append(("ens+temp_dev", "ens", temp_dev(zm), None))
        for method, seed, z, unc in variants:
            row = dict(regime=regime, model=model, method=method, seed=seed)
            for tag, m in (("all", every), ("near", near)):
                row[f"iou_{tag}"] = iou(z[m], y[m])
                row[f"ece_{tag}"] = ece(z[m], y[m])
                row[f"auroc_{tag}"] = auroc(unc[m], (z[m] > 0) != (y[m] == 1)) if unc is not None else np.nan
            crows.append(row)
        print(f"{regime} {model}: done", flush=True)

    s = pd.DataFrame(srows)
    s.to_csv(OUT / "strata_runs.csv", index=False)
    s.groupby(["stratum", "value", "model", "regime"]).agg(
        pixels=("pixels", "first"), water_px=("water_px", "first"), iou=("iou", "mean"), iou_std=("iou", "std"),
        recall=("recall", "mean"), recall_std=("recall", "std"), fpr=("fpr", "mean"), fpr_std=("fpr", "std"),
    ).reset_index().to_csv(OUT / "strata.csv", index=False)
    c = pd.DataFrame(crows)
    c.to_csv(OUT / "calibration_runs.csv", index=False)
    cols = [k for k in c.columns if k.split("_")[0] in ("iou", "ece", "auroc")]
    c.groupby(["model", "regime", "method"])[cols].agg(["mean", "std"]).to_csv(OUT / "calibration.csv")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
