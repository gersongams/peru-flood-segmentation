"""Failure analysis on peru-dev (tasks 6.2-6.4): stratified errors and calibration.

Reads per-chip water probabilities saved by floodseg.callbacks.SaveValPredictions
(outputs/preds_dev/<regime>/<model>/seed<s>/<stem>.npy), the labels, and the
strata of scripts/build_strata.py. Prediction = probability > 0.5 (= argmax).
Only labelled pixels (label != -1) count.

Per stratum it sums TP / FP / FN / TN over all dev pixels and reports
IoU, recall (share of true water found) and false-positive rate (share of dry
pixels called water). Metrics are computed per seed, then mean ± std over seeds.
Calibration: ECE (15 equal-width bins) and error-detection AUROC of the
uncertainty 1 - max(p, 1 - p), on every 7th labelled pixel and, separately, on all
pixels within 10 px of a water edge (most dev pixels are easy desert, which
dilutes the overall ECE).

Writes outputs/analysis_dev/{strata.csv,calibration.csv,chips.csv}.

    uv run python scripts/analyze_dev.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion, distance_transform_edt, label as cc_label

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "outputs" / "preds_dev"
LABEL = REPO / "data" / "peru" / "unosat2023" / "label"
STRATA = REPO / "data" / "peru" / "strata" / "unosat2023"
OUT = REPO / "outputs" / "analysis_dev"
PX_HA = 0.01  # one 10 m pixel = 0.01 ha

WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare/sparse",
              70: "snow", 80: "permanent water", 90: "herbaceous wetland", 95: "mangrove", 100: "moss"}


def bins(x, edges, names):
    return np.array(names, dtype=object)[np.digitize(x, edges)]


def strata_for(stem: str, y: np.ndarray) -> dict[str, np.ndarray]:
    s = np.load(STRATA / f"{stem}.npz")
    water = y == 1
    comp, _ = cc_label(water)
    size = np.bincount(comp.ravel())[comp] * PX_HA  # ha of the water body each pixel is in
    size[~water] = np.nan
    edge = (water ^ binary_erosion(water)) | (~water ^ binary_erosion(~water))
    d_edge = distance_transform_edt(~edge)
    return {
        "date": np.full(y.shape, stem[:10], dtype=object),
        "landcover": np.vectorize(lambda v: WORLDCOVER.get(int(v), "no data"), otypes=[object])(s["landcover"]),
        "elevation": bins(s["elev"], [10, 50, 200], ["<10 m", "10-50 m", "50-200 m", ">200 m"]),
        "slope": bins(s["slope"], [2, 5, 15], ["<2°", "2-5°", "5-15°", ">15°"]),
        "water type": np.where(s["existing"] & water, "existing water", np.where(water, "new flood water", "dry")).astype(object),
        "water body size": np.where(water, bins(np.nan_to_num(size), [1, 10, 100], ["<1 ha", "1-10 ha", "10-100 ha", ">100 ha"]), "dry").astype(object),
        "distance to water edge": bins(d_edge, [1.5, 3.5, 10.5], ["edge (≤1 px)", "2-3 px", "4-10 px", ">10 px"]),
        "NDWI (all pixels)": bins(s["ndwi"], [-10, 0, 10, 30], ["< -0.1", "-0.1-0", "0-0.1", "0.1-0.3", "> 0.3"]),
        "water NDWI": np.where(water, bins(s["ndwi"], [-10, 0, 10, 30], ["< -0.1", "-0.1-0", "0-0.1", "0.1-0.3", "> 0.3"]), "dry").astype(object),
        "distance to thick cloud": bins(s["cloud_dist"], [5, 20, 50], ["<5 px", "5-20 px", "20-50 px", ">50 px"]),
    }


def ece_auroc(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    conf = np.maximum(p, 1 - p)
    correct = (p > 0.5) == (y == 1)
    b = np.minimum((conf - 0.5) / 0.5 * 15, 14).astype(int)
    n = np.bincount(b, minlength=15)
    acc = np.bincount(b, weights=correct, minlength=15)
    cf = np.bincount(b, weights=conf, minlength=15)
    ece = float(np.sum(np.abs(acc - cf)) / len(p))
    # AUROC of uncertainty for detecting errors, from a 1000-bin histogram (ties split evenly)
    u = np.minimum(((1 - conf) * 2 * 1000).astype(int), 999)
    pos = np.bincount(u[~correct], minlength=1000).astype(float)
    neg = np.bincount(u[correct], minlength=1000).astype(float)
    if pos.sum() == 0 or neg.sum() == 0:
        return ece, float("nan")
    auroc = (np.sum(pos * (np.cumsum(neg) - neg)) + 0.5 * np.sum(pos * neg)) / (pos.sum() * neg.sum())
    return ece, float(auroc)


def main() -> None:
    runs = sorted(d for d in PRED.glob("*/*/seed*") if d.is_dir() and len(list(d.glob("*.npy"))) == 174)
    stems = sorted(p.stem for p in runs[0].glob("*.npy"))
    labels = {}
    strata = {}
    for st in stems:
        with rasterio.open(LABEL / f"{st}.tif") as src:
            y = src.read(1)
        labels[st] = y
        strata[st] = strata_for(st, y)
    rows, cal, chips = [], [], []
    # integer-code each stratum once: per chip, code[name] indexes cats[name]
    cats = {name: sorted({v for st in stems for v in np.unique(strata[st][name])}) for name in strata[stems[0]]}
    codes = {st: {n: np.searchsorted(cats[n], strata[st][n]) for n in cats}
             for st in stems}
    for run in runs:
        regime, model, seed = run.parts[-3], run.parts[-2], run.parts[-1]
        acc = {n: np.zeros((len(cats[n]), 4)) for n in cats}
        ps, ys, pn, yn = [], [], [], []
        for st in stems:
            y = labels[st]
            p = np.load(run / f"{st}.npy").astype(np.float32)
            ok = y != -1
            pred, true = (p > 0.5)[ok], (y == 1)[ok]
            ps.append(p[ok][::7]), ys.append(y[ok][::7])  # every 7th pixel for calibration
            near = ok & (strata[st]["distance to water edge"] != ">10 px")  # within 10 px of a water edge
            pn.append(p[near]), yn.append(y[near])
            tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
            chips.append(dict(regime=regime, model=model, seed=seed, chip=st, tp=tp, fp=fp, fn=fn,
                              iou=tp / (tp + fp + fn) if tp + fp + fn else np.nan, water_px=int(true.sum())))
            cm = 2 * (~pred) + (~true)  # 0 TP, 1 FP, 2 FN, 3 TN
            for n in cats:
                k = codes[st][n][ok]
                acc[n] += np.bincount(k * 4 + cm, minlength=4 * len(cats[n])).reshape(-1, 4)
        for n, a in acc.items():
            for v, (tp, fp, fn, tn) in zip(cats[n], a):
                if tp + fp + fn + tn == 0:
                    continue
                rows.append(dict(regime=regime, model=model, seed=seed, stratum=n, value=v, pixels=int(tp + fp + fn + tn),
                                 water_px=int(tp + fn), iou=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                                 recall=tp / (tp + fn) if tp + fn else np.nan, fpr=fp / (fp + tn) if fp + tn else np.nan))
        e, auc = ece_auroc(np.concatenate(ps), np.concatenate(ys))
        en, aucn = ece_auroc(np.concatenate(pn), np.concatenate(yn))
        cal.append(dict(regime=regime, model=model, seed=seed, ece=e, error_auroc=auc,
                        ece_near_water=en, error_auroc_near_water=aucn))
        print(f"{regime} {model} {seed}: done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "strata.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame(chips).to_csv(OUT / "chips.csv", index=False)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
