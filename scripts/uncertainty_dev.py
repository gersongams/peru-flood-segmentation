"""Uncertainty on peru-dev (tasks 5.1, 5.3, 5.4): seed ensembles and temperature scaling.

Uses the per-chip water probabilities of scripts/analyze_dev.py's inputs
(outputs/preds_dev/<regime>/<model>/seed<s>/<stem>.npy, float16).

Per regime and model with 3 seeds:
  single     each seed alone (mean ± std over seeds)
  temp       each seed after temperature scaling; T fitted by NLL on the labelled
             pixels of half the dev chips and applied to the other half (2-fold by
             chip, fixed split), so no pixel is scored with a T fitted on it
  ensemble   mean probability of the 3 seeds; uncertainty = predictive entropy, and
             mutual information (entropy of the mean − mean entropy)
Scores: water IoU, ECE (15 bins) and error-detection AUROC, on all labelled pixels
(every 7th) and on labelled pixels within 10 px of a label water edge.

Logits are recovered as log(p / (1 − p)) from float16 probabilities, so they saturate
at |logit| ≈ 7.6; temperature scaling is approximate for very confident pixels.

Writes outputs/analysis_dev/uncertainty.csv.

    uv run python scripts/uncertainty_dev.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "outputs" / "preds_dev"
LABEL = REPO / "data" / "peru" / "unosat2023" / "label"
OUT = REPO / "outputs" / "analysis_dev"
EPS = 1e-6


def near_water(y: np.ndarray) -> np.ndarray:
    w = y == 1
    edge = (w ^ binary_erosion(w)) | (~w ^ binary_erosion(~w))
    return distance_transform_edt(~edge) <= 10


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ece(p: np.ndarray, y: np.ndarray) -> float:
    conf = np.maximum(p, 1 - p)
    correct = (p > 0.5) == (y == 1)
    b = np.minimum((conf - 0.5) / 0.5 * 15, 14).astype(int)
    acc = np.bincount(b, weights=correct, minlength=15)
    cf = np.bincount(b, weights=conf, minlength=15)
    return float(np.sum(np.abs(acc - cf)) / len(p))


def auroc(score: np.ndarray, err: np.ndarray) -> float:
    """P(score of an error > score of a correct pixel), ties counted half."""
    order = np.argsort(score, kind="mergesort")
    s, e = score[order], err[order]
    _, idx, counts = np.unique(s, return_index=True, return_counts=True)
    pos = np.add.reduceat(e.astype(float), idx)
    neg = counts - pos
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((np.sum(pos * below) + 0.5 * np.sum(pos * neg)) / (npos * nneg))


def iou(p: np.ndarray, y: np.ndarray) -> float:
    pr, t = p > 0.5, y == 1
    return float((pr & t).sum() / ((pr | t).sum()))


def fit_t(logit: np.ndarray, y: np.ndarray) -> float:
    t1 = (y == 1).astype(float)

    def nll(log_t):
        z = logit / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, z) - t1 * z))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def scores(name, p, y, unc, near, every):
    out = {}
    for tag, m in (("all", every), ("near", near)):
        pp, yy, uu = p[m], y[m], unc[m]
        out[f"iou_{tag}"] = iou(pp, yy)
        out[f"ece_{tag}"] = ece(pp, yy)
        out[f"auroc_{tag}"] = auroc(uu, (pp > 0.5) != (yy == 1))
    return dict(method=name, **out)


def main() -> None:
    groups = {}
    for d in sorted(PRED.glob("*/*/seed*")):
        if len(list(d.glob("*.npy"))) == 174:
            groups.setdefault((d.parts[-3], d.parts[-2]), []).append(d)
    stems = sorted(p.stem for p in next(iter(groups.values()))[0].glob("*.npy"))
    ys, nears = [], []
    for st in stems:
        with rasterio.open(LABEL / f"{st}.tif") as src:
            y = src.read(1)
        ys.append(y.ravel()), nears.append(near_water(y).ravel())
    chip = np.repeat(np.arange(len(stems)), ys[0].size)
    y = np.concatenate(ys)
    ok = y != -1
    y, chip, near = y[ok], chip[ok], np.concatenate(nears)[ok]
    every = np.zeros(len(y), bool)
    every[::7] = True
    fold = np.random.default_rng(0).permutation(len(stems)) % 2  # 2-fold split by chip
    f = fold[chip]
    rows = []
    for (regime, model), dirs in groups.items():
        if len(dirs) < 3:
            continue
        ps = [np.concatenate([np.load(d / f"{st}.npy").astype(np.float32).ravel() for st in stems])[ok] for d in dirs]
        for d, p in zip(dirs, ps):
            seed = d.parts[-1]
            rows.append(dict(regime=regime, model=model, seed=seed,
                             **scores("single", p, y, -np.maximum(p, 1 - p), near, every)))
            logit = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
            pt = np.empty_like(p)
            ts = []
            for k in (0, 1):
                t = fit_t(logit[(f != k) & every], y[(f != k) & every])
                ts.append(t)
                pt[f == k] = 1 / (1 + np.exp(-logit[f == k] / t))
            rows.append(dict(regime=regime, model=model, seed=seed, temperature=np.mean(ts),
                             **scores("temp", pt, y, -np.maximum(pt, 1 - pt), near, every)))
        pm = np.mean(ps, axis=0)
        h = entropy(pm)
        mi = h - np.mean([entropy(p) for p in ps], axis=0)
        rows.append(dict(regime=regime, model=model, seed="ens", **scores("ensemble (entropy)", pm, y, h, near, every)))
        rows.append(dict(regime=regime, model=model, seed="ens", **scores("ensemble (MI)", pm, y, mi, near, every)))
        print(f"{regime} {model}: done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "uncertainty.csv", index=False)
    print(f"wrote {(OUT / 'uncertainty.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
