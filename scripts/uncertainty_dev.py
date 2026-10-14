"""Uncertainty on peru-dev (tasks 5.1, 5.3, 5.4): seed ensembles and temperature scaling.

Uses the per-chip water logits z = logit_water − logit_other saved by
floodseg.callbacks.SaveValPredictions (outputs/logits_dev/<regime>/<model>/seed<s>/<stem>.npy,
float32); p = sigmoid(z). Every prediction is handled as a float64 logit:
  single     z of each seed (mean ± std over seeds)
  temp       z / T; T fitted by NLL on the labelled pixels of half the dev chips and
             applied to the other half (2-fold by chip, fixed split), so no pixel is
             scored with a T fitted on it
  ensemble   mean probability of the 3 seeds, as the logit log p̄ − log(1 − p̄) with
             both tails averaged in log space (no saturation); uncertainty =
             predictive entropy (a monotone function of |logit|), and mutual
             information (entropy of the mean − mean entropy)
Scores: water IoU, ECE (15 bins) and error-detection AUROC, on all labelled pixels
(every 7th) and on labelled pixels within 10 px of a label water edge. AUROC is not
reported for `temp`: scaling is monotone within a fold, so pooled over the two folds
it only measures the mismatch between the two temperatures.

An earlier version read float16 probabilities. They saturate asymmetrically (only
0.99951 and 1.0 lie in [0.99951, 1], while near 0 they resolve down to ~1e-7), which
biased AUROC and the temperature fits; see docs/EXPERIMENT_LOG.md.

Writes outputs/analysis_dev/uncertainty.csv.

    uv run python scripts/uncertainty_dev.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "outputs" / "logits_dev"
LABEL = REPO / "data" / "peru" / "unosat2023" / "label"
OUT = REPO / "outputs" / "analysis_dev"


def near_water(y: np.ndarray) -> np.ndarray:
    w = y == 1
    edge = (w ^ binary_erosion(w)) | (~w ^ binary_erosion(~w))
    return distance_transform_edt(~edge) <= 10


def load_z(d: Path, stems: list[str], ok: np.ndarray | None = None) -> np.ndarray:
    z = np.concatenate([np.load(d / f"{st}.npy").ravel() for st in stems]).astype(np.float64)
    return z if ok is None else z[ok]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -z))


def entropy(z: np.ndarray) -> np.ndarray:
    """Binary entropy (nats) of sigmoid(z), stable for any z."""
    a = np.abs(z)
    return np.logaddexp(0, -a) + a * sigmoid(-a)


def ens_logit(zs: list[np.ndarray]) -> np.ndarray:
    """Logit of the mean probability, log p̄ − log(1 − p̄), both tails in log space."""
    n = np.log(len(zs))
    logp = logsumexp([-np.logaddexp(0, -z) for z in zs], axis=0) - n
    logq = logsumexp([-np.logaddexp(0, z) for z in zs], axis=0) - n
    return logp - logq


def ece(z: np.ndarray, y: np.ndarray) -> float:
    conf = sigmoid(np.abs(z))
    correct = (z > 0) == (y == 1)
    b = np.minimum((conf - 0.5) / 0.5 * 15, 14).astype(int)
    acc = np.bincount(b, weights=correct, minlength=15)
    cf = np.bincount(b, weights=conf, minlength=15)
    return float(np.sum(np.abs(acc - cf)) / len(z))


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


def iou(z: np.ndarray, y: np.ndarray) -> float:
    pr, t = z > 0, y == 1
    return float((pr & t).sum() / ((pr | t).sum()))


def fit_t(z: np.ndarray, y: np.ndarray) -> float:
    t1 = (y == 1).astype(float)

    def nll(log_t):
        s = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, s) - t1 * s))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def scores(name, z, y, unc, near, every):
    out = {}
    for tag, m in (("all", every), ("near", near)):
        zz, yy = z[m], y[m]
        out[f"iou_{tag}"] = iou(zz, yy)
        out[f"ece_{tag}"] = ece(zz, yy)
        out[f"auroc_{tag}"] = auroc(unc[m], (zz > 0) != (yy == 1)) if unc is not None else np.nan
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
        zs = [load_z(d, stems, ok) for d in dirs]
        for d, z in zip(dirs, zs):
            seed = d.parts[-1]
            rows.append(dict(regime=regime, model=model, seed=seed, **scores("single", z, y, -np.abs(z), near, every)))
            zt = np.empty_like(z)
            ts = []
            for k in (0, 1):
                t = fit_t(z[(f != k) & every], y[(f != k) & every])
                ts.append(t)
                zt[f == k] = z[f == k] / t
            rows.append(dict(regime=regime, model=model, seed=seed, temperature=np.mean(ts),
                             **scores("temp", zt, y, None, near, every)))
        zm = ens_logit(zs)
        mi = entropy(zm) - np.mean([entropy(z) for z in zs], axis=0)
        rows.append(dict(regime=regime, model=model, seed="ens", **scores("ensemble (entropy)", zm, y, -np.abs(zm), near, every)))
        rows.append(dict(regime=regime, model=model, seed="ens", **scores("ensemble (MI)", zm, y, mi, near, every)))
        print(f"{regime} {model}: done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "uncertainty.csv", index=False)
    print(f"wrote {(OUT / 'uncertainty.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
