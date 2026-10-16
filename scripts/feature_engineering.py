"""Feature engineering for pixel-wise water classification (course deliverable "Avance sem 5";
exploratory, dev-only: no `peru-test` / `peru-2017-test` file is read, and nothing here changes
the frozen phase-D comparison).

Question: do engineered features (spectral indices, local context) beat the raw 13 Sentinel-2
L1C bands for a classical pixel classifier, (a) trained on Peru labels and (b) trained on
Sen1Floods11 only and applied to Peru (the label-free transfer setting of phase D)?

Features (reflectance r = DN / 10,000; bands as in floodseg.normalization.S2_L1C_BANDS):
- `bands`: the 13 L1C bands.
- `indices`: NDWI = (G − NIR)/(G + NIR) with NIR = B08; MNDWI = (G − SWIR1)/(G + SWIR1);
  MNDWI2 = (G − SWIR2)/(G + SWIR2); NDVI = (NIR − R)/(NIR + R); NDMI = (NIR − SWIR1)/(NIR + SWIR1);
  AWEI_nsh = 4(G − SWIR1) − (0.25 NIR + 2.75 SWIR2); AWEI_sh = B + 2.5 G − 1.5 (NIR + SWIR1) − 0.25 SWIR2;
  WRI = (G + R)/(NIR + SWIR1).
- `context`: local mean of MNDWI (3×3, 7×7), local std of MNDWI (7×7), local mean of NIR and
  NDVI (7×7), computed on the full chip before sampling (scipy uniform_filter).
Feature sets: bands; indices; bands+indices; bands+indices+context.

Pixels: labelled (label ≠ −1) and valid (some band ≠ 0). Training: 600 random pixels per
`peru-train` chip (753 chips) or 2,000 per Sen1Floods11 `train` chip (252), natural class
prior. Evaluation on `peru-dev` (174 chips) on every 2nd row and column (1/4 of the pixels),
pooled water IoU / precision / recall, water if p > 0.5. Classifier: sklearn
HistGradientBoostingClassifier (200 iterations, lr 0.1, no early stopping); 3 seeds (seed sets
the pixel sample and the model). Baseline: MNDWI > t with t fitted on the training labels
(best pooled IoU on all training pixels, 0.005 bins), unlike the dev-tuned oracle threshold.
Univariate AUROC of each feature (water vs dry) on the training samples and on dev. Permutation
importance (Jaccard, 3 repeats) of the bands+indices+context model, seed 0, on 1,200 random
dev pixels per chip.

    uv run python scripts/feature_engineering.py
→ outputs/feature_engineering/{results_runs,results,auroc,importance}.csv, summary.md, figs/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from floodseg.normalization import S2_L1C_BANDS

REPO = Path(__file__).resolve().parents[1]
PERU = REPO / "data" / "peru" / "unosat2023"
S1F = REPO / "data" / "sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled"
OUT = REPO / "outputs" / "feature_engineering"
SEEDS = [0, 1, 2]
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
INDEX_NAMES = ["NDWI", "MNDWI", "MNDWI2", "NDVI", "NDMI", "AWEI_nsh", "AWEI_sh", "WRI"]
CONTEXT_NAMES = ["MNDWI_mean3", "MNDWI_mean7", "MNDWI_std7", "NIR_mean7", "NDVI_mean7"]
ALL = BAND_NAMES + INDEX_NAMES + CONTEXT_NAMES
SETS = {"bands": BAND_NAMES, "indices": INDEX_NAMES, "bands+indices": BAND_NAMES + INDEX_NAMES,
        "bands+indices+context": ALL}
PER_CHIP = {"peru-train": 600, "s1f11-train": 2000}


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def features(x: np.ndarray) -> np.ndarray:
    """(F, H, W) float32 features of one chip, in the order of ALL."""
    r = x.astype(np.float32) / 10000
    B, G, R, N, S1, S2 = r[1], r[2], r[3], r[7], r[11], r[12]
    mndwi, ndvi = nd(G, S1), nd(N, R)
    idx = [nd(G, N), mndwi, nd(G, S2), ndvi, nd(N, S1), 4 * (G - S1) - (0.25 * N + 2.75 * S2),
           B + 2.5 * G - 1.5 * (N + S1) - 0.25 * S2, (G + R) / (N + S1 + 1e-6)]
    m7 = uniform_filter(mndwi, 7)
    ctx = [uniform_filter(mndwi, 3), m7, np.sqrt(np.maximum(uniform_filter(mndwi**2, 7) - m7**2, 0)),
           uniform_filter(N, 7), uniform_filter(ndvi, 7)]
    return np.concatenate([r, np.stack(idx + ctx)]).astype(np.float32)


def load(img: Path, lab: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(img) as src:
        x = src.read()
    with rasterio.open(lab) as src:
        y = src.read(1)
    ok = (y >= 0) & (x != 0).any(0)
    return features(x), y, ok


def pairs(name: str) -> list[tuple[Path, Path]]:
    if name in ("peru-train", "peru-dev"):
        stems = (PERU / "splits" / f"{name.split('-')[1]}.txt").read_text().split()
        return [(PERU / "s2" / f"{s}.tif", PERU / "label" / f"{s}.tif") for s in stems]
    ids = [l.split(",")[0].removesuffix("_S1Hand.tif") for l in
           (REPO / "data/sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.txt").read_text().split()]
    return [(S1F / "S2Hand" / f"{i}_S2Hand.tif", S1F / "LabelHand" / f"{i}_LabelHand.tif") for i in ids]


def sample_train(name: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per seed: (X, y) with PER_CHIP[name] random labelled pixels per chip; plus MNDWI histograms."""
    rngs = {s: np.random.default_rng(1000 + s) for s in SEEDS}
    out = {s: ([], []) for s in SEEDS}
    for img, lab in pairs(name):
        f, y, ok = load(img, lab)
        iy, ix = np.nonzero(ok)
        for s in SEEDS:
            k = rngs[s].choice(len(iy), min(PER_CHIP[name], len(iy)), replace=False)
            out[s][0].append(f[:, iy[k], ix[k]].T)
            out[s][1].append(y[iy[k], ix[k]])
        mn = f[ALL.index("MNDWI")][ok]
        HIST.setdefault(name, np.zeros((2, len(EDGES) - 1)))
        for c in (0, 1):
            HIST[name][c] += np.histogram(np.clip(mn[y[ok] == c], -1, 1), EDGES)[0]
    return {s: (np.concatenate(X), np.concatenate(Y)) for s, (X, Y) in out.items()}


EDGES = np.linspace(-1, 1, 401)
HIST: dict[str, np.ndarray] = {}


def fitted_threshold(name: str) -> float:
    """MNDWI threshold with the best pooled IoU on all labelled training pixels."""
    dry, wet = HIST[name]
    tp = wet[::-1].cumsum()[::-1]  # water pixels with MNDWI >= edge i
    fp = dry[::-1].cumsum()[::-1]
    fn = wet.sum() - tp
    iou = tp / (tp + fp + fn)
    return float(EDGES[:-1][np.argmax(iou)])


def pooled(c: np.ndarray) -> dict:
    tp, fp, fn = c
    return dict(iou=tp / (tp + fp + fn), precision=tp / (tp + fp), recall=tp / (tp + fn))


def main() -> None:
    test = set((PERU / "splits" / "test.txt").read_text().split())
    for n in ("train", "dev"):
        assert not test & set((PERU / "splits" / f"{n}.txt").read_text().split()), "test chip in " + n
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)

    train = {n: sample_train(n) for n in ("peru-train", "s1f11-train")}
    thr = {n: fitted_threshold(n) for n in train}
    print("fitted MNDWI thresholds:", thr, flush=True)
    models = {}
    for n, per_seed in train.items():
        for s, (X, y) in per_seed.items():
            for fs, cols in SETS.items():
                idx = [ALL.index(c) for c in cols]
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=s)
                models[(n, fs, s)] = (m.fit(X[:, idx], y), idx)
        print(f"trained {n}: {len(y)} px, water share {y.mean():.3f}", flush=True)

    counts = {k: np.zeros(3) for k in list(models) + [(n, f"MNDWI > {thr[n]:.3f} (fit on {n})", 0) for n in thr]}
    rng = np.random.default_rng(0)
    dev_X, dev_y = [], []
    for img, lab in pairs("peru-dev"):
        f, y, ok = load(img, lab)
        f, y, ok = f[:, ::2, ::2], y[::2, ::2], ok[::2, ::2]
        X, yy = f[:, ok].T, y[ok]
        if not len(yy):
            continue
        w = yy == 1
        for k, (m, idx) in models.items():
            p = m.predict_proba(X[:, idx])[:, 1] > 0.5
            counts[k] += [(p & w).sum(), (p & ~w).sum(), (~p & w).sum()]
        for n in thr:
            p = X[:, ALL.index("MNDWI")] > thr[n]
            counts[(n, f"MNDWI > {thr[n]:.3f} (fit on {n})", 0)] += [(p & w).sum(), (p & ~w).sum(), (~p & w).sum()]
        k = rng.choice(len(yy), min(1200, len(yy)), replace=False)
        dev_X.append(X[k])
        dev_y.append(yy[k])
    dev_X, dev_y = np.concatenate(dev_X), np.concatenate(dev_y)

    runs = pd.DataFrame([dict(train=n, features=fs, seed=s, **pooled(c)) for (n, fs, s), c in counts.items()])
    runs.to_csv(OUT / "results_runs.csv", index=False)
    res = runs.groupby(["train", "features"], sort=False)[["iou", "precision", "recall"]].agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res = res.reset_index()
    res.to_csv(OUT / "results.csv", index=False)

    au = []
    for j, c in enumerate(ALL):
        row = dict(feature=c)
        for n, per_seed in train.items():
            X, y = per_seed[0]
            row[f"auroc_{n}"] = roc_auc_score(y, X[:, j])
        row["auroc_peru-dev"] = roc_auc_score(dev_y, dev_X[:, j])
        au.append(row)
    au = pd.DataFrame(au)
    au.to_csv(OUT / "auroc.csv", index=False)

    imp = []
    for n in train:
        m, idx = models[(n, "bands+indices+context", 0)]
        r = permutation_importance(m, dev_X[:, idx], dev_y, scoring="jaccard", n_repeats=3, random_state=0, n_jobs=-1)
        imp += [dict(train=n, feature=ALL[i], importance=r.importances_mean[j], std=r.importances_std[j]) for j, i in enumerate(idx)]
    imp = pd.DataFrame(imp)
    imp.to_csv(OUT / "importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sep = au.set_index("feature")[["auroc_s1f11-train", "auroc_peru-train", "auroc_peru-dev"]]
    (sep - 0.5).abs().add(0.5).plot.bar(ax=ax, width=0.8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (water vs dry, max(a, 1−a))")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "auroc.png", dpi=150)
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, n in zip(axs, train):
        imp[imp.train == n].set_index("feature").importance.reindex(ALL[::-1]).plot.barh(ax=ax)
        ax.set_title(f"trained on {n}, scored on peru-dev")
        ax.set_xlabel("drop in Jaccard when permuted")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "importance.png", dpi=150)

    pd.set_option("display.width", 200)
    text = ("# Feature engineering (dev-only)\n\nFitted MNDWI thresholds: " + str({k: round(v, 3) for k, v in thr.items()})
            + f"\n\nDev pixels scored: {int(sum(c.sum() for c in [counts[next(iter(models))]]))} (tp+fp+fn of one run)\n\n"
            + "## peru-dev water IoU\n\n```\n" + res.round(4).to_string(index=False) + "\n```"
            + "\n\n## Univariate AUROC\n\n```\n" + au.round(3).to_string(index=False) + "\n```"
            + "\n\n## Permutation importance (bands+indices+context, seed 0)\n\n```\n"
            + imp.sort_values(["train", "importance"], ascending=[True, False]).round(4).to_string(index=False) + "\n```" + "\n")
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
