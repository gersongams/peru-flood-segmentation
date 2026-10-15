"""Reliability diagrams and per-stratum calibration (tasks 5.3, 5.4, 6.4).

1. Sen1Floods11 `valid` (check first): the seed-0 Sen1Floods11 models on their own
   validation split (outputs/logits_valid/s1f11/<model>/seed0/). Reliability bins,
   ECE, error AUROC; the ECE recomputed from the bins must equal ece(). A temperature
   is fitted by NLL on all labelled `valid` pixels (task 5.3) and stored; it is then
   applied to the same models on `peru-dev` (calibration under domain shift).
2. `peru-dev` reliability bins per regime, model and method:
     single    raw logits, the 3 seeds pooled
     temp      Peru-trained models: T fitted 2-fold by chip on dev (as in
               scripts/uncertainty_dev.py); Sen1Floods11-only: T from `valid`
     ensemble  mean probability of the 3 seeds (scripts/uncertainty_dev.ens_logit)
     ens+temp  the ensemble mean, temperature-scaled 2-fold by chip
3. `peru-dev` per-stratum ECE and error AUROC (strata of scripts/analyze_dev.py), with
   pixel and error counts per cell. AUROC is not reported for 2-fold temperature
   scaling: scaling is monotone within a fold, so pooled over the folds it only
   measures the mismatch between the two temperatures.

Works on the float32 logits of floodseg.callbacks.SaveValPredictions (no float16
saturation). Uses all labelled pixels (no subsampling). Bins: 15 equal-width confidence bins on
[0.5, 1], as ece(). "near" = within 10 px of a label water edge.

Writes outputs/analysis_dev/{calib_valid.csv,reliability_bins.csv,calib_strata.csv,
temperatures_valid.json} and figs/reliability_*.png.

    uv run python scripts/calibration_dev.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_dev import strata_for  # noqa: E402
from uncertainty_dev import PHASE_C, auroc, ece, ens_logit, fit_t, iou, load_z, near_water, sigmoid  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PRED_DEV = REPO / "outputs" / "logits_dev"
PRED_VALID = REPO / "outputs" / "logits_valid" / "s1f11"
LABEL_DEV = REPO / "data" / "peru" / "unosat2023" / "label"
LABEL_VALID = REPO / "data" / "sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
OUT = REPO / "outputs" / "analysis_dev"
MODELS = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}
STRATA = ["distance to water edge", "NDWI (all pixels)", "water NDWI", "water body size",
          "water type", "landcover", "distance to thick cloud", "date"]
NB = 15


def rel_bins(z: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    conf = sigmoid(np.abs(z))
    correct = (z > 0) == (y == 1)
    b = np.minimum((conf - 0.5) / 0.5 * NB, NB - 1).astype(int)
    n = np.bincount(b, minlength=NB)
    with np.errstate(invalid="ignore"):
        acc = np.bincount(b, weights=correct, minlength=NB) / n
        cf = np.bincount(b, weights=conf, minlength=NB) / n
    return pd.DataFrame(dict(bin=np.arange(NB), lo=0.5 + np.arange(NB) / NB / 2, n=n, acc=acc, conf=cf))


def ece_from_bins(r: pd.DataFrame) -> float:
    r = r[r.n > 0]
    return float(np.sum(r.n * np.abs(r.acc - r.conf)) / r.n.sum())


def labels(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, nears = [], []
    for path in paths:
        with rasterio.open(path) as src:
            y = src.read(1)
        ys.append(y), nears.append(near_water(y).ravel())
    return np.stack(ys), np.concatenate(nears), np.repeat(np.arange(len(paths)), ys[0].size)


def valid_part(bins_rows: list) -> dict[str, float]:
    stems = sorted(p.stem for p in (PRED_VALID / "unet" / "seed0").glob("*.npy"))
    assert len(stems) == 89, len(stems)
    y2d, near, chip = labels([LABEL_VALID / f"{st.replace('_S2Hand', '_LabelHand')}.tif" for st in stems])
    y = y2d.ravel()
    ok = y != -1
    y, near, chip = y[ok], near[ok], chip[ok]
    fold = np.random.default_rng(0).permutation(len(stems)) % 2
    f = fold[chip]
    rows, temps = [], {}
    for m in MODELS:
        z = load_z(PRED_VALID / m / "seed0", stems, ok)
        temps[m] = fit_t(z, y)
        zt = np.empty_like(z)  # 2-fold, so the valid score is not in-sample
        for k in (0, 1):
            zt[f == k] = z[f == k] / fit_t(z[f != k], y[f != k])
        for method, q in (("single", z), ("temp", zt)):
            for region, msk in (("all", np.ones_like(near)), ("near", near)):
                r = rel_bins(q[msk], y[msk])
                e = ece(q[msk], y[msk])
                assert abs(e - ece_from_bins(r)) < 1e-9, (e, ece_from_bins(r))
                err = (q[msk] > 0) != (y[msk] == 1)
                rows.append(dict(model=m, method=method, region=region, pixels=int(msk.sum()), errors=int(err.sum()),
                                 iou=iou(q[msk], y[msk]), ece=e,
                                 auroc=auroc(-np.abs(q[msk]), err) if method == "single" else np.nan))
                bins_rows.append(r.assign(split="valid", regime="s1f11", model=m, method=method, region=region))
        print(f"valid {m}: T={temps[m]:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "calib_valid.csv", index=False)
    (OUT / "temperatures_valid.json").write_text(json.dumps(temps, indent=2) + "\n")
    return temps


def stratum_cells(code: np.ndarray, ncat: int, z: np.ndarray, y: np.ndarray, unc: np.ndarray | None) -> list[dict]:
    """ECE (and AUROC if unc is given) per category of one stratum."""
    conf = sigmoid(np.abs(z))
    correct = (z > 0) == (y == 1)
    b = np.minimum((conf - 0.5) / 0.5 * NB, NB - 1).astype(int)
    k = code * NB + b
    n = np.bincount(k, minlength=ncat * NB).reshape(ncat, NB)
    acc = np.bincount(k, weights=correct, minlength=ncat * NB).reshape(ncat, NB)
    cf = np.bincount(k, weights=conf, minlength=ncat * NB).reshape(ncat, NB)
    out = []
    for c in range(ncat):
        tot = n[c].sum()
        if tot == 0:
            out.append(None)
            continue
        m = code == c if unc is not None else None
        out.append(dict(pixels=int(tot), errors=int(tot - acc[c].sum()), ece=float(np.abs(acc[c] - cf[c]).sum() / tot),
                        auroc=auroc(unc[m], ~correct[m]) if unc is not None else np.nan))
    return out


def dev_part(t_valid: dict[str, float], bins_rows: list) -> None:
    groups = {}
    for d in sorted(PRED_DEV.glob("*/*/seed*")):
        if d.parts[-3] in PHASE_C and len(list(d.glob("*.npy"))) == 174:
            groups.setdefault((d.parts[-3], d.parts[-2]), []).append(d)
    stems = sorted(p.stem for p in next(iter(groups.values()))[0].glob("*.npy"))
    y2d, near, chip = labels([LABEL_DEV / f"{st}.tif" for st in stems])
    y = y2d.ravel()
    ok = y != -1
    # integer-code each stratum over all dev pixels (as in analyze_dev.py)
    per_chip = [strata_for(st, y2d[i]) for i, st in enumerate(stems)]
    cats = {s: sorted({v for c in per_chip for v in np.unique(c[s])}) for s in STRATA}
    codes = {s: np.concatenate([np.searchsorted(cats[s], c[s]).ravel() for c in per_chip])[ok] for s in STRATA}
    del per_chip
    y, near, chip = y[ok], near[ok], chip[ok]
    fold = np.random.default_rng(0).permutation(len(stems)) % 2  # same split as uncertainty_dev.py
    f = fold[chip]

    def temp2(z):
        zt = np.empty_like(z)
        for k in (0, 1):
            t = fit_t(z[(f != k)][::7], y[(f != k)][::7])  # every 7th pixel, as uncertainty_dev.py
            zt[f == k] = z[f == k] / t
        return zt

    rows = []
    for (regime, model), dirs in groups.items():
        zs = [load_z(d, stems, ok) for d in dirs]
        variants = []  # (method, seed, logits, uncertainty for AUROC or None)
        for d, z in zip(dirs, zs):
            variants.append(("single", d.parts[-1], z, -np.abs(z)))
            zt = z / t_valid[model] if regime == "s1f11" else temp2(z)
            # a single valid-fitted T is monotone, so its AUROC equals the raw one
            variants.append(("temp", d.parts[-1], zt, None))
        if len(zs) >= 3:
            zm = ens_logit(zs)
            variants.append(("ensemble", "ens", zm, -np.abs(zm)))  # entropy is monotone in |logit|
            variants.append(("ens+temp", "ens", temp2(zm), None))
        for method, seed, z, unc in variants:
            for s in STRATA:
                for v, cell in zip(cats[s], stratum_cells(codes[s], len(cats[s]), z, y, unc)):
                    if cell:
                        rows.append(dict(regime=regime, model=model, method=method, seed=seed, stratum=s, value=v, **cell))
            for region, msk in (("all", np.ones_like(near)), ("near", near)):
                rows.append(dict(regime=regime, model=model, method=method, seed=seed, stratum="overall", value=region,
                                 **stratum_cells(np.zeros(msk.sum(), int), 1, z[msk], y[msk],
                                                 unc[msk] if unc is not None else None)[0]))
        # reliability bins: seeds pooled per method
        for method in ("single", "temp", "ensemble", "ens+temp"):
            vs = [v for v in variants if v[0] == method]
            if not vs:
                continue
            for region, msk in (("all", np.ones_like(near)), ("near", near)):
                pp = np.concatenate([v[2][msk] for v in vs])
                yy = np.tile(y[msk], len(vs))
                bins_rows.append(rel_bins(pp, yy).assign(split="peru-dev", regime=regime, model=model,
                                                         method=method, region=region))
        print(f"dev {regime} {model}: done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "calib_strata.csv", index=False)


def plot(bins: pd.DataFrame) -> None:
    figs = OUT / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    panels = [("valid", "s1f11", ["single", "temp"], "Sen1Floods11 valid, Sen1Floods11 models (seed 0)"),
              ("peru-dev", "s1f11", ["single", "temp"], "peru-dev, Sen1Floods11-only (seed 0; T from valid)"),
              ("peru-dev", "peru", ["single", "temp", "ensemble", "ens+temp"], "peru-dev, Peru-only (3 seeds pooled)"),
              ("peru-dev", "s1f11_ft", ["single", "temp", "ensemble", "ens+temp"], "peru-dev, Sen1Floods11 → Peru (3 seeds pooled)")]
    colors = {"single": "#d62728", "temp": "#1f77b4", "ensemble": "#2ca02c", "ens+temp": "#9467bd"}
    for split, regime, methods, title in panels:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
        for j, m in enumerate(MODELS):
            for i, region in enumerate(("all", "near")):
                ax = axes[i, j]
                ax.plot([0.5, 1], [0.5, 1], color="0.6", lw=1, ls="--")
                ax2 = ax.twinx()
                for method in methods:
                    r = bins[(bins.split == split) & (bins.regime == regime) & (bins.model == m)
                             & (bins.method == method) & (bins.region == region) & (bins.n > 0)]
                    if r.empty:
                        continue
                    ax.plot(r.conf, r.acc, marker="o", ms=3, color=colors[method],
                            label=f"{method} (ECE {100 * ece_from_bins(r):.2f} %)")
                    if method == "single":
                        ax2.bar(r.lo + 1 / NB / 4, r.n / r.n.sum(), width=1 / NB / 2, color="0.85", zorder=0)
                ax2.set_yscale("log")
                ax2.set_ylim(1e-5, 1)
                ax2.tick_params(labelsize=7)
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
                ax.set_xlim(0.5, 1), ax.set_ylim(0.3, 1)
                ax.legend(fontsize=7, loc="lower right")
                ax.set_title(f"{MODELS[m]}, {'all labelled pixels' if region == 'all' else '≤ 10 px from water edge'}", fontsize=9)
                if i == 1:
                    ax.set_xlabel("confidence max(p, 1 − p)")
                if j == 0:
                    ax.set_ylabel("accuracy")
                if j == 2:
                    ax2.set_ylabel("share of pixels (bars, log)", fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        name = f"reliability_{split.replace('-', '')}_{regime}.png"
        fig.savefig(figs / name, dpi=130)
        plt.close(fig)
        print(f"wrote figs/{name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bins_rows = []
    t_valid = valid_part(bins_rows)
    dev_part(t_valid, bins_rows)
    bins = pd.concat(bins_rows, ignore_index=True)
    bins.to_csv(OUT / "reliability_bins.csv", index=False)
    plot(bins)


if __name__ == "__main__":
    main()
