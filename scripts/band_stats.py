"""Band distributions: Sen1Floods11 vs Peru (task 6.2). No model is used.

Compares the 13 Sentinel-2 L1C bands (DN = reflectance × 10,000) of Sen1Floods11 S2Hand
`train` (252 chips) with Peru UNOSAT 2023 `peru-train` and `peru-dev`, separately for
labelled water and labelled dry pixels (label −1 and all-zero no-data pixels skipped).
A fixed sample of every 37th labelled pixel per chip is used. Not `peru-test`.

Per band and class: mean, std, percentiles 1/5/50/95/99; shift of Peru vs Sen1Floods11 in
units of the Sen1Floods11 train std of that band (configs/stats/sen1floods11_train_s2.json,
what the U-Net sees after normalization): standardized mean difference and 1-D
Wasserstein distance. Also: water share of labelled pixels, and MNDWI (B03, B11) / NDWI
(B03, B08) percentiles of water pixels. Bands marked * are the 6 HLS bands Prithvi uses.

Writes outputs/band_stats/{bands.csv,indices.csv,summary.md} and figs/hist_*.png.

    uv run python scripts/band_stats.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[1]
S1F = REPO / "data" / "sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled"
SPLITS = REPO / "data" / "sen1floods11" / "v1.1" / "splits" / "flood_handlabeled"
PERU = REPO / "data" / "peru" / "unosat2023"
OUT = REPO / "outputs" / "band_stats"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
HLS = {"B02", "B03", "B04", "B8A", "B11", "B12"}
STEP = 37
Q = [1, 5, 50, 95, 99]


def ratio(a, b):
    return (a - b) / (a + b + 1e-6)


def sample(pairs: list[tuple[Path, Path]]) -> tuple[dict[str, np.ndarray], int, int]:
    xs = {"water": [], "dry": []}
    nw = nd = 0
    for img, lab in pairs:
        with rasterio.open(img) as src:
            x = src.read().astype(np.float32).reshape(13, -1)
        with rasterio.open(lab) as src:
            y = src.read(1).ravel()
        ok = (y >= 0) & np.isfinite(x).all(0) & (x != 0).any(0)
        nw += int((ok & (y == 1)).sum())
        nd += int((ok & (y == 0)).sum())
        idx = np.flatnonzero(ok)[::STEP]
        for cls, v in (("water", 1), ("dry", 0)):
            k = idx[y[idx] == v]
            xs[cls].append(x[:, k])
    return {c: np.concatenate(v, axis=1) for c, v in xs.items()}, nw, nd


def main() -> None:
    std = np.array(json.loads((REPO / "configs" / "stats" / "sen1floods11_train_s2.json").read_text())["stds"])
    ids = [r.split(",")[0].removesuffix("_S1Hand.tif") for r in (SPLITS / "flood_train_data.txt").read_text().split()]
    sets = {"Sen1Floods11 train": [(S1F / "S2Hand" / f"{i}_S2Hand.tif", S1F / "LabelHand" / f"{i}_LabelHand.tif") for i in ids]}
    split = pd.read_csv(PERU / "splits" / "split.csv")
    for sp in ("train", "dev"):
        stems = (split[split.split == sp].date + "_" + split[split.split == sp].chip).tolist()
        sets[f"peru-{sp}"] = [(PERU / "s2" / f"{s}.tif", PERU / "label" / f"{s}.tif") for s in stems]
    data, share = {}, {}
    for name, pairs in sets.items():
        data[name], nw, nd = sample(pairs)
        share[name] = (len(pairs), nw, nd)
        print(f"{name}: {len(pairs)} chips, water share {nw / (nw + nd):.3f}", flush=True)
    ref = "Sen1Floods11 train"
    rows, irows = [], []
    for name, d in data.items():
        for cls, x in d.items():
            for b, band in enumerate(BANDS):
                v = x[b]
                r = dict(dataset=name, cls=cls, band=band + ("*" if band in HLS else ""), n=v.size, mean=v.mean(), std=v.std(),
                         **{f"p{q}": p for q, p in zip(Q, np.percentile(v, Q))})
                if name != ref:
                    u = data[ref][cls][b]
                    r["smd"] = (v.mean() - u.mean()) / std[b]
                    r["w1_std"] = wasserstein_distance(u, v) / std[b]
                rows.append(r)
            for iname, idx in (("MNDWI", ratio(x[2], x[11])), ("NDWI", ratio(x[2], x[7]))):
                irows.append(dict(dataset=name, cls=cls, index=iname,
                                  **{f"p{q}": p for q, p in zip([5, 25, 50, 75, 95], np.percentile(idx, [5, 25, 50, 75, 95]))}))
    OUT.mkdir(parents=True, exist_ok=True)
    bands, ind = pd.DataFrame(rows), pd.DataFrame(irows)
    bands.to_csv(OUT / "bands.csv", index=False)
    ind.to_csv(OUT / "indices.csv", index=False)

    lines = ["## Water share of labelled pixels\n", "| dataset | chips | water px | dry px | water share |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {c} | {w:,} | {d:,} | {w / (w + d):.3f} |" for k, (c, w, d) in share.items()]
    for cls in ("water", "dry"):
        lines += [f"\n## Bands, {cls} pixels: median (p5–p95) DN; shift vs Sen1Floods11 in Sen1Floods11-std units "
                  "(SMD = standardized mean difference, W1 = Wasserstein)\n",
                  "| band | " + " | ".join(sets) + " | SMD peru-train | W1 peru-train | SMD peru-dev | W1 peru-dev |",
                  "|---" * (len(sets) + 5) + "|"]
        for band in bands.band.unique():
            t = bands[(bands.cls == cls) & (bands.band == band)].set_index("dataset")
            cells = [f"{t.loc[n, 'p50']:.0f} ({t.loc[n, 'p5']:.0f}–{t.loc[n, 'p95']:.0f})" for n in sets]
            cells += [f"{t.loc[n, k]:+.2f}" if k == "smd" else f"{t.loc[n, k]:.2f}"
                      for n in ("peru-train", "peru-dev") for k in ("smd", "w1_std")]
            lines.append(f"| {band} | " + " | ".join(cells) + " |")
    lines += ["\n## Water indices: median (p5–p95)\n", "| index | class | " + " | ".join(sets) + " |", "|---" * (len(sets) + 2) + "|"]
    for (iname, cls), t in ind.groupby(["index", "cls"], sort=False):
        t = t.set_index("dataset")
        lines.append(f"| {iname} | {cls} | " + " | ".join(f"{t.loc[n, 'p50']:.2f} ({t.loc[n, 'p5']:.2f}–{t.loc[n, 'p95']:.2f})" for n in sets) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    (OUT / "figs").mkdir(exist_ok=True)
    for cls in ("water", "dry"):
        fig, axes = plt.subplots(3, 5, figsize=(17, 9))
        for b, ax in enumerate(axes.flat):
            if b >= 13:
                ax.axis("off")
                continue
            hi = max(np.percentile(data[n][cls][b], 99.5) for n in sets)
            for n, col in zip(sets, ("#1f77b4", "#d62728", "#ff7f0e")):
                ax.hist(data[n][cls][b], bins=80, range=(0, hi), density=True, histtype="step", color=col, label=n)
            ax.set_title(BANDS[b] + (" (HLS)" if BANDS[b] in HLS else ""), fontsize=9)
            ax.set_yticks([])
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Band DN distributions, {cls} pixels (every {STEP}th labelled pixel)")
        fig.tight_layout()
        fig.savefig(OUT / "figs" / f"hist_{cls}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
