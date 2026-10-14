"""Doubtful-label chips in the UNOSAT 2023 Peru set (task 4.4). No model is used.

For every chip (train, dev, test, buffer), compares the label with the image's MNDWI
(B03 − B11) / (B03 + B11), the index of `scripts/build_peru_unosat.py:agreement()`:

  omission blob   labelled dry, MNDWI > 0.5 (clearly water-like)
  commission blob labelled water, MNDWI < −0.1 (clearly dry-looking)

Thresholds from the label-conditional MNDWI distributions of 300 random train/dev chips
(every 20th labelled pixel): labelled water has median 0.72, 1st percentile 0.05, and 73 %
of it is above 0.5; labelled dry has median −0.16, 99th percentile 0.21, and 0.03 % above
0.5. A first version used ±0.2, which sits inside the tail of dry land and flagged 45 % of
chips (half of the blob area next to cloud, i.e. haze); see docs/METHODS_NOTES.md.

Only labelled pixels more than 2 px from a label water edge count, so slight
misregistration along shores does not form blobs; blobs are 4-connected components of
at least 100 px (1 ha). Each blob is tagged by its mean NDVI (B08 − B04)/(B08 + B04)
("vegetated" if > 0.3: flooded vegetation for commission, irrigated/flooded fields for
omission) and by distance to thick cloud (same rule as the builder; "near cloud" if
< 20 px: shadow, haze, or bright surf that the cloud rule catches). A chip is **doubtful**
if its blobs cover ≥ 5 ha, **minor** if 1–5 ha (`flag`); `flag_no_cloud` counts only blobs
not next to cloud. Rule and thresholds were fixed before any comparison with model errors.

The list annotates the frozen v1 data; it does not filter it.

Writes outputs/label_qa/{chips.csv,blobs.csv} and, with --figs, overlays of the 20
largest doubtful train/dev chips to outputs/label_qa/figs/ (none for test).
--check (after the rule is fixed): on `peru-dev`, compares the flags with the errors of
the 9 Peru-only models (outputs/logits_dev/peru), by direction: omission blobs vs false
positives, commission blobs vs misses. Writes outputs/label_qa/check_dev.txt.

--report writes the tracked list docs/peru_doubtful_chips.md from chips.csv.

    uv run python scripts/label_qa.py [--figs] [--check] [--report]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "peru" / "unosat2023"
OUT = REPO / "outputs" / "label_qa"
T_WATER, T_DRY = 0.5, -0.1  # MNDWI beyond which a pixel is "clearly" water / dry
EDGE_PX = 2
MIN_BLOB = 100  # px = 1 ha
NDVI_VEG = 0.3
CLOUD_PX = 20
DOUBTFUL_HA, MINOR_HA = 5.0, 1.0


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(float), b.astype(float)
    return (a - b) / (a + b + 1e-6)


def thick_cloud(s2: np.ndarray) -> np.ndarray:
    """As build_peru_unosat.thick_cloud (B02 > 4500 and B11 < 1.15 · B02)."""
    blue, swir = s2[1].astype(float), s2[11].astype(float)
    return (blue > 4500) & (swir < 1.15 * blue)


def chip_qa(stem: str) -> tuple[dict, list[dict], dict]:
    with rasterio.open(DATA / "s2" / f"{stem}.tif") as src:
        s2 = src.read()
    with rasterio.open(DATA / "label" / f"{stem}.tif") as src:
        y = src.read(1)
    mndwi, ndvi = ratio(s2[2], s2[11]), ratio(s2[7], s2[3])
    ok = y >= 0
    water = y == 1
    edge = binary_dilation(water ^ binary_dilation(water), iterations=EDGE_PX)
    core = ok & ~edge
    cloud = thick_cloud(s2)
    d_cloud = distance_transform_edt(~cloud) if cloud.any() else np.full(y.shape, np.inf)
    masks = {}
    blobs = []
    for kind, cand in (("omission", core & (y == 0) & (mndwi > T_WATER)),
                       ("commission", core & water & (mndwi < T_DRY))):
        comp, n = cc_label(cand)
        size = np.bincount(comp.ravel(), minlength=n + 1)
        keep = np.flatnonzero(size >= MIN_BLOB)
        keep = keep[keep > 0]
        masks[kind] = np.isin(comp, keep)
        masks[f"{kind}_cloud"] = np.isin(comp, [k for k in keep if d_cloud[comp == k].min() < CLOUD_PX])
        for k in keep:
            m = comp == k
            blobs.append(dict(chip=stem, kind=kind, px=int(size[k]), ndvi=float(ndvi[m].mean()),
                              mndwi=float(mndwi[m].mean()), cloud_dist=float(d_cloud[m].min()),
                              cat="near cloud" if d_cloud[m].min() < CLOUD_PX
                              else "vegetated" if ndvi[m].mean() > NDVI_VEG else "open"))
    masks["dry_hi_other"] = (y == 0) & (mndwi > T_WATER) & ~masks["omission"]  # left out: < 1 ha or edge band
    lab_px = int(ok.sum())
    row = dict(chip=stem, labelled_px=lab_px, water_px=int(water.sum()),
               omission_px=int(masks["omission"].sum()), commission_px=int(masks["commission"].sum()))
    return row, blobs, masks


def overlay(stem: str, masks: dict, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(DATA / "s2" / f"{stem}.tif") as src:
        s2 = src.read()
    with rasterio.open(DATA / "label" / f"{stem}.tif") as src:
        y = src.read(1)
    rgb = np.stack([s2[12], s2[8], s2[3]], -1).astype(float)  # B12, B8A, B04 (as plot_dev_chips.py)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    lab = np.where(y == -1, 0.5, y.astype(float))
    qa = np.stack([lab] * 3, -1) * 0.6
    qa[masks["omission"]] = (0.9, 0.1, 0.1)
    qa[masks["commission"]] = (0.1, 0.4, 1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, img, t in zip(ax, (rgb, lab, qa), ("S2 B12/B8A/B04", "UNOSAT label (grey = ignore)",
                                              "red: labelled dry, looks water\nblue: labelled water, looks dry")):
        a.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.suptitle(f"{stem} ({split})")
    fig.tight_layout()
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figs" / f"{stem}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    split = pd.read_csv(DATA / "splits" / "split.csv")
    split["stem"] = split.date + "_" + split.chip
    rows, blobs = [], []
    for i, r in enumerate(split.itertuples()):
        row, b, _ = chip_qa(r.stem)
        rows.append(dict(row, date=r.date, split=r.split))
        blobs += b
        if i % 200 == 0:
            print(f"{i}/{len(split)}", flush=True)
    chips = pd.DataFrame(rows)
    bl = pd.DataFrame(blobs)
    for kind in ("omission", "commission"):
        for cat in ("open", "vegetated", "near cloud"):
            px = bl[(bl.kind == kind) & (bl.cat == cat)].groupby("chip").px.sum()
            chips[f"{kind}_{cat.replace(' ', '_')}_ha"] = chips.chip.map(px).fillna(0) / 100
    chips["flag_ha"] = (chips.omission_px + chips.commission_px) / 100
    chips["flag"] = np.where(chips.flag_ha >= DOUBTFUL_HA, "doubtful", np.where(chips.flag_ha >= MINOR_HA, "minor", "ok"))
    chips["flag_no_cloud_ha"] = chips.flag_ha - chips.omission_near_cloud_ha - chips.commission_near_cloud_ha
    chips["flag_no_cloud"] = np.where(chips.flag_no_cloud_ha >= DOUBTFUL_HA, "doubtful",
                                      np.where(chips.flag_no_cloud_ha >= MINOR_HA, "minor", "ok"))
    OUT.mkdir(parents=True, exist_ok=True)
    chips.to_csv(OUT / "chips.csv", index=False)
    bl.to_csv(OUT / "blobs.csv", index=False)
    print(chips.groupby(["split", "flag"]).size().unstack(fill_value=0))
    print(chips.groupby(["split", "flag_no_cloud"]).size().unstack(fill_value=0))
    if args.figs:
        top = chips[(chips.flag == "doubtful") & chips.split.isin(["train", "dev"])].nlargest(20, "flag_ha")
        for r in top.itertuples():
            overlay(r.chip, chip_qa(r.chip)[2], r.split)
        print(f"wrote {len(top)} overlays to {(OUT / 'figs').relative_to(REPO)}")
    if args.check:
        check_dev(chips)
    if args.report:
        report(chips)


def report(c: pd.DataFrame) -> None:
    def counts(col):
        t = c.groupby(["split", col]).size().unstack(fill_value=0).reindex(columns=["doubtful", "minor", "ok"], fill_value=0)
        return [f"| {sp} | {t.loc[sp, 'doubtful']} | {t.loc[sp, 'minor']} | {t.loc[sp, 'ok']} |"
                for sp in ("train", "dev", "test", "buffer")]

    area = {k: c[f"{k}_ha"].sum() for k in ("omission_open", "omission_vegetated", "omission_near_cloud",
                                              "commission_open", "commission_vegetated", "commission_near_cloud")}
    L = ["""# Doubtful-label chips, UNOSAT 2023 Peru set v1 (task 4.4)

Generated by `scripts/label_qa.py --report` (rule and thresholds: script docstring and
`docs/METHODS_NOTES.md`); source table `outputs/label_qa/chips.csv`, blobs
`outputs/label_qa/blobs.csv`. No model is used to flag a chip. **The list annotates the
frozen v1 data; nothing is removed.** Any exclusion would be a v2 decision taken before
the final evaluation (section 7) and applied to all models alike.

Rule in short: blobs of ≥ 1 ha, more than 2 px from a label water edge, where the label
says dry but MNDWI > 0.5 ("omission"), or says water but MNDWI < −0.1 ("commission").
Doubtful: blobs ≥ 5 ha; minor: 1–5 ha. Blob categories: "near cloud" (< 20 px from thick
cloud), "vegetated" (mean NDVI > 0.3), "open". A flag means a **clear label/MNDWI
disagreement**, not a proven label error:
- "near cloud" blobs (about half the flagged area) can be cloud shadow or haze that UNOSAT
  correctly labelled dry, or bright surf on the coast caught by the cloud rule (e.g. the sea
  edge of `2023-04-01_E517120_N9308160`); the "without near-cloud blobs" count is the
  stricter one;
- open blobs include clear cases such as a dark pond labelled dry
  (`2023-04-01_E588800_N9328640`, overlays in `outputs/label_qa/figs/`).
The rule is conservative: it misses water whose MNDWI is below 0.5 (turbid or shallow water),
so an unflagged chip is not guaranteed clean.

EMSR 2017 (`peru-2017-test`) is not screened: its labels already set EMS/index
disagreement to −1, so an index-agreement check would be circular.

## Counts, all blobs

| split | doubtful | minor | ok |
|---|---|---|---|"""]
    L += counts("flag")
    L += ["", "## Counts without near-cloud blobs", "", "| split | doubtful | minor | ok |", "|---|---|---|---|"]
    L += counts("flag_no_cloud")
    L += ["", "Flagged area over all chips (ha): " + ", ".join(f"{k.replace('_', ' ')} {v:.1f}" for k, v in area.items() if v > 0) + ".",
          "", "## Chips flagged doubtful or minor (all blobs)", "",
          "| chip | split | flag | flag w/o near-cloud | ha | omission open | omission near cloud | omission vegetated | commission | labelled water px |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    x = c[c.flag != "ok"].assign(order=lambda d: d.split.map({"train": 0, "dev": 1, "test": 2, "buffer": 3}))
    for r in x.sort_values(["order", "flag_ha"], ascending=[True, False]).itertuples():
        com = r.commission_open_ha + r.commission_vegetated_ha + r.commission_near_cloud_ha
        L.append(f"| {r.chip} | {r.split} | {r.flag} | {r.flag_no_cloud} | {r.flag_ha:.1f} | {r.omission_open_ha:.1f} | "
                 f"{r.omission_near_cloud_ha:.1f} | {r.omission_vegetated_ha:.1f} | {com:.1f} | {r.water_px:,} |")
    (REPO / "docs" / "peru_doubtful_chips.md").write_text("\n".join(L) + "\n")
    print("wrote docs/peru_doubtful_chips.md")


FINDING3 = ["2023-04-01_E517120_N9349120", "2023-04-01_E522240_N9349120", "2023-04-01_E527360_N9303040",
            "2023-03-22_E522240_N9354240", "2023-03-22_E527360_N9354240", "2023-03-22_E532480_N9318400",
            "2023-03-22_E512000_N9354240", "2023-04-01_E527360_N9313280", "2023-04-01_E512000_N9354240",
            "2023-03-22_E532480_N9349120"]  # docs/failure_analysis_dev.md, Finding 3


def check_dev(chips: pd.DataFrame) -> None:
    """Flags vs Peru-only model errors on dev (pixel level and per chip)."""
    from scipy.stats import spearmanr

    runs = sorted((REPO / "outputs" / "logits_dev" / "peru").glob("*/seed*"))
    dev = chips[chips.split == "dev"].set_index("chip")
    hit = {k: [0, 0] for k in ("omission, not near cloud", "omission, near cloud", "dry, MNDWI > 0.5, left out by the rule",
                                "dry other", "commission", "water other")}
    per = []
    for stem in dev.index:
        _, _, masks = chip_qa(stem)
        with rasterio.open(DATA / "label" / f"{stem}.tif") as src:
            y = src.read(1)
        pw = np.mean([np.load(r / f"{stem}.npy") > 0 for r in runs], axis=0)  # share of runs calling water
        dry_o = (y == 0) & ~masks["omission"] & ~masks["dry_hi_other"]
        wat_o = (y == 1) & ~masks["commission"]
        om_cl = masks["omission_cloud"]
        for k, m, val in (("omission, not near cloud", masks["omission"] & ~om_cl, pw), ("omission, near cloud", om_cl, pw),
                          ("dry, MNDWI > 0.5, left out by the rule", masks["dry_hi_other"], pw), ("dry other", dry_o, pw),
                          ("commission", masks["commission"], 1 - pw), ("water other", wat_o, 1 - pw)):
            hit[k][0] += val[m].sum()
            hit[k][1] += m.sum()
        lab = max(int((y >= 0).sum()), 1)
        per.append(dict(chip=stem, fp_rate=float(pw[y == 0].sum()) / lab, fn_rate=float((1 - pw)[y == 1].sum()) / lab,
                        om=dev.loc[stem, "omission_px"] / lab, co=dev.loc[stem, "commission_px"] / lab))
    per = pd.DataFrame(per)
    lines = ["Pixel level, mean over the 9 Peru-only runs:"]
    lines += [f"  {k}: {100 * a / n:.1f} % {'missed' if k in ('commission', 'water other') else 'called water'} ({n:,} px)"
              if n else f"  {k}: no pixels" for k, (a, n) in hit.items()]
    lines.append("  (omission and 'left out' pixels are high-MNDWI by construction: models that follow the"
                 " index call them water whether or not the label is wrong; compare the two)")
    rho_o = spearmanr(per.om, per.fp_rate)
    lines.append(f"Per dev chip, Spearman(omission share, FP rate) = {rho_o.statistic:.3f} (p = {rho_o.pvalue:.2g}, n = {len(per)})")
    if per.co.any():
        rho_c = spearmanr(per.co, per.fn_rate)
        lines.append(f"Per dev chip, Spearman(commission share, FN rate) = {rho_c.statistic:.3f} (p = {rho_c.pvalue:.2g})")
    lines.append("Finding-3 chips (failure analysis): " + "; ".join(
        f"{c[:10]}_{c[-15:]} {dev.loc[c, 'flag']} ({dev.loc[c, 'flag_ha']:.1f} ha)" for c in FINDING3))
    (OUT / "check_dev.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
