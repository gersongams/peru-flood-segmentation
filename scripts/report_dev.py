"""Tables (mean ± std over seeds) from outputs/analysis_dev/*.csv, as Markdown.

    uv run python scripts/report_dev.py > outputs/analysis_dev/tables.md
"""

from pathlib import Path

import pandas as pd

A = Path(__file__).resolve().parents[1] / "outputs" / "analysis_dev"
MODELS = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}
REGIMES = {"peru": "Peru-only", "s1f11_ft": "S1F11→Peru", "s1f11": "S1F11-only"}
ORDER = {
    "water type": ["existing water", "new flood water"],
    "NDWI (all pixels)": ["< -0.1", "-0.1-0", "0-0.1", "0.1-0.3", "> 0.3"],
    "water NDWI": ["< -0.1", "-0.1-0", "0-0.1", "0.1-0.3", "> 0.3"],
    "water body size": ["<1 ha", "1-10 ha", "10-100 ha", ">100 ha"],
    "distance to water edge": ["edge (≤1 px)", "2-3 px", "4-10 px", ">10 px"],
    "distance to thick cloud": ["<5 px", "5-20 px", "20-50 px", ">50 px"],
    "elevation": ["<10 m", "10-50 m", "50-200 m", ">200 m"],
    "slope": ["<2°", "2-5°", "5-15°", ">15°"],
}
# Which metric is meaningful per stratum: recall where the stratum splits true water,
# false-positive rate where it splits dry land, IoU where it splits all pixels.
METRIC = {"water type": "recall", "water NDWI": "recall", "water body size": "recall", "landcover": "fpr",
          "elevation": "fpr", "slope": "fpr", "date": "iou",
          "distance to water edge": "iou", "NDWI (all pixels)": "fpr", "distance to thick cloud": "iou"}


def fmt(s: pd.Series, pct: bool) -> str:
    k = 100 if pct else 1
    m, sd, n = s.mean() * k, s.std() * k, len(s)
    d = 2 if pct else 3
    return f"{m:.{d}f}" if n == 1 else f"{m:.{d}f} ± {sd:.{d}f}"


def cols(df):
    return [(r, m) for r in REGIMES for m in MODELS if ((df.regime == r) & (df.model == m)).any()]


def main() -> None:
    s = pd.read_csv(A / "strata.csv")
    c = pd.read_csv(A / "calibration.csv")
    ch = pd.read_csv(A / "chips.csv")
    cs = cols(s)
    head = "| | " + " | ".join(f"{REGIMES[r]} {MODELS[m]}" for r, m in cs) + " |"
    sep = "|---" * (len(cs) + 1) + "|"
    n = ch.groupby(["regime", "model"]).seed.nunique()
    print("Seeds per column: " + ", ".join(f"{REGIMES[r]} {MODELS[m]} {n[(r, m)]}" for r, m in cs) + "\n")

    print("## Overall dev water IoU\n")
    print(head); print(sep)
    g = ch.groupby(["regime", "model", "seed"])[["tp", "fp", "fn"]].sum()
    iou = (g.tp / (g.tp + g.fp + g.fn)).rename("iou").reset_index()
    print("| water IoU | " + " | ".join(fmt(iou[(iou.regime == r) & (iou.model == m)].iou, False) for r, m in cs) + " |\n")

    for name, metric in METRIC.items():
        t = s[s.stratum == name]
        if name in ("water type", "water body size", "water NDWI"):
            t = t[t.value != "dry"]
        pct = metric != "iou"
        unit = {"recall": "recall, % of true water found", "fpr": "false-positive rate, % of dry pixels called water",
                "iou": "water IoU"}[metric]
        values = ORDER.get(name) or list(t.groupby("value").pixels.mean().sort_values(ascending=False).index)
        print(f"## {name} ({unit})\n")
        print("| " + name + " (dev pixels, water pixels) | " + head[4:]); print(sep)
        for v in values:
            tv = t[t.value == v]
            if tv.empty:
                continue
            px = tv.pixels.iloc[0]; wp = tv.water_px.iloc[0]
            if metric == "fpr" and px - wp < 20000:
                continue
            cells = [fmt(tv[(tv.regime == r) & (tv.model == m)][metric], pct) for r, m in cs]
            print(f"| {v} ({px:,}, {wp:,}) | " + " | ".join(cells) + " |")
        print()

    print("## Calibration (ECE %, error-detection AUROC)\n")
    print(head); print(sep)
    print("| ECE (%) | " + " | ".join(fmt(c[(c.regime == r) & (c.model == m)].ece, True) for r, m in cs) + " |")
    print("| error AUROC | " + " | ".join(fmt(c[(c.regime == r) & (c.model == m)].error_auroc, False) for r, m in cs) + " |")
    print("| ECE (%), ≤ 10 px from water edge | " + " | ".join(fmt(c[(c.regime == r) & (c.model == m)].ece_near_water, True) for r, m in cs) + " |")
    print("| error AUROC, ≤ 10 px from water edge | " + " | ".join(fmt(c[(c.regime == r) & (c.model == m)].error_auroc_near_water, False) for r, m in cs) + " |\n")

    print("## Errors concentrated in few chips (Peru-only, summed over models and seeds)\n")
    p = ch[ch.regime == "peru"].groupby("chip")[["fp", "fn", "water_px"]].sum()
    p["err"] = p.fp + p.fn
    tot = p.err.sum()
    top = p.sort_values("err", ascending=False)
    share10 = top.err.iloc[:10].sum() / tot
    print(f"Top 10 of {len(p)} chips hold {share10:.0%} of all errors.\n")
    print("| chip | FP | FN | water px (summed) |"); print("|---|---|---|---|")
    for chip, r in top.head(10).iterrows():
        print(f"| {chip} | {r.fp:,} | {r.fn:,} | {r.water_px:,} |")


if __name__ == "__main__":
    main()
