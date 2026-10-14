"""Tables from scripts/calibration_dev.py outputs, as Markdown.

    uv run python scripts/report_calibration.py > outputs/analysis_dev/calib_tables.md
"""

import json
from pathlib import Path

import pandas as pd

from report_dev import MODELS, ORDER, REGIMES, fmt

A = Path(__file__).resolve().parents[1] / "outputs" / "analysis_dev"
MIN_ERR = 1000  # cells with fewer errors (mean over seeds) get no AUROC and are marked
# input-defined strata first; the label-defined ones (edge distance, water-only strata)
# partly restate the error rate, because they are conditioned on the label
STRATA = ["NDWI (all pixels)", "landcover", "distance to thick cloud", "date",
          "distance to water edge", "water NDWI", "water body size", "water type"]
FN_ONLY = {"water NDWI", "water body size", "water type"}


def main() -> None:
    v = pd.read_csv(A / "calib_valid.csv")
    t = json.loads((A / "temperatures_valid.json").read_text())
    s = pd.read_csv(A / "calib_strata.csv")

    print("## Sen1Floods11 `valid`, Sen1Floods11 models (seed 0)\n")
    print("| Model | T (fit on all valid) | IoU | ECE all % raw → temp | ECE near % raw → temp | AUROC all | AUROC near |")
    print("|---|---|---|---|---|---|---|")
    for m, name in MODELS.items():
        x = v[v.model == m].set_index(["method", "region"])
        print(f"| {name} | {t[m]:.2f} | {x.loc[('single', 'all'), 'iou']:.4f} "
              f"| {100 * x.loc[('single', 'all'), 'ece']:.2f} → {100 * x.loc[('temp', 'all'), 'ece']:.2f} "
              f"| {100 * x.loc[('single', 'near'), 'ece']:.2f} → {100 * x.loc[('temp', 'near'), 'ece']:.2f} "
              f"| {x.loc[('single', 'all'), 'auroc']:.3f} | {x.loc[('single', 'near'), 'auroc']:.3f} |")
    print("\n`temp` on valid: T fitted 2-fold by chip, so no pixel is scored with a T fitted on it.\n")

    print("## `peru-dev`, Sen1Floods11-only (seed 0) with the temperature from `valid`\n")
    print("| Model | ECE all % raw → T_valid | ECE near % raw → T_valid |")
    print("|---|---|---|")
    o = s[(s.regime == "s1f11") & (s.stratum == "overall")].set_index(["model", "method", "value"])
    for m, name in MODELS.items():
        print(f"| {name} | {100 * o.loc[(m, 'single', 'all'), 'ece']:.2f} → {100 * o.loc[(m, 'temp', 'all'), 'ece']:.2f} "
              f"| {100 * o.loc[(m, 'single', 'near'), 'ece']:.2f} → {100 * o.loc[(m, 'temp', 'near'), 'ece']:.2f} |")
    print()

    for regime in ("peru", "s1f11_ft"):
        r = s[s.regime == regime]
        print(f"## Per-stratum calibration on `peru-dev`, {REGIMES[regime]}\n")
        print("ECE %: single seed (mean ± std over 3 seeds) → seed ensemble temperature-scaled 2-fold. "
              "AUROC: error detection, single seed (1 − max(p, 1 − p)) / seed ensemble (entropy). "
              f"Pixels and errors: U-Net, mean over seeds. † that model has fewer than {MIN_ERR:,} errors "
              "in the cell (mean over seeds): AUROC not shown, ECE unreliable.\n")
        for st in ["overall"] + STRATA:
            x = r[r.stratum == st]
            if st in FN_ONLY:
                x = x[x.value != "dry"]
            if x.empty:
                continue
            note = " (true water only, so every error is a miss)" if st in FN_ONLY else ""
            print(f"### {st}{note}\n")
            print("| value | pixels | errors | " + " | ".join(f"{n} ECE % | {n} AUROC" for n in MODELS.values()) + " |")
            print("|---" * (3 + 2 * len(MODELS)) + "|")
            values = ORDER.get(st) or list(x.groupby("value").pixels.mean().sort_values(ascending=False).index)
            for val in values:
                xv = x[x.value == val]
                if xv.empty:
                    continue
                u = xv[(xv.model == "unet") & (xv.method == "single")]
                px, err = u.pixels.mean(), u.errors.mean()
                cells = []
                for m in MODELS:
                    xm = xv[xv.model == m]
                    single, ens = xm[xm.method == "single"], xm[xm.method == "ensemble"]
                    enst = xm[xm.method == "ens+temp"]
                    few = single.errors.mean() < MIN_ERR
                    cells.append(f"{fmt(single.ece, True)} → {fmt(enst.ece, True)}{' †' if few else ''}")
                    cells.append("—" if few else f"{fmt(single.auroc, False)} / {fmt(ens.auroc, False)}")
                print(f"| {val} | {px:,.0f} | {err:,.0f} | " + " | ".join(cells) + " |")
            print()


if __name__ == "__main__":
    main()
