"""Share of pixels and of errors per stratum on peru-dev (Chapter III, HE2 strata).

Reads outputs/phaseD/analysis/strata.csv (scripts/phaseD_analysis.py). Errors per stratum =
water_px * (1 - recall) + (pixels - water_px) * fpr, with the seed-mean recall and FPR, so the
counts are seed-mean error counts. Shares are within one stratum family (the values of a family
partition the labelled dev pixels).
"""
import sys
from pathlib import Path

import pandas as pd

FAMILIES = ["landcover", "distance to thick cloud", "distance to water edge"]
REGIMES = ["s1f11_src", "peru", "trhm", "trfda", "adabn", "hm"]


def main(src: Path, out: Path) -> None:
    d = pd.read_csv(src)
    d["errors"] = d.water_px * (1 - d.recall) + (d.pixels - d.water_px) * d.fpr
    d = d[d.stratum.isin(FAMILIES) & d.regime.isin(REGIMES)].copy()
    g = d.groupby(["stratum", "model", "regime"])
    d["pixel_share"] = d.pixels / g.pixels.transform("sum")
    d["error_share"] = d.errors / g.errors.transform("sum")
    d["concentration"] = d.error_share / d.pixel_share
    cols = ["stratum", "value", "model", "regime", "pixels", "errors", "pixel_share", "error_share", "concentration"]
    d[cols].sort_values(cols[:4]).to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out} ({len(d)} rows)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/phaseD/analysis/strata.csv"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/phaseD/analysis/error_shares.csv"))
