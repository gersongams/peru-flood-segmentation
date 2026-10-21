"""Check that the system's scene-level EMS 2017 reference reproduces the 2017 chip labels.

For every peru-2017-test chip inside a processed 2017-04-07 scene, compares the chip label
(data/peru/emsr2017/label: 1 water, 0 dry, -1 ignore) with the same 512 × 512 window of the
scene reference (floodseg.system.reference: 1, 0, 255) where the system's cloud / no-data mask is
valid. The chip labels also drop thick cloud, so pixels are compared only where both are scored,
and the share of pixels scored by one side only is reported.
→ outputs/system_eval/rf/rf5_ref2017.json

    uv run python scripts/system_ref2017_check.py 17MNP_2017-04-07 17MNQ_2017-04-07
"""

import json
import sys

import numpy as np
import pandas as pd
import rasterio

from floodseg.system.pipeline import scene_dir
from floodseg.system.reference import EMS_MANIFEST, IGNORE, metrics, reference
from floodseg.target_pool import REPO


def main() -> None:
    m = pd.read_csv(EMS_MANIFEST)
    out = {}
    for scene in sys.argv[1:]:
        d = scene_dir(*scene.split("_"))
        ref = reference(scene)
        with rasterio.open(d / "mask.tif") as src:
            mask, t = src.read(1), src.transform
        n = same = only_chip = only_scene = chips = 0
        for chip, x0, y0 in m[["chip", "minx", "miny"]].itertuples(index=False):
            c0, r0 = int(round((x0 - t.c) / t.a)), int(round((t.f - (y0 + 5120)) / -t.e))
            if c0 < 0 or r0 < 0 or c0 + 512 > ref.shape[1] or r0 + 512 > ref.shape[0]:
                continue
            chips += 1
            with rasterio.open(EMS_MANIFEST.parent / "label" / f"{chip}.tif") as src:
                lab = src.read(1)
            r = ref[r0:r0 + 512, c0:c0 + 512]
            ok = mask[r0:r0 + 512, c0:c0 + 512] == 0
            a, b = (lab != -1) & ok, (r != IGNORE) & ok
            both = a & b
            n += int(both.sum())
            same += int((both & ((lab == 1) == (r == 1))).sum())
            only_chip += int((a & ~b).sum())
            only_scene += int((b & ~a).sum())
        tile = metrics(scene, part="test")
        out[scene] = {"chips": chips, "pixels_both": n, "agreement": same / n if n else None,
                      "only_chip_label": only_chip, "only_scene_reference": only_scene,
                      "scene_metrics_test_chips": {k: tile[k] for k in ("pixels", "iou", "precision", "recall")}}
        print(scene, json.dumps(out[scene]), flush=True)
    dest = REPO / "outputs" / "system_eval" / "rf"
    (dest / "rf5_ref2017.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
