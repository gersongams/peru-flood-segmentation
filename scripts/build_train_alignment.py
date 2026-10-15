"""Parameters for training-time input alignment (floodseg.train_alignment), from images only.

- q_s1f, q_pool: the per-band 1,001-quantile tables of the test-time histogram matching
  (outputs/phaseD/alignment/params.npz, scripts/input_alignment_dev.py: Sen1Floods11 `train`
  images and the unlabelled `peru-train` pool), reused unchanged.
- amp_win: for each of the 753 pool images, the centred low-frequency amplitude square of
  every band, half-width ⌊0.01 · 512⌋ = 5 → (753, 13, 11, 11), read under the Peru-label guard.
Writes outputs/phaseD/alignment/train_params.npz (small enough to upload; no image leaves the Mac).

    uv run python scripts/build_train_alignment.py
"""

import numpy as np
import rasterio

from floodseg.input_alignment import amplitude
from floodseg.target_pool import REPO, forbid_peru_labels, read_manifest

BETA = 0.01
OUT = REPO / "outputs" / "phaseD" / "alignment"


def main() -> None:
    prm = np.load(OUT / "params.npz")
    wins = []
    with forbid_peru_labels():
        paths = read_manifest("peru-train")
        for p in paths:
            with rasterio.open(p) as src:
                x = src.read()
            h, w = x.shape[-2:]
            b = int(np.floor(BETA * min(h, w)))
            ch, cw = h // 2, w // 2
            wins.append(amplitude(x)[:, ch - b:ch + b + 1, cw - b:cw + b + 1].astype(np.float32))
    amp_win = np.stack(wins)
    np.savez(OUT / "train_params.npz", q_s1f=prm["q_s1f"], q_pool=prm["q_pool"], amp_win=amp_win, beta=BETA)
    print(f"pool images {len(paths)}, amp_win {amp_win.shape}, wrote {(OUT / 'train_params.npz').relative_to(REPO)}")


if __name__ == "__main__":
    main()
