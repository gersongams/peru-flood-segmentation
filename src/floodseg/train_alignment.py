"""Training-time input alignment Sen1Floods11 → Peru pool (phase-D task 3.4 part 2; design.md
decision 7; details pre-registered in docs/METHODS_NOTES.md 2026-10-15).

Albumentations image-only transforms applied to the raw Sentinel-2 L1C DN of each source chip
(TerraTorch hands them channels-last after NaN → no_data_replace and band filtering, before the
model normalization). Pixels with all bands 0 (no data) stay 0. Parameters come from
`outputs/phaseD/alignment/train_params.npz` (scripts/build_train_alignment.py), fitted on images
only: no label and no dev/test image is involved.
- HistMatchToPool: per band, the global map v → F_pool⁻¹(F_s1f(v)) (1,001 quantiles each), the
  inverse direction of the test-time histogram matching.
- FDAToPool: per band, the centred low-frequency amplitude square (half-width ⌊β·min(H, W)⌋)
  of the chip is replaced by that of a random pool image (phase kept), β = 0.01. With
  `mean_amp=True` (validation) the mean pool amplitude is used instead, so validation is
  deterministic (as the test-time FDA, which used a mean amplitude).
`bands` names the bands the transform receives (all 13 for U-Net/TerraMind, Prithvi's 6).
"""

from pathlib import Path

import albumentations as A
import numpy as np

S2_BANDS = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
            "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"]
REPO = Path(__file__).resolve().parents[2]
PARAMS = "outputs/phaseD/alignment/train_params.npz"


def _load(params: str) -> dict:
    p = Path(params)
    return dict(np.load(p if p.is_absolute() else REPO / p))


def _idx(bands: list[str] | None) -> list[int]:
    return list(range(len(S2_BANDS))) if bands is None else [S2_BANDS.index(b) for b in bands]


class HistMatchToPool(A.ImageOnlyTransform):
    def __init__(self, bands: list[str] | None = None, params: str = PARAMS, p: float = 1.0):
        super().__init__(p=p)
        prm, idx = _load(params), _idx(bands)
        self.q_from, self.q_to = prm["q_s1f"][:, idx], prm["q_pool"][:, idx]

    def apply(self, img, **params):  # img: (H, W, C) float32
        nodata = ~(img != 0).any(-1)
        out = np.stack([np.interp(img[..., b].astype(np.float64), self.q_from[:, b], self.q_to[:, b])
                        for b in range(img.shape[-1])], axis=-1)
        out[nodata] = 0
        return out.astype(np.float32)


class FDAToPool(A.ImageOnlyTransform):
    def __init__(self, bands: list[str] | None = None, params: str = PARAMS, beta: float = 0.01,
                 mean_amp: bool = False, p: float = 1.0):
        super().__init__(p=p)
        prm, idx = _load(params), _idx(bands)
        self.win = prm["amp_win"][:, idx]  # (n_pool, C, 2b+1, 2b+1) centred low-frequency amplitudes
        if mean_amp:
            self.win = self.win.mean(0, keepdims=True)
        self.beta = beta

    def get_params(self):
        return {"k": int(self.random_generator.integers(len(self.win)))}

    def apply(self, img, k=0, **params):
        x = np.moveaxis(img, -1, 0).astype(np.float64)
        nodata = ~(x != 0).any(0)
        h, w = x.shape[-2:]
        b = int(np.floor(self.beta * min(h, w)))
        assert self.win.shape[-1] == 2 * b + 1, (self.win.shape, b)
        f = np.fft.fftshift(np.fft.fft2(x), axes=(-2, -1))
        amp, pha = np.abs(f), np.angle(f)
        ch, cw = h // 2, w // 2
        amp[:, ch - b:ch + b + 1, cw - b:cw + b + 1] = self.win[k]
        y = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha), axes=(-2, -1))))
        y = np.clip(y, 0, None)
        y[:, nodata] = 0
        return np.moveaxis(y, 0, -1).astype(np.float32)
