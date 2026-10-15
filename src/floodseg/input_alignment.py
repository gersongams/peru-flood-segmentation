"""Label-free input alignment between Peru and Sen1Floods11 (phase-D task 3.4).

Both work on raw Sentinel-2 L1C DN with all 13 bands, before the model's normalization;
pixels with all bands 0 (no data) stay 0. Parameters are fitted on images only (the target
pool and the Sen1Floods11 `train` images), never on dev/test images.
- Histogram matching: per band, the global monotone map v → F_ref⁻¹(F_from(v)) between two
  empirical CDFs, stored as 1,001 quantiles each.
- FDA (Yang & Soatto, CVPR 2020), per band: the low-frequency amplitude of the centred
  spectrum (a square of half-width ⌊β·min(H, W)⌋) is replaced by a reference amplitude; the
  phase is kept.
"""

from pathlib import Path

import numpy as np
import rasterio

N_QUANTILES = 1001
STEP = 50


def sample_pixels(paths: list[Path], step: int = STEP) -> np.ndarray:
    """(bands, N) raw DN of every `step`-th pixel with a non-zero band, over all images."""
    out = []
    for p in paths:
        with rasterio.open(p) as src:
            x = src.read().astype(np.float64)
        out.append(x[:, (x != 0).any(0)][:, ::step])
    return np.concatenate(out, axis=1)


def quantiles(samples: np.ndarray, n: int = N_QUANTILES) -> np.ndarray:
    return np.quantile(samples, np.linspace(0, 1, n), axis=1)  # (n, bands)


def hist_match(x: np.ndarray, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Map each band of a raw image (bands, H, W) from the `q_from` to the `q_to` distribution."""
    nodata = ~(x != 0).any(0)
    y = np.stack([np.interp(x[b].astype(np.float64), q_from[:, b], q_to[:, b]) for b in range(x.shape[0])])
    y[:, nodata] = 0
    return y


def amplitude(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fftshift(np.fft.fft2(x.astype(np.float64)), axes=(-2, -1)))


def mean_amplitude(paths: list[Path]) -> np.ndarray:
    acc = None
    for p in paths:
        with rasterio.open(p) as src:
            a = amplitude(src.read())
        acc = a if acc is None else acc + a
    return acc / len(paths)


def fda(x: np.ndarray, amp_ref: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """Replace the low-frequency amplitude of each band of x (bands, H, W) by `amp_ref`."""
    nodata = ~(x != 0).any(0)
    f = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)), axes=(-2, -1))
    amp, pha = np.abs(f), np.angle(f)
    h, w = x.shape[-2:]
    b = int(np.floor(beta * min(h, w)))
    ch, cw = h // 2, w // 2
    amp[:, ch - b:ch + b + 1, cw - b:cw + b + 1] = amp_ref[:, ch - b:ch + b + 1, cw - b:cw + b + 1]
    y = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha), axes=(-2, -1))))
    y = np.clip(y, 0, None)
    y[:, nodata] = 0
    return y
