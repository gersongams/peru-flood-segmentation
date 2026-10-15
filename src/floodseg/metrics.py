"""Pixel counts for the phase-D metrics (spec `adaptation-evaluation`, requirement Metrics).

Plain water IoU over labelled pixels (label != −1), and the boundary-tolerant water IoU that
also leaves out labelled pixels within 1 px of a label water edge. Edge pixels: pixels on
either side of a water/not-water boundary (ignore counts as not water when finding it, as in
scripts/analyze_dev.py and scripts/edge_tolerance.py; unlike them, the chip frame is not a
boundary). "Within 1 px" = Euclidean distance to an edge pixel < 1.5, so the edge pixels
themselves and their 8-neighbours are excluded.
Counts are pooled over chips by summing, then turned into IoU, precision and recall.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

TOLERANCE_PX = 1


def masks(y: np.ndarray, k: int = TOLERANCE_PX) -> tuple[np.ndarray, np.ndarray]:
    """(plain, tolerant) boolean masks of the pixels scored, for a label array y in {−1, 0, 1}."""
    w = y == 1
    # border_value=1: the chip frame is not an edge (with the default 0 every border pixel
    # would be one; scripts/edge_tolerance.py and analyze_dev.py have that, see METHODS_NOTES)
    edge = (w ^ binary_erosion(w, border_value=1)) | (~w ^ binary_erosion(~w, border_value=1))
    ok = y != -1
    return ok, ok & (distance_transform_edt(~edge) >= k + 0.5)


def counts(pred: np.ndarray, y: np.ndarray, k: int = TOLERANCE_PX) -> np.ndarray:
    """[tp, fp, fn] for plain, then tolerant: shape (2, 3). pred is boolean water."""
    w = y == 1
    out = np.zeros((2, 3), np.int64)
    for i, m in enumerate(masks(y, k)):
        p, t = pred[m], w[m]
        out[i] = ((p & t).sum(), (p & ~t).sum(), (~p & t).sum())
    return out


def scores(c: np.ndarray) -> dict:
    """IoU, precision, recall from pooled counts of `counts` (plain and tolerant)."""
    (tp, fp, fn), (ttp, tfp, tfn) = c
    return dict(iou=tp / (tp + fp + fn), precision=tp / (tp + fp), recall=tp / (tp + fn),
                iou_tol=ttp / (ttp + tfp + tfn), precision_tol=ttp / (ttp + tfp), recall_tol=ttp / (ttp + tfn))
