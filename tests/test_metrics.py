import numpy as np

from floodseg.metrics import counts, masks, scores


def test_tolerant_mask_drops_edge_band():
    y = np.zeros((9, 9), np.int64)
    y[:, :4] = 1  # water in columns 0-3, edge pixels in columns 3 and 4
    y[0, 8] = -1
    plain, tol = masks(y)
    assert plain.sum() == 80  # ignore excluded
    assert not tol[:, 2:6].any()  # edge columns 3, 4 and their neighbours 2, 5 dropped
    assert tol[:, :2].all() and tol[1:, 6:].all()


def test_counts_and_scores():
    y = np.zeros((9, 9), np.int64)
    y[:, :4] = 1
    pred = np.zeros((9, 9), bool)
    pred[:, :5] = True  # one column too wide: fp only on the edge column 4
    c = counts(pred, y)
    assert c[0].tolist() == [36, 9, 0] and c[1].tolist() == [18, 0, 0]
    s = scores(c)
    assert s["iou"] == 36 / 45 and s["recall"] == 1 and s["iou_tol"] == 1
