import numpy as np

from floodseg.train_alignment import FDAToPool, HistMatchToPool


def params(tmp_path):
    q = np.linspace(0, 1, 1001)[:, None]
    f = tmp_path / "p.npz"
    rng = np.random.default_rng(0)
    np.savez(f, q_s1f=np.repeat(q * 1000, 13, 1), q_pool=np.repeat(q * 2000 + 100, 13, 1),
             amp_win=rng.uniform(1, 2, (4, 13, 11, 11)).astype(np.float32), beta=0.01)
    return str(f)


def test_hist_match_maps_quantiles_and_keeps_nodata(tmp_path):
    img = np.full((512, 512, 13), 500, np.float32)
    img[:10] = 0  # no-data rows
    out = HistMatchToPool(params=params(tmp_path)).apply(img)
    assert out.dtype == np.float32
    assert (out[:10] == 0).all()
    assert np.allclose(out[10:], 1100)  # 500 is the median of U(0, 1000) → median of U(100, 2100)


def test_band_subset_matches_full(tmp_path):
    p = params(tmp_path)
    img = np.random.default_rng(1).uniform(1, 1000, (64, 64, 13)).astype(np.float32)
    sub = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    full = HistMatchToPool(params=p).apply(img)
    part = HistMatchToPool(bands=sub, params=p).apply(img[..., [1, 2, 3, 8, 11, 12]])
    assert np.allclose(part, full[..., [1, 2, 3, 8, 11, 12]])


def test_fda_keeps_nodata_and_phase_shape(tmp_path):
    img = np.random.default_rng(2).uniform(100, 3000, (512, 512, 13)).astype(np.float32)
    img[:, :5] = 0
    t = FDAToPool(params=params(tmp_path))
    out = t.apply(img, k=1)
    assert out.shape == img.shape and (out[:, :5] == 0).all() and (out >= 0).all()
    assert FDAToPool(params=params(tmp_path), mean_amp=True).win.shape[0] == 1
