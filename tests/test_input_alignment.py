import numpy as np

from floodseg.input_alignment import amplitude, fda, hist_match, quantiles


def test_hist_match_maps_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(1000, 100, (2, 64, 64)).clip(1)
    b = rng.normal(3000, 300, (2, 20000)).clip(1)
    y = hist_match(a, quantiles(a.reshape(2, -1)), quantiles(b))
    assert abs(y.mean() - 3000) < 30 and abs(y.std() - 300) < 30
    assert np.all(np.diff(y[0].ravel()[np.argsort(a[0].ravel())]) >= 0)  # monotone


def test_nodata_kept():
    x = np.full((2, 32, 32), 500.0)
    x[:, :4] = 0
    q = quantiles(np.array([[1.0, 2000.0], [1.0, 2000.0]]))
    assert (hist_match(x, q, q * 2)[:, :4] == 0).all()
    assert (fda(x, amplitude(x) * 3)[:, :4] == 0).all()


def test_fda_identity_and_low_freq_only():
    rng = np.random.default_rng(1)
    x = rng.uniform(100, 2000, (1, 64, 64))
    np.testing.assert_allclose(fda(x, amplitude(x), beta=0.05), x, atol=1e-6)
    ref = amplitude(x).copy()
    ref[:, 32, 32] *= 2  # doubling the DC term doubles the mean, nothing else changes
    np.testing.assert_allclose(fda(x, ref, beta=0.05).mean(), 2 * x.mean(), rtol=1e-6)
