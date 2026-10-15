import torch

from floodseg.self_training import IGNORE, dacs_mix, patch_mask, pseudo_labels, student_aug


def test_pseudo_labels_per_class_top_q():
    # 100 dry pixels with confidence 0.51..1.00, 10 water pixels with confidence 0.6..0.69
    conf_dry = torch.linspace(0.51, 0.999, 100)
    conf_w = torch.linspace(0.6, 0.69, 10)
    p_water = torch.cat([1 - conf_dry, conf_w]).view(1, 1, 1, 110)
    logits = torch.cat([torch.zeros_like(p_water), torch.logit(p_water.double()).float()], 1)
    y, cuts = pseudo_labels(logits, q=0.2)
    assert (y == 0).sum() == 20 and (y == 1).sum() == 2  # 20 % of each class, water not crowded out
    assert (y[..., 100:] == 1).sum() == 2 and y[0, 0, 108] == 1 and y[0, 0, 109] == 1
    assert abs(torch.sigmoid(torch.tensor(cuts[0])) - conf_dry[80]) < 1e-5 and abs(torch.sigmoid(torch.tensor(cuts[1])) - 0.68) < 1e-5
    assert (y == IGNORE).sum() == 88


def test_pseudo_labels_missing_class():
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 0] = 30.0  # saturated: all ties at probability 1.0
    y, cuts = pseudo_labels(logits, 0.5)
    assert (y == 0).sum() == 8 and cuts[1] != cuts[1]  # nan


def test_dacs_mix_pastes_source_water_only():
    sx, tx = torch.ones(2, 3, 4, 4), torch.zeros(2, 3, 4, 4)
    sy = torch.zeros(2, 4, 4, dtype=torch.long)
    sy[:, 0] = 1
    sy[:, 1] = -1
    tpl = torch.full((2, 4, 4), IGNORE)
    x, y = dacs_mix(sx, sy, tx, tpl, perm=torch.tensor([1, 0]))
    assert (x[:, :, 0] == 1).all() and (x[:, :, 1:] == 0).all()
    assert (y[:, 0] == 1).all() and (y[:, 1:] == IGNORE).all()


def test_patch_mask_ratio():
    g = torch.Generator().manual_seed(0)
    m = patch_mask(torch.ones(64, 1, 512, 512), 64, 0.7, gen=g)
    assert abs((m == 0).float().mean().item() - 0.7) < 0.03
    blocks = m[:, 0, ::64, ::64]
    assert torch.equal(m[:, 0], blocks.repeat_interleave(64, 1).repeat_interleave(64, 2))


def test_student_aug_shape_and_finite():
    x = torch.randn(4, 13, 32, 32)
    y = student_aug(x, gen=torch.Generator().manual_seed(0))
    assert y.shape == x.shape and torch.isfinite(y).all()
