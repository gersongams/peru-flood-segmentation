from types import SimpleNamespace

import torch

from floodseg.callbacks import InitFromCheckpoint


def test_init_from_checkpoint_loads_weights(tmp_path):
    src, dst = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    path = tmp_path / "best.ckpt"
    torch.save({"state_dict": src.state_dict(), "epoch": 50}, path)
    InitFromCheckpoint(str(path)).setup(SimpleNamespace(ckpt_path=None), dst, "fit")
    assert torch.equal(dst.weight, src.weight) and torch.equal(dst.bias, src.bias)


def test_init_from_checkpoint_skips_resume_and_validate(tmp_path):
    src, dst = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    path = tmp_path / "best.ckpt"
    torch.save({"state_dict": src.state_dict()}, path)
    before = dst.weight.clone()
    cb = InitFromCheckpoint(str(path))
    cb.setup(SimpleNamespace(ckpt_path="last.ckpt"), dst, "fit")
    cb.setup(SimpleNamespace(ckpt_path=None), dst, "validate")
    assert torch.equal(dst.weight, before)
