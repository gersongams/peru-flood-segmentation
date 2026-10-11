import json

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from floodseg import provenance
from floodseg.callbacks import MPSMemoryGuard
from floodseg.provenance import RunProvenance


class Tiny(pl.LightningModule):
    def __init__(self, lr: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def _loader():
    return DataLoader(TensorDataset(torch.randn(8, 2), torch.randn(8, 1)), batch_size=4)


def _fit(tmp_path, monkeypatch, dirty: bool, limit: float):
    state = {"commit": "abc", "branch": "main", "dirty": dirty, "diff": "x" if dirty else "", "untracked": []}
    monkeypatch.setattr(provenance, "git_state", lambda: state)
    trainer = pl.Trainer(
        accelerator="cpu", max_epochs=1, default_root_dir=tmp_path, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        limit_train_batches=limit, callbacks=[RunProvenance()],
    )
    pl.seed_everything(7)
    trainer.fit(Tiny(), _loader())
    return tmp_path


def test_run_json_written(tmp_path, monkeypatch):
    run = json.loads((_fit(tmp_path, monkeypatch, dirty=False, limit=1.0) / "run.json").read_text())
    assert run["git"]["commit"] == "abc"
    assert run["seed"] == 7
    assert run["hparams"] == {"lr": 0.1}
    assert run["device"] == "cpu" and run["demo"] is False
    assert run["wall_clock_s"] >= 0


def test_full_run_refuses_dirty_tree(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="dirty"):
        _fit(tmp_path, monkeypatch, dirty=True, limit=1.0)


def test_demo_run_allows_dirty_tree_and_saves_diff(tmp_path, monkeypatch):
    out = _fit(tmp_path, monkeypatch, dirty=True, limit=0.5)
    assert (out / "git.diff").read_text() == "x"
    assert json.loads((out / "run.json").read_text())["demo"] is True


def test_mps_guard_aborts_over_limit(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 30 * 2**30)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 28 * 2**30)
    module = Tiny().to("mps")
    with pytest.raises(MemoryError, match="exceeds limit"):
        MPSMemoryGuard().on_train_batch_end(None, module, None, None, 0)


def test_mps_guard_ignores_other_devices():
    MPSMemoryGuard().on_train_batch_end(None, Tiny(), None, None, 0)
