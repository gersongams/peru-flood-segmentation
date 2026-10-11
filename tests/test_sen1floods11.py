import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from terratorch.tasks.segmentation_tasks import init_loss
from torchmetrics.classification import MulticlassJaccardIndex

from floodseg import normalization
from floodseg.sen1floods11 import check_chip, official_ids


def _write_csv(d, split, n):
    rows = "".join(f"C_{i}_S1Hand.tif,C_{i}_LabelHand.tif\n" for i in range(n))
    (d / f"flood_{split}_data.csv").write_text(rows)


def test_official_sizes_enforced(tmp_path):
    _write_csv(tmp_path, "valid", 88)
    with pytest.raises(ValueError, match="expected 89"):
        official_ids("valid", tmp_path)
    _write_csv(tmp_path, "valid", 89)
    assert official_ids("valid", tmp_path)[0] == "C_0"


def test_bolivia_rejected(tmp_path):
    with pytest.raises(ValueError, match="excluded"):
        official_ids("bolivia", tmp_path)


def _tif(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=arr.dtype, transform=from_origin(0, 0, 10, 10),
    ) as dst:
        dst.write(arr)


def test_check_chip_flags_bad_labels_and_bands(tmp_path):
    _tif(tmp_path / "S2Hand" / "X_S2Hand.tif", np.zeros((12, 4, 4), np.int16))
    _tif(tmp_path / "LabelHand" / "X_LabelHand.tif", np.full((1, 4, 4), 2, np.int16))
    problems = check_chip("X", tmp_path)
    assert any("12 bands" in p for p in problems)
    assert any("outside" in p for p in problems)


def test_check_chip_missing_file(tmp_path):
    assert "unreadable" in check_chip("missing", tmp_path)[0]


def test_ignore_pixels_do_not_affect_loss_or_metric():
    """-1 pixels must not change the training loss or water IoU, whatever is predicted there."""
    logits = torch.randn(1, 2, 8, 8)
    target = torch.randint(0, 2, (1, 8, 8))
    target[:, :4] = -1
    other = logits.clone()
    other[:, :, :4] = torch.randn(1, 2, 4, 8) * 10  # change predictions only on ignored pixels

    loss = init_loss("dice", ignore_index=-1)
    assert torch.isclose(loss(logits, target), loss(other, target))

    iou = MulticlassJaccardIndex(num_classes=2, ignore_index=-1, average=None)
    assert torch.equal(iou(logits.argmax(1), target), iou(other.argmax(1), target))


@pytest.mark.parametrize("model", ["unet", "terramind_v1_base", "prithvi_eo_v2_300"])
def test_normalization_shapes(model):
    n = normalization.get(model)
    assert len(n["bands"]) == len(n["means"]) == len(n["stds"])
    assert all(s > 0 for s in n["stds"])


@pytest.mark.parametrize("model", ["unet", "terramind_v1_base", "prithvi_eo_v2_300"])
def test_model_configs_use_official_normalization(model):
    import yaml

    from floodseg.sen1floods11 import REPO

    data = yaml.safe_load((REPO / "configs" / f"{model}.yaml").read_text())["data"]["init_args"]
    n = normalization.get(model)
    assert data["output_bands"] == n["bands"]
    assert data["means"] == pytest.approx(n["means"])
    assert data["stds"] == pytest.approx(n["stds"])
