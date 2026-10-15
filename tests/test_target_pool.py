import numpy as np
import pandas as pd
import pytest
import rasterio
import torch

from floodseg.target_pool import (
    PERU_IMAGES,
    PERU_LABEL_ROOTS,
    REPO,
    TargetPoolDataset,
    check_pool,
    default_pool_stems,
    forbid_peru_labels,
    guard_worker_init,
    model_input_spec,
    read_manifest,
)

HAVE_DATA = PERU_IMAGES.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="Peru data not on disk")


@pytest.fixture
def layout(tmp_path):
    split = tmp_path / "split.csv"
    pd.DataFrame({"chip": ["E1_N1", "E2_N2", "E3_N3", "E4_N4"], "date": ["2023-03-22"] * 4,
                  "split": ["train", "dev", "test", "buffer"]}).to_csv(split, index=False)
    labels = tmp_path / "emsr"
    labels.mkdir()
    (labels / "E9_N9.tif").touch()
    return {"unosat_split": split, "emsr_labels": labels}


def test_train_pool_passes(layout):
    check_pool(["2023-03-22_E1_N1"], **layout)


@pytest.mark.parametrize("stem", ["2023-03-22_E2_N2", "2023-03-22_E3_N3", "2023-03-22_E4_N4", "E9_N9"])
def test_held_out_refused(layout, stem):
    with pytest.raises(RuntimeError, match="held-out") as e:
        check_pool(["2023-03-22_E1_N1", stem], **layout)
    assert stem in str(e.value)


def test_unknown_chip_refused(layout):
    with pytest.raises(RuntimeError, match="not in the Peru v1 split"):
        check_pool(["2023-03-22_E7_N7"], **layout)


def test_label_guard_blocks_and_restores(tmp_path):
    label = PERU_LABEL_ROOTS[0] / "anything.tif"
    with forbid_peru_labels():
        with pytest.raises(RuntimeError, match="Peru label read during adaptation: .*anything.tif"):
            rasterio.open(label)
        with pytest.raises(rasterio.errors.RasterioIOError):  # non-label paths go through
            rasterio.open(tmp_path / "missing.tif")
    with pytest.raises(rasterio.errors.RasterioIOError):  # guard removed after the block
        rasterio.open(label)


@needs_data
def test_default_pool_is_peru_train():
    stems = default_pool_stems()
    assert len(stems) == 753
    check_pool(stems)
    assert [p.stem for p in read_manifest(verify=False)] == stems


@needs_data
@pytest.mark.parametrize("model", ["unet", "terramind_v1_base", "prithvi_eo_v2_300"])
def test_matches_terratorch_input(tmp_path, model):
    """A pool image equals what TerraTorch's datamodule feeds the model (val path, normalized)."""
    from terratorch.datamodules import GenericNonGeoSegmentationDataModule

    stem = default_pool_stems()[0]
    split = tmp_path / "one.txt"
    split.write_text(stem + "\n")
    spec = model_input_spec(REPO / "configs" / f"{model}.yaml")
    root, labels = str(PERU_IMAGES), str(REPO / "data/peru/unosat2023/label")
    dm = GenericNonGeoSegmentationDataModule(
        batch_size=1, num_workers=0, num_classes=2, dataset_bands=spec["dataset_bands"],
        output_bands=spec["output_bands"], means=spec["means"], stds=spec["stds"],
        no_data_replace=spec["no_data_replace"], no_label_replace=-1,
        train_data_root=root, val_data_root=root, test_data_root=root,
        train_label_data_root=labels, val_label_data_root=labels, test_label_data_root=labels,
        train_split=split, val_split=split, test_split=split, img_grep="*.tif", label_grep="*.tif",
    )
    dm.setup("validate")
    with forbid_peru_labels(), pytest.raises(RuntimeError, match="Peru label read"):
        dm.val_dataset[0]  # TerraTorch reads the mask via rioxarray: the guard catches it
    batch = next(iter(dm.val_dataloader()))
    ref = dm.aug(batch)["image"][0]
    ours = TargetPoolDataset([PERU_IMAGES / f"{stem}.tif"], spec)[0]["image"]
    assert ours.dtype == torch.float32 and ours.shape == ref.shape
    np.testing.assert_allclose(ours.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


class _LabelReader(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, i):
        rasterio.open(PERU_LABEL_ROOTS[0] / "anything.tif")


def test_guard_in_spawned_workers():
    loader = torch.utils.data.DataLoader(_LabelReader(), num_workers=1, multiprocessing_context="spawn",
                                         worker_init_fn=guard_worker_init)
    with pytest.raises(RuntimeError, match="Peru label read"):
        next(iter(loader))
