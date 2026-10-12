import pandas as pd
import pytest

from floodseg.peru import check_train_split


@pytest.fixture
def layout(tmp_path):
    split = tmp_path / "split.csv"
    pd.DataFrame({"chip": ["E1_N1", "E2_N2", "E3_N3"], "date": ["2023-03-22"] * 3,
                  "split": ["train", "dev", "test"]}).to_csv(split, index=False)
    labels = tmp_path / "emsr"
    labels.mkdir()
    (labels / "E9_N9.tif").touch()
    return {"unosat_split": split, "emsr_labels": labels}


def test_train_only_passes(tmp_path, layout):
    f = tmp_path / "train.txt"
    f.write_text("2023-03-22_E1_N1\n")
    check_train_split(f, **layout)


@pytest.mark.parametrize("stem", ["2023-03-22_E2_N2", "2023-03-22_E3_N3", "E9_N9"])
def test_held_out_rejected(tmp_path, layout, stem):
    f = tmp_path / "train.txt"
    f.write_text(f"2023-03-22_E1_N1\n{stem}\n")
    with pytest.raises(RuntimeError, match="held-out"):
        check_train_split(f, **layout)


def test_no_split_is_fine():
    check_train_split(None)
