import pytest

from TractoPL.data.loader import Dataset, Subject


def test_dataset_root_can_be_read_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACTOPL_DATASET_ROOT", str(tmp_path))

    dataset = Dataset()
    subject = Subject("01")

    assert dataset.db_root == str(tmp_path)
    assert subject.db_root == str(tmp_path)


def test_dataset_root_is_required_without_argument_or_environment(monkeypatch):
    monkeypatch.delenv("TRACTOPL_DATASET_ROOT", raising=False)

    with pytest.raises(ValueError, match="TRACTOPL_DATASET_ROOT"):
        Dataset()