from TractoPL.data.loader import Dataset


def test_dataset_describe_indexes_raw_and_derivative_files(tmp_path):
    raw_dwi = tmp_path / "sub-01" / "dwi" / "sub-01_dwi.nii.gz"
    derivative_dwi = (
        tmp_path
        / "derivatives"
        / "tractometry"
        / "sub-01"
        / "metric"
        / "sub-01_bundle-CSTleft_metric-FA_mean.csv"
    )
    raw_dwi.parent.mkdir(parents=True)
    derivative_dwi.parent.mkdir(parents=True)
    raw_dwi.touch()
    derivative_dwi.touch()

    description = Dataset(tmp_path).describe()

    assert description == {
        "root": str(tmp_path),
        "subjects": ["01"],
        "file_count": 2,
        "derivatives": ["tractometry"],
    }