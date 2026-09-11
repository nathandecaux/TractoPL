import json

from TractoPL.dashboard.streamlit_dashboard import get_available_pipelines


def test_dashboard_detects_tractometry_from_bids_description(tmp_path):
    derivatives = tmp_path / "derivatives"
    tractometry = derivatives / "custom_tractometry"
    tractometry.mkdir(parents=True)
    (tractometry / "dataset_description.json").write_text(
        json.dumps({"PipelineDescription": {"Name": "tractometry"}}),
        encoding="utf-8",
    )
    (derivatives / "other_pipeline").mkdir()

    assert get_available_pipelines(str(tmp_path)) == ["custom_tractometry"]


def test_dashboard_keeps_legacy_hcp_association_detection(tmp_path):
    legacy_pipeline = tmp_path / "derivatives" / "hcp_association_100pts"
    legacy_pipeline.mkdir(parents=True)

    assert get_available_pipelines(str(tmp_path)) == ["hcp_association_100pts"]


def test_dashboard_detects_tractometry_marker(tmp_path):
    tractometry = tmp_path / "derivatives" / "study-output"
    tractometry.mkdir(parents=True)
    (tractometry / ".tag_tractometry").touch()

    assert get_available_pipelines(str(tmp_path)) == ["study-output"]


def test_dashboard_detects_existing_tractometry_metric_files(tmp_path):
    tractometry = tmp_path / "derivatives" / "existing-output" / "sub-01" / "metric"
    tractometry.mkdir(parents=True)
    (tractometry / "sub-01_bundle-CSTleft_model-MCM_stats.csv").touch()

    assert get_available_pipelines(str(tmp_path)) == ["existing-output"]