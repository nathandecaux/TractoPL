from TractoPL.pipeline.tractometry import mark_tractometry_derivative, process_hcp_association


class FakeSubject:
    pass


def test_tractometry_forwards_configured_input_pipelines(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "TractoPL.pipeline.tractometry.bundle_association_multiclusters",
        lambda subject, pipeline, **options: captured.update(
            {"pipeline": pipeline, **options}
        ),
    )

    process_hcp_association(
        FakeSubject(),
        atlas=object(),
        pipeline="custom-tractometry_42pts",
        n_pts=42,
        pipeline_list=["bundle_association_multiclusters"],
        mcm_pipeline="custom-mcm",
        bundle_pipeline="custom-bundles",
        clustering_method="quickbundles",
        model_clustering=7.5,
    )

    assert captured == {
        "pipeline": "custom-tractometry_42pts",
        "n_pts": 42,
        "mcm_pipeline": "custom-mcm",
        "bundle_pipeline": "custom-bundles",
        "clustering_method": "quickbundles",
        "model_clustering": 7.5,
    }


def test_tractometry_marks_custom_derivative_for_dashboard(tmp_path):
    mark_tractometry_derivative(tmp_path, "study-tractometry")

    assert (tmp_path / "derivatives" / "study-tractometry" / ".tag_tractometry").is_file()
