import os

from TractoPL.pipeline.mcm import _bundle_names, mcm_to_bundleseg_tracts, process_mcm_pipeline


class FakeBundle:
    def __init__(self, name):
        self.name = name

    def get_entities(self):
        return {"bundle": self.name}


class FakeSubject:
    def __init__(self, bundles):
        self.bundles = bundles

    def get(self, **kwargs):
        return [FakeBundle(name) for name in self.bundles]


def test_mcm_all_uses_available_bundle_names():
    subject = FakeSubject(["CSTleft", "CSTleft", "ILFright"])

    assert _bundle_names(subject, "ALL", "bundle_seg") == ["CSTleft", "ILFright"]


def test_mcm_explicit_bundle_selection_is_preserved():
    subject = FakeSubject([])

    assert _bundle_names(subject, ["CSTleft"], "bundle_seg") == ["CSTleft"]


def test_mcm_pipeline_forwards_explicit_estimation_options(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "TractoPL.pipeline.mcm.process_mcm_estimation",
        lambda subject, pipeline, **options: captured.update(options),
    )

    process_mcm_pipeline(
        FakeSubject([]),
        pipeline_list=["mcm_estimation"],
        estimation_options={"n": 2, "F": False},
    )

    assert captured == {"n": 2, "F": False}

class FakeOutput:
    def __init__(self, path, bundle):
        self.bids_path = str(path)
        self.bundle = bundle

    def get_full_entities(self):
        return {"bundle": self.bundle} if self.bundle else {}


class ProjectionSubject:
    """bundle_seg holds `segmented`; the MCM pipeline holds one vtk per bundle in `projected`."""

    def __init__(self, tmp_path, segmented, projected):
        self.segmented = segmented
        self.outputs = []
        for bundle in projected:
            path = tmp_path / f"{bundle}.vtk"
            path.write_text("old")
            self.outputs.append(FakeOutput(path, bundle))
        mcm = tmp_path / "subject.mcmx"
        mcm.write_text("mcm")
        self.mcm = FakeOutput(mcm, None)

    def get_unique(self, **kwargs):
        return self.mcm

    def get(self, **kwargs):
        if kwargs.get("metric") == "FA":
            return [object()]
        if kwargs.get("pipeline") == "bundle_seg":
            wanted = kwargs.get("bundle")
            return [FakeBundle(name) for name in self.segmented if wanted in (None, name)]
        return [output for output in self.outputs if os.path.exists(output.bids_path)]


def run_projection(monkeypatch, subject, **kwargs):
    projected = []
    monkeypatch.setattr("TractoPL.pipeline.mcm.add_mcm_to_tracts",
                        lambda tracts, **options: projected.append(tracts.name) or {})
    monkeypatch.setattr("TractoPL.pipeline.mcm.copy_from_dict", lambda *args, **options: None)
    mcm_to_bundleseg_tracts(subject, "mcm_tensors_restricted", "ALL", **kwargs)
    return sorted(projected)


def test_projection_skips_existing_bundles_by_default(tmp_path, monkeypatch):
    subject = ProjectionSubject(tmp_path, ["CC1", "UFleft"], ["UFleft"])

    assert run_projection(monkeypatch, subject) == ["CC1"]
    assert (tmp_path / "UFleft.vtk").exists()


def test_projection_overwrite_recomputes_and_drops_outdated_bundles(tmp_path, monkeypatch):
    subject = ProjectionSubject(tmp_path, ["CC1", "UFleft"], ["UFleft", "CA"])

    assert run_projection(monkeypatch, subject, overwrite=True) == ["CC1", "UFleft"]
    assert not (tmp_path / "CA.vtk").exists()
    assert (tmp_path / "subject.mcmx").exists()


def test_mcm_pipeline_forwards_overwrite_to_projection(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "TractoPL.pipeline.mcm.mcm_to_bundleseg_tracts",
        lambda subject, pipeline, bundle_name, **options: captured.update(options),
    )

    process_mcm_pipeline(FakeSubject([]), pipeline_list=["mcm_to_bundleseg_tracts"], overwrite=True)

    assert captured["overwrite"] is True
