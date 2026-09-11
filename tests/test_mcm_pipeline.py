from TractoPL.pipeline.mcm import _bundle_names, process_mcm_pipeline


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