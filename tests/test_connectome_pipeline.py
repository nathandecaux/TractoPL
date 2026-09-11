from TractoPL.pipeline.connectome import _available_bundle_names


class FakeBundle:
    def __init__(self, name):
        self.name = name

    def get_entities(self):
        return {"bundle": self.name}


class FakeSubject:
    def __init__(self):
        self.query = None

    def get(self, **kwargs):
        self.query = kwargs
        return [FakeBundle("CSTleft"), FakeBundle("CSTleft"), FakeBundle("ILFright")]


def test_connectome_bundle_discovery_uses_requested_entities():
    subject = FakeSubject()

    names = _available_bundle_names(
        subject,
        "connectome",
        suffix="density",
        atlas_name="custom-atlas",
    )

    assert names == ["CSTleft", "ILFright"]
    assert subject.query == {
        "pipeline": "connectome",
        "datatype": "tracto",
        "suffix": "density",
        "atlas": "custom-atlas",
    }