import json

from TractoPL.data.loader import LayoutProxy
from TractoPL.utils.tools import create_pipeline_description


def test_create_pipeline_description_is_dataset_neutral(tmp_path):
    create_pipeline_description("tractometry", LayoutProxy(str(tmp_path)))

    description_path = tmp_path / "derivatives" / "tractometry" / "dataset_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))

    assert description["Name"] == "TractoPL tractometry pipeline"
    assert description["PipelineDescription"]["Name"] == "tractometry"