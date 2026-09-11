import json

import pytest

from TractoPL.analysis.AFQ_analysis import load_config, validate_config


def test_afq_config_requires_an_existing_dataset(tmp_path):
    config = {"dataset": str(tmp_path), "hcp_asso_pipeline": "tractometry"}

    validate_config(config)


def test_afq_config_rejects_placeholder_dataset():
    with pytest.raises(ValueError, match="dataset"):
        validate_config({"dataset": "/PATH/TO/DATASET", "hcp_asso_pipeline": "tractometry"})


def test_afq_config_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(str(tmp_path / "missing.json"))


def test_afq_config_rejects_invalid_json(tmp_path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        load_config(str(config_path))


def test_afq_config_loads_valid_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"dataset": str(tmp_path)}), encoding="utf-8")

    assert load_config(str(config_path))["dataset"] == str(tmp_path)