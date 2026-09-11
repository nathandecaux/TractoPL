import json

import pytest

from TractoPL.configuration import (
    ConfigurationError,
    load_atlas_config,
    load_hcp_bundle_mapping,
    load_tool_config,
)
from TractoPL.pipeline.msmt_csd import DEFAULT_STEPS, validate_steps


def test_load_atlas_config_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "atlas.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "custom-atlas",
                "reference": "images/reference.nii.gz",
                "bundles_dir": "bundles",
                "centroids_dir": "centroids",
                "bundle_mapping": {"CSTleft": "CST_left"},
            }
        ),
        encoding="utf-8",
    )

    atlas = load_atlas_config(manifest)

    assert atlas.reference == tmp_path / "images/reference.nii.gz"
    assert atlas.bundles_dir == tmp_path / "bundles"
    assert atlas.centroids_dir == tmp_path / "centroids"
    assert atlas.bundle_name("CSTleft") == "CST_left"
    assert atlas.bundle_name("custom") == "custom"


def test_load_atlas_config_rejects_missing_required_resource(tmp_path):
    manifest = tmp_path / "atlas.json"
    manifest.write_text(json.dumps({"name": "custom-atlas"}), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="reference, bundles_dir"):
        load_atlas_config(manifest)


def test_atlas_requires_optional_resource_when_an_operation_needs_it(tmp_path):
    manifest = tmp_path / "atlas.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "custom-atlas",
                "reference": "reference.nii.gz",
                "bundles_dir": "bundles",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="centroids"):
        load_atlas_config(manifest).require("centroids")


def test_hcp_legacy_profile_preserves_known_bundle_name():
    assert load_hcp_bundle_mapping()["CSTleft"] == "CST_left"


def test_tool_config_uses_environment_without_legacy_files(tmp_path):
    config, tools = load_tool_config(
        anima_config_path=tmp_path / "missing-anima.ini",
        tractseg_config_path=tmp_path / "missing-tractseg.ini",
        environ={
            "TRACTOPL_ANIMA_DIR": "/tools/anima/bin",
            "TRACTOPL_ANIMA_DATA_DIR": "/tools/anima/data",
            "TRACTOPL_ANIMA_SCRIPTS_DIR": "/tools/anima/scripts",
            "TRACTOPL_ANIMA_PRIVATE_SCRIPTS_DIR": "/tools/anima/private",
            "TRACTOPL_TRACTSEG_DIR": "/tools/tractseg/bin",
            "TRACTOPL_MRTRIX_DIR": "/tools/mrtrix/bin",
        },
    )

    assert config["animaDir"] == "/tools/anima/bin"
    assert tools["animaDTIScalarMaps"] == "/tools/anima/bin/animaDTIScalarMaps"


def test_msmt_csd_steps_default_and_validation():
    assert validate_steps(()) == DEFAULT_STEPS
    assert validate_steps(("fod", "ifod2_tracto")) == ("fod", "ifod2_tracto")

    with pytest.raises(ValueError, match="unknown"):
        validate_steps(("unknown",))