"""Configuration portable des ressources externes de TractoPL."""

from __future__ import annotations

import json
import os
from configparser import RawConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


class ConfigurationError(ValueError):
    """Raised when a TractoPL configuration file is incomplete or invalid."""


@dataclass(frozen=True)
class AtlasConfig:
    """Description résolue des fichiers nécessaires à un atlas de bundles."""

    name: str
    reference: Path
    bundles_dir: Path
    centroids_dir: Optional[Path] = None
    parcellations_dir: Optional[Path] = None
    bundle_mapping: Optional[Mapping[str, str]] = None

    def bundle_name(self, name: str) -> str:
        """Return the atlas filename stem associated with a user bundle name."""
        return (self.bundle_mapping or {}).get(name, name)

    def require(self, *resources: str) -> None:
        """Validate that an operation's atlas resources have been configured."""
        missing = [
            resource
            for resource in resources
            if getattr(self, f"{resource}_dir", None) is None
        ]
        if missing:
            raise ConfigurationError(
                f"Atlas '{self.name}' is missing required resources: {', '.join(missing)}"
            )


def load_atlas_config(path: Union[str, Path]) -> AtlasConfig:
    """Load an atlas manifest and resolve relative paths from its location.

    The JSON manifest requires ``name``, ``reference`` and ``bundles_dir``.
    ``centroids_dir``, ``parcellations_dir`` and ``bundle_mapping`` are optional.
    """
    manifest_path = Path(path).expanduser().resolve()
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            data: Dict[str, Any] = json.load(manifest_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Atlas manifest does not exist: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"Atlas manifest is not valid JSON: {manifest_path}") from error

    required = ("name", "reference", "bundles_dir")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ConfigurationError(
            f"Atlas manifest is missing required keys: {', '.join(missing)}"
        )

    mapping = data.get("bundle_mapping", {})
    if not isinstance(mapping, dict) or not all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in mapping.items()
    ):
        raise ConfigurationError("bundle_mapping must be an object mapping names to names")

    manifest_dir = manifest_path.parent

    def resolve_path(key: str, required_path: bool = False) -> Optional[Path]:
        value = data.get(key)
        if value is None and not required_path:
            return None
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"{key} must be a non-empty path string")
        configured_path = Path(value).expanduser()
        return configured_path if configured_path.is_absolute() else manifest_dir / configured_path

    return AtlasConfig(
        name=data["name"],
        reference=resolve_path("reference", required_path=True),
        bundles_dir=resolve_path("bundles_dir", required_path=True),
        centroids_dir=resolve_path("centroids_dir"),
        parcellations_dir=resolve_path("parcellations_dir"),
        bundle_mapping=mapping,
    )


def load_hcp_bundle_mapping() -> Dict[str, str]:
    """Load the legacy HCP bundle-name mapping distributed with TractoPL."""
    profile_path = Path(__file__).parent / "config_profiles" / "hcp105_legacy.json"
    try:
        with profile_path.open(encoding="utf-8") as profile_file:
            mapping = json.load(profile_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Missing bundled HCP profile: {profile_path}") from error
    if not isinstance(mapping, dict) or not all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in mapping.items()
    ):
        raise ConfigurationError("The bundled HCP profile contains an invalid mapping")
    return mapping


def load_tool_config(
    anima_config_path: Optional[Union[str, Path]] = None,
    tractseg_config_path: Optional[Union[str, Path]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, Union[str, List[str]]]]:
    """Load external-tool locations from environment variables or legacy files.

    Environment variables have precedence over the optional legacy INI files:
    ``TRACTOPL_ANIMA_DIR``, ``TRACTOPL_ANIMA_DATA_DIR``,
    ``TRACTOPL_ANIMA_SCRIPTS_DIR``, ``TRACTOPL_ANIMA_PRIVATE_SCRIPTS_DIR``,
    ``TRACTOPL_TRACTSEG_DIR`` and ``TRACTOPL_MRTRIX_DIR``.
    """
    environment = os.environ if environ is None else environ
    anima_path = Path(anima_config_path or "~/.anima/config.txt").expanduser()
    tractseg_path = Path(
        tractseg_config_path or "~/.tractseg-config/config.txt"
    ).expanduser()

    anima_values = _read_ini_section(anima_path, "anima-scripts")
    tractseg_values = _read_ini_section(tractseg_path, "tractseg")
    config = {
        "animaDir": environment.get("TRACTOPL_ANIMA_DIR", anima_values.get("anima", "")),
        "animaDataDir": environment.get(
            "TRACTOPL_ANIMA_DATA_DIR", anima_values.get("extra-data-root", "")
        ),
        "animaScriptsDir": environment.get(
            "TRACTOPL_ANIMA_SCRIPTS_DIR", anima_values.get("anima-scripts-public-root", "")
        ),
        "animaPrivScriptsDir": environment.get(
            "TRACTOPL_ANIMA_PRIVATE_SCRIPTS_DIR", anima_values.get("anima-scripts-root", "")
        ),
        "tractsegDir": environment.get("TRACTOPL_TRACTSEG_DIR", tractseg_values.get("tractseg-bin", "")),
        "mrtrixDir": environment.get("TRACTOPL_MRTRIX_DIR", tractseg_values.get("mrtrix-bin", "")),
    }
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise ConfigurationError(
            "External tool configuration is incomplete. Set the corresponding "
            f"TRACTOPL_* variables or configure the legacy files. Missing: {', '.join(missing)}"
        )

    anima_dir = config["animaDir"]
    scripts_dir = config["animaScriptsDir"]
    private_scripts_dir = config["animaPrivScriptsDir"]
    configuration_dir = os.path.dirname(os.path.abspath(__file__))
    tools: Dict[str, Union[str, List[str]]] = {
        "animaConvertImage": os.path.join(anima_dir, "animaConvertImage"),
        "animaApplyTransformSerie": os.path.join(anima_dir, "animaApplyTransformSerie"),
        "animaTensorApplyTransformSerie": os.path.join(anima_dir, "animaTensorApplyTransformSerie"),
        "animaMCMApplyTransformSerie": os.path.join(anima_dir, "animaMCMApplyTransformSerie"),
        "animaCropImage": os.path.join(anima_dir, "animaCropImage"),
        "animaGMMT2RelaxometryEstimation": os.path.join(anima_dir, "animaGMMT2RelaxometryEstimation"),
        "animaDTIScalarMaps": os.path.join(anima_dir, "animaDTIScalarMaps"),
        "animaMCMScalarMaps": os.path.join(anima_dir, "animaMCMScalarMaps"),
        "animaBrainExtraction": os.path.join(scripts_dir, "brain_extraction", "animaAtlasBasedBrainExtraction.py"),
        "myAnimaSubjectsMCMFiberPreparation": os.path.join(private_scripts_dir, "diffusion", "mcm_fiber_atlas_comparison", "myAnimaSubjectsMCMFiberPreparation.py"),
        "animaRegister3DImageOnAtlas": os.path.join(scripts_dir, "registration", "animaRegister3DImageOnAtlas.py"),
        "antsRegister3DImageOnAtlas": os.path.join(private_scripts_dir, "registration", "antsRegister3DImageOnAtlas.py"),
        "Tractometry": os.path.join(private_scripts_dir, "tractometry_Julie.py"),
        "animaDiffusionImagePreprocessing": os.path.join(configuration_dir, "scripts", "animaDiffusionImagePreprocessing.py"),
        "fast": ["fast"] if _which("fast") else ["fsl", "fast"],
    }
    return config, tools


def _read_ini_section(path: Path, section: str) -> Dict[str, str]:
    if not path.exists():
        return {}
    parser = RawConfigParser()
    parser.read(path)
    return dict(parser.items(section)) if parser.has_section(section) else {}


def _which(command: str) -> Optional[str]:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None