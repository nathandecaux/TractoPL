#!/usr/bin/env python
"""
Population tractometry analysis
===============================

Pointwise statistics along the bundles of a tractometry pipeline (one CSV per
subject and bundle), for every metric column found in the CSVs:

- group variables (categorical): difference between groups at each point;
- correlation variables (continuous): association with the metric at each point.

Each variable has its own list of confounds. Two statistical models are
available (config key "STAT_MODEL"):

"residualize" (default, historical behaviour)
    At each point the confounds are regressed out of the metric by OLS (and
    out of the variable for correlations, unless CORRECT_METRIC_CONFOND_ONLY),
    then a Welch t-test (2 groups), a one-way ANOVA (> 2 groups) or a
    Pearson / Spearman correlation is computed on the residuals.

"glm"
    An OLS model is fitted with statsmodels at each point:

        metric ~ 1 + variable + confounds

    - group variables: ANCOVA with robust standard errors (ANCOVA_COV_TYPE,
      HC3 by default, valid with unequal group sizes and variances). With 2
      groups, diff is the adjusted difference (group 1 - group 0) and p comes
      from its t-test; with more groups, Wald F-test of the group dummies.
    - correlation variables: classical OLS, t-test of the slope; r is the
      partial correlation (degrees of freedom account for the confounds).

    CORRELATION_TEST and CORRECT_METRIC_CONFOND_ONLY do not apply to the tests.

RANK_TRANSFORM (both models): before the test, the metric at each point and the
correlation variable are replaced by their normal scores, Phi^-1((rank - 0.5) / n),
computed on the subjects of the analysis. Confounds and group labels are left as is.
With the GLM, the slope test then approximates a partial Spearman correlation.

The family-wise error along each bundle is controlled by permutations
(AFQ_NPERM): min p-value threshold (alphaFWE), cluster size threshold
(clusterFWE), both (mixed), or Threshold-Free Cluster Enhancement (TFCE).
The thresholds of all the methods are estimated from the same permutations
and saved in the per-point CSVs (sig_<method> columns), so that the readers
(population_analysis_report.py, report_analysis.py) can switch method without
re-running the analysis; FWE_METHOD sets sig_afq / n_sig and the figures.

- "residualize" model: AFQ-style permutations on the metric and the variable
  with the confounds regressed out (group labels or variable shuffled, same
  Welch t-test / one-way ANOVA / correlation as the observed one).
- "glm" model: Freedman-Lane permutations. The residuals of the confounds-only
  model are permuted and added back to its fitted values, and the same GLM
  test as the observed one (same covariance type) is computed on each
  permuted dataset.

TFCE (Smith & Nichols, 2009), 1D version along the profile: the p-values are
converted to z-scores, z = Phi^-1(1 - p/2), signed by the direction of the
effect for single-term tests (2 groups, correlations), unsigned for F-tests.
At each point, for the positive and negative parts separately,

    TFCE(v) = sum_h  e(h)^TFCE_E * h^TFCE_H * TFCE_DH,   h = TFCE_DH, 2 TFCE_DH, ...

where e(h) is the length (in points) of the run of points with z >= h that
contains v. The null distribution of max |TFCE| over the profile is built
with the same permutations as the other methods (Freedman-Lane for the GLM,
permutations of the group labels / variable on the confound-corrected data for
the "residualize" model), giving a FWE-corrected p-value per point (p_tfce);
a point is significant when p_tfce <= AFQ_ALPHA.

With CORRECT_MULTI_TRACT the thresholds are computed once over the
concatenated profiles of all bundles; clusters (clusterFWE, mixed, TFCE) never
span two bundles/centroids of the concatenated profiles.

Categorical confounds (non-numeric columns) are dummy-coded in both models.

Usage:
    python population_analysis.py -c config.json [--subjects-table subjects.csv] [--output-dir DIR]
"""

import argparse
import gc
import json
import os
import re
import warnings
import zlib
from collections import defaultdict
from dataclasses import dataclass, field, replace
from os.path import join as opj
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import f_oneway, norm, rankdata, ttest_ind
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist
from tqdm import tqdm

from TractoPL.analysis.fwe_utils import FWE_METHODS, n_significant, significance_mask, true_runs
from TractoPL.data.loader import Dataset
from TractoPL.set_config import get_HCP_bundle_names


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "pipeline": "default",
    "dataset": "/PATH/TO/DATASET",
    "hcp_asso_pipeline": "tractometry",

    # --- Variables to analyse ---
    # Each variable maps to {"confounds": [...]}, e.g.
    # "group_variables": {"group": {"confounds": ["age"]}},
    # "corr_variables": {"score": {"confounds": ["age"]}},
    "group_variables": {},
    "corr_variables": {},

    # --- Optional: split the correlation variables by group levels ---
    # Creates <var>_by_<group>_<level> variables (correlation within each group)
    "test_by_group": None,

    # --- Statistical model ---
    "STAT_MODEL": "residualize",    # "residualize" or "glm" (see module docstring)
    "ANCOVA_COV_TYPE": "HC3",       # "glm" model, group variables: statsmodels cov_type

    # --- Multiple comparison correction ---
    "FWE_METHOD": "clusterFWE",     # "clusterFWE", "alphaFWE", "mixed" or "TFCE"
    "AFQ_ALPHA": 0.05,
    "AFQ_NPERM": 1000,
    "CORRECT_MULTI_TRACT": False,   # Thresholds computed across all bundles (slower)
    "RANDOM_SEED": None,            # Integer for reproducible permutations
    "TFCE_E": 0.5,                  # TFCE extent exponent
    "TFCE_H": 2.0,                  # TFCE height exponent
    "TFCE_DH": 0.1,                 # TFCE height step (z units)

    # --- Data quality ---
    "MISSING_SUBJECTS_TOLERANCE": 0,    # Max subjects with a missing value per point
    "INTERPOLATE_MISSING_POINTS": 2,    # Max gap for linear interpolation (0 = off)
    "KEEP_LARGEST_CONTIGUOUS": True,    # Keep only the longest valid segment
    "RESAMPLE_N_POINTS": None,          # Resample to N points per bundle

    # --- Metric detection: columns <METRIC>_<STAT>, or <METRIC> ---
    "METRIC_COLUMNS_CANDIDATES": ["FA"],
    "STAT_TYPES": ["mean"],

    # --- "residualize" model only ---
    "CORRELATION_TEST": "pearson",      # "pearson" or "spearman"
    "CORRECT_METRIC_CONFOND_ONLY": False,  # Do not regress the confounds out of the variable

    # --- Both models: normal scores of the metric (per point) and of the variable ---
    "RANK_TRANSFORM": False,

    # --- Optional table merged by participant_id (ignored with --subjects-table) ---
    "additional_info_path": None,

    # --- Exclusion filter: {column: [values to exclude]} ---
    "EXCLUDE": {},

    # --- Execution / outputs ---
    "MULTIPROCESSING": True,
    "n_jobs": None,         # None = auto (cpu_count // 3)
    "SAVE_FIGURES": False,  # Also save a PNG next to each result CSV
    "output_dir": None,     # None = auto-generated
}

STAT_MODELS = ("residualize", "glm")
ANCOVA_COV_TYPES = ("HC3", "HC2", "HC1", "HC0", "nonrobust")
CORRELATION_TESTS = ("pearson", "spearman")
MIN_SUBJECTS = 4


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load the JSON configuration, or return the default one when no path is given."""
    if not config_path:
        return dict(DEFAULT_CONFIG)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Configuration file is not valid JSON: {config_path}") from error
    print(f"Configuration loaded from: {config_path}")
    return config


def validate_config(config: Dict) -> None:
    """Validate the minimum configuration required before starting an analysis."""
    dataset_path = config.get("dataset")
    if not isinstance(dataset_path, str) or not dataset_path or dataset_path == DEFAULT_CONFIG["dataset"]:
        raise ValueError("Configuration key 'dataset' must contain a BIDS dataset path")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Configured dataset does not exist: {dataset_path}")
    pipeline_name = config.get("hcp_asso_pipeline")
    if not isinstance(pipeline_name, str) or not pipeline_name:
        raise ValueError("Configuration key 'hcp_asso_pipeline' must be a pipeline name")
    for key, allowed in (("STAT_MODEL", STAT_MODELS), ("ANCOVA_COV_TYPE", ANCOVA_COV_TYPES),
                         ("FWE_METHOD", FWE_METHODS),
                         ("CORRELATION_TEST", CORRELATION_TESTS)):
        value = config.get(key) or DEFAULT_CONFIG[key]
        if value not in allowed:
            raise ValueError(f"Configuration key '{key}' must be one of {allowed}, got '{value}'")
    for key in ("TFCE_E", "TFCE_H", "TFCE_DH"):
        value = config.get(key)
        if value is not None and not (isinstance(value, (int, float)) and value > 0):
            raise ValueError(f"Configuration key '{key}' must be a positive number, got '{value}'")


def save_config(config: Dict, output_path: str) -> None:
    """Save the configuration dictionary to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {output_path}")


@dataclass(frozen=True)
class Settings:
    """Analysis parameters, read once from the config and passed to the workers."""
    group_variables: Dict[str, List[str]]   # variable -> confounds
    corr_variables: Dict[str, List[str]]    # variable -> confounds
    test_by_group: Optional[str] = None
    stat_model: str = "residualize"
    ancova_cov_type: str = "HC3"
    fwe_method: str = "clusterFWE"
    alpha: float = 0.05
    nperm: int = 1000
    multi_tract: bool = False
    random_seed: Optional[int] = None
    tfce_e: float = 0.5
    tfce_h: float = 2.0
    tfce_dh: float = 0.1
    correlation_test: str = "pearson"
    correct_metric_only: bool = False
    missing_subjects_tolerance: int = 0
    interpolate_max_gap: int = 2
    keep_largest_contiguous: bool = True
    resample_n_points: Optional[int] = None
    metric_candidates: Tuple[str, ...] = ("FA",)
    stat_types: Tuple[str, ...] = ("mean",)
    exclude: Dict[str, list] = field(default_factory=dict)
    save_figures: bool = False
    rank_transform: bool = False

    @classmethod
    def from_config(cls, config: Dict) -> "Settings":
        def get(key):
            value = config.get(key)
            return DEFAULT_CONFIG[key] if value is None else value

        def variables(key):
            out = {}
            for var, spec in (config.get(key) or {}).items():
                confounds = spec.get("confounds") if isinstance(spec, dict) else spec
                out[var] = list(confounds or [])
            return out

        return cls(
            group_variables=variables("group_variables"),
            corr_variables=variables("corr_variables"),
            test_by_group=config.get("test_by_group") or None,
            stat_model=get("STAT_MODEL"),
            ancova_cov_type=get("ANCOVA_COV_TYPE"),
            fwe_method=get("FWE_METHOD"),
            alpha=float(get("AFQ_ALPHA")),
            nperm=int(get("AFQ_NPERM")),
            multi_tract=bool(get("CORRECT_MULTI_TRACT")),
            random_seed=config.get("RANDOM_SEED"),
            tfce_e=float(get("TFCE_E")),
            tfce_h=float(get("TFCE_H")),
            tfce_dh=float(get("TFCE_DH")),
            correlation_test=get("CORRELATION_TEST"),
            correct_metric_only=bool(get("CORRECT_METRIC_CONFOND_ONLY")),
            missing_subjects_tolerance=int(get("MISSING_SUBJECTS_TOLERANCE")),
            interpolate_max_gap=int(get("INTERPOLATE_MISSING_POINTS")),
            keep_largest_contiguous=bool(get("KEEP_LARGEST_CONTIGUOUS")),
            resample_n_points=config.get("RESAMPLE_N_POINTS") or None,
            metric_candidates=tuple(get("METRIC_COLUMNS_CANDIDATES")),
            stat_types=tuple(get("STAT_TYPES")),
            exclude=dict(config.get("EXCLUDE") or {}),
            save_figures=bool(get("SAVE_FIGURES")),
            rank_transform=bool(get("RANK_TRANSFORM")),
        )

    def variables(self) -> List[Tuple[str, str, List[str]]]:
        """(kind, variable, confounds) of every analysis, kind being 'group' or 'corr'."""
        return ([("group", var, conf) for var, conf in self.group_variables.items()]
                + [("corr", var, conf) for var, conf in self.corr_variables.items()])

    @property
    def model_label(self) -> str:
        return "GLM" if self.stat_model == "glm" else "residualized"


# ============================================================================
# SUBJECT TABLES
# ============================================================================

def normalize_participant_id(value) -> Optional[str]:
    """Normalize a participant ID to the 'sub-XXX' format."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if text.startswith("sub-") else f"sub-{text}"


def read_table(path: str, **kwargs) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def load_subject_info(path: str) -> pd.DataFrame:
    """
    Load a subjects table (CSV/XLSX) with a 'participant_id' column. IDs are
    normalised to 'sub-XXX' (the format of the bundle CSVs) and each
    participant keeps a single row.
    """
    df = read_table(path, dtype={"participant_id": str})  # keep IDs such as '001' as text
    if "participant_id" not in df.columns:
        raise ValueError(f"Table '{path}' must contain a 'participant_id' column.")
    df = df.assign(participant_id=df["participant_id"].map(normalize_participant_id))
    return df.dropna(subset=["participant_id"]).drop_duplicates("participant_id")


def check_config_columns(subject_info: pd.DataFrame, settings: Settings) -> None:
    """Warn about the configured variables and confounds missing from the subjects table."""
    wanted = set()
    for _, var, confounds in settings.variables():
        wanted.add(var)
        wanted.update(confounds)
    if settings.test_by_group:
        wanted.add(settings.test_by_group)
    missing = wanted - set(subject_info.columns)
    present = wanted & set(subject_info.columns)
    if missing:
        print(f"[WARNING] Config variables not found in subjects table: {sorted(missing)}")
    if present:
        print(f"[OK] Config variables found in subjects table: {sorted(present)}")


# ============================================================================
# BUNDLE DATA LOADING
# ============================================================================

def load_bundle(bundle_csvs: list, subject_info: Optional[pd.DataFrame] = None,
                keep_only_listed: bool = False) -> pd.DataFrame:
    """
    Concatenate the per-subject CSVs of one bundle and left-join the subjects
    table on participant_id (columns already in the CSVs are kept from the CSVs).
    With keep_only_listed, subjects absent from the table are dropped.
    """
    frames = []
    for f in bundle_csvs:
        frame = pd.read_csv(f.path)
        frame["subject"] = f.get_full_entities()["subject"]
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["participant_id"] = "sub-" + df["subject"].astype(str)

    if subject_info is not None:
        if keep_only_listed:
            df = df[df["participant_id"].isin(subject_info["participant_id"])]
        extra_cols = [c for c in subject_info.columns if c not in df.columns]
        df = df.merge(subject_info[["participant_id"] + extra_cols], on="participant_id", how="left")
    return df


def apply_exclusions(df: pd.DataFrame, exclude: Dict[str, list]) -> pd.DataFrame:
    """Drop the rows whose value in a column is one of the excluded values."""
    for column, values in exclude.items():
        if column in df.columns:
            df = df[~df[column].isin(values)]
    return df


def _level_label(value) -> str:
    """Text label of a group level (1.0 -> '1')."""
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def add_group_split_columns(df: pd.DataFrame, settings: Settings
                            ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    test_by_group: for each correlation variable and each level of the group
    column, add a <var>_by_<group>_<level> column holding the variable for the
    subjects of that level only (NaN elsewhere).

    Returns the new DataFrame and the new correlation variables (same confounds).
    """
    group_col = settings.test_by_group
    if not group_col or group_col not in df.columns:
        return df, {}

    new_cols, new_vars = {}, {}
    for var, confounds in settings.corr_variables.items():
        if var not in df.columns:
            continue
        for level in df[group_col].dropna().unique():
            name = f"{var}_by_{group_col}_{_level_label(level)}"
            new_cols[name] = df[var].where(df[group_col] == level)
            new_vars[name] = confounds
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df, new_vars


# ============================================================================
# COLUMN DETECTION
# ============================================================================

def detect_point_column(df: pd.DataFrame) -> str:
    """Find the column holding the point index along the tract ('point_id' or 'point')."""
    for name in ("point_id", "point"):
        if name in df.columns:
            return name
    for c in df.columns:
        if re.fullmatch(r"point(_?id)?", c.lower()):
            return c
    raise ValueError("Point column not found in DataFrame. Expected 'point_id' or 'point'.")


def detect_metric_columns(df: pd.DataFrame, metric_candidates, stat_types) -> List[str]:
    """Metric columns present in df: <METRIC>_<STAT> first, else the bare <METRIC> names."""
    detected = [f"{m}_{s}" for m in metric_candidates for s in stat_types
                if f"{m}_{s}" in df.columns]
    return detected or [m for m in metric_candidates if m in df.columns]


def get_all_metric_related_columns(df: pd.DataFrame, metric_candidates=()) -> List[str]:
    """Columns holding metric values: <METRIC>, <METRIC>_*, 'value' and 'value_*'."""
    candidates = list(metric_candidates) + ["value"]
    return [c for c in df.columns
            if c in candidates or any(c.startswith(f"{m}_") for m in candidates)]


def to_long(df: pd.DataFrame, metric_col: str, metric_candidates=()) -> pd.DataFrame:
    """One row per (subject, point): metric renamed 'value', point column renamed 'point',
    other metric columns dropped, subject-level columns kept."""
    point_col = detect_point_column(df)
    metric_related = set(get_all_metric_related_columns(df, metric_candidates))
    meta_cols = [c for c in df.columns if c not in metric_related]
    return df[meta_cols + [metric_col]].rename(columns={metric_col: "value", point_col: "point"})


# ============================================================================
# RESAMPLING & MISSING DATA
# ============================================================================

def resample_bundle_data(df: pd.DataFrame, n_points: int, xu_max=None,
                         metric_candidates=()) -> pd.DataFrame:
    """
    Resample each subject's profile to a common grid of n_points by linear
    interpolation (consistent point counts are required by the permutations).
    Resampling is done per centroid when 'centroid_id' is present. Rows where
    all the metric values are zero are dropped.
    """
    if not n_points or n_points < 2:
        return df
    try:
        point_col = detect_point_column(df)
    except ValueError:
        return df
    metric_cols = get_all_metric_related_columns(df, metric_candidates)
    if not metric_cols:
        return df

    has_centroid = "centroid_id" in df.columns
    meta_cols = [c for c in df.columns if c not in metric_cols and c != point_col]
    resampled = []

    for cent_id in (df["centroid_id"].unique() if has_centroid else [None]):
        cent_df = df[df["centroid_id"] == cent_id] if has_centroid else df
        # Common target grid for this centroid
        x_target = np.linspace(cent_df[point_col].min(),
                               cent_df[point_col].max() if xu_max is None else xu_max, n_points)
        idx_target = np.arange(1, n_points + 1)

        for _, sub_df in cent_df.groupby("subject", sort=False):
            sub_df = sub_df.sort_values(point_col)
            x = sub_df[point_col].to_numpy(dtype=float)
            if len(x) < 2:
                resampled.append(sub_df)
                continue
            xu, idx = np.unique(x, return_index=True)
            # Part of the target grid within this subject's range
            mask = (x_target >= xu.min()) & (x_target <= xu.max())
            if mask.sum() < 2:
                resampled.append(sub_df)
                continue
            new_data = {point_col: idx_target[mask]}
            for mc in metric_cols:
                new_data[mc] = np.interp(x_target[mask], xu, sub_df[mc].to_numpy(dtype=float)[idx])
            for col in meta_cols:
                new_data[col] = sub_df.iloc[0][col]
            resampled.append(pd.DataFrame(new_data))

    if not resampled:
        return df
    resampled = pd.concat(resampled, ignore_index=True)
    group_cols = ["subject", "centroid_id"] if has_centroid else ["subject"]
    resampled = resampled[(resampled[metric_cols] != 0).any(axis=1)]
    return resampled.sort_values(group_cols, kind="stable").reset_index(drop=True)


def interpolate_missing_points(long_df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """
    Linearly interpolate the missing points of each subject's profile, up to
    max_gap consecutive points.
    """
    if long_df.empty or max_gap < 1:
        return long_df

    pivot = long_df.pivot_table(index="subject", columns="point", values="value", aggfunc="mean")
    pivot = pivot.interpolate(method="linear", axis=1, limit=max_gap, limit_direction="both")
    out = pivot.reset_index().melt(id_vars="subject", var_name="point", value_name="value")

    meta_cols = [c for c in long_df.columns if c not in {"subject", "point", "value"}]
    if meta_cols:
        meta = long_df[["subject"] + meta_cols].drop_duplicates(subset=["subject"])
        out = out.merge(meta, on="subject", how="left")
    return out


def clean_missing_data(long_df: pd.DataFrame, tolerance: int = 0, interpolate_max_gap: int = 2,
                       keep_largest_contiguous: bool = True) -> Tuple[pd.DataFrame, list, list]:
    """
    Handle missing values along the bundle:

    1. interpolate small gaps (interpolate_max_gap points at most);
    2. keep the points where at most `tolerance` subjects have a missing value
       (only the longest contiguous run of them with keep_largest_contiguous);
    3. drop the subjects that still have a missing value on the kept points.

    Returns (cleaned long_df, removed subjects, removed points).
    """
    if long_df.empty:
        return long_df, [], []
    if interpolate_max_gap and interpolate_max_gap > 0:
        long_df = interpolate_missing_points(long_df, max_gap=interpolate_max_gap)

    pivot = long_df.pivot_table(index="subject", columns="point", values="value", aggfunc="mean")
    if pivot.empty:
        return long_df, [], []
    points = pivot.columns.to_numpy()
    valid = (pivot.isna().sum(axis=0) <= tolerance).to_numpy()

    if keep_largest_contiguous:
        runs = true_runs(valid)
        if not runs:
            return long_df.iloc[0:0].copy(), [], points.tolist()
        start, end = max(runs, key=lambda run: run[1] - run[0])  # first longest run
        valid = np.zeros(len(points), dtype=bool)
        valid[start:end] = True

    kept_points = points[valid]
    profiles = pivot[kept_points].reindex(long_df["subject"].unique())
    removed_subjects = profiles.index[profiles.isna().any(axis=1)].tolist()
    long_clean = long_df[long_df["point"].isin(kept_points)
                         & ~long_df["subject"].isin(removed_subjects)].copy()
    return long_clean, removed_subjects, points[~valid].tolist()


# ============================================================================
# SUBJECT-LEVEL DATA OF ONE ANALYSIS
# ============================================================================

@dataclass
class UnitData:
    """Complete-case data of one analysis (one variable x metric x bundle/centroid)."""
    Y: np.ndarray           # (n_subjects, n_points) metric values
    points: np.ndarray      # point ids (columns of Y)
    subjects: np.ndarray
    x: np.ndarray           # group codes 0..k-1 (group) or variable values (corr)
    confounds: np.ndarray   # (n_subjects, n_columns), categorical confounds dummy-coded
    levels: list            # group levels in code order (group analyses only)


def encode_confounds(meta: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """Design columns of the confounds: numeric columns as is, other columns dummy-coded
    (first level dropped). Constant columns are dropped."""
    blocks = []
    for col in columns:
        numeric = pd.to_numeric(meta[col], errors="coerce")
        if numeric.notna().all():
            blocks.append(numeric.astype(float).to_frame(col))
        else:
            blocks.append(pd.get_dummies(meta[col].astype(str), prefix=col,
                                         drop_first=True, dtype=float))
    if not blocks:
        return np.empty((len(meta), 0))
    design = pd.concat(blocks, axis=1).to_numpy(dtype=float)
    return design[:, design.std(axis=0) > 0]


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Normal scores Phi^-1((rank - 0.5) / n) along axis 0 (average ranks for ties)."""
    values = np.asarray(values, dtype=float)
    return norm.ppf((rankdata(values, axis=0) - 0.5) / len(values))


def build_unit_data(pivot: pd.DataFrame, meta: pd.DataFrame, var: str, confounds: List[str],
                    kind: str, rank_transform: bool = False) -> Optional[UnitData]:
    """
    Keep the subjects with a complete profile and non-missing variable and
    confounds, and encode the variable (group codes, or numeric values).
    Returns None when there is not enough data to test.

    pivot: subjects x points metric values; meta: subject-level columns, same index.
    """
    if var not in meta.columns:
        return None
    conf_cols = [c for c in dict.fromkeys(confounds) if c in meta.columns and c != var]
    x = meta[var] if kind == "group" else pd.to_numeric(meta[var], errors="coerce")
    complete = pivot.notna().all(axis=1) & x.notna() & meta[conf_cols].notna().all(axis=1)
    pivot, x, meta = pivot[complete], x[complete], meta.loc[complete, conf_cols]
    if len(x) < MIN_SUBJECTS:
        return None

    levels = []
    if kind == "group":
        levels = sorted(x.unique(), key=str)
        if len(levels) < 2:
            return None
        x_values = x.map({level: i for i, level in enumerate(levels)}).to_numpy(dtype=int)
        if np.bincount(x_values).min() < 2:
            return None
    else:
        x_values = x.to_numpy(dtype=float)
        if rank_transform:
            x_values = normal_scores(x_values)

    Y = pivot.to_numpy(dtype=float)
    if rank_transform:
        Y = normal_scores(Y)
    return UnitData(
        Y=Y,
        points=pivot.columns.to_numpy(),
        subjects=pivot.index.to_numpy(),
        x=x_values,
        confounds=encode_confounds(meta, conf_cols),
        levels=levels,
    )


# ============================================================================
# POINTWISE TESTS
# ============================================================================
# A test returns its observed statistics (one value per point) and a function
# giving the p-values (and effect signs) of one permutation: Freedman-Lane for
# the GLM, permutation of the group labels / variable for "residualize".
# ============================================================================

# random generator -> (p-values, effect signs or None) of one permutation
Permutation = Callable[[np.random.Generator], Tuple[np.ndarray, Optional[np.ndarray]]]


@dataclass
class PointwiseTest:
    stats: Dict[str, np.ndarray]    # observed statistics per point, always with "p_raw"
    values: np.ndarray              # (n_subjects, n_points) values the group means are computed on
    permute: Permutation            # p-values and signs of one permutation of the same test
    sign: Optional[np.ndarray] = None   # observed effect sign (single-term tests), None for F-tests


def _sign(stat: np.ndarray) -> np.ndarray:
    return np.sign(np.nan_to_num(stat))


def _residualize(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Regress the confounds C out of Y (each column), keeping the intercept."""
    if C.shape[1] == 0:
        return Y
    Z = np.column_stack([np.ones(len(C)), C])
    beta = np.linalg.lstsq(Z, Y, rcond=None)[0]
    return Y - C @ beta[1:]


def _pearson(x: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pearson r and two-sided p-value between x and each column of Y."""
    n = len(x)
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    ss_y = (Yc ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.clip((xc @ Yc) / np.sqrt((xc @ xc) * ss_y), -1.0, 1.0)
        r[np.sqrt(ss_y) < 1e-13 * np.abs(Y.mean(axis=0))] = np.nan  # (near-)constant profile
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return r, 2 * t_dist.sf(np.abs(t), n - 2)


def _group_test(Y: np.ndarray, codes: np.ndarray, n_groups: int):
    """Welch t-test (group 0 vs group 1) or one-way ANOVA at each point."""
    groups = [Y[codes == i] for i in range(n_groups)]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # constant profiles -> NaN
        if n_groups == 2:
            return ttest_ind(groups[0], groups[1], axis=0, equal_var=False)
        return f_oneway(*groups, axis=0)


def _residualized_inputs(data: UnitData, kind: str, settings: Settings
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Metric with the confounds regressed out, and group codes (group) or the variable
    with the confounds regressed out (corr, unless CORRECT_METRIC_CONFOND_ONLY)."""
    Y = _residualize(data.Y, data.confounds)
    if kind == "group":
        return Y, data.x
    x = data.x if settings.correct_metric_only else _residualize(data.x, data.confounds)
    return Y, x


def residualize_test(data: UnitData, kind: str, settings: Settings) -> PointwiseTest:
    """Confounds regressed out by OLS, then t-test/ANOVA or correlation on the residuals.
    Permutations: group labels (group) or variable (corr) shuffled, same test."""
    Y, x = _residualized_inputs(data, kind, settings)

    if kind == "group":
        n_groups = len(data.levels)

        def group_test(codes):
            res = _group_test(Y, codes, n_groups)
            return res, (_sign(res.statistic) if n_groups == 2 else None)

        res, sign = group_test(data.x)
        diff = (Y[data.x == 1].mean(axis=0) - Y[data.x == 0].mean(axis=0) if n_groups == 2
                else np.full(Y.shape[1], np.nan))

        def permute(rng):
            res, sign = group_test(rng.permutation(data.x))
            return res.pvalue, sign

        return PointwiseTest(stats={"p_raw": res.pvalue, "diff": diff, "stat": res.statistic},
                             values=Y, permute=permute, sign=sign)

    if settings.correlation_test == "spearman":
        xt, Yt = rankdata(x), rankdata(Y, axis=0)
    else:
        xt, Yt = x, Y
    r, p = _pearson(xt, Yt)

    def permute(rng):
        r, p = _pearson(rng.permutation(xt), Yt)
        return p, _sign(r)

    return PointwiseTest(stats={"r": r, "p_raw": p}, values=Y, permute=permute, sign=_sign(r))


def _glm_design(data: UnitData, kind: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Design matrices of the GLM: full model X (confounds + variable terms), reduced
    model Z (confounds only), and the names of the variable terms in X."""
    Z = pd.DataFrame(data.confounds, columns=[f"conf{i}" for i in range(data.confounds.shape[1])])
    Z.insert(0, "const", 1.0)
    if kind == "group":
        # Dummy coding, level 0 = reference
        terms = pd.DataFrame(np.eye(len(data.levels))[data.x][:, 1:],
                             columns=[f"x_{_level_label(level)}" for level in data.levels[1:]])
    else:
        terms = pd.DataFrame({"x": data.x})
    return pd.concat([Z, terms], axis=1), Z, list(terms.columns)


class VectorizedGLM:
    """
    OLS of every column of Y on a fixed design X, vectorized over the columns, with
    the same tests as glm_test (statsmodels): t-test of a single term, Wald F-test of
    several terms, with classical or HC0-HC3 standard errors. Used for the permutations.
    """

    def __init__(self, X: np.ndarray, term_idx: List[int], cov_type: str):
        self.X = X
        self.pinv = np.linalg.pinv(X)
        self.term_idx = term_idx
        self.cov_type = cov_type
        self.n = len(X)
        self.dof = self.n - np.linalg.matrix_rank(X)
        self.leverage = np.einsum("ij,ji->i", X, self.pinv)

    def pvalues(self, Y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """p-values and, for a single term, sign of its coefficient."""
        beta = self.pinv @ Y
        resid = Y - self.X @ beta
        A = self.pinv[self.term_idx]                            # (n_terms, n): beta_terms = A @ Y
        b = beta[self.term_idx].T                               # (n_points, n_terms)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.cov_type == "nonrobust":
                sigma2 = (resid ** 2).sum(axis=0) / self.dof
                cov = (A @ A.T)[None] * sigma2[:, None, None]
            else:
                w = resid ** 2
                if self.cov_type == "HC1":
                    w = w * self.n / self.dof
                elif self.cov_type == "HC2":
                    w = w / (1 - self.leverage)[:, None]
                elif self.cov_type == "HC3":
                    w = w / ((1 - self.leverage) ** 2)[:, None]
                cov = np.einsum("ki,ij,li->jkl", A, w, A)       # (n_points, n_terms, n_terms)
            if len(self.term_idx) == 1:
                t = b[:, 0] / np.sqrt(cov[:, 0, 0])
                return 2 * t_dist.sf(np.abs(t), self.dof), _sign(t)
            F = np.einsum("jk,jkl,jl->j", b, np.linalg.pinv(cov), b) / len(self.term_idx)
            return f_dist.sf(F, len(self.term_idx), self.dof), None


def _freedman_lane(X: np.ndarray, Z: np.ndarray, Y: np.ndarray, term_idx: List[int],
                   cov_type: str) -> Permutation:
    """p-values (and signs) of one Freedman-Lane permutation: the residuals of the reduced model Z
    (confounds only) are permuted and added back to its fitted values, then the full
    model X is tested."""
    glm = VectorizedGLM(X, term_idx, cov_type)
    fitted = Z @ (np.linalg.pinv(Z) @ Y)
    resid = Y - fitted
    return lambda rng: glm.pvalues(fitted + resid[rng.permutation(len(Y))])


def glm_test(data: UnitData, kind: str, settings: Settings) -> Optional[PointwiseTest]:
    """
    OLS fitted with statsmodels at each point, metric ~ 1 + confounds + variable terms.

    - group: ANCOVA with robust standard errors (ANCOVA_COV_TYPE, HC3 by default):
      t-test of the adjusted difference (2 groups) or Wald F-test of the group terms;
    - corr: classical OLS, t-test of the slope; r is the partial correlation.

    Permutations: Freedman-Lane, with the same test.
    """
    X, Z, terms = _glm_design(data, kind)
    if (np.linalg.matrix_rank(X) - np.linalg.matrix_rank(Z) < len(terms)
            or len(X) - np.linalg.matrix_rank(X) < 1):
        return None  # variable collinear with the confounds, or too few subjects
    cov_type = settings.ancova_cov_type if kind == "group" else "nonrobust"
    contrast = np.zeros((len(terms), X.shape[1]))              # selects the variable terms
    for i, term in enumerate(terms):
        contrast[i, X.columns.get_loc(term)] = 1.0

    n_points = data.Y.shape[1]
    beta, se, t, p = (np.full(n_points, np.nan) for _ in range(4))
    conf_cols = list(Z.columns[1:])
    conf_beta = np.zeros((len(conf_cols), n_points))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant profiles -> NaN
        for j in range(n_points):
            fit = sm.OLS(data.Y[:, j], X).fit(cov_type=cov_type, use_t=True)
            conf_beta[:, j] = fit.params[conf_cols]
            if len(terms) == 1:
                term = terms[0]
                beta[j], se[j], t[j], p[j] = (fit.params[term], fit.bse[term],
                                              fit.tvalues[term], fit.pvalues[term])
            else:
                wald = fit.wald_test(contrast, use_f=True, scalar=True)
                t[j], p[j] = float(wald.statistic), float(wald.pvalue)  # F statistic
    dof = fit.df_resid

    if kind == "group":
        if len(terms) == 1:
            # diff = adjusted mean of group 1 - group 0; stat follows the sign
            # convention of ttest_ind(group 0, group 1) used by the residualize model
            stats = {"p_raw": p, "diff": beta, "diff_se": se, "stat": -t}
        else:
            stats = {"p_raw": p, "diff": np.full(n_points, np.nan), "stat": t}
    else:
        with np.errstate(invalid="ignore"):
            r = t / np.sqrt(t ** 2 + dof)                      # partial correlation
        stats = {"r": r, "p_raw": p, "beta": beta, "t": t}
    # Adjusted values: confounds set to their mean using the full-model coefficients,
    # so the group difference of the adjusted means is the tested one
    C = data.confounds
    values = data.Y - (C - C.mean(axis=0)) @ conf_beta
    permute = _freedman_lane(X.to_numpy(dtype=float), Z.to_numpy(dtype=float), data.Y,
                             [X.columns.get_loc(term) for term in terms], cov_type)
    return PointwiseTest(stats=stats, values=values, permute=permute,
                         sign=_sign(beta) if len(terms) == 1 else None)


def make_test(data: UnitData, kind: str, settings: Settings) -> Optional[PointwiseTest]:
    model = glm_test if settings.stat_model == "glm" else residualize_test
    return model(data, kind, settings)


# ============================================================================
# FAMILY-WISE ERROR CORRECTION (AFQ / TFCE)
# ============================================================================

def _rng(settings: Settings, key: str) -> np.random.Generator:
    """Random generator of one analysis; with RANDOM_SEED it only depends on the seed and
    the analysis key, so results do not depend on how jobs are scheduled."""
    if settings.random_seed is None:
        return np.random.default_rng()
    return np.random.default_rng([int(settings.random_seed), zlib.crc32(key.encode())])


def segment_starts(points: np.ndarray) -> Optional[np.ndarray]:
    """Points starting a new bundle/centroid in profiles stacked by stack_bundles (their
    points are (label, point) tuples), where TFCE clusters must not cross; None otherwise."""
    if len(points) == 0 or not isinstance(points[0], tuple):
        return None
    labels = np.array([p[0] for p in points], dtype=object)
    return np.r_[False, labels[1:] != labels[:-1]]


def tfce(p: np.ndarray, sign: Optional[np.ndarray], settings: Settings,
         starts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    1D Threshold-Free Cluster Enhancement of a p-value profile (see module docstring).
    The p-values are converted to z-scores, signed when `sign` is given (positive and
    negative effects enhanced separately); the result carries the sign of the effect.
    `starts` marks the points where a new segment starts (no cluster across them).
    """
    p = np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=1.0), 1e-300, 1.0)
    z = norm.isf(p / 2)
    if sign is not None:
        z = z * sign
    dh, n = settings.tfce_dh, len(z)
    out = np.zeros(n)
    for direction in ((1, -1) if sign is not None else (1,)):
        zs = np.maximum(direction * z, 0.0)
        if zs.max() < dh:
            continue
        heights = np.arange(1, int(zs.max() / dh + 1e-9) + 1) * dh
        above = zs[None, :] >= heights[:, None]                         # (n_heights, n)
        new_run = above & ~np.pad(above, ((0, 0), (1, 0)))[:, :-1]      # first point of each run
        if starts is not None:
            new_run |= above & starts[None, :]
        run_id = np.cumsum(new_run.ravel()).reshape(above.shape)
        extent = np.bincount(run_id[above], minlength=run_id.max() + 1)[run_id]
        enhanced = np.where(above, extent.astype(float) ** settings.tfce_e, 0.0)
        out += direction * (enhanced * (heights ** settings.tfce_h)[:, None]).sum(axis=0) * dh
    return out


@dataclass(frozen=True)
class FWEThresholds:
    """FWE thresholds of one analysis. tfce_null: sorted (descending) max |TFCE| over
    the profile of each permutation, only computed with the TFCE method."""
    alpha_fwe: float = np.nan
    cluster_fwe: float = np.nan
    tfce_null: Optional[np.ndarray] = None

    def tfce_pvalues(self, tfce_values: np.ndarray) -> np.ndarray:
        """FWE-corrected p-values: (1 + #{permutations with max|TFCE| >= |TFCE|}) / (nperm + 1)."""
        if self.tfce_null is None or len(self.tfce_null) == 0:
            return np.full(len(tfce_values), np.nan)
        null = np.sort(self.tfce_null)
        n_ge = len(null) - np.searchsorted(null, np.abs(tfce_values) - 1e-12, side="left")
        return (1 + n_ge) / (len(null) + 1)

    def tfce_fwe(self, alpha: float) -> float:
        """Smallest |TFCE| level exceeded by the significant points (p_tfce <= alpha)."""
        if self.tfce_null is None or len(self.tfce_null) == 0:
            return np.nan
        c = int(np.floor(alpha * (len(self.tfce_null) + 1) + 1e-9)) - 1
        return float(self.tfce_null[c]) if c >= 0 else np.inf


def fwe_thresholds(test: PointwiseTest, settings: Settings, rng: np.random.Generator,
                   starts: Optional[np.ndarray] = None) -> FWEThresholds:
    """
    Thresholds of every FWE method from the same AFQ_NPERM permutations of the test
    (same convention as AFQ_MultiCompCorrection for alphaFWE / clusterFWE):

    - alphaFWE: alpha-quantile of the min p-value over the points;
    - clusterFWE: (1 - alpha)-quantile of the longest run of points with p < alpha,
      plus one, i.e. the minimum length of a significant run (strictly longer than
      the null quantile, valid despite the ties of the discrete run lengths);
    - TFCE: null distribution of max |TFCE|.

    `starts` (stacked multi-tract profiles) marks the first point of each bundle /
    centroid: runs and TFCE clusters are cut there, as they are in the per-bundle
    observed profiles.
    """
    nperm = settings.nperm
    if nperm < 1:
        return FWEThresholds()
    p_min = np.full(nperm, np.nan)
    cluster_max = np.zeros(nperm)
    tfce_max = np.zeros(nperm)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # constant profiles -> NaN
        for i in range(nperm):
            p, sign = test.permute(rng)
            if np.isfinite(p).any():
                p_min[i] = np.nanmin(p)
            cluster_max[i] = max((end - start for start, end
                                  in true_runs(p < settings.alpha, starts)), default=0)
            tfce_max[i] = np.abs(tfce(p, sign, settings, starts)).max(initial=0.0)
    k = min(int(round(settings.alpha * nperm)), nperm - 1)
    return FWEThresholds(alpha_fwe=float(np.sort(p_min)[k]),
                         cluster_fwe=float(np.sort(cluster_max)[::-1][k]) + 1,
                         tfce_null=np.sort(tfce_max)[::-1])


def run_pointwise_analysis(data: UnitData, kind: str, settings: Settings,
                           thresholds: Optional[FWEThresholds], rng_key: str
                           ) -> Optional[Tuple[pd.DataFrame, PointwiseTest]]:
    """
    Pointwise test + FWE correction. `thresholds` are computed beforehand
    (multi-tract), else computed here.

    Result columns: point, the test statistics (group: p_raw, diff, stat, ...;
    corr: r, p_raw, n, ...), tfce, p_tfce, sig_<method> for every FWE method,
    sig_afq (= sig_<FWE_METHOD>), alphaFWE, clusterFWE, tfceFWE, and for groups
    mean_g<i> / n<i> per group.
    """
    test = make_test(data, kind, settings)
    if test is None:
        return None
    starts = segment_starts(data.points)
    if thresholds is None:
        thresholds = fwe_thresholds(test, settings, _rng(settings, rng_key), starts)

    res = pd.DataFrame({"point": data.points, **test.stats})
    if kind == "corr":
        res.insert(3, "n", len(data.x))
    p = res["p_raw"].to_numpy(dtype=float)
    res["tfce"] = tfce(p, test.sign, settings, starts)
    res["p_tfce"] = thresholds.tfce_pvalues(res["tfce"].to_numpy())
    for method in FWE_METHODS:
        res[f"sig_{method}"] = (res["p_tfce"].to_numpy() <= settings.alpha if method == "TFCE"
                                else significance_mask(p, thresholds.alpha_fwe,
                                                       thresholds.cluster_fwe, method,
                                                       settings.alpha))
    res["sig_afq"] = res[f"sig_{settings.fwe_method}"]
    res["alphaFWE"] = thresholds.alpha_fwe
    res["clusterFWE"] = thresholds.cluster_fwe
    res["tfceFWE"] = thresholds.tfce_fwe(settings.alpha)
    if kind == "group":
        for i in range(len(data.levels)):
            group_values = test.values[data.x == i]
            res[f"mean_g{i}"] = group_values.mean(axis=0)
            res[f"n{i}"] = len(group_values)
    return res, test


# ============================================================================
# OUTPUTS: CSV & FIGURES
# ============================================================================

def group_table(data: UnitData, values: np.ndarray) -> pd.DataFrame:
    """Mean / std / n of the analysed values per group and point."""
    table = {"point": data.points}
    for i, level in enumerate(data.levels):
        group_values = values[data.x == i]
        table[f"mean_{level}"] = group_values.mean(axis=0)
        table[f"std_{level}"] = group_values.std(axis=0, ddof=1)
        table[f"n_{level}"] = len(group_values)
    return pd.DataFrame(table)


def _shade_clusters(ax, points, sig):
    for start, end in true_runs(sig):
        ax.axvspan(points[start], points[end - 1], color="red", alpha=0.18, linewidth=0)


def _plot_pvalues(ax, res: pd.DataFrame, settings: Settings):
    """p-value bars and FWE thresholds on a secondary axis."""
    ax2 = ax.twinx()
    p = res["p_raw"].clip(0, 1)
    ax2.bar(res["point"], p, color="gray", alpha=0.2, width=1.0, label="p-value")
    ymax = min(1.0, max(0.06, float(p.max()) * 1.15)) if p.notna().any() else 1.0
    alpha_fwe, cluster_fwe = res["alphaFWE"].iloc[0], res["clusterFWE"].iloc[0]
    if settings.fwe_method in ("alphaFWE", "mixed") and np.isfinite(alpha_fwe):
        ax2.axhline(alpha_fwe, color="green", linestyle=":", linewidth=1.2,
                    label=f"alphaFWE={alpha_fwe:.3g}")
    if settings.fwe_method in ("clusterFWE", "mixed") and np.isfinite(cluster_fwe):
        ax2.text(0.01, 0.95, f"clusterFWE>={int(cluster_fwe)} (alpha={settings.alpha})",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="gray"))
    if settings.fwe_method == "TFCE":
        p_tfce = res["p_tfce"].clip(0, 1)
        ax2.plot(res["point"], p_tfce, color="purple", linewidth=1.2, label="p TFCE (FWE)")
        ax2.axhline(settings.alpha, color="purple", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("p-value")
    ax2.set_ylim(0, ymax)
    return ax2


def _finish_figure(fig, ax, ax2, title, out_png, removed_subjects, removed_points):
    lines = []
    for name, items, max_items in (("subjects", removed_subjects, 8), ("points", removed_points, 12)):
        if items:
            text = ", ".join(map(str, items[:max_items])) + ("..." if len(items) > max_items else "")
            lines.append(f"Removed {name} ({len(items)}): {text}")
    if lines:
        ax.text(0.01, 0.99, "\n".join(lines), ha="left", va="top", transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.65, edgecolor="gray"))
    handles, labels = [], []
    for a in (ax, ax2):
        h, l = a.get_legend_handles_labels()
        handles += h
        labels += l
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Point")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_group_figure(data, values, res, var, metric_col, label, out_png, settings,
                      removed_subjects, removed_points):
    """Mean profile +/- 95% CI per group, p-values and significant clusters."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, level in enumerate(data.levels):
        group_values = values[data.x == i]
        mean = group_values.mean(axis=0)
        ci = 1.96 * group_values.std(axis=0, ddof=1) / np.sqrt(len(group_values))
        ax.plot(data.points, mean, label=f"{var}={level} (n={len(group_values)})")
        ax.fill_between(data.points, mean - ci, mean + ci, alpha=0.2)
    _shade_clusters(ax, data.points, res["sig_afq"].to_numpy())
    ax.set_ylabel(f"{metric_col} (confound-adjusted)" if settings.stat_model == "glm"
                  else f"{metric_col} (confound-corrected)")
    ax2 = _plot_pvalues(ax, res, settings)
    title = f"{label} - {metric_col} - {var} ({settings.model_label}, {settings.fwe_method})"
    _finish_figure(fig, ax, ax2, title, out_png, removed_subjects, removed_points)


def save_corr_figure(res, var, metric_col, label, out_png, settings,
                     removed_subjects, removed_points):
    """Pointwise r, p-values and significant clusters."""
    fig, ax = plt.subplots(figsize=(10, 5))
    points = res["point"].to_numpy()
    _shade_clusters(ax, points, res["sig_afq"].to_numpy())
    ax.plot(points, res["r"], color="tab:blue", label="r")
    sig = res[res["sig_afq"]]
    if not sig.empty:
        ax.scatter(sig["point"], sig["r"], color="red", s=22, label="significant")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("partial r (GLM)" if settings.stat_model == "glm"
                  else f"r ({settings.correlation_test})")
    ax2 = _plot_pvalues(ax, res, settings)
    title = f"{label} - {metric_col} - correlation {var} ({settings.model_label}, {settings.fwe_method})"
    _finish_figure(fig, ax, ax2, title, out_png, removed_subjects, removed_points)


# ============================================================================
# ANALYSES OF ONE BUNDLE
# ============================================================================

@dataclass
class Profiles:
    """Cleaned profiles of one metric x bundle/centroid, for the subjects with the variable."""
    pivot: pd.DataFrame         # subjects x points
    meta: pd.DataFrame          # subject-level columns, same index
    removed_subjects: list
    removed_points: list
    candidates: Set[str]        # subjects with the variable, before cleaning


def iter_units(bundle_name: str, df: pd.DataFrame) -> Iterator[Tuple[Optional[int], str, pd.DataFrame]]:
    """(centroid_id, label, data) of each analysis unit of a bundle: one per centroid when
    the CSVs have a 'centroid_id' column, else the whole bundle."""
    if "centroid_id" not in df.columns:
        yield None, bundle_name, df
        return
    for cid, df_unit in df.groupby("centroid_id", sort=True):
        if isinstance(cid, (int, float, np.integer, np.floating)) and float(cid).is_integer():
            cid = int(cid)
        yield cid, f"{bundle_name}_cent{cid}", df_unit


def prepare_profiles(df_unit: pd.DataFrame, metric_col: str, var: str,
                     settings: Settings) -> Optional[Profiles]:
    if metric_col not in df_unit.columns or var not in df_unit.columns:
        return None
    try:
        long_df = to_long(df_unit, metric_col, settings.metric_candidates)
    except ValueError:
        return None
    long_df = long_df[long_df[var].notna()]
    candidates = set(long_df["subject"].dropna().astype(str))
    long_df, removed_subjects, removed_points = clean_missing_data(
        long_df, tolerance=settings.missing_subjects_tolerance,
        interpolate_max_gap=settings.interpolate_max_gap,
        keep_largest_contiguous=settings.keep_largest_contiguous)
    if long_df.empty:
        return None
    pivot = long_df.pivot_table(index="subject", columns="point", values="value", aggfunc="mean")
    meta = long_df.drop_duplicates("subject").set_index("subject").reindex(pivot.index)
    return Profiles(pivot, meta, removed_subjects, removed_points, candidates)


def analyse_bundle(bundle_name: str, df: pd.DataFrame, metric_set: List[str], settings: Settings,
                   report_dir: str, thresholds: Dict[Tuple[str, str, str], FWEThresholds]
                   ) -> Dict:
    """
    Run every variable x metric x centroid analysis of one bundle, write one
    CSV (and figure) per analysis in <report_dir>/figures and return the
    summary rows and the participants used.

    thresholds: precomputed FWE thresholds per (kind, variable, metric)
    when multi-tract correction is on.
    """
    figure_dir = opj(report_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    rows, included, candidates = [], set(), set()

    for kind, var, confounds in settings.variables():
        for metric_col in metric_set:
            for cid, label, df_unit in iter_units(bundle_name, df):
                profiles = prepare_profiles(df_unit, metric_col, var, settings)
                if profiles is None:
                    continue
                data = build_unit_data(profiles.pivot, profiles.meta, var, confounds, kind,
                                       settings.rank_transform)
                outcome = None
                if data is not None:
                    outcome = run_pointwise_analysis(
                        data, kind, settings, thresholds.get((kind, var, metric_col)),
                        rng_key=f"{label}:{metric_col}:{kind}:{var}")

                used = set(map(str, data.subjects)) if outcome else set()
                included |= used
                candidates |= profiles.candidates

                res = outcome[0] if outcome else pd.DataFrame()
                if outcome:
                    suffix = "group_corrected" if kind == "group" else "corr_partial"
                    stem = opj(figure_dir, f"{label}_{metric_col}_{var}_{suffix}")
                    if kind == "group":
                        table = group_table(data, outcome[1].values).join(res.drop(columns="point"))
                    else:
                        table = res.copy()
                    table = table.assign(
                        bundle=bundle_name,
                        centroid_id=cid if cid is not None else np.nan,
                        metric=metric_col,
                        **{"classif" if kind == "group" else "variable": var},
                        analysis="corrected" if kind == "group" else "partial",
                        stat_model=settings.stat_model,
                    )
                    table.to_csv(stem + ".csv", index=False)

                    if settings.save_figures:
                        if kind == "group":
                            save_group_figure(data, outcome[1].values, res, var, metric_col, label,
                                              stem + ".png", settings,
                                              profiles.removed_subjects, profiles.removed_points)
                        else:
                            save_corr_figure(res, var, metric_col, label, stem + ".png", settings,
                                             profiles.removed_subjects, profiles.removed_points)

                rows.append({
                    "bundle": bundle_name,
                    "centroid_id": cid if cid is not None else np.nan,
                    "metric": metric_col,
                    "type": f"{kind}_{var}",
                    "n_sig": 0 if res.empty else n_significant(res["sig_afq"], settings.fwe_method),
                    **{f"n_sig_{m}": 0 if res.empty else n_significant(res[f"sig_{m}"], m)
                       for m in FWE_METHODS},
                    "min_p": np.nan if res.empty else float(res["p_raw"].min()),
                    "min_p_tfce": np.nan if res.empty else float(res["p_tfce"].min()),
                    "max_abs_r": (float(res["r"].abs().max())
                                  if kind == "corr" and not res.empty else np.nan),
                    "removed_points": len(profiles.removed_points),
                    "subjects_used": len(used),
                    "removed_subjects": len(profiles.candidates - used),
                    "confounds_used": ",".join(confounds) if confounds else "none",
                    **{key: np.nan if res.empty else float(res[key].iloc[0])
                       for key in ("alphaFWE", "clusterFWE", "tfceFWE")},
                    "fwe_method": settings.fwe_method,
                    "stat_model": settings.stat_model,
                })
        gc.collect()

    return {"summary": rows,
            "included": {normalize_participant_id(s) for s in included},
            "candidates": {normalize_participant_id(s) for s in candidates}}


# ============================================================================
# MULTI-TRACT CORRECTION
# ============================================================================

def stack_bundles(bundle_data: Dict[str, pd.DataFrame], kind: str, var: str, confounds: List[str],
                  metric_col: str, settings: Settings) -> Optional[UnitData]:
    """
    Concatenate the profiles of all bundles/centroids side by side (one row per
    subject), to estimate FWE thresholds valid over all the bundles. Values of
    a subject missing from a bundle are replaced by the point's mean.
    """
    pivots, metas = [], []
    for bundle_name, df in bundle_data.items():
        for _, label, df_unit in iter_units(bundle_name, df):
            profiles = prepare_profiles(df_unit, metric_col, var, settings)
            if profiles is None:
                continue
            pivot = profiles.pivot.copy()
            pivot.columns = pd.MultiIndex.from_product([[label], pivot.columns])
            pivots.append(pivot)
            metas.append(profiles.meta)
    if not pivots:
        return None

    pivot = pd.concat(pivots, axis=1)
    pivot = pivot.fillna(pivot.mean())
    meta = pd.concat(metas).groupby(level=0).first().reindex(pivot.index)
    return build_unit_data(pivot, meta, var, confounds, kind, settings.rank_transform)


def multi_tract_thresholds(data: UnitData, kind: str, settings: Settings,
                           rng_key: str) -> FWEThresholds:
    test = make_test(data, kind, settings)
    if test is None:
        return FWEThresholds()
    return fwe_thresholds(test, settings, _rng(settings, rng_key), segment_starts(data.points))


# ============================================================================
# REPORT
# ============================================================================

def _run_jobs(func, jobs: list, n_jobs: int, desc: str) -> list:
    """Run func(*job) for each job, in parallel processes when n_jobs > 1."""
    if n_jobs > 1 and len(jobs) > 1:
        print(f"{desc}: {len(jobs)} jobs on {n_jobs} processes")
        return Parallel(n_jobs=n_jobs)(delayed(func)(*job) for job in jobs)
    return [func(*job) for job in tqdm(jobs, desc=desc)]


def load_bundles(csv_files: list, bundle_names: List[str], settings: Settings,
                 subject_info: Optional[pd.DataFrame], keep_only_listed: bool
                 ) -> Tuple[Dict[str, pd.DataFrame], Settings]:
    """Load every bundle (merge, test_by_group split, resampling, exclusions).
    Returns the bundles and the settings extended with the test_by_group variables."""
    files_by_bundle = defaultdict(list)
    for f in csv_files:
        files_by_bundle[f.get_entities()["bundle"]].append(f)

    bundle_data, split_vars = {}, {}
    for bundle_name in tqdm([bn.replace("_", "") for bn in bundle_names], desc="Loading bundles"):
        if not files_by_bundle.get(bundle_name):
            continue
        df = load_bundle(files_by_bundle[bundle_name], subject_info, keep_only_listed)
        df, new_vars = add_group_split_columns(df, settings)
        split_vars.update(new_vars)
        if settings.resample_n_points:
            df = resample_bundle_data(df, settings.resample_n_points,
                                      metric_candidates=settings.metric_candidates)
        bundle_data[bundle_name] = apply_exclusions(df, settings.exclude)

    corr_variables = {**settings.corr_variables,
                      **{v: c for v, c in split_vars.items() if v not in settings.corr_variables}}
    return bundle_data, replace(settings, corr_variables=corr_variables)


def generate_report(settings: Settings, csv_files: list, bundle_names: List[str], report_dir: str,
                    n_jobs: int = 1, subject_info: Optional[pd.DataFrame] = None,
                    keep_only_listed: bool = False) -> None:
    """
    1. Load all the bundles.
    2. Detect the available metrics.
    3. Multi-tract: estimate the FWE thresholds over all the bundles.
    4. Run the analyses bundle by bundle (in parallel when n_jobs > 1).
    5. Save the summary CSV and the list of removed participants.
    """
    os.makedirs(report_dir, exist_ok=True)
    print(f"Statistical model: {settings.model_label}")
    if settings.stat_model == "glm" and settings.correlation_test != "pearson":
        print(f"[NOTE] CORRELATION_TEST='{settings.correlation_test}' is ignored by the GLM model")
    print(f"FWE correction: {settings.fwe_method} (alpha={settings.alpha}, {settings.nperm} permutations)")
    if settings.fwe_method == "TFCE":
        print(f"TFCE: E={settings.tfce_e}, H={settings.tfce_h}, dh={settings.tfce_dh}")
    print(f"Multi-tract correction: {'ENABLED' if settings.multi_tract else 'disabled'}")
    print(f"Resampling: {f'{settings.resample_n_points} points' if settings.resample_n_points else 'disabled'}")
    print(f"Stat types: {list(settings.stat_types) or 'all (bare columns)'}")
    print(f"Rank transform (normal scores): {'ENABLED' if settings.rank_transform else 'disabled'}")

    bundle_data, settings = load_bundles(csv_files, bundle_names, settings,
                                         subject_info, keep_only_listed)
    if not bundle_data:
        print("No bundles to analyse.")
        return
    print(f"Analysing bundles: {', '.join(bundle_data)}")

    metric_set = sorted({m for df in bundle_data.values()
                         for m in detect_metric_columns(df, settings.metric_candidates, settings.stat_types)})
    print(f"Detected metrics: {', '.join(metric_set) if metric_set else '(none)'}")

    thresholds = {}
    if settings.multi_tract:
        keys, jobs = [], []
        for kind, var, confounds in settings.variables():
            for metric_col in metric_set:
                stacked = stack_bundles(bundle_data, kind, var, confounds, metric_col, settings)
                if stacked is not None:
                    keys.append((kind, var, metric_col))
                    jobs.append((stacked, kind, settings, f"multi:{metric_col}:{kind}:{var}"))
        results = _run_jobs(multi_tract_thresholds, jobs, n_jobs, "Multi-tract thresholds")
        thresholds = dict(zip(keys, results))
        # Analyses without a stacked dataset get no threshold (nothing significant)
        for kind, var, _ in settings.variables():
            for metric_col in metric_set:
                thresholds.setdefault((kind, var, metric_col), FWEThresholds())

    jobs = [(name, df, metric_set, settings, report_dir, thresholds)
            for name, df in bundle_data.items()]
    results = _run_jobs(analyse_bundle, jobs, n_jobs, "Bundles")

    summary_rows, included, candidates = [], set(), set()
    for r in results:
        summary_rows += r["summary"]
        included |= r["included"]
        candidates |= r["candidates"]
    if not summary_rows:
        print("No data for the report.")
        return

    summary_csv = opj(report_dir, "summary_results.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Summary CSV: {summary_csv}")

    # Participants never used: reference = subjects table when given
    reference = set(subject_info["participant_id"]) if subject_info is not None else candidates
    removed = sorted(reference - included)
    removed_csv = opj(report_dir, "removed_participants.csv")
    pd.DataFrame({"participant_id": removed}).to_csv(removed_csv, index=False)
    print(f"Removed participants: {len(removed)} (see {removed_csv})")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AFQ tractometry analysis report generator")
    parser.add_argument("-c", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--subjects-table", "-s", type=str, default=None,
                        help="CSV/XLSX with a 'participant_id' column listing the subjects to "
                             "include; its other columns are available as variables")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for reports (overrides config)")
    args = parser.parse_args()

    config = load_config(args.c)
    validate_config(config)
    settings = Settings.from_config(config)

    # --- Subject-level variables ---
    subject_info, keep_only_listed = None, False
    if args.subjects_table:
        if not os.path.exists(args.subjects_table):
            raise FileNotFoundError(f"Subjects table not found: {args.subjects_table}")
        subject_info, keep_only_listed = load_subject_info(args.subjects_table), True
        print(f"Subjects table loaded: {len(subject_info)} participant(s) to process")
    elif config.get("additional_info_path"):
        subject_info = load_subject_info(config["additional_info_path"])
        print(f"Additional info table loaded: {len(subject_info)} participant(s)")
    if subject_info is not None:
        check_config_columns(subject_info, settings)

    # --- Dataset ---
    ds = Dataset(config["dataset"])
    csv_files = ds.get_global(pipeline=config["hcp_asso_pipeline"], extension="csv",
                              datatype="metric", suffix="mean")
    bundle_names = list(get_HCP_bundle_names().keys())

    # --- Output directory ---
    output_dir = args.output_dir or config.get("output_dir")
    if output_dir is None:
        pipeline = config.get("pipeline", DEFAULT_CONFIG["pipeline"])
        tag = "glm" if settings.stat_model == "glm" else settings.correlation_test
        output_dir = opj(os.getcwd(), f"report_{pipeline}_{settings.fwe_method}_{tag}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Report output: {output_dir}")
    save_config(config, opj(output_dir, "config.json"))

    # --- Parallel jobs ---
    n_jobs = config.get("n_jobs") or max(1, os.cpu_count() // 3)
    if not config.get("MULTIPROCESSING", DEFAULT_CONFIG["MULTIPROCESSING"]):
        n_jobs = 1

    generate_report(settings, csv_files, bundle_names, output_dir, n_jobs=n_jobs,
                    subject_info=subject_info, keep_only_listed=keep_only_listed)


if __name__ == "__main__":
    main()
