#!/usr/bin/env python
"""
AFQ Analysis Report Generator
=============================

This script performs tractometry statistical analysis on AFQ (Automated Fiber
Quantification) outputs. It supports:

- Group comparisons (t-tests / ANOVA) between categorical variables
- Pointwise correlations with continuous variables
- Confound correction via OLS residualization
- Multiple comparison correction (clusterFWE, alphaFWE, or mixed)
- Optional multi-tract correction (across all bundles)
- Automated report generation with plots and CSV summaries

Configuration is driven by a JSON file. See DEFAULT_CONFIG for all available
options. Each variable (group or correlation) can specify its own list of
confounds to control for.

Usage:
    python AFQ_analysis.py -c config.json [--subjects-table subjects.csv] [--output-dir /path/to/output]
"""

import os
import re
import gc
import json
import argparse
from os.path import join as opj
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import ttest_ind, f_oneway, spearmanr, pearsonr

# Third-party imports for tractometry analysis
from tractseg.libs.AFQ_MultiCompCorrection import get_significant_areas
from TractoPL.analysis.afq_optimized import AFQ_MultiCompCorrection
from TractoPL.set_config import get_HCP_bundle_names
from TractoPL.data.loader import Dataset


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================
# This serves as a fallback when no config file is provided, and documents
# all available options. Each variable (group or correlation) maps to a dict
# with a "confounds" key listing the confounding variables to regress out.
# ============================================================================

DEFAULT_CONFIG = {
    "pipeline": "default",
    "dataset": "/PATH/TO/DATASET",
    "hcp_asso_pipeline": "tractometry",

    # --- Variables to analyse ---
    # Each variable maps to a dict with a "confounds" key listing the
    # confounding variables to regress out before testing.
    # "group_variables": {
    #     "group": {"confounds": ["age"]},
    # },
    # "corr_variables": {
    #     "some_continuous_var": {"confounds": ["age"]},
    # },

    # --- Optional: split a continuous variable by group levels ---
    # If set, creates var_by_<group>_<level> columns for interaction analysis
    "test_by_group": None,

    # --- Multiple comparison correction ---
    "FWE_METHOD": "clusterFWE",   # "clusterFWE", "alphaFWE", or "mixed"
    "AFQ_ALPHA": 0.05,
    "AFQ_NPERM": 1000,
    "CORRECT_MULTI_TRACT": False,  # Correct across all bundles (slower)

    # --- Data quality ---
    "MISSING_SUBJECTS_TOLERANCE": 0,   # Max NaN subjects per point
    "INTERPOLATE_MISSING_POINTS": 2,    # Max gap for linear interpolation
    "KEEP_LARGEST_CONTIGUOUS": True,    # Keep only the longest valid segment
    "RESAMPLE_N_POINTS": None,          # Resample to N points per bundle

    # --- Metric detection ---
    "METRIC_COLUMNS_CANDIDATES": ["FA"],
    "STAT_TYPES": ["mean"],

    # --- Correlation settings ---
    "CORRELATION_TEST": "pearson",  # "pearson" or "spearman"
    "CORRECT_METRIC_CONFOND_ONLY": False,

    # --- Optional external data (paths to Excel/CSV files) ---
    # If provided, these tables will be merged into the bundle DataFrames
    # by participant_id. Leave as None to skip.
    "additional_info_path": None,

    # --- Exclusion filter ---
    # Key = column name, Value = list of values to exclude
    "EXCLUDE": {},

    # --- Execution ---
    "MULTIPROCESSING": True,
    "n_jobs": None,  # None = auto (cpu_count // 3)
    "output_dir": None,  # None = auto-generated
}


# ============================================================================
# CONFIGURATION LOADING / SAVING
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from a JSON file, or return the default config.

    Parameters
    ----------
    config_path : str or None
        Path to a JSON configuration file. If None or not found,
        the default configuration is used.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"Configuration loaded from: {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config ({config_path}): {e}")
            print("Falling back to default configuration.")
            return DEFAULT_CONFIG.copy()
    else:
        if config_path:
            print(f"Config file not found ({config_path}). Using default configuration.")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict, output_path: str) -> None:
    """Save the configuration dictionary to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {output_path}")


# ============================================================================
# GLOBAL STATE
# ============================================================================
# These are populated at startup (in main()) and shared across worker
# functions via the module namespace.
# ============================================================================

config: Dict = {}
dataset: str = ""
hcp_asso_pipeline: str = ""
csv_files: list = []
bundle_names: list = []
ds = None

# Variable definitions (populated from config)
# Each entry: {var_name: {"confounds": [...]}}
group_variables: Dict = {}
corr_variables: Dict = {}

# Analysis hyperparameters (populated from config)
AFQ_ALPHA: float = 0.05
AFQ_NPERM: int = 1000
FWE_METHOD: str = "clusterFWE"
CORRECT_MULTI_TRACT: bool = False
RESAMPLE_N_POINTS: Optional[int] = None
CORRECT_METRIC_CONFOND_ONLY: bool = False
CORRELATION_TEST: str = "pearson"
METRIC_COLUMNS_CANDIDATES: List[str] = []
STAT_TYPES: List[str] = []

# Optional external data
_ADDITIONAL_INFO_DF: pd.DataFrame = pd.DataFrame()

# Subject filtering
_SUBJECTS_FILTER: Optional[Set[str]] = None
_SUBJECTS_TABLE_DF: Optional[pd.DataFrame] = None


# ============================================================================
# SUBJECT TABLE LOADING
# ============================================================================

def load_subjects_table(table_path: str) -> Tuple[Set[str], pd.DataFrame]:
    """
    Load a subjects table from CSV or XLSX.

    The table must contain a 'participant_id' column. IDs are normalised
    to the 'sub-XXX' format.

    Returns
    -------
    participant_ids : set of str
        Normalised participant IDs to include in the analysis.
    df : pd.DataFrame
        The full table (for column validation and merging).
    """
    if table_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(table_path)
    else:
        df = pd.read_csv(table_path)

    if "participant_id" not in df.columns:
        raise ValueError(
            f"Subjects table '{table_path}' must contain a 'participant_id' column."
        )

    participant_ids = set()
    for val in df["participant_id"].dropna():
        pid = str(val).strip()
        if pid:
            participant_ids.add(pid if pid.startswith("sub-") else f"sub-{pid}")

    return participant_ids, df


def validate_config_columns_in_table(subjects_df: pd.DataFrame, config: Dict) -> None:
    """
    Check that all variables referenced in the config exist as columns
    in the subjects table. Print warnings for any missing variables.
    """
    all_config_vars = set()

    # Collect group variable names and their confounds
    for var_name, var_cfg in config.get("group_variables", {}).items():
        all_config_vars.add(var_name)
        all_config_vars.update(var_cfg.get("confounds", []))

    # Collect correlation variable names and their confounds
    for var_name, var_cfg in config.get("corr_variables", {}).items():
        all_config_vars.add(var_name)
        all_config_vars.update(var_cfg.get("confounds", []))

    # Add the test_by_group variable if specified
    if config.get("test_by_group"):
        all_config_vars.add(config["test_by_group"])

    table_cols = set(subjects_df.columns)
    missing = all_config_vars - table_cols
    present = all_config_vars & table_cols

    if missing:
        print(f"[WARNING] Config variables not found in subjects table: {sorted(missing)}")
    if present:
        print(f"[OK] Config variables found in subjects table: {sorted(present)}")


# ============================================================================
# EXTERNAL TABLE LOADING (optional)
# ============================================================================

@lru_cache(maxsize=1)
def _load_external_tables(
    additional_info_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load optional external data tables (additional info).
    Returns empty DataFrames if paths are None or files cannot be read.
    """
    add_df = pd.DataFrame()
    act_df = pd.DataFrame()

    if additional_info_path and os.path.exists(additional_info_path):
        try:
            add_df = pd.read_excel(additional_info_path)
        except Exception as e:
            print(f"Could not load additional info table: {e}")

    return add_df, act_df


# ============================================================================
# DATA MERGING & PREPARATION
# ============================================================================

def load_and_merge_bundle_csvs(
    bundle_name: str, bundle_csvs: list
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and merge per-subject CSV files for a single bundle.

    This function:
    1. Reads all CSV files for the given bundle.
    2. Filters to the requested subjects (if a subjects table was provided).
    3. Merges in optional external data (additional info).
    4. If 'test_by_group' is configured, creates per-group interaction columns
       for each correlation variable.

    Returns
    -------
    metrics_df : pd.DataFrame
        Merged DataFrame with all subjects and optional external columns.
    metric_files_dict : dict
        Mapping from subject ID to the original file object.
    """

    metric_files_dict = {f.get_full_entities()["subject"]: f for f in bundle_csvs}
    metric_files = [pd.read_csv(f.path) for f in bundle_csvs]

    for df, f in zip(metric_files, bundle_csvs):
        df["subject"] = f.get_full_entities()["subject"]
        df["participant_id"] = "sub-" + df["subject"].astype(str)

    metrics_df = pd.concat(metric_files, ignore_index=True)

    # --- Filter to requested subjects ---
    if _SUBJECTS_FILTER is not None:
        metrics_df = metrics_df[metrics_df["participant_id"].isin(_SUBJECTS_FILTER)]

    # --- Merge subjects table columns (if provided) ---
    if _SUBJECTS_TABLE_DF is not None and not _SUBJECTS_TABLE_DF.empty:
        extra_cols = ["participant_id"] + [
            c for c in _SUBJECTS_TABLE_DF.columns if c not in metrics_df.columns
        ]
        metrics_df = metrics_df.merge(
            _SUBJECTS_TABLE_DF[extra_cols], on="participant_id", how="left"
        )

    # --- Merge additional info (if available) ---
    if not _ADDITIONAL_INFO_DF.empty:
        add_cols = ["participant_id"] + [
            c
            for c in _ADDITIONAL_INFO_DF.columns
            if c != "participant_id" and c not in metrics_df.columns
        ]
        metrics_df = metrics_df.merge(
            _ADDITIONAL_INFO_DF[add_cols], on="participant_id", how="left"
        )
 
    # --- Create per-group interaction columns (if test_by_group is set) ---
    test_by_group = config.get("test_by_group")
    if test_by_group and test_by_group in metrics_df.columns:
        groups = metrics_df[test_by_group].dropna().unique()
        new_cols_data = {}

        for var_name in list(corr_variables.keys()):
            if var_name not in metrics_df.columns:
                continue
            for g in groups:
                g_str = (
                    str(int(g))
                    if isinstance(g, (int, float)) and int(g) == g
                    else str(g)
                )
                new_col = f"{var_name}_by_{test_by_group}_{g_str}"
                new_cols_data[new_col] = metrics_df[var_name].where(
                    metrics_df[test_by_group] == g
                )
                # Register as a new correlation variable with same confounds
                if new_col not in corr_variables:
                    corr_variables[new_col] = {
                        "confounds": corr_variables[var_name].get("confounds", [])
                    }

        if new_cols_data:
            metrics_df = pd.concat([metrics_df, pd.DataFrame(new_cols_data)], axis=1)

    del metric_files
    gc.collect()
    return metrics_df, metric_files_dict


# ============================================================================
# COLUMN DETECTION HELPERS
# ============================================================================

def detect_point_column(df: pd.DataFrame) -> str:
    """
    Find the column representing point/node ID along the tract.
    Looks for 'point_id', 'point', or similar names.
    """
    if "point_id" in df.columns:
        return "point_id"
    if "point" in df.columns:
        return "point"
    for c in df.columns:
        if re.fullmatch(r"point(_?id)?", c.lower()):
            return c
    raise ValueError("Point column not found in DataFrame. Expected 'point_id' or 'point'.")


def detect_metric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect which metric columns are present in the DataFrame.
    Tries combinations of METRIC_COLUMNS_CANDIDATES x STAT_TYPES first,
    then falls back to bare metric names.
    """
    detected = []
    if STAT_TYPES:
        for metric in METRIC_COLUMNS_CANDIDATES:
            for stat_type in STAT_TYPES:
                col_name = f"{metric}_{stat_type}"
                if col_name in df.columns:
                    detected.append(col_name)
    if not detected:
        detected = [c for c in METRIC_COLUMNS_CANDIDATES if c in df.columns]
    return detected


def get_metric_base_name(col_name: str) -> str:
    """Extract the base metric name from a column like 'FA_mean' -> 'FA'."""
    for metric in METRIC_COLUMNS_CANDIDATES:
        if col_name == metric or col_name.startswith(f"{metric}_"):
            return metric
    return col_name


def get_stat_type(col_name: str) -> Optional[str]:
    """Extract the stat type suffix from a column like 'FA_mean' -> 'mean'."""
    for metric in METRIC_COLUMNS_CANDIDATES:
        if col_name.startswith(f"{metric}_"):
            return col_name[len(metric) + 1 :]
    return None


def get_all_metric_related_columns(df: pd.DataFrame) -> List[str]:
    """Return all columns that are metric values (including 'value' itself)."""
    metric_cols = []
    candidates = METRIC_COLUMNS_CANDIDATES + ["value"]
    for col in df.columns:
        if col in candidates:
            metric_cols.append(col)
            continue
        for metric in candidates:
            if col.startswith(f"{metric}_"):
                metric_cols.append(col)
                break
    return metric_cols


# ============================================================================
# CONFOUND CORRECTION (OLS RESIDUALIZATION)
# ============================================================================

def ols_residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """
    Regress out X from y using OLS, return residuals + intercept.

    Parameters
    ----------
    y : pd.Series
        Target variable to residualize.
    X : pd.DataFrame
        Confound variables (predictors).

    Returns
    -------
    pd.Series
        Residualized target (same index as y).
    """
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty:
        return pd.Series(index=y.index, data=np.nan)

    y_clean = df.iloc[:, 0]
    X_clean = df.iloc[:, 1:]
    if X_clean.empty:
        return y

    Xc = sm.add_constant(X_clean, has_constant="add")
    try:
        model = sm.OLS(y_clean, Xc).fit()
        resid = model.resid + model.params.get("const", 0.0)
        out = pd.Series(index=y.index, data=np.nan)
        out.loc[resid.index] = resid.astype(float)
        return out
    except Exception:
        return y


def residualize_on_confond(
    df_subject_level: pd.DataFrame, target_col: str, confounds: List[str]
) -> pd.Series:
    """
    Residualize a target column against a list of confound variables.
    Categorical confounds are converted to numeric codes.
    """
    cols = [c for c in confounds if c in df_subject_level.columns]
    if not cols:
        return df_subject_level[target_col]

    X = df_subject_level[cols].copy()
    for c in cols:
        if X[c].dtype == "object" or str(X[c].dtype).startswith("category"):
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return ols_residualize(df_subject_level[target_col], X)


# ============================================================================
# RESAMPLING
# ============================================================================

def resample_bundle_data(
    df: pd.DataFrame, n_points: int, xu_max=None
) -> pd.DataFrame:
    """
    Resample bundle data to a fixed number of points per subject via
    linear interpolation. Ensures consistent point counts across subjects
    (required for clusterFWE and permutations).

    If 'centroid_id' is present, resampling is done per-centroid.
    """
    if not n_points or n_points < 2:
        return df

    try:
        point_col = detect_point_column(df)
    except Exception:
        return df

    metric_cols = get_all_metric_related_columns(df)
    if not metric_cols:
        return df

    has_centroid = "centroid_id" in df.columns
    meta_cols = [c for c in df.columns if c not in metric_cols and c != point_col]

    resampled = []
    centroid_ids = df["centroid_id"].unique() if has_centroid else [None]

    for cent_id in centroid_ids:
        cent_df = df[df["centroid_id"] == cent_id] if has_centroid else df
        xu_max_global = cent_df[point_col].max() if xu_max is None else xu_max
        xu_min_global = cent_df[point_col].min()

        # Common target grid for this centroid
        x_target = np.linspace(xu_min_global, xu_max_global, n_points)
        idx_target = np.arange(1, n_points + 1)

        for sub in cent_df["subject"].unique():
            sub_df = cent_df[cent_df["subject"] == sub].sort_values(point_col)
            x = sub_df[point_col].to_numpy(dtype=float)
            if len(x) < 2:
                resampled.append(sub_df)
                continue

            xu, idx = np.unique(x, return_index=True)
            xu_min_sub, xu_max_sub = xu.min(), xu.max()

            # Subset of target grid within this subject's range
            mask = (x_target >= xu_min_sub) & (x_target <= xu_max_sub)
            x_new = x_target[mask]
            x_new_normalized = idx_target[mask]

            if len(x_new) < 2:
                resampled.append(sub_df)
                continue

            new_data = {point_col: x_new_normalized}
            for mc in metric_cols:
                if mc in sub_df.columns:
                    y = sub_df[mc].to_numpy(dtype=float)
                    yu = y[idx]
                    new_data[mc] = np.interp(x_new, xu, yu)

            for col in meta_cols:
                if col in sub_df.columns:
                    new_data[col] = sub_df.iloc[0][col]

            resampled.append(pd.DataFrame(new_data))

    resampled = pd.concat(resampled, ignore_index=True) if resampled else df

    # Remove rows where all metric values are zero
    new_df = []
    group_cols = ["subject", "centroid_id"] if has_centroid else ["subject"]
    for _, grp in resampled.groupby(group_cols):
        grp = grp[(grp[metric_cols] != 0).any(axis=1)]
        new_df.append(grp)

    new_df = pd.concat(new_df, ignore_index=True) if new_df else pd.DataFrame()
    return new_df


# ============================================================================
# MISSING DATA MANAGEMENT
# ============================================================================

def interpolate_missing_points(
    long_df: pd.DataFrame, max_gap: int = 2
) -> pd.DataFrame:
    """
    Linearly interpolate missing points per subject, up to a maximum gap.
    Points missing at the start or end of a series cannot be interpolated
    and are left as NaN.
    """
    if long_df.empty or max_gap < 1:
        return long_df

    meta_cols = [c for c in long_df.columns if c not in {"subject", "point", "value"}]
    if meta_cols:
        meta_df = (
            long_df[["subject"] + meta_cols]
            .drop_duplicates(subset=["subject"])
            .groupby("subject", as_index=False)
            .first()
        )
    else:
        meta_df = None

    pivot = long_df.pivot_table(
        index="subject", columns="point", values="value", aggfunc="mean"
    )
    pivot_interpolated = pivot.copy()

    for subj in pivot.index:
        series = pivot.loc[subj]
        if series.isna().all():
            continue
        pivot_interpolated.loc[subj] = series.interpolate(
            method="linear", limit=max_gap, limit_direction="both"
        )

    long_df_interp = pivot_interpolated.reset_index().melt(
        id_vars="subject", var_name="point", value_name="value"
    )

    if meta_df is not None:
        long_df_interp = long_df_interp.merge(meta_df, on="subject", how="left")

    return long_df_interp


def clean_missing_data(
    long_df: pd.DataFrame,
    threshold: float = 0.15,
    missing_subjects_tolerance: Optional[int] = None,
) -> Tuple[pd.DataFrame, List, List]:
    """
    Manage NaN data by selecting the longest contiguous run of points where
    the number of subjects with missing values is within tolerance.

    Steps:
    1. Optionally interpolate small gaps (controlled by
       INTERPOLATE_MISSING_POINTS).
    2. For each point, check if NaN count <= missing_subjects_tolerance.
    3. Find the longest contiguous run of valid points.
    4. Remove points outside this run and report removed subjects.

    Returns
    -------
    long_df_clean : pd.DataFrame
        Filtered long-format DataFrame.
    removed_subjects : list
        Subject IDs that were dropped.
    points_to_remove : list
        Point IDs that were excluded.
    """
    if long_df.empty:
        return long_df, [], []

    # Step 1: Interpolate small gaps
    interpolate_max_gap = config.get(
        "INTERPOLATE_MISSING_POINTS", DEFAULT_CONFIG["INTERPOLATE_MISSING_POINTS"]
    )
    if interpolate_max_gap and interpolate_max_gap > 0:
        long_df = interpolate_missing_points(long_df, max_gap=interpolate_max_gap)

    # Step 2: Build pivot and check per-point completeness
    pivot = long_df.pivot_table(
        index="subject", columns="point", values="value", aggfunc="mean"
    )
    if pivot.shape[0] == 0:
        return long_df, [], []

    all_points = sorted(pivot.columns.tolist())
    if missing_subjects_tolerance is None:
        missing_subjects_tolerance = config.get(
            "MISSING_SUBJECTS_TOLERANCE",
            DEFAULT_CONFIG["MISSING_SUBJECTS_TOLERANCE"],
        )
    complete = [
        pivot[p].isna().sum() <= missing_subjects_tolerance for p in all_points
    ]

    # Step 3: Find longest contiguous run (or all valid points)
    if config.get("KEEP_LARGEST_CONTIGUOUS", DEFAULT_CONFIG["KEEP_LARGEST_CONTIGUOUS"]):
        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0
        for i, ok in enumerate(complete):
            if ok:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
            else:
                cur_len = 0

        if best_len == 0:
            return long_df.iloc[0:0].copy(), [], all_points

        kept_points = set(all_points[best_start : best_start + best_len])
    else:
        kept_points = set(p for p, ok in zip(all_points, complete) if ok)

    points_to_remove = [p for p in all_points if p not in kept_points]
    long_df_clean = long_df[long_df["point"].isin(kept_points)].copy()

    # Step 4: Identify removed subjects
    removed_subjects = []
    for p in kept_points:
        n_removed = pivot[p].isna().sum()
        if n_removed > missing_subjects_tolerance:
            removed_subjects.append(pivot[p][pivot[p].isna()].index.tolist())
    removed_subjects = [subj for sublist in removed_subjects for subj in sublist]

    return long_df_clean, removed_subjects, points_to_remove


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def correlation_test(x, y, method: str = "pearson"):
    """Run a correlation test (Pearson or Spearman)."""
    if method == "pearson":
        return pearsonr(x, y)
    elif method == "spearman":
        return spearmanr(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}")


# ============================================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================================

def _estimate_cluster_threshold_group(
    pivot_values: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    nperm: int = 500,
    random_state=None,
) -> Tuple[float, float]:
    """
    Estimate cluster-size and alpha thresholds via permutation for group tests.

    This is a fallback when the full AFQ_MultiCompCorrection fails or is
    not applicable.

    Returns
    -------
    cluster_threshold : int
        Minimum cluster size for significance (95th percentile of max
        cluster sizes across permutations).
    alpha_threshold : float
        P-value threshold for significance.
    """
    rng = np.random.default_rng(random_state)
    n_points = pivot_values.shape[1]
    unique_groups = np.unique(y[~np.isnan(y)])
    max_clusters = []
    min_pvals = []

    for i in range(min(nperm, 2000)):
        perm = rng.permutation(y)
        groups_data = [pivot_values[perm == g] for g in unique_groups]

        pvals = []
        for j in range(n_points):
            point_data = [gd[:, j] for gd in groups_data]
            valid_counts = [np.sum(~np.isnan(pd)) for pd in point_data]
            if all(vc >= 2 for vc in valid_counts):
                try:
                    if len(unique_groups) == 2:
                        _, p = ttest_ind(
                            point_data[0],
                            point_data[1],
                            equal_var=False,
                            nan_policy="omit",
                        )
                    else:
                        clean_data = [pd[~np.isnan(pd)] for pd in point_data]
                        _, p = f_oneway(*clean_data)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
            pvals.append(p)

        pvals_arr = np.array(pvals)
        if np.all(np.isnan(pvals_arr)):
            min_pvals.append(np.nan)
            max_clusters.append(0)
            continue

        min_pvals.append(np.nanmin(pvals_arr))

        # Find longest contiguous significant run
        mask = pvals_arr < alpha
        cur = max_len = 0
        for b in mask:
            if b:
                cur += 1
                max_len = max(max_len, cur)
            else:
                cur = 0
        max_clusters.append(max_len)

    if not max_clusters:
        return np.nan, np.nan

    cluster_thr = int(np.nanpercentile(max_clusters, 95))
    alpha_thr = np.nanpercentile(min_pvals, alpha * 100)
    return max(1, cluster_thr), alpha_thr


def _count_clusters(sig_iterable) -> int:
    """Count the number of contiguous significant clusters in a boolean array."""
    cnt = 0
    in_cluster = False
    for v in list(sig_iterable):
        if v and not in_cluster:
            cnt += 1
            in_cluster = True
        elif not v and in_cluster:
            in_cluster = False
    return cnt


def get_significant_areas_mixed(
    pvalues: np.ndarray,
    cluster_threshold: int,
    alphaFWE: float,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Mixed correction method: a cluster is significant if:
    - Its length >= cluster_threshold, AND
    - It contains at least one point with p-value < alphaFWE.
    """
    if np.isnan(cluster_threshold) or cluster_threshold < 1:
        return np.zeros_like(pvalues, dtype=bool)
    if np.isnan(alphaFWE):
        return np.zeros_like(pvalues, dtype=bool)

    n = len(pvalues)
    sig_mask = np.zeros(n, dtype=bool)
    cluster_mask = pvalues < alpha

    i = 0
    while i < n:
        if cluster_mask[i]:
            start = i
            while i < n and cluster_mask[i]:
                i += 1
            end = i
            cluster_length = end - start

            if cluster_length >= cluster_threshold:
                cluster_pvals = pvalues[start:end]
                if np.any(cluster_pvals < alphaFWE):
                    sig_mask[start:end] = True
        else:
            i += 1

    return sig_mask


def compute_multi_tract_thresholds_group(
    bundle_data: Dict,
    metric_col: str,
    classif_col: str,
    alpha: float = 0.05,
    nperm: int = 1000,
) -> Tuple[float, float]:
    """
    Compute alphaFWE and clusterFWE thresholds across ALL bundles for a
    group comparison. This implements multi-tract correction by concatenating
    pointwise data from all bundles.
    """
    all_values = []
    subject_groups = {}

    for bundle_name, df in bundle_data.items():
        if metric_col not in df.columns or classif_col not in df.columns:
            continue
        try:
            point_col = detect_point_column(df)
        except Exception:
            continue

        pivot = df.pivot_table(
            index="subject", columns=point_col, values=metric_col, aggfunc="mean"
        )
        if pivot.empty:
            continue

        subj_groups = (
            df[["subject", classif_col]]
            .drop_duplicates()
            .set_index("subject")
            .reindex(pivot.index)
        )

        for subj in pivot.index:
            if subj not in subject_groups:
                subject_groups[subj] = subj_groups.loc[subj, classif_col]
            values_subj = pivot.loc[subj].values.tolist()

            found = False
            for entry in all_values:
                if entry["subject"] == subj:
                    entry["values"].extend(values_subj)
                    found = True
                    break
            if not found:
                all_values.append({"subject": subj, "values": values_subj})

    if not all_values:
        return np.nan, np.nan

    valid_subjects = [
        e
        for e in all_values
        if e["subject"] in subject_groups
        and pd.notna(subject_groups[e["subject"]])
    ]
    if len(valid_subjects) < 4:
        return np.nan, np.nan

    subjects = [e["subject"] for e in valid_subjects]
    n_points = min(len(e["values"]) for e in valid_subjects)
    values_matrix = np.array([e["values"][:n_points] for e in valid_subjects])

    groups = [subject_groups[s] for s in subjects]
    sorted_groups = sorted(
        set(g for g in groups if pd.notna(g)), key=lambda x: str(x)
    )
    if len(sorted_groups) < 2:
        return np.nan, np.nan

    group_map = {g: i for i, g in enumerate(sorted_groups)}
    y = np.array([group_map.get(g, np.nan) for g in groups])

    mask_valid = ~np.isnan(y)
    values_matrix = values_matrix[mask_valid]
    y = y[mask_valid].astype(int)
    values_matrix = np.nan_to_num(values_matrix, nan=np.nanmean(values_matrix))

    if len(y) < 4:
        return np.nan, np.nan

    try:
        alphaFWE, _, clusterFWE, _ = AFQ_MultiCompCorrection(
            values_matrix, y, alpha, nperm=int(nperm)
        )
        return alphaFWE, clusterFWE
    except Exception:
        return np.nan, np.nan


def compute_multi_tract_thresholds_corr(
    bundle_data: Dict,
    metric_col: str,
    var_col: str,
    confounds: List[str],
    alpha: float = 0.05,
    nperm: int = 1000,
) -> Tuple[float, float]:
    """
    Compute alphaFWE and clusterFWE thresholds across ALL bundles for a
    correlation analysis. See compute_multi_tract_thresholds_group for details.
    """
    all_values = []
    subject_targets = {}

    for bundle_name, df in bundle_data.items():
        if metric_col not in df.columns or var_col not in df.columns:
            continue
        try:
            point_col = detect_point_column(df)
        except Exception:
            continue

        pivot = df.pivot_table(
            index="subject", columns=point_col, values=metric_col, aggfunc="mean"
        )
        if pivot.empty:
            continue

        subj_targets = (
            df[["subject", var_col]]
            .drop_duplicates()
            .set_index("subject")
            .reindex(pivot.index)
        )

        for subj in pivot.index:
            if subj not in subject_targets:
                val = subj_targets.loc[subj, var_col]
                subject_targets[subj] = (
                    pd.to_numeric(val, errors="coerce")
                    if not isinstance(val, (int, float))
                    else val
                )
            values_subj = pivot.loc[subj].values.tolist()

            found = False
            for entry in all_values:
                if entry["subject"] == subj:
                    entry["values"].extend(values_subj)
                    found = True
                    break
            if not found:
                all_values.append({"subject": subj, "values": values_subj})

    if not all_values:
        return np.nan, np.nan

    valid_subjects = [
        e
        for e in all_values
        if e["subject"] in subject_targets
        and pd.notna(subject_targets[e["subject"]])
    ]
    if len(valid_subjects) < 4:
        return np.nan, np.nan

    subjects = [e["subject"] for e in valid_subjects]
    n_points = min(len(e["values"]) for e in valid_subjects)
    values_matrix = np.array([e["values"][:n_points] for e in valid_subjects])
    y = np.array([subject_targets[s] for s in subjects], dtype=float)

    mask_valid = ~np.isnan(y)
    values_matrix = values_matrix[mask_valid]
    y = y[mask_valid]
    values_matrix = np.nan_to_num(values_matrix, nan=np.nanmean(values_matrix))

    if len(y) < 4:
        return np.nan, np.nan

    try:
        alphaFWE, _, clusterFWE, _ = AFQ_MultiCompCorrection(
            values_matrix, y, alpha, nperm=int(nperm)
        )
        return alphaFWE, clusterFWE
    except Exception:
        return np.nan, np.nan


# ============================================================================
# LONG FORMAT PREPARATION
# ============================================================================

def prepare_long(
    df_bundle: pd.DataFrame, metric_col: str, point_col: str
) -> pd.DataFrame:
    """
    Convert a bundle DataFrame to long format with a single 'value' column.

    Also applies exclusion filters defined in config["EXCLUDE"].
    """
    # Apply exclusion filters
    exclude_dict = config.get("EXCLUDE", {})
    if exclude_dict:
        for var, values in exclude_dict.items():
            if var in df_bundle.columns:
                df_bundle = df_bundle[~df_bundle[var].isin(values)]

    needed = ["subject", point_col, metric_col]
    if not all(c in df_bundle.columns for c in needed):
        raise ValueError(f"Missing columns for {metric_col}: needed {needed}")

    all_metric_cols = get_all_metric_related_columns(df_bundle)
    meta_cols = [c for c in df_bundle.columns if c not in all_metric_cols]
    long_df = df_bundle[meta_cols + [metric_col]].rename(
        columns={metric_col: "value", point_col: "point"}
    )
    return long_df


# ============================================================================
# GROUP COMPARISON (t-test / ANOVA)
# ============================================================================

def pointwise_ttest(long_df: pd.DataFrame, classif_col: str) -> pd.DataFrame:
    """
    Pointwise t-test or ANOVA between groups (no FWE correction).
    Results include raw p-values and FDR-corrected p-values.
    """
    if classif_col not in long_df.columns:
        return pd.DataFrame()

    groups = long_df[classif_col].dropna().unique()
    if len(groups) < 2:
        return pd.DataFrame()

    sorted_groups = sorted(groups, key=lambda x: str(x))
    rows = []

    for pid, g in long_df.groupby("point"):
        sub = g[[classif_col, "value"]].dropna()
        group_vals = [
            sub.loc[sub[classif_col] == grp, "value"].astype(float)
            for grp in sorted_groups
        ]

        if all(len(v) >= 2 for v in group_vals):
            try:
                if len(sorted_groups) == 2:
                    stat, p = ttest_ind(
                        group_vals[0],
                        group_vals[1],
                        equal_var=False,
                        nan_policy="omit",
                    )
                    diff = group_vals[1].mean() - group_vals[0].mean()
                else:
                    stat, p = f_oneway(*group_vals)
                    diff = np.nan
                    stat = np.nan
            except Exception:
                p = diff = stat = np.nan
        else:
            p = diff = stat = np.nan

        row = {"point": pid, "p_raw": p, "diff": diff, "stat": stat}
        for i, grp in enumerate(sorted_groups):
            row[f"mean_g{i}"] = group_vals[i].mean()
            row[f"n{i}"] = len(group_vals[i])
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows).sort_values("point")
    res["p_raw"] = res["p_raw"].astype(float)

    # FDR correction
    valid_mask = res["p_raw"].notna()
    if valid_mask.any():
        _, p_fdr, _, _ = multipletests(
            res.loc[valid_mask, "p_raw"], method="fdr_bh"
        )
        res["p_fdr"] = np.nan
        res.loc[valid_mask, "p_fdr"] = p_fdr
        res["sig_fdr"] = res["p_fdr"] < 0.05
    else:
        res["p_fdr"] = np.nan
        res["sig_fdr"] = False

    return res


def group_test_afq(
    long_df: pd.DataFrame,
    classif_col: str,
    alpha: float = 0.05,
    nperm: int = 1000,
    precomputed_alphaFWE=None,
    precomputed_clusterFWE=None,
) -> pd.DataFrame:
    """
    Group comparison with FWE correction (clusterFWE, alphaFWE, or mixed).

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data with columns: subject, point, value, classif_col.
    classif_col : str
        Column name defining the groups.
    precomputed_alphaFWE, precomputed_clusterFWE : float, optional
        Pre-computed thresholds (e.g. from multi-tract correction).
    """
    if classif_col not in long_df.columns or long_df.empty:
        return pd.DataFrame()

    long_df = long_df[long_df[classif_col].notna()]
    if long_df.empty:
        return pd.DataFrame()

    groups = long_df[classif_col].dropna().unique()
    if len(groups) < 2:
        return pd.DataFrame()

    sorted_groups = sorted(groups, key=lambda x: str(x))

    # Build pivot table: subjects x points
    pivot = long_df.pivot_table(
        index="subject", columns="point", values="value", aggfunc="mean"
    )
    subj_groups = (
        long_df[["subject", classif_col]]
        .drop_duplicates()
        .set_index("subject")
        .loc[pivot.index]
    )

    # Filter to subjects with valid group labels
    mask_valid_label = subj_groups[classif_col].notna()
    pivot = pivot.loc[mask_valid_label]
    subj_groups = subj_groups.loc[mask_valid_label]
    if pivot.shape[0] < 4:
        return pd.DataFrame()

    # Drop subjects with any NaN across points
    pivot = pivot.dropna(axis=0, how="any")
    subj_groups = subj_groups.loc[pivot.index]
    if pivot.shape[0] < 4:
        return pd.DataFrame()

    # Encode groups as integers
    group_map = {g: i for i, g in enumerate(sorted_groups)}
    y_series = subj_groups[classif_col].map(group_map)
    mask_y_valid = y_series.notna()
    pivot = pivot.loc[mask_y_valid]
    y_series = y_series.loc[mask_y_valid]
    if pivot.shape[0] < 4:
        return pd.DataFrame()

    y = y_series.values.astype(int)

    # --- Determine FWE thresholds ---
    alphaFWE = (
        precomputed_alphaFWE
        if precomputed_alphaFWE is not None and not np.isnan(precomputed_alphaFWE)
        else np.nan
    )
    clusterFWE = (
        precomputed_clusterFWE
        if precomputed_clusterFWE is not None and not np.isnan(precomputed_clusterFWE)
        else np.nan
    )

    if np.isnan(alphaFWE) or np.isnan(clusterFWE):
        if len(np.unique(y)) == 2:
            try:
                computed_alpha, _, computed_cluster, _ = AFQ_MultiCompCorrection(
                    pivot.values, y, alpha, nperm=int(nperm)
                )
                if np.isnan(alphaFWE):
                    alphaFWE = computed_alpha
                if np.isnan(clusterFWE):
                    clusterFWE = computed_cluster
            except Exception:
                pass

        if (np.isnan(clusterFWE) or clusterFWE < 1) or np.isnan(alphaFWE):
            est_cluster, est_alpha = _estimate_cluster_threshold_group(
                pivot.values, y, alpha=alpha, nperm=int(nperm / 2)
            )
            if np.isnan(clusterFWE) or clusterFWE < 1:
                clusterFWE = est_cluster
            if np.isnan(alphaFWE):
                alphaFWE = est_alpha

    # --- Pointwise tests ---
    pvalues, diffs, stats = [], [], []
    means_list = [[] for _ in sorted_groups]
    n_list = [[] for _ in sorted_groups]

    for point in pivot.columns:
        group_vals = [
            pivot.loc[y == i, point].astype(float)
            for i in range(len(sorted_groups))
        ]

        if all(v.notna().sum() >= 2 for v in group_vals):
            try:
                if len(sorted_groups) == 2:
                    stat, p = ttest_ind(
                        group_vals[0],
                        group_vals[1],
                        equal_var=False,
                        nan_policy="omit",
                    )
                    diff = group_vals[1].mean() - group_vals[0].mean()
                else:
                    clean_vals = [v.dropna() for v in group_vals]
                    if all(len(cv) >= 2 for cv in clean_vals):
                        stat, p = f_oneway(*clean_vals)
                    else:
                        p = np.nan
                    diff = np.nan
                    stat = np.nan
            except Exception:
                p = diff = stat = np.nan
        else:
            p = diff = stat = np.nan

        pvalues.append(p)
        diffs.append(diff)
        stats.append(stat)
        for i in range(len(sorted_groups)):
            means_list[i].append(group_vals[i].mean())
            n_list[i].append(group_vals[i].notna().sum())

    pvalues = np.array(pvalues, dtype=float)

    # --- Apply FWE correction ---
    if FWE_METHOD == "alphaFWE":
        sig_mask = (
            (pvalues < alphaFWE)
            if not np.isnan(alphaFWE)
            else np.zeros_like(pvalues, dtype=bool)
        )
    elif FWE_METHOD == "mixed":
        sig_mask = get_significant_areas_mixed(
            pvalues,
            int(clusterFWE) if not np.isnan(clusterFWE) else 1,
            alphaFWE,
            alpha=alpha,
        )
    else:  # clusterFWE
        if np.isnan(clusterFWE) or clusterFWE < 1:
            sig_mask = np.zeros_like(pvalues, dtype=bool)
        else:
            sig_mask = get_significant_areas(
                pvalues, int(clusterFWE), alpha=alpha
            ).astype(bool)

    out_data = {
        "point": pivot.columns,
        "p_raw": pvalues,
        "diff": diffs,
        "stat": stats,
        "sig_afq": sig_mask,
        "alphaFWE": alphaFWE,
        "clusterFWE": clusterFWE,
    }
    for i in range(len(sorted_groups)):
        out_data[f"mean_g{i}"] = means_list[i]
        out_data[f"n{i}"] = n_list[i]

    out_df = pd.DataFrame(out_data)
    del pivot, subj_groups, y_series
    gc.collect()
    return out_df


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def _correlation_with_alphaFWE(
    pivot_values: pd.DataFrame,
    y_vec,
    nperm: int = 1000,
    alpha: float = 0.05,
    precomputed_alphaFWE=None,
    precomputed_clusterFWE=None,
) -> Dict:
    """
    Pointwise correlation with FWE correction.

    Supports clusterFWE, alphaFWE, and mixed correction methods.
    Pre-computed thresholds (from multi-tract correction) can be supplied.
    """
    # Align y with pivot index
    if isinstance(y_vec, pd.Series):
        y_series = y_vec.copy()
    else:
        y_series = pd.Series(y_vec, index=pivot_values.index[: len(y_vec)])

    common_index = pivot_values.index.intersection(y_series.index)
    pivot_values = pivot_values.loc[common_index]
    y_series = y_series.loc[common_index]

    if len(y_series) < 4:
        return {"alphaFWE": np.nan, "df": pd.DataFrame()}

    # Drop NaN y values
    mask_y = y_series.notna()
    pivot_values = pivot_values.loc[mask_y]
    y_series = y_series.loc[mask_y]
    if len(y_series) < 4:
        return {"alphaFWE": np.nan, "df": pd.DataFrame()}

    # Drop subjects with any NaN in metric values
    pivot_values = pivot_values.dropna(axis=0, how="any")
    y_series = y_series.reindex(pivot_values.index)
    if len(y_series) < 4:
        return {"alphaFWE": np.nan, "df": pd.DataFrame()}

    # --- Determine FWE thresholds ---
    alphaFWE = (
        precomputed_alphaFWE
        if precomputed_alphaFWE is not None and not np.isnan(precomputed_alphaFWE)
        else np.nan
    )
    clusterFWE = (
        precomputed_clusterFWE
        if precomputed_clusterFWE is not None and not np.isnan(precomputed_clusterFWE)
        else np.nan
    )

    if np.isnan(alphaFWE) or np.isnan(clusterFWE):
        try:
            computed_alpha, _, computed_cluster, _ = AFQ_MultiCompCorrection(
                pivot_values.values, y_series.values, alpha, nperm=int(nperm)
            )
            if np.isnan(alphaFWE):
                alphaFWE = computed_alpha
            if np.isnan(clusterFWE):
                clusterFWE = computed_cluster
        except Exception as e:
            print(f"AFQ_MultiCompCorrection error: {e}")

    # --- Pointwise correlations ---
    r_list, p_list = [], []
    n_subj = len(y_series)

    for col in pivot_values.columns:
        xv = pivot_values[col].astype(float)
        mask = xv.notna() & y_series.notna()
        if mask.sum() >= 3:
            try:
                if np.linalg.norm(xv[mask] - np.mean(xv[mask])) < 1e-13 * abs(
                    np.mean(xv[mask])
                ):
                    r, p = np.nan, np.nan
                else:
                    r, p = correlation_test(
                        xv[mask], y_series[mask], method=CORRELATION_TEST
                    )
            except Exception:
                r, p = np.nan, np.nan
        else:
            r, p = np.nan, np.nan
        r_list.append(r)
        p_list.append(p)

    r_arr = np.array(r_list, dtype=float)
    p_arr = np.array(p_list, dtype=float)

    # --- Apply FWE correction ---
    if FWE_METHOD == "alphaFWE":
        sig_mask = (
            (p_arr < alphaFWE)
            if not np.isnan(alphaFWE)
            else np.zeros_like(p_arr, dtype=bool)
        )
    elif FWE_METHOD == "mixed":
        sig_mask = get_significant_areas_mixed(
            p_arr,
            int(clusterFWE) if not np.isnan(clusterFWE) else 1,
            alphaFWE,
            alpha=alpha,
        )
    else:  # clusterFWE
        if np.isnan(clusterFWE) or clusterFWE < 1:
            sig_mask = np.zeros_like(p_arr, dtype=bool)
        else:
            sig_mask = get_significant_areas(
                p_arr, int(clusterFWE), alpha=alpha
            ).astype(bool)

    df_out = pd.DataFrame(
        {
            "point": pivot_values.columns,
            "r": r_arr,
            "p_raw": p_arr,
            "n": n_subj,
            "sig_afq": sig_mask,
            "alphaFWE": alphaFWE,
            "clusterFWE": clusterFWE,
        }
    )

    del r_list, p_list, r_arr, p_arr
    gc.collect()
    return {"alphaFWE": alphaFWE, "df": df_out}


def pointwise_correlation(long_df: pd.DataFrame, var_col: str) -> pd.DataFrame:
    """Simple pointwise correlation (no confound correction)."""
    if var_col not in long_df.columns:
        return pd.DataFrame()
    pivot_val = long_df.pivot_table(
        index="subject", columns="point", values="value", aggfunc="mean"
    )
    y = (
        long_df[["subject", var_col]]
        .drop_duplicates()
        .set_index("subject")
        .reindex(pivot_val.index)[var_col]
    )
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = y.astype("category").cat.codes.replace(-1, np.nan).astype(float)
    else:
        y = pd.to_numeric(y, errors="coerce")
    return _correlation_with_alphaFWE(pivot_val, y)["df"]


def pointwise_partial_correlation(
    long_df: pd.DataFrame,
    var_col: str,
    confounds: List[str],
    precomputed_alphaFWE=None,
    precomputed_clusterFWE=None,
) -> pd.DataFrame:
    """
    Pointwise partial correlation: residualize both the metric values and the
    target variable against confounds, then compute correlations.
    """
    if var_col not in long_df.columns:
        return pd.DataFrame()

    pivot_val = long_df.pivot_table(
        index="subject", columns="point", values="value", aggfunc="mean"
    )
    cols = list(
        dict.fromkeys(
            ["subject", var_col] + [c for c in confounds if c in long_df.columns]
        )
    )
    subj_meta = long_df[cols].drop_duplicates().set_index("subject")
    pivot_val = pivot_val.reindex(subj_meta.index)

    # Residualize the target variable
    if CORRECT_METRIC_CONFOND_ONLY:
        var_res = subj_meta[var_col].copy()
    else:
        var_res = residualize_on_confond(
            subj_meta.assign(target=subj_meta[var_col]), "target", confounds
        )

    # Residualize each metric point
    X_res_cols = []
    for pid in pivot_val.columns:
        series_metric = pivot_val[pid]
        df_tmp = subj_meta.copy()
        df_tmp = df_tmp.assign(metric=series_metric)
        metric_resid = residualize_on_confond(
            df_tmp.assign(target=df_tmp["metric"]), "target", confounds
        )
        X_res_cols.append(metric_resid)

    if not X_res_cols:
        return pd.DataFrame()

    X_res = pd.concat(X_res_cols, axis=1)
    X_res.columns = pivot_val.columns

    # Align indices
    common_index = X_res.index.intersection(var_res.index)
    X_res = X_res.loc[common_index]
    var_res_aligned = var_res.loc[common_index]

    res = _correlation_with_alphaFWE(
        X_res,
        var_res_aligned,
        precomputed_alphaFWE=precomputed_alphaFWE,
        precomputed_clusterFWE=precomputed_clusterFWE,
    )
    return res["df"]


# ============================================================================
# PARTICIPANT ID UTILITIES
# ============================================================================

def _normalize_participant_id(value) -> Optional[str]:
    """Normalize a participant ID to 'sub-XXX' format."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    val = str(value).strip()
    if not val:
        return None
    return val if val.startswith("sub-") else f"sub-{val}"


def _build_subject_participant_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build a mapping from subject ID to normalized participant_id."""
    mapping = {}
    if df is None or df.empty or "subject" not in df.columns:
        return mapping
    if "participant_id" in df.columns:
        tmp = (
            df[["subject", "participant_id"]]
            .dropna(subset=["subject"])
            .drop_duplicates(subset="subject")
        )
        for _, row in tmp.iterrows():
            subj_key = str(row["subject"])
            pid_norm = (
                _normalize_participant_id(row["participant_id"])
                or _normalize_participant_id(subj_key)
            )
            mapping[subj_key] = pid_norm
    else:
        for subj in df["subject"].dropna().unique():
            subj_key = str(subj)
            mapping[subj_key] = _normalize_participant_id(subj_key)
    return mapping


def _subjects_to_participant_ids(
    subjects, subject_pid_map: Dict[str, str]
) -> Set[str]:
    """Convert a list of subject IDs to normalized participant IDs."""
    ids = set()
    if subjects is None:
        return ids
    for subj in subjects:
        if pd.isna(subj):
            continue
        subj_key = str(subj)
        pid = subject_pid_map.get(subj_key) or _normalize_participant_id(subj_key)
        if pid:
            ids.add(pid)
    return ids


def _get_reference_participants() -> Set[str]:
    """
    Get the set of participant IDs from the additional info table (if loaded).
    Used to determine which subjects were expected but not included.
    """
    if _ADDITIONAL_INFO_DF is None or _ADDITIONAL_INFO_DF.empty:
        return set()
    if "participant_id" not in _ADDITIONAL_INFO_DF.columns:
        return set()
    refs = set()
    for val in _ADDITIONAL_INFO_DF["participant_id"].dropna().unique():
        pid = _normalize_participant_id(val)
        if pid:
            refs.add(pid)
    return refs


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _annotate_missing(ax, removed_subjects, removed_points):
    """Add a text box listing removed subjects and points."""
    lines = []
    if removed_subjects:
        ex = ", ".join(map(str, removed_subjects[:8]))
        if len(removed_subjects) > 8:
            ex += "..."
        lines.append(f"Removed subjects ({len(removed_subjects)}): {ex}")
    if removed_points:
        exp = ", ".join(map(str, removed_points[:12]))
        if len(removed_points) > 12:
            exp += "..."
        lines.append(f"Removed points ({len(removed_points)}): {exp}")
    if lines:
        ax.text(
            0.01,
            0.99,
            "\n".join(lines),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.65, edgecolor="gray"
            ),
        )


def _annotate_pvalues(ax, ttest_df, sig_col, y_offset_factor=0.02, max_labels=30):
    """Annotate significant points with their p-values."""
    if ttest_df is None or ttest_df.empty:
        return
    sig_df = (
        ttest_df[ttest_df[sig_col]] if sig_col in ttest_df.columns else pd.DataFrame()
    )
    if sig_df.empty:
        return

    # Limit the number of labels to avoid clutter
    if "p_fdr" in sig_df.columns and sig_col != "sig_afq":
        sig_df = sig_df.sort_values("p_fdr").head(max_labels)
        p_label_col = "p_fdr"
    else:
        sig_df = sig_df.sort_values("p_raw").head(max_labels)
        p_label_col = "p_raw"

    ylim = ax.get_ylim()
    y_span = ylim[1] - ylim[0]
    for _, row in sig_df.iterrows():
        pval = row.get(p_label_col, np.nan)
        if np.isnan(pval):
            continue
        x = row["point"]
        y = ylim[1] - y_span * 0.05
        label = (
            f"p={'{:.2e}'.format(pval) if pval < 0.001 else '{:.3f}'.format(pval)}"
        )
        ax.text(
            x, y, label, rotation=90, ha="center", va="top", fontsize=7, color="red"
        )


def _highlight_clusters(ax, points, sig_mask, color="red", alpha=0.18):
    """Shade contiguous significant clusters on the plot."""
    if len(points) == 0:
        return
    pts = list(points)
    sig = list(sig_mask)
    if not any(sig):
        return

    start = None
    prev = None
    for p, s in zip(pts, sig):
        if s and start is None:
            start = p
        if (not s) and start is not None:
            ax.axvspan(start, prev, color=color, alpha=alpha, linewidth=0)
            start = None
        prev = p
    if start is not None:
        ax.axvspan(start, prev, color=color, alpha=alpha, linewidth=0)


def _export_group_csv(
    long_df,
    ttest_df,
    classif_col,
    metric_name,
    bundle_name,
    centroid_id,
    out_csv,
    analysis_label,
):
    """Export group comparison results to CSV with aggregated statistics."""
    agg = (
        long_df.groupby(["point", classif_col])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    pivot_mean = agg.pivot(index="point", columns=classif_col, values="mean")
    pivot_std = agg.pivot(index="point", columns=classif_col, values="std")
    pivot_n = agg.pivot(index="point", columns=classif_col, values="count")

    df_out = pd.DataFrame({"point": pivot_mean.index}).set_index("point")
    for g in pivot_mean.columns:
        df_out[f"mean_{g}"] = pivot_mean[g]
        df_out[f"std_{g}"] = pivot_std[g]
        df_out[f"n_{g}"] = pivot_n[g]

    if ttest_df is not None and not ttest_df.empty:
        ttest_indexed = ttest_df.set_index("point")
        fixed_cols = [
            "p_raw",
            "p_fdr",
            "sig_afq",
            "sig_fdr",
            "alphaFWE",
            "clusterFWE",
        ]
        group_mean_cols = sorted(
            [c for c in ttest_indexed.columns if c.startswith("mean_g")],
            key=lambda x: int(x.replace("mean_g", "")),
        )
        group_n_cols = sorted(
            [
                c
                for c in ttest_indexed.columns
                if c.startswith("n") and c[1:].isdigit()
            ],
            key=lambda x: int(x[1:]),
        )
        n_groups = len(group_mean_cols)
        if n_groups == 2 and "diff" in ttest_indexed.columns:
            fixed_cols.extend(["diff", "stat"])
        for col in fixed_cols + group_mean_cols + group_n_cols:
            if col in ttest_indexed.columns:
                df_out[col] = ttest_indexed[col]

    df_out["bundle"] = bundle_name
    df_out["centroid_id"] = centroid_id
    df_out["metric"] = metric_name
    df_out["classif"] = classif_col
    df_out["analysis"] = analysis_label
    df_out.reset_index().to_csv(out_csv, index=False)


def _export_corr_csv(
    corr_df,
    metric_name,
    bundle_name,
    centroid_id,
    var_name,
    out_csv,
    analysis_label,
):
    """Export correlation results to CSV."""
    if corr_df is None or corr_df.empty:
        pd.DataFrame(
            {
                "info": ["no data"],
                "bundle": [bundle_name],
                "centroid_id": [centroid_id],
                "metric": [metric_name],
                "variable": [var_name],
                "analysis": [analysis_label],
            }
        ).to_csv(out_csv, index=False)
        return

    df_out = corr_df.copy()
    df_out["bundle"] = bundle_name
    df_out["centroid_id"] = centroid_id
    df_out["metric"] = metric_name
    df_out["variable"] = var_name
    df_out["analysis"] = analysis_label
    df_out.to_csv(out_csv, index=False)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_group_diff(
    long_df,
    ttest_df,
    classif_col,
    metric_name,
    bundle_name,
    out_path,
    title_suffix,
    removed_subjects=None,
    removed_points=None,
    export_csv=True,
    canonical_bundle=None,
    centroid_id=None,
):
    """
    Plot group comparison: mean profiles +/- 95% CI, p-values bar, and
    significant cluster highlights.

    The significance indicator line and cluster shading adapt to the
    configured FWE_METHOD.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=long_df,
        x="point",
        y="value",
        hue=classif_col,
        estimator="mean",
        errorbar=("ci", 95),
    )

    sig_col = (
        "sig_afq"
        if ttest_df is not None
        and not ttest_df.empty
        and "sig_afq" in ttest_df.columns
        else "sig_fdr"
    )

    # Highlight significant clusters (clusterFWE or mixed only)
    if (
        FWE_METHOD in ("clusterFWE", "mixed")
        and ttest_df is not None
        and not ttest_df.empty
        and "sig_afq" in ttest_df.columns
    ):
        _highlight_clusters(plt.gca(), ttest_df["point"], ttest_df["sig_afq"])

    # Plot significance indicator line
    if (
        ttest_df is not None
        and not ttest_df.empty
        and ttest_df[sig_col].any()
    ):
        all_vals = pd.to_numeric(long_df["value"], errors="coerce")
        if all_vals.notna().any():
            low = np.nanquantile(all_vals, 0.02)
            high = np.nanquantile(all_vals, 0.98)
        else:
            low, high = 0, 1

        sig_line = ttest_df[["point"]].copy()
        sig_line["sig_line"] = np.where(ttest_df[sig_col], high, low)

        if FWE_METHOD == "alphaFWE":
            label_txt = "Significant (alphaFWE)"
        elif FWE_METHOD == "mixed":
            label_txt = "Significant (mixed: cluster+alpha)"
        else:
            label_txt = "Significant (clusterFWE)"
        plt.plot(
            sig_line["point"],
            sig_line["sig_line"],
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=label_txt,
        )

    ax = plt.gca()

    # P-value bar on secondary axis
    if (
        ttest_df is not None
        and not ttest_df.empty
        and "p_raw" in ttest_df.columns
    ):
        ttest_df = ttest_df.copy()
        ttest_df["p_raw"] = ttest_df["p_raw"].clip(lower=0, upper=1)
        ax2 = ax.twinx()
        ax2.bar(
            ttest_df["point"],
            ttest_df["p_raw"],
            color="gray",
            alpha=0.18,
            width=1.0,
            label="p-value",
        )

        pmax = (
            float(ttest_df["p_raw"].max())
            if ttest_df["p_raw"].notna().any()
            else 0.05
        )
        ymax = min(1.0, max(0.06, pmax * 1.15))

        if "alphaFWE" in ttest_df.columns and not ttest_df["alphaFWE"].isna().all():
            alphaFWE_val = ttest_df["alphaFWE"].iloc[0]
            if not np.isnan(alphaFWE_val) and FWE_METHOD in ("alphaFWE", "mixed"):
                ax2.axhline(
                    alphaFWE_val,
                    color="green",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"alphaFWE={alphaFWE_val:.3g}",
                )
                ymax = max(ymax, alphaFWE_val * 1.15)

        if (
            "clusterFWE" in ttest_df.columns
            and not ttest_df["clusterFWE"].isna().all()
        ):
            cluster_val = ttest_df["clusterFWE"].iloc[0]
            if not np.isnan(cluster_val) and FWE_METHOD in ("clusterFWE", "mixed"):
                ax2.text(
                    0.01,
                    0.95,
                    f"clusterFWE>={int(cluster_val)} (alpha={AFQ_ALPHA})",
                    transform=ax2.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.6,
                        edgecolor="gray",
                    ),
                )

        ax2.set_ylabel("p-value")
        ax2.set_ylim(0, ymax)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    plt.title(
        f"{bundle_name} - {metric_name} - {classif_col} {title_suffix} ({FWE_METHOD})"
    )
    plt.xlabel("Point")
    plt.ylabel(metric_name)
    _annotate_missing(ax, removed_subjects or [], removed_points or [])
    _annotate_pvalues(ax, ttest_df, sig_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    if export_csv:
        out_csv = os.path.splitext(out_path)[0] + ".csv"
        analysis_lbl = "raw" if "raw" in out_path else "corrected"
        bundle_for_csv = canonical_bundle if canonical_bundle else bundle_name
        if centroid_id is None:
            centroid_id = (
                long_df["centroid_id"].iloc[0]
                if "centroid_id" in long_df.columns
                and not long_df["centroid_id"].isna().all()
                else np.nan
            )
        _export_group_csv(
            long_df,
            ttest_df,
            classif_col,
            metric_name,
            bundle_for_csv,
            centroid_id,
            out_csv,
            analysis_lbl,
        )


def plot_correlation(
    corr_df,
    metric_name,
    bundle_name,
    var_name,
    out_path,
    title_suffix,
    removed_subjects=None,
    removed_points=None,
    export_csv=True,
    canonical_bundle=None,
    centroid_id=None,
):
    """
    Plot correlation: r values, p-values bar, and significant point markers.
    """
    plt.figure(figsize=(10, 5))
    if corr_df.empty:
        plt.text(0.5, 0.5, "No data", ha="center")
        plt.title(
            f"{bundle_name} - {metric_name} - Correlation {var_name} "
            f"{title_suffix} ({FWE_METHOD})"
        )
    else:
        ax = plt.gca()

        # Highlight significant clusters
        if FWE_METHOD in ("clusterFWE", "mixed") and "sig_afq" in corr_df.columns:
            _highlight_clusters(ax, corr_df["point"], corr_df["sig_afq"])

        ax.plot(corr_df["point"], corr_df["r"], color="tab:blue", label="r")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"r ({CORRELATION_TEST})")

        # Clip p-values for plotting
        if "p_raw" in corr_df.columns:
            corr_df = corr_df.copy()
            corr_df["p_raw"] = corr_df["p_raw"].clip(lower=0, upper=1)

        ax2 = ax.twinx()
        ax2.bar(
            corr_df["point"],
            corr_df["p_raw"],
            color="gray",
            alpha=0.25,
            width=1.0,
            label="p-value",
        )
        pmax = (
            float(corr_df["p_raw"].max())
            if corr_df["p_raw"].notna().any()
            else 0.05
        )
        ymax = min(1.0, max(0.06, pmax * 1.15))

        if (
            "alphaFWE" in corr_df.columns
            and not corr_df["alphaFWE"].isna().all()
            and FWE_METHOD in ("alphaFWE", "mixed")
        ):
            alphaFWE_val = corr_df["alphaFWE"].iloc[0]
            if not np.isnan(alphaFWE_val):
                ax2.axhline(
                    alphaFWE_val,
                    color="red",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"alphaFWE={alphaFWE_val:.3g}",
                )
                ymax = max(ymax, alphaFWE_val * 1.15)

        if (
            "clusterFWE" in corr_df.columns
            and not corr_df["clusterFWE"].isna().all()
            and FWE_METHOD in ("clusterFWE", "mixed")
        ):
            cluster_val = corr_df["clusterFWE"].iloc[0]
            if not np.isnan(cluster_val):
                ax2.text(
                    0.01,
                    0.95,
                    f"clusterFWE>={int(cluster_val)} (alpha={AFQ_ALPHA})",
                    transform=ax2.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.6,
                        edgecolor="gray",
                    ),
                )

        ax2.set_ylabel("p-value")
        ax2.set_ylim(0, ymax)

        if "sig_afq" in corr_df.columns and corr_df["sig_afq"].any():
            sig = corr_df[corr_df["sig_afq"]]
            ax.scatter(
                sig["point"], sig["r"], color="red", s=22, label="significant"
            )
            tmp = corr_df.copy()
            tmp["sig_afq"] = corr_df["sig_afq"]
            _annotate_pvalues(
                ax2, tmp.rename(columns={"sig_afq": "sig_fdr"}), "sig_fdr"
            )

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

        plt.title(
            f"{bundle_name} - {metric_name} - Correlation {var_name} "
            f"{title_suffix} ({FWE_METHOD})"
        )
        _annotate_missing(ax, removed_subjects or [], removed_points or [])

    plt.xlabel("Point")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    if export_csv:
        out_csv = os.path.splitext(out_path)[0] + ".csv"
        analysis_lbl = "raw" if "raw" in out_path else "partial"
        bundle_for_csv = canonical_bundle if canonical_bundle else bundle_name
        if centroid_id is None:
            centroid_id = (
                corr_df.get("centroid_id").iloc[0]
                if isinstance(corr_df, pd.DataFrame)
                and "centroid_id" in corr_df.columns
                and not corr_df["centroid_id"].isna().all()
                else np.nan
            )
        _export_corr_csv(
            corr_df,
            metric_name,
            bundle_for_csv,
            centroid_id,
            var_name,
            out_csv,
            analysis_lbl,
        )


# ============================================================================
# WORKER FUNCTIONS
# ============================================================================

def _worker_classif_variable(
    classif_col,
    bundle_data,
    metric_set,
    report_dir,
    AFQ_NPERM=1000,
    AFQ_ALPHA=0.05,
    precomputed_thresholds=None,
):
    """
    Worker for a single group variable: runs confound-corrected group tests
    across all bundles and metrics.

    Parameters
    ----------
    classif_col : str
        Name of the grouping variable.
    bundle_data : dict
        {bundle_name: DataFrame} for all bundles.
    metric_set : list of str
        Metric columns to analyse.
    precomputed_thresholds : dict, optional
        {metric_col: (alphaFWE, clusterFWE)} from multi-tract correction.
    """
    figure_dir = opj(report_dir, "figures")
    ensure_dir(figure_dir)
    summary_rows = []
    included_participants = set()
    removed_participants = set()

    if not bundle_data:
        return {
            "summary": summary_rows,
            "included_participants": included_participants,
            "removed_participants": removed_participants,
        }

    sig_key = "alphaFWE" if FWE_METHOD == "alphaFWE" else "clusterFWE"

    # Pre-compute multi-tract thresholds if requested
    if CORRECT_MULTI_TRACT and precomputed_thresholds is None:
        precomputed_thresholds = {}
        for metric_col in metric_set:
            alphaFWE, clusterFWE = compute_multi_tract_thresholds_group(
                bundle_data,
                metric_col,
                classif_col,
                alpha=AFQ_ALPHA,
                nperm=AFQ_NPERM,
            )
            precomputed_thresholds[metric_col] = (alphaFWE, clusterFWE)

    # Get confounds for this group variable
    conf_list = group_variables.get(classif_col, {}).get("confounds", [])
    confonds_str = ",".join(conf_list) if conf_list else "none"

    loop_counter = 0
    for metric_col in tqdm(metric_set, desc=f"group:{classif_col}"):
        for bundle_name, df in tqdm(
            bundle_data.items(), desc=f"{classif_col}-{metric_col}", leave=False
        ):
            # Handle multiple centroids per bundle
            if "centroid_id" in df.columns:
                try:
                    centroid_vals = (
                        pd.to_numeric(df["centroid_id"], errors="coerce")
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                except Exception:
                    centroid_vals = df["centroid_id"].dropna().unique().tolist()
                centroid_vals = sorted(centroid_vals)
            else:
                centroid_vals = [None]

            for cid in centroid_vals:
                df_sub = df if cid is None else df[df["centroid_id"] == cid]
                if df_sub is None or df_sub.empty:
                    continue
                bundle_label = (
                    f"{bundle_name}_cent{cid}" if cid is not None else bundle_name
                )

                try:
                    point_col = detect_point_column(df_sub)
                except Exception:
                    continue
                if (
                    metric_col not in df_sub.columns
                    or classif_col not in df_sub.columns
                ):
                    continue

                # Prepare long format and clean missing data
                try:
                    long_raw = prepare_long(df_sub, metric_col, point_col)
                except Exception:
                    continue
                long_raw = long_raw[long_raw[classif_col].notna()]
                if long_raw.empty:
                    del long_raw
                    continue

                subject_pid_map = _build_subject_participant_map(long_raw)
                (
                    long_raw_clean,
                    removed_subjects,
                    removed_points,
                ) = clean_missing_data(long_raw, threshold=0.15)
                del long_raw
                if long_raw_clean.empty:
                    del long_raw_clean
                    continue

                included_participants.update(
                    _subjects_to_participant_ids(
                        long_raw_clean["subject"].dropna().unique(),
                        subject_pid_map,
                    )
                )
                removed_participants.update(
                    _subjects_to_participant_ids(removed_subjects, subject_pid_map)
                )

                # --- Confound correction ---
                if conf_list:
                    corrected_segments = []
                    for pid, gpt in long_raw_clean.groupby("point"):
                        subj_level = (
                            gpt[
                                ["subject", "value"]
                                + [c for c in conf_list if c in gpt.columns]
                            ]
                            .drop_duplicates(subset="subject")
                            .set_index("subject")
                        )
                        y_res = residualize_on_confond(
                            subj_level.assign(target=subj_level["value"]),
                            "target",
                            conf_list,
                        )
                        g_corr = gpt.copy()
                        g_corr["value"] = g_corr["subject"].map(y_res)
                        corrected_segments.append(g_corr)
                    long_corr = (
                        pd.concat(corrected_segments, ignore_index=True)
                        if corrected_segments
                        else pd.DataFrame()
                    )
                    del corrected_segments
                else:
                    long_corr = long_raw_clean.copy()

                # --- Retrieve pre-computed thresholds ---
                pre_alpha, pre_cluster = None, None
                if (
                    CORRECT_MULTI_TRACT
                    and precomputed_thresholds
                    and metric_col in precomputed_thresholds
                ):
                    pre_alpha, pre_cluster = precomputed_thresholds[metric_col]

                # --- Run group test ---
                corr_res = (
                    group_test_afq(
                        long_corr,
                        classif_col,
                        alpha=AFQ_ALPHA,
                        nperm=AFQ_NPERM,
                        precomputed_alphaFWE=pre_alpha,
                        precomputed_clusterFWE=pre_cluster,
                    )
                    if not long_corr.empty
                    else pd.DataFrame()
                )

                if not corr_res.empty:
                    out_plot = opj(
                        figure_dir,
                        f"{bundle_label}_{metric_col}_{classif_col}_group_corrected.png",
                    )
                    plot_group_diff(
                        long_corr,
                        corr_res,
                        classif_col,
                        metric_col,
                        bundle_label,
                        out_plot,
                        "(confound-corrected)"
                        + (" [multi-tract]" if CORRECT_MULTI_TRACT else ""),
                        removed_subjects,
                        removed_points,
                        export_csv=True,
                        canonical_bundle=bundle_name,
                        centroid_id=cid,
                    )

                # --- Record summary ---
                n_subjects_used = (
                    int(long_corr["subject"].dropna().nunique())
                    if not long_corr.empty
                    else 0
                )
                n_subjects_removed = len(removed_subjects)
                summary_rows.append(
                    {
                        "bundle": bundle_name,
                        "centroid_id": cid if cid is not None else np.nan,
                        "metric": metric_col,
                        "type": f"group_{classif_col}",
                        "n_sig": (
                            _count_clusters(corr_res["sig_afq"])
                            if FWE_METHOD == "clusterFWE"
                            else int(corr_res["sig_afq"].sum())
                        )
                        if not corr_res.empty
                        else 0,
                        "min_p": float(corr_res["p_raw"].min())
                        if not corr_res.empty
                        else np.nan,
                        "max_abs_r": np.nan,
                        "removed_points": len(removed_points),
                        "subjects_used": n_subjects_used,
                        "removed_subjects": n_subjects_removed,
                        "confounds_used": confonds_str,
                        sig_key: float(corr_res[sig_key].iloc[0])
                        if not corr_res.empty
                        else np.nan,
                    }
                )

                del long_raw_clean, long_corr, corr_res
                loop_counter += 1
                if loop_counter % 10 == 0:
                    gc.collect()

    gc.collect()
    return {
        "summary": summary_rows,
        "included_participants": included_participants,
        "removed_participants": removed_participants,
    }


def _worker_corr_variable(
    var_col,
    bundle_data,
    metric_set,
    report_dir,
    precomputed_thresholds=None,
):
    """
    Worker for a single correlation variable: runs confound-corrected partial
    correlations across all bundles and metrics.

    Parameters
    ----------
    var_col : str
        Name of the continuous variable to correlate with metrics.
    bundle_data : dict
        {bundle_name: DataFrame} for all bundles.
    metric_set : list of str
        Metric columns to analyse.
    precomputed_thresholds : dict, optional
        {metric_col: (alphaFWE, clusterFWE)} from multi-tract correction.
    """
    rows = []
    included_participants = set()
    removed_participants = set()

    if not bundle_data:
        return {
            "summary": rows,
            "included_participants": included_participants,
            "removed_participants": removed_participants,
        }

    figure_dir = opj(report_dir, "figures")
    ensure_dir(figure_dir)
    sig_key = "alphaFWE" if FWE_METHOD == "alphaFWE" else "clusterFWE"

    # Get confounds for this correlation variable
    confonds_for_var = corr_variables.get(var_col, {}).get("confounds", [])
    confonds_str = ",".join(confonds_for_var) if confonds_for_var else "none"

    # Pre-compute multi-tract thresholds if requested
    if CORRECT_MULTI_TRACT and precomputed_thresholds is None:
        precomputed_thresholds = {}
        for metric_col in metric_set:
            alphaFWE, clusterFWE = compute_multi_tract_thresholds_corr(
                bundle_data,
                metric_col,
                var_col,
                confonds_for_var,
                alpha=AFQ_ALPHA,
                nperm=AFQ_NPERM,
            )
            precomputed_thresholds[metric_col] = (alphaFWE, clusterFWE)

    loop_counter = 0
    for bundle_name, df in tqdm(
        bundle_data.items(), desc=f"corr:{var_col} bundles"
    ):
        if "centroid_id" in df.columns:
            try:
                centroid_vals = (
                    pd.to_numeric(df["centroid_id"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
            except Exception:
                centroid_vals = df["centroid_id"].dropna().unique().tolist()
            centroid_vals = sorted(centroid_vals)
        else:
            centroid_vals = [None]

        for cid in centroid_vals:
            df_sub = df if cid is None else df[df["centroid_id"] == cid]
            if df_sub is None or df_sub.empty:
                continue
            bundle_label = (
                f"{bundle_name}_cent{cid}" if cid is not None else bundle_name
            )

            try:
                point_col = detect_point_column(df_sub)
            except Exception:
                continue

            metric_cols = [m for m in metric_set if m in df_sub.columns]
            for metric_col in tqdm(
                metric_cols, desc=f"{var_col}-{bundle_name} metrics", leave=False
            ):
                long_df = prepare_long(df_sub, metric_col, point_col)
                subject_pid_map = _build_subject_participant_map(long_df)
                (
                    long_df_clean,
                    removed_subjects,
                    removed_points,
                ) = clean_missing_data(long_df, threshold=0.15)
                del long_df
                if long_df_clean.empty or var_col not in long_df_clean.columns:
                    if not long_df_clean.empty:
                        del long_df_clean
                    continue

                included_participants.update(
                    _subjects_to_participant_ids(
                        long_df_clean["subject"].dropna().unique(),
                        subject_pid_map,
                    )
                )
                removed_participants.update(
                    _subjects_to_participant_ids(removed_subjects, subject_pid_map)
                )

                # --- Retrieve pre-computed thresholds ---
                pre_alpha, pre_cluster = None, None
                if (
                    CORRECT_MULTI_TRACT
                    and precomputed_thresholds
                    and metric_col in precomputed_thresholds
                ):
                    pre_alpha, pre_cluster = precomputed_thresholds[metric_col]

                # --- Run partial correlation ---
                corr_partial = pointwise_partial_correlation(
                    long_df_clean,
                    var_col,
                    confonds_for_var,
                    precomputed_alphaFWE=pre_alpha,
                    precomputed_clusterFWE=pre_cluster,
                )

                out_corr = opj(
                    figure_dir,
                    f"{bundle_label}_{metric_col}_{var_col}_corr_partial.png",
                )
                plot_correlation(
                    corr_partial,
                    metric_col,
                    bundle_label,
                    var_name=var_col,
                    out_path=out_corr,
                    title_suffix="(confound-corrected)"
                    + (" [multi-tract]" if CORRECT_MULTI_TRACT else ""),
                    removed_subjects=removed_subjects,
                    removed_points=removed_points,
                    export_csv=True,
                    canonical_bundle=bundle_name,
                    centroid_id=cid,
                )

                n_subjects_used = int(
                    long_df_clean["subject"].dropna().nunique()
                )
                n_subjects_removed = len(removed_subjects)
                rows.append(
                    {
                        "bundle": bundle_name,
                        "centroid_id": cid if cid is not None else np.nan,
                        "metric": metric_col,
                        "type": f"corr_{var_col}",
                        "n_sig": (
                            _count_clusters(corr_partial["sig_afq"])
                            if FWE_METHOD == "clusterFWE"
                            else int(corr_partial["sig_afq"].sum())
                        )
                        if not corr_partial.empty
                        else 0,
                        "min_p": float(corr_partial["p_raw"].min())
                        if not corr_partial.empty
                        else np.nan,
                        "max_abs_r": float(corr_partial["r"].abs().max())
                        if not corr_partial.empty
                        else np.nan,
                        "removed_points": len(removed_points),
                        "subjects_used": n_subjects_used,
                        "removed_subjects": n_subjects_removed,
                        "confounds_used": confonds_str,
                        sig_key: float(corr_partial[sig_key].iloc[0])
                        if not corr_partial.empty
                        else np.nan,
                    }
                )

                del long_df_clean, corr_partial
                loop_counter += 1
                if loop_counter % 10 == 0:
                    gc.collect()

    gc.collect()
    return {
        "summary": rows,
        "included_participants": included_participants,
        "removed_participants": removed_participants,
    }


def _worker_bundle(bundle_name, df, metric_set, report_dir):
    """
    Worker for a single bundle — used when multiprocessing is enabled
    and multi-tract correction is OFF (no cross-bundle dependency).
    """
    bundle_data = {bundle_name: df}
    summary_rows = []
    included = set()
    removed = set()

    for var in group_variables.keys():
        r = _worker_classif_variable(var, bundle_data, metric_set, report_dir)
        summary_rows.extend(r.get("summary", []))
        included.update(r.get("included_participants", set()))
        removed.update(r.get("removed_participants", set()))

    for var in corr_variables.keys():
        r = _worker_corr_variable(var, bundle_data, metric_set, report_dir)
        summary_rows.extend(r.get("summary", []))
        included.update(r.get("included_participants", set()))
        removed.update(r.get("removed_participants", set()))

    return {
        "summary": summary_rows,
        "included_participants": included,
        "removed_participants": removed,
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(
    bundle_names, csv_files, report_dir="report_output", n_jobs=1
):
    """
    Generate the full analysis report.

    Steps:
    1. Load all bundle DataFrames (with optional resampling).
    2. Detect available metrics.
    3. For each group/correlation variable, run the appropriate worker
       (in parallel if MULTIPROCESSING is enabled).
    4. Aggregate results and save summary CSV + removed participants list.
    """
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    bundle_data = {}
    bundle_names = list(bundle_names)

    print(f"Analysing bundles: {', '.join(bundle_names)}")
    print(
        f"Multi-tract correction: {'ENABLED' if CORRECT_MULTI_TRACT else 'disabled'}"
    )
    print(
        f"Resampling: {f'{RESAMPLE_N_POINTS} points' if RESAMPLE_N_POINTS else 'disabled'}"
    )
    print(
        f"Stat types: {STAT_TYPES if STAT_TYPES else 'all (bare columns)'}"
    )

    # --- Load bundles ---
    for bundle_name in tqdm(
        [bn.replace("_", "") for bn in bundle_names], desc="Loading bundles"
    ):
        bundle_csvs = [
            f for f in csv_files if f.get_entities()["bundle"] == bundle_name
        ]
        if not bundle_csvs:
            continue
        df, _ = load_and_merge_bundle_csvs(bundle_name, bundle_csvs)
        if RESAMPLE_N_POINTS:
            df = resample_bundle_data(df, RESAMPLE_N_POINTS)
        bundle_data[bundle_name] = df
        if len(bundle_data) % 5 == 0:
            gc.collect()

    if not bundle_data:
        print("No bundles to analyse.")
        return

    # --- Detect metrics ---
    metric_set = sorted(
        {m for df in bundle_data.values() for m in detect_metric_columns(df)}
    )
    print(
        f"Detected metrics: {', '.join(metric_set) if metric_set else '(none)'}"
    )

    # --- Build variable job list ---
    var_jobs = ([("group", v) for v in group_variables.keys()] +
                [("corr", v) for v in corr_variables.keys()])

    summary_rows = []
    included_participants = set()
    removed_participants = set()

    # --- Choose parallelisation strategy ---
    use_bundle_parallel = (
        config.get("MULTIPROCESSING", True)
        and not CORRECT_MULTI_TRACT
        and n_jobs > 1
        and bundle_data
    )

    if use_bundle_parallel:
        from joblib import Parallel, delayed

        print(
            f"Parallelising by bundle ({len(bundle_data)} bundles, {n_jobs} jobs)"
        )
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_worker_bundle)(name, df, metric_set, report_dir)
            for name, df in bundle_data.items()
        )
        for r in results:
            if not r:
                continue
            summary_rows.extend(r.get("summary", []))
            included_participants.update(r.get("included_participants", set()))
            removed_participants.update(r.get("removed_participants", set()))
        del results
        gc.collect()

    elif n_jobs > 1 and var_jobs:
        from joblib import Parallel, delayed

        def _dispatch(job):
            kind, var = job
            if kind == "group":
                return _worker_classif_variable(
                    var, bundle_data, metric_set, report_dir
                )
            else:
                return _worker_corr_variable(
                    var, bundle_data, metric_set, report_dir
                )

        if config.get("MULTIPROCESSING", True):
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_dispatch)(job) for job in var_jobs
            )
        else:
            results = [_dispatch(job) for job in var_jobs]

        for r in results:
            if not r:
                continue
            summary_rows.extend(r.get("summary", []))
            included_participants.update(r.get("included_participants", set()))
            removed_participants.update(r.get("removed_participants", set()))
        del results
        gc.collect()

    else:
        # Sequential execution
        for kind, var in tqdm(var_jobs, desc="Variables"):
            if kind == "group":
                res = _worker_classif_variable(
                    var, bundle_data, metric_set, report_dir
                )
            else:
                res = _worker_corr_variable(
                    var, bundle_data, metric_set, report_dir
                )
            if res:
                summary_rows.extend(res.get("summary", []))
                included_participants.update(
                    res.get("included_participants", set())
                )
                removed_participants.update(
                    res.get("removed_participants", set())
                )
            gc.collect()

    if not summary_rows:
        print("No data for the report.")
        return

    # --- Save summary CSV ---
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = opj(report_dir, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV: {summary_csv}")

    # --- Save removed participants ---
    reference_participants = _get_reference_participants()
    if reference_participants:
        final_removed = sorted(reference_participants - included_participants)
    else:
        final_removed = sorted(removed_participants - included_participants)
    removed_csv = opj(report_dir, "removed_participants.csv")
    pd.DataFrame({"participant_id": final_removed}).to_csv(removed_csv, index=False)
    print(f"Removed participants: {len(final_removed)} (see {removed_csv})")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Entry point: parse arguments, load config, set up globals,
    and run the report generator.
    """
    global config, dataset, hcp_asso_pipeline, csv_files, bundle_names, ds
    global group_variables, corr_variables
    global AFQ_ALPHA, AFQ_NPERM, FWE_METHOD, CORRECT_MULTI_TRACT
    global RESAMPLE_N_POINTS, CORRECT_METRIC_CONFOND_ONLY, CORRELATION_TEST
    global METRIC_COLUMNS_CANDIDATES, STAT_TYPES
    global _ADDITIONAL_INFO_DF
    global _SUBJECTS_FILTER, _SUBJECTS_TABLE_DF

    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="AFQ tractometry analysis report generator"
    )
    parser.add_argument(
        "-c",
        type=str,
        default=None,
        help="Path to JSON configuration file",
        required=True,
    )
    parser.add_argument(
        "--subjects-table",
        "-s",
        type=str,
        default=None,
        help="Path to CSV/XLSX with a 'participant_id' column "
        "listing subjects to include",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (overrides config)",
        required=True
    )
    args = parser.parse_args()

    # --- Load configuration ---
    config = load_config(args.c)

    # --- Extract config values into globals ---
    dataset = config.get("dataset", DEFAULT_CONFIG["dataset"])
    hcp_asso_pipeline = config.get(
        "hcp_asso_pipeline", DEFAULT_CONFIG["hcp_asso_pipeline"]
    )
    pipeline = config.get("pipeline", DEFAULT_CONFIG["pipeline"])

    # Variable definitions: each maps to {"confounds": [...]}
    group_variables = config.get(
        "group_variables", DEFAULT_CONFIG["group_variables"]
    )
    corr_variables = config.get(
        "corr_variables", DEFAULT_CONFIG["corr_variables"]
    )

    # Analysis hyperparameters
    AFQ_ALPHA = config.get("AFQ_ALPHA", DEFAULT_CONFIG["AFQ_ALPHA"])
    AFQ_NPERM = config.get("AFQ_NPERM", DEFAULT_CONFIG["AFQ_NPERM"])
    FWE_METHOD = config.get("FWE_METHOD", DEFAULT_CONFIG["FWE_METHOD"])
    CORRECT_MULTI_TRACT = config.get(
        "CORRECT_MULTI_TRACT", DEFAULT_CONFIG["CORRECT_MULTI_TRACT"]
    )
    RESAMPLE_N_POINTS = config.get(
        "RESAMPLE_N_POINTS", DEFAULT_CONFIG["RESAMPLE_N_POINTS"]
    )
    CORRECT_METRIC_CONFOND_ONLY = config.get(
        "CORRECT_METRIC_CONFOND_ONLY",
        DEFAULT_CONFIG["CORRECT_METRIC_CONFOND_ONLY"],
    )
    CORRELATION_TEST = config.get(
        "CORRELATION_TEST", DEFAULT_CONFIG["CORRELATION_TEST"]
    )
    METRIC_COLUMNS_CANDIDATES = config.get(
        "METRIC_COLUMNS_CANDIDATES",
        DEFAULT_CONFIG["METRIC_COLUMNS_CANDIDATES"],
    )
    STAT_TYPES = config.get("STAT_TYPES", DEFAULT_CONFIG["STAT_TYPES"])

    # --- Load subjects table (optional) ---
    _SUBJECTS_FILTER = None
    _SUBJECTS_TABLE_DF = None
    if args.subjects_table:
        if not os.path.exists(args.subjects_table):
            raise FileNotFoundError(
                f"Subjects table not found: {args.subjects_table}"
            )
        _SUBJECTS_FILTER, _SUBJECTS_TABLE_DF = load_subjects_table(
            args.subjects_table
        )
        print(
            f"Subjects table loaded: {len(_SUBJECTS_FILTER)} participant(s) to process"
        )
        validate_config_columns_in_table(_SUBJECTS_TABLE_DF, config)

    # --- Load external tables (optional, driven by config) ---
    additional_info_path = config.get("additional_info_path")
    _ADDITIONAL_INFO_DF = _load_external_tables(
        additional_info_path
    )

    # --- Load dataset ---
    ds = Dataset(dataset)
    csv_files = ds.get_global(
        pipeline=hcp_asso_pipeline,
        extension="csv",
        datatype="metric",
        suffix="mean",
    )
    bundle_names = list(get_HCP_bundle_names().keys())

    # --- Determine output directory ---
    output_dir = args.output_dir or config.get("output_dir")
    if output_dir is None:
        # Auto-generate from key config values
        output_dir = opj(
            os.getcwd(),
            f"report_{pipeline}_{FWE_METHOD}_{CORRELATION_TEST}",
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Report output: {output_dir}")

    # --- Save the used configuration ---
    save_config(config, opj(output_dir, "config.json"))

    # --- Determine number of parallel jobs ---
    n_jobs = config.get("n_jobs")
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() // 3)

    # --- Run the analysis ---
    generate_report(
        bundle_names, csv_files, report_dir=output_dir, n_jobs=n_jobs
    )


if __name__ == "__main__":
    main()