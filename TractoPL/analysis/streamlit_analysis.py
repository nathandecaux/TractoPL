#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit interface — TractoPL / population_analysis.py
================================================
Builds the config.json expected by population_analysis.py, lets you
load/save/download it, and runs the analysis with live logs.

Usage:
    streamlit run streamlit_analysis.py
"""

import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from TractoPL.data.loader import Dataset, Subject
import pandas as pd
import streamlit as st
import datetime as date

# ---------------------------------------------------------------------------
# Default config (mirror of DEFAULT_CONFIG in population_analysis.py)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "pipeline": date.datetime.now().strftime("%Y%m%d%H%M%S"),
    "dataset": "",
    "hcp_asso_pipeline": "tractometry",
    "group_variables": {},           # e.g. {"group": {"confounds": ["age"]}}
    "corr_variables": {},            # e.g. {"score": {"confounds": ["age"]}}
    "test_by_group": None,
    "FWE_METHOD": "clusterFWE",      # clusterFWE | alphaFWE | mixed
    "AFQ_ALPHA": 0.05,
    "AFQ_NPERM": 1000,
    "CORRECT_MULTI_TRACT": False,
    "MISSING_SUBJECTS_TOLERANCE": 0,
    "INTERPOLATE_MISSING_POINTS": 2,
    "KEEP_LARGEST_CONTIGUOUS": True,
    "RESAMPLE_N_POINTS": None,
    "METRIC_COLUMNS_CANDIDATES": ["FA"],
    "STAT_TYPES": ["mean"],
    "CORRELATION_TEST": "pearson",   # pearson | spearman
    "CORRECT_METRIC_CONFOND_ONLY": False,
    "additional_info_path": None,
    "EXCLUDE": {},                   # e.g. {"group": ["excluded"]}
    "MULTIPROCESSING": True,
    "n_jobs": None,                  # None = auto
    "output_dir": None,
}

FWE_METHODS = ["clusterFWE", "alphaFWE", "mixed"]
CORRELATION_TESTS = ["pearson", "spearman"]
METRIC_CHOICES = ["FA", "IFW",'IRF']
STAT_CHOICES = ["mean"]

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "streamlit_workdir"
WORK_DIR.mkdir(exist_ok=True)
CONFIG_PATH = WORK_DIR / "config.json"
MERGED_SUBJECTS_PATH = WORK_DIR / "subjects_with_additional_info.xlsx"
DEFAULT_SCRIPT = APP_DIR / "population_analysis.py"
NO_GROUP = "(none)"
DEFAULT_JOIN_KEY = "participant_id"

st.set_page_config(page_title="TractoPL — AFQ Analysis", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def deep_copy(d):
    return json.loads(json.dumps(d))

def state_as_list(key: str) -> list:
    """Ensure session_state[key] is a list (for multiselect)."""
    raw = st.session_state.get(key, [])
    if isinstance(raw, str):
        st.session_state[key] = [s.strip() for s in raw.split(",") if s.strip()]
    return st.session_state.get(key, [])


def state_as_str(key: str) -> str:
    """Ensure session_state[key] is a string (for text_input)."""
    raw = st.session_state.get(key, [])
    if isinstance(raw, (list, tuple)):
        st.session_state[key] = ", ".join(map(str, raw))
    return st.session_state.get(key, "")


def conf_as_list(var: str) -> list:
    """Ensure session_state['conf::<var>'] is a list (for multiselect)."""
    return state_as_list(f"conf::{var}")


def get_active_subjects_df():
    """Subjects table used in the UI: joined with the additional info table if any,
    and filtered on the HCP pipeline if one is selected."""
    ss = st.session_state
    filtered = ss.get("filtered_subjects_df")
    if filtered is not None:
        return filtered
    merged = ss.get("merged_subjects_df")
    if merged is not None:
        return merged
    return ss.get("subjects_df")


def conf_as_str(var: str) -> str:
    """Ensure session_state['conf::<var>'] is a string (for text_input)."""
    return state_as_str(f"conf::{var}")

def push_config_to_widgets(cfg):
    """Inject a config into the widgets' session keys."""
    ss = st.session_state
    ss["dataset"] = cfg.get("dataset") or ""
    ss["pipeline"] = cfg.get("pipeline", "default")
    ss["hcp_pipeline"] = cfg.get("hcp_asso_pipeline", "tractometry")
    ss["output_dir"] = cfg.get("output_dir") or ""

    gv = list(cfg.get("group_variables", {}).keys())
    cv = list(cfg.get("corr_variables", {}).keys())
    ss["group_vars_sel"] = gv
    ss["corr_vars_sel"] = cv
    ss["manual_group_vars"] = ", ".join(gv)
    ss["manual_corr_vars"] = ", ".join(cv)
    for var, spec in {**cfg.get("group_variables", {}), **cfg.get("corr_variables", {})}.items():
        ss[f"conf::{var}"] = list(spec.get("confounds", []))

    ss["test_by_group"] = cfg.get("test_by_group") or NO_GROUP
    ss["fwe_method"] = cfg.get("FWE_METHOD", "clusterFWE")
    ss["afq_alpha"] = float(cfg.get("AFQ_ALPHA", 0.05))
    ss["afq_nperm"] = int(cfg.get("AFQ_NPERM", 1000))
    ss["corr_test"] = cfg.get("CORRELATION_TEST", "pearson")
    ss["correct_multi_tract"] = bool(cfg.get("CORRECT_MULTI_TRACT", False))

    ss["missing_tol"] = int(cfg.get("MISSING_SUBJECTS_TOLERANCE", 0))
    ss["interp_gap"] = int(cfg.get("INTERPOLATE_MISSING_POINTS", 2))
    ss["keep_contig"] = bool(cfg.get("KEEP_LARGEST_CONTIGUOUS", True))
    ss["resample_n"] = int(cfg.get("RESAMPLE_N_POINTS") or 0)

    ss["metrics"] = list(cfg.get("METRIC_COLUMNS_CANDIDATES", ["FA"]))
    ss["stat_types"] = list(cfg.get("STAT_TYPES", ["mean"]))
    ss["correct_metric_confond_only"] = bool(cfg.get("CORRECT_METRIC_CONFOND_ONLY", False))

    ss["multiproc"] = bool(cfg.get("MULTIPROCESSING", True))
    ss["n_jobs"] = int(cfg.get("n_jobs") or 0)

    for rid in ss.get("exclude_ids", []):
        ss.pop(f"excl_col::{rid}", None)
        ss.pop(f"excl_vals::{rid}", None)
    ss["exclude_ids"] = []
    for column, values in cfg.get("EXCLUDE", {}).items():
        add_exclude_row(column, values)
    for var, spec in {**cfg.get("group_variables", {}), **cfg.get("corr_variables", {})}.items():
        if ss.get("subjects_df") is not None:
            ss[f"conf::{var}"] = list(spec.get("confounds", []))
        else:
            ss[f"conf::{var}"] = ", ".join(map(str, spec.get("confounds", [])))


def init_state():
    ss = st.session_state
    if ss.get("app_initialized"):
        return
    ss.app_initialized = True
    ss.config = deep_copy(DEFAULT_CONFIG)
    ss.subjects_df = None
    ss.subjects_file = None
    ss.subjects_error = None
    ss.additional_info_df = None
    ss.additional_info_file = None
    ss.additional_info_error = None
    ss.merged_subjects_df = None
    ss.merged_subjects_file = None
    ss.join_report = None
    ss.run = None
    ss.script_path = str(DEFAULT_SCRIPT)
    ss.python_exe = sys.executable
    ss.mean_csv_files = []
    ss.filtered_subjects_df = None
    ss.pipeline_subjects = []
    ss.available_metrics = []
    ss.available_stat_types = []
    ss.metrics_read_error = None
    push_config_to_widgets(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Subjects table / additional info join
# ---------------------------------------------------------------------------

def read_table(f) -> pd.DataFrame:
    """Read an uploaded CSV/XLSX file."""
    if f.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    return pd.read_csv(f)


def default_join_key(columns, current):
    """Keep the current join column if still valid, else 'participant_id', else the first column."""
    if current in columns:
        return current
    if DEFAULT_JOIN_KEY in columns:
        return DEFAULT_JOIN_KEY
    return columns[0] if columns else None


def left_join_additional_info(subjects_df, add_df, subjects_key, add_key):
    """
    Left-join the additional info table onto the subjects table.

    Keys are compared as text, ignoring surrounding spaces and a leading 'sub-'.
    Columns already present in the subjects table are kept from the subjects
    table (same rule as population_analysis.py). Duplicate keys in the additional
    table keep their first row, so each subject stays on a single row.
    """
    join_col, status_col = "__join_key__", "__join_status__"

    def normalize(s):
        return s.astype(str).str.strip().str.replace(r"^sub-", "", regex=True).where(s.notna())

    skipped = [c for c in add_df.columns if c != add_key and c in subjects_df.columns]
    added = [c for c in add_df.columns if c != add_key and c not in subjects_df.columns]

    right = add_df[added].assign(**{join_col: normalize(add_df[add_key])})
    right = right[right[join_col].notna()]
    n_duplicates = int(right[join_col].duplicated().sum())
    right = right.drop_duplicates(join_col)

    left = subjects_df.assign(**{join_col: normalize(subjects_df[subjects_key])})
    merged = left.merge(right, on=join_col, how="left", indicator=status_col)
    n_matched = int((merged[status_col] == "both").sum())
    merged = merged.drop(columns=[join_col, status_col])

    report = {
        "n_subjects": len(subjects_df),
        "n_matched": n_matched,
        "n_duplicates": n_duplicates,
        "added": added,
        "skipped": skipped,
    }
    return merged, report


def refresh_subjects_view():
    """
    Rebuild the subjects table used by the UI and the analysis: left join with the
    additional info table (if any), then filter on the subjects available in the
    selected HCP pipeline.
    """
    ss = st.session_state
    subjects_df = ss.get("subjects_df")
    add_df = ss.get("additional_info_df")

    ss["merged_subjects_df"] = None
    ss["merged_subjects_file"] = None
    ss["join_report"] = None
    if subjects_df is not None and add_df is not None:
        ss["join_key_subjects"] = default_join_key(list(subjects_df.columns),
                                                   ss.get("join_key_subjects"))
        ss["join_key_additional"] = default_join_key(list(add_df.columns),
                                                     ss.get("join_key_additional"))
        if ss["join_key_subjects"] and ss["join_key_additional"]:
            merged, report = left_join_additional_info(
                subjects_df, add_df, ss["join_key_subjects"], ss["join_key_additional"])
            # Saved as xlsx so that text IDs (e.g. '001') keep their type when re-read
            merged.to_excel(MERGED_SUBJECTS_PATH, index=False)
            ss["merged_subjects_df"] = merged
            ss["merged_subjects_file"] = str(MERGED_SUBJECTS_PATH)
            ss["join_report"] = report

    base_df = ss["merged_subjects_df"] if ss["merged_subjects_df"] is not None else subjects_df
    pipeline_subjects = ss.get("pipeline_subjects") or []
    if base_df is not None and pipeline_subjects:
        pid_norm = base_df["participant_id"].astype(str).str.replace(r"^sub-", "", regex=True)
        ss["filtered_subjects_df"] = base_df[pid_norm.isin(pipeline_subjects)].reset_index(drop=True)
    else:
        ss["filtered_subjects_df"] = None

    # Clean up selections that are no longer valid
    if base_df is not None:
        cols = list(base_df.columns)
        ss["group_vars_sel"] = [v for v in ss.get("group_vars_sel", []) if v in cols]
        ss["corr_vars_sel"] = [v for v in ss.get("corr_vars_sel", []) if v in cols]


# ---------------------------------------------------------------------------
# Exclusions (EXCLUDE): one row = a column + the values to exclude
# ---------------------------------------------------------------------------

def add_exclude_row(column=None, values=None):
    ss = st.session_state
    rid = ss.get("exclude_next_id", 0)
    ss["exclude_next_id"] = rid + 1
    ss["exclude_ids"] = ss.get("exclude_ids", []) + [rid]
    ss[f"excl_col::{rid}"] = column
    ss[f"excl_vals::{rid}"] = list(values or [])


def remove_exclude_row(rid):
    ss = st.session_state
    ss["exclude_ids"] = [r for r in ss.get("exclude_ids", []) if r != rid]
    ss.pop(f"excl_col::{rid}", None)
    ss.pop(f"excl_vals::{rid}", None)


def clear_exclude_values(rid):
    """The column changed: its previously selected values no longer apply."""
    st.session_state[f"excl_vals::{rid}"] = []


def column_values(df, column) -> list:
    """Distinct non-missing values of a column, as JSON-serialisable Python values.

    Native types are kept (not converted to text) because population_analysis.py
    matches them with isin(): 1 and '1' are different values.
    """
    if column not in df.columns:
        return []
    values = [v if isinstance(v, (bool, int, float, str)) else str(v)
              for v in pd.Series(df[column].dropna().unique()).tolist()]
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def render_exclude_row(rid, df):
    """Column + values to exclude, suggested from the subjects table when one is loaded."""
    ss = st.session_state
    col_key, vals_key = f"excl_col::{rid}", f"excl_vals::{rid}"
    c1, c2, c3 = st.columns([2, 5, 1], vertical_alignment="bottom")
    if df is not None:
        columns = list(df.columns)
        column = ss.get(col_key) or None
        if column is not None and column not in columns:
            columns.append(column)  # keep a column coming from a loaded config visible
        ss[col_key] = column
        c1.selectbox("Column", columns, key=col_key, placeholder="Choose a column",
                     on_change=clear_exclude_values, args=(rid,))
        options = column_values(df, column) if column else []
        # Map values typed as text (older configs) onto the table's values, e.g. '1' -> 1
        by_text = {str(o): o for o in options}
        current = [v if v in options else by_text.get(str(v), v) for v in state_as_list(vals_key)]
        options += [v for v in current if v not in options]
        ss[vals_key] = current
        c2.multiselect("Values to exclude", options, key=vals_key, disabled=column is None,
                       placeholder="Choose values")
    else:
        if ss.get(col_key) is None:
            ss[col_key] = ""
        c1.text_input("Column", key=col_key)
        state_as_str(vals_key)
        c2.text_input("Values to exclude (comma-separated)", key=vals_key)
    c3.button("🗑️", key=f"excl_del::{rid}", help="Remove this exclusion",
              on_click=remove_exclude_row, args=(rid,))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_reset():
    st.session_state.config = deep_copy(DEFAULT_CONFIG)
    push_config_to_widgets(DEFAULT_CONFIG)


def on_load_cfg():
    ss = st.session_state
    ss.cfg_load_error = None
    f = ss.get("cfg_uploader")
    if f is None:
        return
    try:
        cfg = json.loads(f.getvalue().decode("utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("The root JSON value must be an object.")
        ss.config = cfg
        push_config_to_widgets(cfg)
    except Exception as e:
        ss.cfg_load_error = str(e)
        return
    apply_dataset_and_pipeline()


def on_apply_json():
    """Apply the manually edited JSON. Run as a callback, before the widgets are
    instantiated, so that their session keys can still be modified."""
    ss = st.session_state
    ss.json_apply_error = None
    try:
        cfg = json.loads(ss.get("json_editor", ""))
        if not isinstance(cfg, dict):
            raise ValueError("The root JSON value must be an object.")
    except Exception as e:
        ss.json_apply_error = f"Invalid JSON: {e}"
        return
    ss.config = cfg
    push_config_to_widgets(cfg)
    apply_dataset_and_pipeline()


def apply_dataset_and_pipeline():
    """After a config is loaded: load its dataset, then its HCP pipeline (if set),
    as if both had been entered in the Data tab."""
    ss = st.session_state
    pipeline = ss.get("hcp_pipeline")
    if not ss.get("dataset", "").strip():
        return
    on_dataset_setting()
    if ss.get("dataset_setting_error") or not pipeline:
        return
    if pipeline not in ss.get("hcp_pipeline_choices", []):
        ss["hcp_pipeline"] = None
        ss["dataset_setting_error"] = (f"HCP pipeline '{pipeline}' from the config has no "
                                       "mean metric CSV in this dataset.")
        return
    ss["hcp_pipeline"] = pipeline
    on_hcp_pipeline_setting()


def on_subjects_upload():
    ss = st.session_state
    ss.subjects_error = None
    ss.subjects_df = None
    ss.subjects_file = None
    f = ss.get("subjects_uploader")
    if f is not None:
        try:
            df = read_table(f)
        except Exception as e:
            ss.subjects_error = f"Could not read file: {e}"
        else:
            if "participant_id" not in df.columns:
                ss.subjects_error = "The table must contain a 'participant_id' column."
            else:
                dest = WORK_DIR / f.name
                dest.write_bytes(f.getbuffer())
                ss.subjects_file = str(dest)
                ss.subjects_df = df
    # Rebuild the join and re-apply the pipeline filter on the new subjects table
    refresh_subjects_view()


def on_additional_info_upload():
    ss = st.session_state
    ss.additional_info_error = None
    ss.additional_info_df = None
    ss.additional_info_file = None
    f = ss.get("additional_info_uploader")
    if f is not None:
        try:
            df = read_table(f)
        except Exception as e:
            ss.additional_info_error = f"Could not read file: {e}"
        else:
            dest = WORK_DIR / f.name
            dest.write_bytes(f.getbuffer())
            ss.additional_info_file = str(dest)
            ss.additional_info_df = df
    refresh_subjects_view()


def on_dataset_setting():
    ss = st.session_state
    ss.dataset_setting_error = None
    try:
        ds= Dataset(ss.get("dataset", "").strip())
        ds.build_dataframe()
        mean_csv=ds.get_global(suffix='mean',extension='csv',datatype='metric')
    except Exception as e:
        ss.pop("dataset_obj", None)
        ss["hcp_pipeline_choices"] = []
        ss["dataset_setting_error"] = f"Could not load dataset '{ss.get('dataset', '').strip()}': {e}"
        return
    pipelines_with_mean=set([p.pipeline for p in mean_csv])
    print(pipelines_with_mean)
    # #Set choices of hcp_asso_pipeline based on pipelines with mean
    ss["dataset_obj"] = ds
    ss["mean_csv_files"] = mean_csv
    ss["hcp_pipeline_choices"] = list(pipelines_with_mean)
    # The dataset changed: previously derived filters/metrics are no longer valid
    ss["filtered_subjects_df"] = None
    ss["pipeline_subjects"] = []
    ss["available_metrics"] = []
    ss["available_stat_types"] = []
    st.success(f"Dataset '{ss.get('dataset', '').strip()}' is available.")


def on_hcp_pipeline_setting():
    """
    Retrieve the subjects available for the selected HCP pipeline(s), filter
    the subjects table accordingly, and read the first available metrics CSV
    to infer the usable metric columns.
    """
    ss = st.session_state
    selected_pipelines = ss.get("hcp_pipeline") or []
    print(selected_pipelines)
    pipeline_files = ss["dataset_obj"].get_global(datatype='metric', extension='csv')

    # --- Subjects available in the selected pipeline(s) ---
    pipeline_subjects = sorted({f.subject for f in pipeline_files if f.subject})
    ss["pipeline_subjects"] = pipeline_subjects
    refresh_subjects_view()

    # --- Available metrics: read the first CSV found for the pipeline ---
    ss["available_metrics"] = []
    ss["available_stat_types"] = []
    ss["metrics_read_error"] = None
    if pipeline_files:
        first_csv = sorted(pipeline_files, key=lambda f: f.path)[0]
        try:
            df_metrics = pd.read_csv(first_csv.path, nrows=1)
            metrics=[
                c for c in df_metrics.columns if c not in ("point_id", "centroid_id")
            ]
            ss["available_metrics"] = set([m.split('_')[0] for m in metrics])
            print("STAT TYPES : ",set([m.split('_')[1] for m in metrics]))
            ss["available_stat_types"] = set([m.split('_')[1] for m in metrics])
        except Exception as e:
            ss["metrics_read_error"] = f"Could not read ({first_csv.path}): {e}"

# ---------------------------------------------------------------------------
# Config building / validation
# ---------------------------------------------------------------------------

def build_config() -> dict:
    ss = st.session_state
    cfg = deep_copy(DEFAULT_CONFIG)

    cfg["dataset"] = ss.get("dataset", "").strip()
    cfg["pipeline"] = ss.get("pipeline", date.datetime.now().strftime("%Y%m%d%H%M%S"))
    cfg["hcp_asso_pipeline"] = ss.get("hcp_pipeline", "tractometry")
    cfg["output_dir"] = ss.get("output_dir", "").strip() or None
    # For traceability only: the join is already applied in the table passed via --subjects-table
    cfg["additional_info_path"] = (ss.get("additional_info_file")
                                   if ss.get("merged_subjects_file") else None)

    if ss.get("subjects_df") is not None:
        gvars = list(ss.get("group_vars_sel", []))
        cvars = list(ss.get("corr_vars_sel", []))
    else:
        gvars = [v.strip() for v in ss.get("manual_group_vars", "").split(",") if v.strip()]
        cvars = [v.strip() for v in ss.get("manual_corr_vars", "").split(",") if v.strip()]

    def confounds(var):
        raw = ss.get(f"conf::{var}", [])
        if isinstance(raw, str):
            return [s.strip() for s in raw.split(",") if s.strip()]
        return list(raw or [])

    cfg["group_variables"] = {v: {"confounds": confounds(v)} for v in gvars}
    cfg["corr_variables"] = {v: {"confounds": confounds(v)} for v in cvars}

    tb = ss.get("test_by_group", NO_GROUP)
    cfg["test_by_group"] = None if tb in (None, "", NO_GROUP) else tb

    cfg["FWE_METHOD"] = ss.get("fwe_method", "clusterFWE")
    cfg["AFQ_ALPHA"] = float(ss.get("afq_alpha", 0.05))
    cfg["AFQ_NPERM"] = int(ss.get("afq_nperm", 1000))
    cfg["CORRELATION_TEST"] = ss.get("corr_test", "pearson")
    cfg["CORRECT_MULTI_TRACT"] = bool(ss.get("correct_multi_tract", False))

    cfg["MISSING_SUBJECTS_TOLERANCE"] = int(ss.get("missing_tol", 0))
    cfg["INTERPOLATE_MISSING_POINTS"] = int(ss.get("interp_gap", 2))
    cfg["KEEP_LARGEST_CONTIGUOUS"] = bool(ss.get("keep_contig", True))
    cfg["RESAMPLE_N_POINTS"] = int(ss["resample_n"]) if ss.get("resample_n") else None

    cfg["METRIC_COLUMNS_CANDIDATES"] = list(ss.get("metrics") or ["FA"])
    cfg["STAT_TYPES"] = list(ss.get("stat_types") or ["mean"])
    cfg["CORRECT_METRIC_CONFOND_ONLY"] = bool(ss.get("correct_metric_confond_only", False))

    cfg["MULTIPROCESSING"] = bool(ss.get("multiproc", True))
    cfg["n_jobs"] = int(ss["n_jobs"]) if ss.get("n_jobs") else None

    cfg["EXCLUDE"] = {}
    for rid in ss.get("exclude_ids", []):
        column = str(ss.get(f"excl_col::{rid}") or "").strip()
        values = ss.get(f"excl_vals::{rid}") or []
        if isinstance(values, str):
            values = [v.strip() for v in values.split(",") if v.strip()]
        if column and values:
            excluded = cfg["EXCLUDE"].setdefault(column, [])
            excluded += [v for v in values if v not in excluded]
    return cfg


def validate(cfg, script_path):
    problems, warnings = [], []
    ds = cfg.get("dataset", "")
    if not ds:
        problems.append("BIDS dataset path is required ('dataset').")
    elif not Path(ds).is_dir():
        problems.append(f"Dataset not found: {ds}")
    if not cfg.get("hcp_asso_pipeline"):
        problems.append("'hcp_asso_pipeline' cannot be empty.")
    if not (cfg.get("group_variables") or cfg.get("corr_variables")):
        warnings.append("No group or correlation variable selected.")
    if not Path(script_path).is_file():
        problems.append(f"Script not found: {script_path}")
    return problems, warnings


def start_analysis(cmd, cwd, cfg):
    """
    Write the config and start population_analysis.py in the background.
    The output goes to a log file, so the UI stays usable while it runs.
    """
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path = WORK_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    run = {"proc": None, "log": str(log_path), "error": None, "stopped": False,
           "output_dir": cfg.get("output_dir")}
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            # Own process group, so that Stop also kills the multiprocessing workers
            run["proc"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=cwd, env=env, start_new_session=True)
    except Exception as e:
        run["error"] = f"Could not start the process: {e}"
    st.session_state["run"] = run


def stop_analysis():
    """Terminate the running analysis and its worker processes (SIGTERM, then SIGKILL)."""
    run = st.session_state.get("run") or {}
    proc = run.get("proc")
    if proc is None or proc.poll() is not None:
        return
    run["stopped"] = True
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            continue


def is_running() -> bool:
    run = st.session_state.get("run")
    return bool(run and run["proc"] is not None and run["proc"].poll() is None)


def read_log_tail(path, n_lines=400, max_bytes=256_000) -> str:
    """Last lines of the log, without reading the whole file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-n_lines:])


def render_run_status(polling: bool):
    """Status and log of the last run. Re-run every second (st.fragment) while it runs."""
    run = st.session_state.get("run")
    if not run:
        return
    if run["error"]:
        st.error(run["error"])
        return
    rc = run["proc"].poll()
    if polling and rc is not None:
        st.rerun()  # full rerun: stop polling and refresh the Run/Stop buttons
    if rc is None:
        st.info(f"⏳ Analysis running (PID {run['proc'].pid})…")
    elif run["stopped"]:
        st.warning("⏹️ Analysis stopped.")
    elif rc == 0:
        st.success("✅ Analysis completed without errors.")
    else:
        st.error(f"Analysis finished with exit code {rc}.")
    st.caption(f"Config: `{CONFIG_PATH}` — log: `{run['log']}`")
    st.code(read_log_tail(run["log"]), language="console")
    if rc is not None and run["output_dir"]:
        st.markdown(f"Results expected in: `{run['output_dir']}`")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

init_state()
cfg = build_config()
st.session_state.config = cfg


def render_sidebar():
    with st.sidebar:
        st.title("TractoPL — Tractometry analysis")
        st.caption("Config generator + launcher for `population_analysis.py`")
        st.button("♻️ Reset", on_click=on_reset, use_container_width=True)
        st.file_uploader("Load a config.json", type=["json"],
                         key="cfg_uploader", on_change=on_load_cfg)
        if st.session_state.get("cfg_load_error"):
            st.error(st.session_state["cfg_load_error"])
        st.download_button("⬇️ Download config.json",
                           json.dumps(cfg, indent=2, ensure_ascii=False),
                           file_name="config.json", mime="application/json",
                           use_container_width=True)
        if st.button("💾 Save to workdir", use_container_width=True):
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            st.success(f"Saved: {CONFIG_PATH}")
        with st.expander("Info"):
            st.markdown(f"- Workdir: `{WORK_DIR}`\n- Default script: `{DEFAULT_SCRIPT}`")


render_sidebar()

st.title("TractoPL — Tractometry analysis")
st.caption("Configure the parameters, preview the JSON, run the analysis.")

tab_data, tab_vars, tab_adv, tab_json, tab_run = st.tabs(
    ["📁 Data", "🔬 Variables & tests", "⚙️ Advanced options",
     "📄 config.json", "▶️ Run"])

# --------------------------------- Data ------------------------------------
with tab_data:
    st.markdown("#### Dataset")
    c1, c2 = st.columns(2)
    c1.text_input("BIDS dataset (path)", key="dataset", on_change=on_dataset_setting)
    c2.selectbox("Associated HCP pipeline (e.g. tractometry)", key="hcp_pipeline", options=st.session_state.get("hcp_pipeline_choices", []),on_change=on_hcp_pipeline_setting)
    if st.session_state.get("dataset_setting_error"):
        st.error(st.session_state["dataset_setting_error"])
    st.text_input("ID", key="pipeline",disabled=True)
    st.text_input("Output directory (empty = auto)", key="output_dir")

    if st.session_state.get("hcp_pipeline"):
        n_pipe_subj = len(st.session_state.get("pipeline_subjects", []))
        st.caption(f"Subjects available in the selected pipeline: {n_pipe_subj}")
        if st.session_state.get("metrics_read_error"):
            st.warning(st.session_state["metrics_read_error"])
        elif st.session_state.get("available_metrics"):
            st.caption("Metrics detected in the pipeline's first CSV: "
                       + ", ".join(st.session_state["available_metrics"]))

    st.divider()
    st.markdown("#### Subjects table (optional)")
    st.file_uploader("CSV / XLSX — 'participant_id' column required",
                     type=["csv", "xlsx", "xls"],
                     key="subjects_uploader", on_change=on_subjects_upload)
    if st.session_state.get("subjects_error"):
        st.error(st.session_state["subjects_error"])
    df_sub_raw = st.session_state.get("subjects_df")

    st.markdown("#### Additional info table (optional)")
    st.file_uploader("CSV / XLSX — left-joined onto the subjects table",
                     type=["csv", "xlsx", "xls"],
                     key="additional_info_uploader", on_change=on_additional_info_upload)
    if st.session_state.get("additional_info_error"):
        st.error(st.session_state["additional_info_error"])
    df_add = st.session_state.get("additional_info_df")
    if df_add is not None:
        if df_sub_raw is None:
            st.info("Load a subjects table first: the additional info table is left-joined onto it.")
        else:
            c1, c2 = st.columns(2)
            c1.selectbox("Join column — subjects table", list(df_sub_raw.columns),
                         key="join_key_subjects", on_change=refresh_subjects_view)
            c2.selectbox("Join column — additional info table", list(df_add.columns),
                         key="join_key_additional", on_change=refresh_subjects_view)
            st.caption("Left join: every subject of the subjects table is kept. Keys are "
                       "compared as text, ignoring surrounding spaces and a leading 'sub-'.")
            report = st.session_state.get("join_report")
            if report:
                n_missing = report["n_subjects"] - report["n_matched"]
                if report["n_matched"] == 0:
                    st.error("No subject matched the additional info table — check the join columns.")
                elif n_missing:
                    st.warning(f"{report['n_matched']}/{report['n_subjects']} subject(s) matched — "
                               f"{n_missing} without additional info (empty values).")
                else:
                    st.success(f"All {report['n_subjects']} subject(s) matched.")
                if report["n_duplicates"]:
                    st.warning(f"{report['n_duplicates']} duplicate key(s) in the additional info "
                               "table: only the first row of each was kept.")
                st.caption(f"Columns added: {', '.join(report['added']) or '(none)'}")
                if report["skipped"]:
                    st.caption("Columns already in the subjects table (kept from the subjects "
                               f"table): {', '.join(report['skipped'])}")

    df_sub = get_active_subjects_df()
    if df_sub_raw is not None:
        table_file = st.session_state.get("merged_subjects_file") or st.session_state.get("subjects_file")
        if st.session_state.get("filtered_subjects_df") is not None:
            st.success(f"Table loaded: {len(df_sub_raw)} rows — "
                      f"{len(df_sub)} subject(s) available in the selected pipeline.")
        else:
            st.success(f"Table loaded: {len(df_sub_raw)} rows — will be passed via `--subjects-table`.")
        st.caption(f"Table passed to the analysis: `{table_file}`")
        st.dataframe(df_sub, use_container_width=True)


# --------------------------- Variables & tests -----------------------------
with tab_vars:
    df_sub = get_active_subjects_df()
    st.markdown("#### Variables to analyse")
    if df_sub is not None:
        cols = list(df_sub.columns)
        st.caption("Selected from the subjects table columns.")
        c1, c2 = st.columns(2)
        gvars = c1.multiselect("Group variables (categorical → t-test/ANOVA)",
                               cols, key="group_vars_sel")
        cvars = c2.multiselect("Correlation variables (continuous)",
                               cols, key="corr_vars_sel")
    else:
        st.caption("No table loaded: enter the variables manually (comma-separated).")
        st.text_input("Group variables (categorical)", key="manual_group_vars")
        st.text_input("Correlation variables (continuous)", key="manual_corr_vars")
        gvars = [v.strip() for v in st.session_state.get("manual_group_vars", "").split(",") if v.strip()]
        cvars = [v.strip() for v in st.session_state.get("manual_corr_vars", "").split(",") if v.strip()]

    all_vars = gvars + cvars
    if all_vars:
        st.markdown("#### Confounds (variables regressed out by OLS before testing)")
        conf_opts = list(df_sub.columns) if df_sub is not None else None
        for v in all_vars:
            with st.expander(f"Confounds — {v}"):
                if conf_opts:
                    # Multiselect mode: coerce to a list and keep only valid columns
                    opts = [c for c in conf_opts if c != v]
                    current = conf_as_list(v)
                    valid = [c for c in current if c in opts]
                    if len(valid) != len(current):
                        st.session_state[f"conf::{v}"] = valid
                        dropped = [c for c in current if c not in opts]
                        st.caption(f"⚠️ Confounds ignored (not in the table): {dropped}")
                    st.multiselect("Columns to control for", opts, key=f"conf::{v}")
                else:
                    # Text mode: coerce to a string BEFORE instantiating the widget
                    conf_as_str(v)
                    st.text_input("Confounds (comma-separated)", key=f"conf::{v}")

    st.divider()
    st.markdown("#### Statistical parameters")
    tb_opts = [NO_GROUP] + list(gvars)
    if st.session_state.get("test_by_group") not in tb_opts:
        st.session_state["test_by_group"] = NO_GROUP
    st.selectbox("test_by_group — stratify correlations by group level",
                 tb_opts, key="test_by_group")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("FWE correction", FWE_METHODS, key="fwe_method")
    c2.number_input("Alpha", 0.0, 1.0, step=0.01, key="afq_alpha")
    c3.number_input("Permutations (AFQ_NPERM)", 0, 10000, step=100, key="afq_nperm")
    c1, c2 = st.columns(2)
    c1.selectbox("Correlation test", CORRELATION_TESTS, key="corr_test")
    c2.checkbox("Multi-tract correction (slower)", key="correct_multi_tract")

# ---------------------------- Advanced options -----------------------------
with tab_adv:
    st.markdown("#### Data quality")
    c1, c2 = st.columns(2)
    c1.number_input("Missing subjects tolerated per point", 0, 100, key="missing_tol")
    c2.number_input("Interpolation: max gap (points, 0 = off)", 0, 50, key="interp_gap")
    st.checkbox("Keep the longest contiguous segment (KEEP_LARGEST_CONTIGUOUS)", key="keep_contig")
    st.number_input("Resample to N points (0 = none)", 0, 1000, key="resample_n")

    st.markdown("#### Metrics")
    if st.session_state.get("available_metrics"):
        st.caption("Columns detected in the pipeline CSV (excluding point_id/centroid_id): "
                   + ", ".join(st.session_state["available_metrics"]))
    metric_options = sorted(
        set(METRIC_CHOICES)
        | set(st.session_state.get("metrics", []))
        | set(st.session_state.get("available_metrics", []))
    )
    st.multiselect("Metrics (METRIC_COLUMNS_CANDIDATES)", metric_options, key="metrics")
    stat_options = sorted(
        set(STAT_CHOICES)
        | set(st.session_state.get("stat_types", []))
        | set(st.session_state.get("available_stat_types", []))
    )
    st.multiselect("Statistics (columns looked up: <METRIC>_<STAT> or <METRIC>)",
                   stat_options, key="stat_types")
    st.checkbox("Only correct the metric for confounds", key="correct_metric_confond_only")

    st.markdown("#### Execution")
    st.checkbox("Multiprocessing", key="multiproc")
    st.number_input("n_jobs (0 = auto → cpu_count//3)", 0, 256, key="n_jobs")

    st.markdown("#### Exclusions (EXCLUDE)")
    df_excl = get_active_subjects_df()
    if df_excl is not None:
        st.caption("Subjects whose value in the chosen column is one of the selected values "
                   "are excluded from the analysis.")
    else:
        st.caption("No subjects table loaded: enter the column and the values to exclude "
                   "(comma-separated).")
    for rid in list(st.session_state.get("exclude_ids", [])):
        render_exclude_row(rid, df_excl)
    st.button("➕ Add exclusion", on_click=add_exclude_row)

# ------------------------------- config.json -------------------------------
with tab_json:
    st.markdown("#### Preview (live update)")
    st.json(cfg, expanded=False)
    st.markdown("#### Manual editing")
    txt = st.text_area("JSON", json.dumps(cfg, indent=2, ensure_ascii=False),height=420, key="json_editor")
    st.button("✅ Apply this JSON", on_click=on_apply_json)
    if st.session_state.get("json_apply_error"):
        st.error(st.session_state["json_apply_error"])

# ---------------------------------- Run ------------------------------------
with tab_run:
    st.markdown("#### Running population_analysis.py")
    c1, c2 = st.columns(2)
    c1.text_input("Path to population_analysis.py", key="script_path")
    c2.text_input("Python interpreter (TractoPL environment)", key="python_exe")

    script_path = st.session_state.get("script_path", "")
    cmd = [st.session_state.get("python_exe", sys.executable), script_path, "-c", str(CONFIG_PATH)]
    subjects_table = (st.session_state.get("merged_subjects_file")
                      or st.session_state.get("subjects_file"))
    if subjects_table:
        cmd += ["--subjects-table", subjects_table]
    if cfg.get("output_dir"):
        cmd += ["--output-dir", cfg["output_dir"]]

    st.markdown("**Command:**")
    st.code(shlex.join(cmd), language="bash")

    problems, warnings = validate(cfg, script_path)
    for p in problems:
        st.error("⛔ " + p)
    for w in warnings:
        st.warning("⚠️ " + w)

    running = is_running()
    c1, c2 = st.columns(2)
    if c1.button("▶️ Run analysis", type="primary", disabled=bool(problems) or running,
                 use_container_width=True):
        cwd = str(Path(script_path).resolve().parent) if Path(script_path).is_file() else str(APP_DIR)
        start_analysis(cmd, cwd, cfg)
        st.rerun()
    c2.button("⏹️ Stop", disabled=not running, on_click=stop_analysis,
              use_container_width=True)
    st.fragment(render_run_status, run_every=1 if running else None)(running)
