import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from TractoPL.data.loader import Dataset, Subject
import pandas as pd
import streamlit as st
import datetime as date

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "streamlit_workdir"
WORK_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = APP_DIR / "streamlit_output"
OUTPUT_DIR.mkdir(exist_ok=True)
def deep_copy(d):
    return json.loads(json.dumps(d))
def init_state():
    ss = st.session_state
    if ss.get("app_initialized"):
        return
    ss.app_initialized = True
    ss.subjects_df = None
    ss.subjects_file = None
    ss.subjects_error = None
    ss.exclude_rows = []
    ss.python_exe = sys.executable
    ss.mean_csv_files = []
    ss.filtered_subjects_df = None
    ss.pipeline_subjects = []
    ss.available_metrics = []
    ss.stat_types = []
    ss.metrics_read_error = None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def on_load_csv():
    ss = st.session_state
    ss.subjects_file = ss.get("summary_results")
    # Implement the logic to load and process the CSV file here
    ss.subjects_df = pd.read_csv(ss.subjects_file)
    #Get path of the uploaded CSV file
    ss.report_dir= ss.subjects_file

    print(ss.subjects_df)
init_state()


def render_sidebar():
    with st.sidebar:
        st.title("TractoPL — Tractometry report")
        st.button("♻️ Reset", use_container_width=True)
        st.file_uploader("Upload a summary_results.csv", type=["csv"],on_change=on_load_csv,key="summary_results")
        st.markdown("---")

st.title("Tractometry report")
st.session_state['report_dir'] = None
if st.session_state.get("subjects_df") is not None:
    st.caption("Preview of the uploaded summary results.")
    st.dataframe(st.session_state.subjects_df)
    st.caption(f"Report directory: {st.session_state.report_dir }")

render_sidebar()
