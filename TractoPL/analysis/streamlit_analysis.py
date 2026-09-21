#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interface Streamlit — TractoPL / population_analysis.py
================================================
Génère le config.json attendu par population_analysis.py, permet de le
charger/sauvegarder/télécharger, et lance l'analyse avec logs en direct.

Lancement :
    streamlit run streamlit_app.py
"""

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

# ---------------------------------------------------------------------------
# Config par défaut (miroir de DEFAULT_CONFIG dans population_analysis.py)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "pipeline": date.datetime.now().strftime("%Y%m%d%H%M%S"),
    "dataset": "",
    "hcp_asso_pipeline": "tractometry",
    "group_variables": {},           # ex: {"groupe": {"confounds": ["age"]}}
    "corr_variables": {},            # ex: {"score": {"confounds": ["age"]}}
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
    "EXCLUDE": {},                   # ex: {"groupe": ["exclu"]}
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
DEFAULT_SCRIPT = APP_DIR / "population_analysis.py"
NO_GROUP = "(aucun)"

st.set_page_config(page_title="TractoPL — AFQ Analysis", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Helpers session
# ---------------------------------------------------------------------------

def deep_copy(d):
    return json.loads(json.dumps(d))

def conf_as_list(var: str) -> list:
    """Garantit que session_state['conf::<var>'] est une liste (pour multiselect)."""
    key = f"conf::{var}"
    raw = st.session_state.get(key, [])
    if isinstance(raw, str):
        st.session_state[key] = [s.strip() for s in raw.split(",") if s.strip()]
    return st.session_state.get(key, [])


def get_active_subjects_df():
    """Table de sujets à utiliser dans l'UI : filtrée sur la pipeline HCP si sélectionnée."""
    ss = st.session_state
    filtered = ss.get("filtered_subjects_df")
    if filtered is not None:
        return filtered
    return ss.get("subjects_df")


def conf_as_str(var: str) -> str:
    """Garantit que session_state['conf::<var>'] est une string (pour text_input)."""
    key = f"conf::{var}"
    raw = st.session_state.get(key, [])
    if isinstance(raw, (list, tuple)):
        st.session_state[key] = ", ".join(map(str, raw))
    return st.session_state.get(key, "")

def push_config_to_widgets(cfg):
    """Injecte une config dans les clés de session des widgets."""
    ss = st.session_state
    ss["dataset"] = cfg.get("dataset") or ""
    ss["pipeline"] = cfg.get("pipeline", "default")
    ss["hcp_pipeline"] = cfg.get("hcp_asso_pipeline", "tractometry")
    ss["output_dir"] = cfg.get("output_dir") or ""
    ss["additional_info_path"] = cfg.get("additional_info_path") or ""

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

    ss["exclude_rows"] = [
        {"colonne": k, "valeurs": ", ".join(map(str, v))}
        for k, v in cfg.get("EXCLUDE", {}).items()
    ]
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
    ss.exclude_rows = []
    ss.script_path = str(DEFAULT_SCRIPT)
    ss.python_exe = sys.executable
    ss.mean_csv_files = []
    ss.filtered_subjects_df = None
    ss.pipeline_subjects = []
    ss.available_metrics = []
    ss.stat_types = []
    ss.metrics_read_error = None
    push_config_to_widgets(DEFAULT_CONFIG)


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
            raise ValueError("Le JSON racine doit être un objet.")
        ss.config = cfg
        push_config_to_widgets(cfg)
    except Exception as e:
        ss.cfg_load_error = str(e)


def on_subjects_upload():
    ss = st.session_state
    ss.subjects_error = None
    f = ss.get("subjects_uploader")
    if f is None:
        ss.subjects_df = None
        ss.subjects_file = None
        return
    try:
        if f.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f)
    except Exception as e:
        ss.subjects_df = None
        ss.subjects_error = f"Lecture impossible : {e}"
        return
    if "participant_id" not in df.columns:
        ss.subjects_df = None
        ss.subjects_error = "La table doit contenir une colonne 'participant_id'."
        return
    dest = WORK_DIR / f.name
    dest.write_bytes(f.getbuffer())
    ss.subjects_file = str(dest)
    ss.subjects_df = df
    # Nettoyage des sélections devenues invalides
    cols = list(df.columns)
    ss["group_vars_sel"] = [v for v in ss.get("group_vars_sel", []) if v in cols]
    ss["corr_vars_sel"] = [v for v in ss.get("corr_vars_sel", []) if v in cols]
    # Réappliquer le filtre pipeline sur la nouvelle table de sujets
    on_hcp_pipeline_setting()


def on_dataset_setting():
    ss = st.session_state
    ss.dataset_setting_error = None
    ds= Dataset(ss.get("dataset", "").strip())
    ds.build_dataframe()
    mean_csv=ds.get_global(suffix='mean',extension='csv',datatype='metric')
    pipelines_with_mean=set([p.pipeline for p in mean_csv])
    print(pipelines_with_mean)
    # #Set choices of hcp_asso_pipeline based on pipelines with mean
    ss["dataset_obj"] = ds
    ss["mean_csv_files"] = mean_csv
    ss["hcp_pipeline_choices"] = list(pipelines_with_mean)
    # Le dataset a changé : les filtres/métriques précédemment déduits ne sont plus valides
    ss["filtered_subjects_df"] = None
    ss["pipeline_subjects"] = []
    ss["available_metrics"] = []
    ss["stat_types"] = []
    st.success(f"Dataset '{ss.get('dataset', '').strip()}' is available.")


def on_hcp_pipeline_setting():
    """
    Récupère les sujets disponibles pour la/les pipeline(s) HCP sélectionnée(s),
    filtre le tableau de sujets en conséquence, et lit le premier CSV de métriques
    disponible pour en déduire les colonnes de métriques exploitables.
    """
    ss = st.session_state
    selected_pipelines = ss.get("hcp_pipeline") or []
    mean_csv = ss.get("mean_csv_files") or []
    pipeline_files = [f for f in mean_csv if getattr(f, "pipeline", None) in selected_pipelines]

    # --- Sujets disponibles dans la/les pipeline(s) sélectionnée(s) ---
    pipeline_subjects = sorted({f.subject for f in pipeline_files if f.subject})
    ss["pipeline_subjects"] = pipeline_subjects

    subjects_df = ss.get("subjects_df")
    if subjects_df is not None and pipeline_subjects:
        pid_norm = subjects_df["participant_id"].astype(str).str.replace(r"^sub-", "", regex=True)
        ss["filtered_subjects_df"] = subjects_df[pid_norm.isin(pipeline_subjects)].reset_index(drop=True)
    else:
        ss["filtered_subjects_df"] = None

    # --- Métriques disponibles : lecture du premier CSV trouvé pour la pipeline ---
    ss["available_metrics"] = []
    ss["stat_types"] = []
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
            ss["stat_types"] = set([m.split('_')[1] for m in metrics])
        except Exception as e:
            ss["metrics_read_error"] = f"Lecture impossible ({first_csv.path}) : {e}"

# ---------------------------------------------------------------------------
# Construction / validation de la config
# ---------------------------------------------------------------------------

def build_config() -> dict:
    ss = st.session_state
    cfg = deep_copy(DEFAULT_CONFIG)

    cfg["dataset"] = ss.get("dataset", "").strip()
    cfg["pipeline"] = ss.get("pipeline", date.datetime.now().strftime("%Y%m%d%H%M%S"))
    cfg["hcp_asso_pipeline"] = ss.get("hcp_pipeline", "tractometry")
    cfg["output_dir"] = ss.get("output_dir", "").strip() or None
    cfg["additional_info_path"] = ss.get("additional_info_path", "").strip() or None

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
    for row in ss.get("exclude_rows", []):
        col = str(row.get("colonne") or "").strip()
        vals = [v.strip() for v in str(row.get("valeurs") or "").split(",") if v.strip()]
        if col and vals:
            cfg["EXCLUDE"][col] = vals
    return cfg


def validate(cfg, script_path):
    problems, warnings = [], []
    ds = cfg.get("dataset", "")
    if not ds:
        problems.append("Chemin du dataset BIDS requis ('dataset').")
    elif not Path(ds).is_dir():
        problems.append(f"Dataset introuvable : {ds}")
    if not cfg.get("hcp_asso_pipeline"):
        problems.append("'hcp_asso_pipeline' ne peut pas être vide.")
    if not (cfg.get("group_variables") or cfg.get("corr_variables")):
        warnings.append("Aucune variable de groupe ni de corrélation sélectionnée.")
    if not Path(script_path).is_file():
        problems.append(f"Script introuvable : {script_path}")
    return problems, warnings


def run_analysis(cmd, cwd) -> int:
    box = st.empty()
    lines = []
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=cwd, env=env,
        )
    except Exception as e:
        st.error(f"Impossible de lancer le processus : {e}")
        return -1
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-400:]), language="console")  # garde les 400 dernières lignes
    proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

init_state()
cfg = build_config()
st.session_state.config = cfg


def render_sidebar():
    with st.sidebar:
        st.title("TractoPL — Tractometry analysis")
        st.caption("Générateur de config + lanceur pour `population_analysis.py`")
        st.button("♻️ Réinitialiser", on_click=on_reset, use_container_width=True)
        st.file_uploader("Charger un config.json", type=["json"],
                         key="cfg_uploader", on_change=on_load_cfg)
        if st.session_state.get("cfg_load_error"):
            st.error(st.session_state["cfg_load_error"])
        st.download_button("⬇️ Télécharger config.json",
                           json.dumps(cfg, indent=2, ensure_ascii=False),
                           file_name="config.json", mime="application/json",
                           use_container_width=True)
        if st.button("💾 Enregistrer dans le workdir", use_container_width=True):
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            st.success(f"Enregistré : {CONFIG_PATH}")
        with st.expander("Infos"):
            st.markdown(f"- Workdir : `{WORK_DIR}`\n- Script par défaut : `{DEFAULT_SCRIPT}`")


render_sidebar()

st.title("TractoPL — Tractometry analysis")
st.caption("Configurez les paramètres, prévisualisez le JSON, lancez l'analyse.")

tab_data, tab_vars, tab_adv, tab_json, tab_run = st.tabs(
    ["📁 Données", "🔬 Variables & tests", "⚙️ Options avancées",
     "📄 config.json", "▶️ Exécution"])

# ------------------------------- Données -----------------------------------
with tab_data:
    st.markdown("#### Dataset")
    c1, c2 = st.columns(2)
    c1.text_input("Dataset BIDS (chemin)", key="dataset", on_change=on_dataset_setting)
    c2.selectbox("Pipeline associé HCP (ex. tractometry)", key="hcp_pipeline", options=st.session_state.get("hcp_pipeline_choices", []),on_change=on_hcp_pipeline_setting)
    st.text_input("ID", key="pipeline",disabled=True)
    st.text_input("Dossier de sortie (vide = auto)", key="output_dir")
    st.text_input("Table d'infos additionnelles (xlsx/csv, optionnel)",
                  key="additional_info_path")

    if st.session_state.get("hcp_pipeline"):
        n_pipe_subj = len(st.session_state.get("pipeline_subjects", []))
        st.caption(f"Sujets disponibles dans la pipeline sélectionnée : {n_pipe_subj}")
        if st.session_state.get("metrics_read_error"):
            st.warning(st.session_state["metrics_read_error"])
        elif st.session_state.get("available_metrics"):
            st.caption("Métriques détectées dans le premier CSV de la pipeline : "
                       + ", ".join(st.session_state["available_metrics"]))

    st.divider()
    st.markdown("#### Table sujets (optionnelle)")
    st.file_uploader("CSV / XLSX — colonne 'participant_id' requise",
                     type=["csv", "xlsx", "xls"],
                     key="subjects_uploader", on_change=on_subjects_upload)
    if st.session_state.get("subjects_error"):
        st.error(st.session_state["subjects_error"])
    df_sub_raw = st.session_state.get("subjects_df")
    df_sub = get_active_subjects_df()
    if df_sub_raw is not None:
        if st.session_state.get("filtered_subjects_df") is not None:
            st.success(f"Table chargée : {len(df_sub_raw)} lignes — "
                      f"{len(df_sub)} sujet(s) disponible(s) dans la pipeline sélectionnée.")
        else:
            st.success(f"Table chargée : {len(df_sub_raw)} lignes — sera passée via `--subjects-table`.")
        st.dataframe(df_sub, use_container_width=True)


# --------------------------- Variables & tests -----------------------------
with tab_vars:
    df_sub = get_active_subjects_df()
    st.markdown("#### Variables à analyser")
    if df_sub is not None:
        cols = list(df_sub.columns)
        st.caption("Sélection depuis les colonnes de la table sujets.")
        c1, c2 = st.columns(2)
        gvars = c1.multiselect("Variables de groupe (catégorielles → t-test/ANOVA)",
                               cols, key="group_vars_sel")
        cvars = c2.multiselect("Variables de corrélation (continues)",
                               cols, key="corr_vars_sel")
    else:
        st.caption("Aucune table chargée : saisissez les variables manuellement (séparées par des virgules).")
        st.text_input("Variables de groupe (catégorielles)", key="manual_group_vars")
        st.text_input("Variables de corrélation (continues)", key="manual_corr_vars")
        gvars = [v.strip() for v in st.session_state.get("manual_group_vars", "").split(",") if v.strip()]
        cvars = [v.strip() for v in st.session_state.get("manual_corr_vars", "").split(",") if v.strip()]

    all_vars = gvars + cvars
    if all_vars:
        st.markdown("#### Confounds (variables régressées par OLS avant test)")
        conf_opts = list(df_sub.columns) if df_sub is not None else None
        for v in all_vars:
            with st.expander(f"Confounds — {v}"):
                if conf_opts:
                    # Mode multiselect : coercer en liste et filtrer sur les colonnes valides
                    opts = [c for c in conf_opts if c != v]
                    current = conf_as_list(v)
                    valid = [c for c in current if c in opts]
                    if len(valid) != len(current):
                        st.session_state[f"conf::{v}"] = valid
                        dropped = [c for c in current if c not in opts]
                        st.caption(f"⚠️ Confounds ignorés (absents de la table) : {dropped}")
                    st.multiselect("Colonnes à contrôler", opts, key=f"conf::{v}")
                else:
                    # Mode texte : coercer en string AVANT d'instancier le widget
                    conf_as_str(v)
                    st.text_input("Confounds (séparés par ,)", key=f"conf::{v}")

    st.divider()
    st.markdown("#### Paramètres statistiques")
    tb_opts = [NO_GROUP] + list(gvars)
    if st.session_state.get("test_by_group") not in tb_opts:
        st.session_state["test_by_group"] = NO_GROUP
    st.selectbox("test_by_group — stratifier les corrélations par niveau de groupe",
                 tb_opts, key="test_by_group")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Correction FWE", FWE_METHODS, key="fwe_method")
    c2.number_input("Alpha", 0.0, 1.0, step=0.01, key="afq_alpha")
    c3.number_input("Permutations (AFQ_NPERM)", 0, 10000, step=100, key="afq_nperm")
    c1, c2 = st.columns(2)
    c1.selectbox("Test de corrélation", CORRELATION_TESTS, key="corr_test")
    c2.checkbox("Correction multi-faisceaux (plus lent)", key="correct_multi_tract")

# ---------------------------- Options avancées -----------------------------
with tab_adv:
    st.markdown("#### Qualité des données")
    c1, c2 = st.columns(2)
    c1.number_input("Sujets manquants tolérés par point", 0, 100, key="missing_tol")
    c2.number_input("Interpolation : écart max (points, 0 = off)", 0, 50, key="interp_gap")
    st.checkbox("Garder le plus long segment continu (KEEP_LARGEST_CONTIGUOUS)", key="keep_contig")
    st.number_input("Rééchantillonner à N points (0 = aucun)", 0, 1000, key="resample_n")

    st.markdown("#### Métriques")
    if st.session_state.get("available_metrics"):
        st.caption("Colonnes détectées dans le CSV de la pipeline (hors point_id/centroid_id) : "
                   + ", ".join(st.session_state["available_metrics"]))
    metric_options = sorted(
        set(METRIC_CHOICES)
        | set(st.session_state.get("metrics", []))
        | set(st.session_state.get("available_metrics", []))
    )
    st.multiselect("Métriques (METRIC_COLUMNS_CANDIDATES)", metric_options, key="metrics")
    stat_options = sorted(set(STAT_CHOICES) | set(st.session_state.get("stat_types", [])))
    st.multiselect("Statistiques (colonnes cherchées : <METRIC>_<STAT> ou <METRIC>)",
                   stat_options, key="stats")
    st.checkbox("Ne corriger que la métrique pour les confounds", key="correct_metric_confond_only")

    st.markdown("#### Exécution")
    st.checkbox("Multiprocessing", key="multiproc")
    st.number_input("n_jobs (0 = auto → cpu_count//3)", 0, 256, key="n_jobs")

    st.markdown("#### Exclusions (EXCLUDE)")
    st.caption("Une ligne par variable : valeurs à exclure, séparées par des virgules.")
    excl_df = pd.DataFrame(st.session_state.get("exclude_rows", []),
                           columns=["colonne", "valeurs"])
    edited = st.data_editor(excl_df, num_rows="dynamic", key="exclude_editor",
                            use_container_width=True)
    st.session_state["exclude_rows"] = edited.to_dict("records")

# ------------------------------- config.json -------------------------------
with tab_json:
    st.markdown("#### Prévisualisation (mise à jour en direct)")
    st.json(cfg, expanded=False)
    st.markdown("#### Édition manuelle")
    txt = st.text_area("JSON", json.dumps(cfg, indent=2, ensure_ascii=False),height=420, key="json_editor")
    if st.button("✅ Appliquer ce JSON"):
        try:
            new_cfg = json.loads(txt)
            st.session_state.config = new_cfg
            push_config_to_widgets(new_cfg)
            st.rerun()
        except Exception as e:
            st.error(f"JSON invalide : {e}")

# -------------------------------- Exécution --------------------------------
with tab_run:
    st.markdown("#### Exécution de population_analysis.py")
    c1, c2 = st.columns(2)
    c1.text_input("Chemin de population_analysis.py", key="script_path")
    c2.text_input("Interpréteur Python (environnement TractoPL)", key="python_exe")

    script_path = st.session_state.get("script_path", "")
    cmd = [st.session_state.get("python_exe", sys.executable), script_path, "-c", str(CONFIG_PATH)]
    if st.session_state.get("subjects_file"):
        cmd += ["--subjects-table", st.session_state["subjects_file"]]
    if cfg.get("output_dir"):
        cmd += ["--output-dir", cfg["output_dir"]]

    st.markdown("**Commande exécutée :**")
    st.code(shlex.join(cmd), language="bash")

    problems, warnings = validate(cfg, script_path)
    for p in problems:
        st.error("⛔ " + p)
    for w in warnings:
        st.warning("⚠️ " + w)

    if st.button("▶️ Lancer l'analyse", type="primary", disabled=bool(problems)):
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        st.info(f"Config écrite : `{CONFIG_PATH}`")
        cwd = str(Path(script_path).resolve().parent) if Path(script_path).is_file() else str(APP_DIR)
        rc = run_analysis(cmd, cwd)
        if rc == 0:
            st.success("✅ Analyse terminée sans erreur.")
        else:
            st.error(f"Analyse terminée avec le code de sortie {rc}.")
        if cfg.get("output_dir"):
            st.markdown(f"Résultats attendus dans : `{cfg['output_dir']}`")