#!/usr/bin/env python3
"""
generate_report_v3.py
=====================
Version non-linéaire du rapport tractométrique.

Au lieu de résidualiser puis corréler (v2), on ajuste à chaque point du faisceau
un modèle OLS polynomial :

    metric_at_point = β0 + β1·var + β2·var² + β3·age [+ autres confonds] + ε

La significativité est évaluée par un test F partiel
(modèle complet vs modèle réduit = confonds seuls).
La correction FWE (alphaFWE / clusterFWE / mixed) est obtenue par permutations
de type Freedman-Lane.

Les fonctions de statistiques, modèles et visualisation sont dans report_stats_v3.py.
Ce fichier ne conserve que la configuration, le chargement des données,
les workers d'orchestration, et la génération du rapport.
"""

import os
import json
from os.path import join as opj
import gc
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm

from TractoPL.set_config import get_HCP_bundle_names
from TractoPL.data.loader import Dataset
import argparse
# --- Fonctions stats / modèles / visualisation depuis report_stats_v3 ---
from TractoPL.analysis.generate_report_v2 import resample_bundle_data
from TractoPL.analysis.report_stats_v3 import (
    # Constantes & configuration
    STAT_TYPES, AFQ_ALPHA, AFQ_NPERM, FWE_METHOD, POLY_DEGREE,
    CORRECT_MULTI_TRACT, configure_hyperparams,
    # Chargement & nettoyage des données
    compute_12h_averages, apply_exclusion_filter, ensure_dir,
    detect_point_column, detect_metric_columns, prepare_long, clean_missing_data,
    # Identifiants participants
    _normalize_participant_id, _build_subject_participant_map, _subjects_to_participant_ids,
    # Tests de groupe & modèles OLS
    residualize_on_confond, group_test_afq, compute_multi_tract_thresholds_group,
    pointwise_ols_model, pointwise_ols_interaction_model, select_best_model_pointwise,
    compute_multi_tract_thresholds_ols, _harmonize_model_points,
    compute_per_model_cluster_summary, _count_clusters,
    # Visualisation
    plot_group_diff, plot_ols_results,
)

# =====================================================================
#                        CONFIGURATION
# =====================================================================

parser = argparse.ArgumentParser(description="Génération de rapport d'analyse AFQ")
parser.add_argument('-c', type=str, default=None, help="Chemin vers le fichier de configuration JSON")
parser.add_argument('--subjects-table', type=str, default=None,
                    help="Chemin vers un CSV/XLSX avec une colonne 'participant_id' listant les sujets à traiter. "
                         "Les colonnes 'confond_variables', 'corr_variables' et 'classif_variables' (séparées par ';') "
                         "servent à vérifier que ces variables existent dans le tableau.")
args = parser.parse_args()
config_path = args.c

DEFAULT_CONFIG = {
    "pipeline": "INTERACTION_50PTS",
    "dataset": "actidep",
    "hcp_asso_pipeline": "hcp_association_new_50pts_mcm_tensors_staniz_longcentral",
    "confond_variables_with_control": ["age"],
    "confond_variables_without_control": ["age"],
    "confond_variables_for_actimetry": ["age"],
    "corr_variables": ["aes", "ami","activity_rate_3d"],
    "test_by_group": "apathy",
    "FWE_METHOD": "clusterFWE",
    "AFQ_ALPHA": 0.05,
    "AFQ_NPERM": 1000,
    "CORRECT_MULTI_TRACT": False,
    "POLY_DEGREE": 2,
    "MULTIPROCESSING": True,
    "classif_variables": {
        "apathy": "no_controls",
    },
    "METRIC_COLUMNS_CANDIDATES": ["FA", "IFW"],
    "STAT_TYPES": ["mean"],
    "EXCLUDE": {"group": ["hc"]},
    "MISSING_SUBJECTS_TOLERANCE": 0,
    "RESAMPLE_N_POINTS": None,
}


def load_config(config_path=None):
    """
    Charge la configuration depuis un fichier JSON, ou retourne la config par défaut.
    Si config_path est fourni et existe, le fichier est chargé et fusionné avec DEFAULT_CONFIG.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        config = {**DEFAULT_CONFIG, **user_config}
        print(f"Configuration chargée depuis: {config_path}")
    else:
        if config_path:
            print(f"Config introuvable ({config_path}), config par défaut utilisée.")
        config = DEFAULT_CONFIG.copy()
    return config


def save_config(config, output_path):
    """
    Sauvegarde la configuration dans un fichier JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration sauvegardée dans: {output_path}")


config = load_config(config_path)

dataset = config["dataset"]
db_root = f'/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids'
ds = Dataset(db_root)

hcp_asso_pipeline = config["hcp_asso_pipeline"]
csv_files = ds.get_global(pipeline=hcp_asso_pipeline, extension='csv', datatype='metric', suffix='mean')
bundle_names = list(get_HCP_bundle_names().keys())

test_by_group = config.get("test_by_group", DEFAULT_CONFIG["test_by_group"])
corr_variables = list(config.get("corr_variables", DEFAULT_CONFIG["corr_variables"]))
classif_variables = dict(config.get("classif_variables", DEFAULT_CONFIG["classif_variables"]))

confond_variables_with_control = config.get("confond_variables_with_control",
                                            DEFAULT_CONFIG["confond_variables_with_control"])
confond_variables_without_control = config.get("confond_variables_without_control",
                                               DEFAULT_CONFIG["confond_variables_without_control"])
confond_variables_for_actimetry = config.get("confond_variables_for_actimetry",
                                             DEFAULT_CONFIG["confond_variables_for_actimetry"])

# Hyperparamètres algorithmiques (surchargent les constantes de report_stats_v3)
AFQ_ALPHA = config.get("AFQ_ALPHA", DEFAULT_CONFIG["AFQ_ALPHA"])
AFQ_NPERM = config.get("AFQ_NPERM", DEFAULT_CONFIG["AFQ_NPERM"])
FWE_METHOD = config.get("FWE_METHOD", DEFAULT_CONFIG["FWE_METHOD"])
CORRECT_MULTI_TRACT = config.get("CORRECT_MULTI_TRACT", DEFAULT_CONFIG["CORRECT_MULTI_TRACT"])
POLY_DEGREE = config.get("POLY_DEGREE", DEFAULT_CONFIG["POLY_DEGREE"])
MULTIPROCESSING = config.get("MULTIPROCESSING", DEFAULT_CONFIG["MULTIPROCESSING"])
MISSING_SUBJECTS_TOLERANCE = config.get("MISSING_SUBJECTS_TOLERANCE", DEFAULT_CONFIG["MISSING_SUBJECTS_TOLERANCE"])
RESAMPLE_N_POINTS = config.get("RESAMPLE_N_POINTS", DEFAULT_CONFIG["RESAMPLE_N_POINTS"])

# Propager les hyperparamètres vers les globals de report_stats_v3
configure_hyperparams(
    afq_alpha=AFQ_ALPHA,
    afq_nperm=AFQ_NPERM,
    fwe_method=FWE_METHOD,
    poly_degree=POLY_DEGREE,
    correct_multi_tract=CORRECT_MULTI_TRACT,
    metric_columns_candidates=config.get("METRIC_COLUMNS_CANDIDATES",
                                          DEFAULT_CONFIG["METRIC_COLUMNS_CANDIDATES"]),
    stat_types=config.get("STAT_TYPES", DEFAULT_CONFIG["STAT_TYPES"]),
    missing_subjects_tolerance=MISSING_SUBJECTS_TOLERANCE,
)

# Dict serialisable passé aux workers (pour les sous-processus joblib)
_HYPERPARAMS = {
    'afq_alpha': AFQ_ALPHA,
    'afq_nperm': AFQ_NPERM,
    'fwe_method': FWE_METHOD,
    'poly_degree': POLY_DEGREE,
    'correct_multi_tract': CORRECT_MULTI_TRACT,
    'metric_columns_candidates': config.get("METRIC_COLUMNS_CANDIDATES",
                                             DEFAULT_CONFIG["METRIC_COLUMNS_CANDIDATES"]),
    'stat_types': config.get("STAT_TYPES", DEFAULT_CONFIG["STAT_TYPES"]),
    'missing_subjects_tolerance': MISSING_SUBJECTS_TOLERANCE,
}

_ADDITIONAL_INFO_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/participants_full_info.xlsx"
_ACTIMETRY_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/actimetry_features.xlsx"

CLASSIF_CONFOUND_MAP = {
    k: (confond_variables_with_control if v == "with_controls" else confond_variables_without_control)
    for k, v in classif_variables.items()
}

# --- Filtre d'exclusion par valeurs catégorielles ---
# Exemple : EXCLUDE = {'group': ['hc']}  → exclut les sujets dont group == 'hc'
EXCLUDE = config.get("EXCLUDE", DEFAULT_CONFIG["EXCLUDE"])

# =====================================================================
#                    CHARGEMENT DONNÉES EXTERNES
# =====================================================================

@lru_cache(maxsize=1)
def _load_external_tables(additional_info_path: str, actimetry_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        add_df = pd.read_excel(additional_info_path)
    except Exception:
        add_df = pd.DataFrame()
    try:
        act_df = pd.read_excel(actimetry_path)
    except Exception:
        act_df = pd.DataFrame()
    return add_df, act_df

_ADDITIONAL_INFO_DF, _ACTIMETRY_DF = _load_external_tables(_ADDITIONAL_INFO_PATH, _ACTIMETRY_PATH)
actimetry_columns = [c for c in _ACTIMETRY_DF.columns if c not in ['subject_id', 'participant_id']] if not _ACTIMETRY_DF.empty else []

# =====================================================================
#                        UTILITAIRES COMMUNS
# =====================================================================

def load_and_merge_bundle_csvs(bundle_name, bundle_csvs, with_actimetry=True):
    global actimetry_columns, corr_variables

    metric_files_dict = {f.get_full_entities()['subject']: f for f in bundle_csvs}
    metric_files = [pd.read_csv(f.path) for f in bundle_csvs]
    for df, f in zip(metric_files, bundle_csvs):
        df['subject'] = f.get_full_entities()['subject']
        df['participant_id'] = 'sub-' + df["subject"].astype(str)
    metrics_df = pd.concat(metric_files, ignore_index=True)

    if not _ADDITIONAL_INFO_DF.empty:
        metrics_df = metrics_df.merge(_ADDITIONAL_INFO_DF, on='participant_id', how='left')
    if not _ACTIMETRY_DF.empty and with_actimetry:
        metrics_df = metrics_df.merge(_ACTIMETRY_DF, on='participant_id', how='left')

        acti_cols = [c for c in metrics_df.columns if '12h_' in c and c.endswith(tuple(str(i) for i in range(7)))]
        if acti_cols:
            metrics_df, avg_features = compute_12h_averages(metrics_df, acti_cols)
            if avg_features:
                for feat in avg_features:
                    if feat not in actimetry_columns:
                        actimetry_columns.append(feat)
        #Drop acti_cols individuels
        metrics_df.drop(columns=acti_cols, inplace=True, errors='ignore')

    # Apply exclusion filter
    metrics_df = apply_exclusion_filter(metrics_df, EXCLUDE)

    del metric_files
    gc.collect()
    return metrics_df, metric_files_dict


def _iter_bundle_centroids(bundle_name, df):
    """Itère sur les centroïdes d'un bundle.

    Cède des triplets ``(centroid_id, sous_dataframe, label)``. Si la colonne
    ``centroid_id`` est absente, cède un unique élément ``(None, df, bundle_name)``.
    Les sous-DataFrames vides sont ignorés.
    """
    if 'centroid_id' in df.columns:
        try:
            centroid_vals = (pd.to_numeric(df['centroid_id'], errors='coerce')
                             .dropna().astype(int).unique().tolist())
        except Exception:
            centroid_vals = df['centroid_id'].dropna().unique().tolist()
        centroid_vals = sorted(centroid_vals)
    else:
        centroid_vals = [None]

    for cid in centroid_vals:
        df_sub = df if cid is None else df[df['centroid_id'] == cid]
        if df_sub is None or df_sub.empty:
            continue
        label = f"{bundle_name}_cent{cid}" if cid is not None else bundle_name
        yield cid, df_sub, label


def _get_reference_participants():
    if _ADDITIONAL_INFO_DF is None or _ADDITIONAL_INFO_DF.empty:
        return set()
    if 'participant_id' not in _ADDITIONAL_INFO_DF.columns:
        return set()
    refs = set()
    for val in _ADDITIONAL_INFO_DF['participant_id'].dropna().unique():
        pid = _normalize_participant_id(val)
        if pid:
            refs.add(pid)
    return refs



# =====================================================================
#            WORKERS (classification identique v2, corrélation v3)
# =====================================================================

def _worker_classif_variable(classif_col, bundle_data, metric_set, report_dir,
                             AFQ_NPERM=1000, AFQ_ALPHA=0.05, MULTI_BUNDLE_AFQ=False,
                             precomputed_thresholds=None, hyperparams=None):
    """
    Analyse de groupe corrigée des confondants (identique v2).
    """
    # Réapplication des hyperparamètres dans ce sous-processus
    if hyperparams:
        configure_hyperparams(**hyperparams)
    figure_dir = opj(report_dir, "figures")
    ensure_dir(figure_dir)
    summary_rows = []
    included_participants = set()
    removed_participants = set()
    if not bundle_data:
        return {'summary': summary_rows,
                'included_participants': included_participants,
                'removed_participants': removed_participants}
    loop_counter = 0
    sig_key = 'alphaFWE' if FWE_METHOD == 'alphaFWE' else 'clusterFWE'

    if CORRECT_MULTI_TRACT and precomputed_thresholds is None:
        precomputed_thresholds = {}
        for metric_col in metric_set:
            aFWE, cFWE = compute_multi_tract_thresholds_group(
                bundle_data, metric_col, classif_col, alpha=AFQ_ALPHA, nperm=AFQ_NPERM)
            precomputed_thresholds[metric_col] = (aFWE, cFWE)

    for metric_col in tqdm(metric_set, desc=f"classif:{classif_col}"):
        for bundle_name, df in tqdm(bundle_data.items(), desc=f"{classif_col}-{metric_col}", leave=False):
            for cid, df_sub, bundle_label in _iter_bundle_centroids(bundle_name, df):
                try:
                    point_col = detect_point_column(df_sub)
                except Exception:
                    continue
                if metric_col not in df_sub.columns or classif_col not in df_sub.columns:
                    continue
                try:
                    long_raw = prepare_long(df_sub, metric_col, point_col)
                except Exception:
                    continue
                long_raw = long_raw[long_raw[classif_col].notna()]
                if long_raw.empty:
                    del long_raw
                    continue
                subject_pid_map = _build_subject_participant_map(long_raw)
                long_raw_clean, removed_subjects, removed_points = clean_missing_data(long_raw, threshold=0.15, missing_subjects_tolerance=MISSING_SUBJECTS_TOLERANCE)
                del long_raw
                if long_raw_clean.empty:
                    del long_raw_clean
                    continue

                included_participants.update(
                    _subjects_to_participant_ids(long_raw_clean['subject'].dropna().unique(), subject_pid_map))
                removed_participants.update(
                    _subjects_to_participant_ids(removed_subjects, subject_pid_map))

                conf_list = CLASSIF_CONFOUND_MAP.get(classif_col, confond_variables_with_control)
                confonds_str = ",".join(conf_list) if conf_list else "none"
                if conf_list:
                    corrected_segments = []
                    for pid, gpt in long_raw_clean.groupby('point'):
                        subj_level = (gpt[['subject', 'value'] + [c for c in conf_list if c in gpt.columns]]
                                      .drop_duplicates(subset='subject')
                                      .set_index('subject'))
                        y_res = residualize_on_confond(
                            subj_level.assign(target=subj_level['value']), 'target', conf_list)
                        g_corr = gpt.copy()
                        g_corr['value'] = g_corr['subject'].map(y_res)
                        corrected_segments.append(g_corr)
                    long_corr = (pd.concat(corrected_segments, ignore_index=True)
                                 if corrected_segments else pd.DataFrame())
                    del corrected_segments
                else:
                    long_corr = long_raw_clean.copy()

                pre_alpha, pre_cluster = None, None
                if CORRECT_MULTI_TRACT and precomputed_thresholds and metric_col in precomputed_thresholds:
                    pre_alpha, pre_cluster = precomputed_thresholds[metric_col]

                corr_res = (group_test_afq(long_corr, classif_col, alpha=AFQ_ALPHA, nperm=AFQ_NPERM,
                                           precomputed_alphaFWE=pre_alpha,
                                           precomputed_clusterFWE=pre_cluster)
                            if not long_corr.empty else pd.DataFrame())
                if not corr_res.empty:
                    out_plot_corr = opj(figure_dir,
                                        f"{bundle_label}_{metric_col}_{classif_col}_group_corrected.png")
                    plot_group_diff(long_corr, corr_res, classif_col, metric_col,
                                    bundle_label, out_plot_corr,
                                    "(corrigé confond)" + (" [multi-tract]" if CORRECT_MULTI_TRACT else ""),
                                    removed_subjects, removed_points,
                                    export_csv=True, canonical_bundle=bundle_name, centroid_id=cid)

                n_subjects_used = int(long_corr['subject'].dropna().nunique()) if not long_corr.empty else 0
                n_subjects_removed = len(removed_subjects)
                summary_rows.append({
                    'bundle': bundle_name,
                    'centroid_id': cid if cid is not None else np.nan,
                    'metric': metric_col,
                    'type': f"group_{classif_col}",
                    'n_sig': (_count_clusters(corr_res['sig_afq']) if FWE_METHOD == 'clusterFWE'
                              else int(corr_res['sig_afq'].sum())) if not corr_res.empty else 0,
                    'min_p': float(corr_res['p_raw'].min()) if not corr_res.empty else np.nan,
                    'max_r2_partial': np.nan,
                    'removed_points': len(removed_points),
                    'subjects_used': n_subjects_used,
                    'removed_subjects': n_subjects_removed,
                    'confonds_used': confonds_str,
                    sig_key: float(corr_res[sig_key].iloc[0]) if not corr_res.empty else np.nan
                })
                del long_raw_clean, long_corr, corr_res
                loop_counter += 1
                if loop_counter % 10 == 0:
                    gc.collect()
    gc.collect()
    return {'summary': summary_rows,
            'included_participants': included_participants,
            'removed_participants': removed_participants}


def _worker_corr_variable(var_col, bundle_data, metric_set, report_dir,
                           precomputed_thresholds=None, hyperparams=None):
    """
    Analyse de corrélation non-linéaire (v3).

    Au lieu de résidualiser puis corréler, ajuste à chaque point :
        metric = β0 + β1·var + β2·var² + confonds + ε
    et teste la significativité des termes polynomiaux.
    """
    # Réapplication des hyperparamètres dans ce sous-processus
    if hyperparams:
        configure_hyperparams(**hyperparams)
    rows = []
    included_participants = set()
    removed_participants = set()
    if not bundle_data:
        return {'summary': rows,
                'included_participants': included_participants,
                'removed_participants': removed_participants}
    figure_dir = opj(report_dir, "figures")
    ensure_dir(figure_dir)
    loop_counter = 0
    sig_key = 'alphaFWE' if FWE_METHOD == 'alphaFWE' else 'clusterFWE'

    # Confonds pour cette variable
    confonds_for_var = (confond_variables_for_actimetry if var_col in actimetry_columns
                        else confond_variables_without_control)

    # Pré-calcul des seuils multi-tracts si demandé
    if CORRECT_MULTI_TRACT and precomputed_thresholds is None:
        precomputed_thresholds = {}
        for metric_col in metric_set:
            aFWE, cFWE = compute_multi_tract_thresholds_ols(
                bundle_data, metric_col, var_col, confonds_for_var,
                poly_degree=POLY_DEGREE, alpha=AFQ_ALPHA, nperm=AFQ_NPERM)
            precomputed_thresholds[metric_col] = (aFWE, cFWE)

    for bundle_name, df in tqdm(bundle_data.items(), desc=f"ols:{var_col} bundles"):
        for cid, df_sub, bundle_label in _iter_bundle_centroids(bundle_name, df):
            try:
                point_col = detect_point_column(df_sub)
            except Exception:
                continue

            metric_cols = [m for m in metric_set if m in df_sub.columns]
            for metric_col in tqdm(metric_cols, desc=f"{var_col}-{bundle_name} métriques", leave=False):
                long_df = prepare_long(df_sub, metric_col, point_col)
                subject_pid_map = _build_subject_participant_map(long_df)
                long_df_clean, removed_subjects, removed_points = clean_missing_data(long_df, threshold=0.15, missing_subjects_tolerance=MISSING_SUBJECTS_TOLERANCE)
                del long_df
                if long_df_clean.empty or var_col not in long_df_clean.columns:
                    if not long_df_clean.empty:
                        del long_df_clean
                    continue

                included_participants.update(
                    _subjects_to_participant_ids(long_df_clean['subject'].dropna().unique(), subject_pid_map))
                removed_participants.update(
                    _subjects_to_participant_ids(removed_subjects, subject_pid_map))

                confonds_to_use = (confond_variables_for_actimetry if var_col in actimetry_columns
                                   else confond_variables_without_control)
                confonds_str = ",".join(confonds_to_use) if confonds_to_use else "none"

                # Récupérer les seuils pré-calculés si correction multi-tracts
                pre_alpha, pre_cluster = None, None
                if CORRECT_MULTI_TRACT and precomputed_thresholds and metric_col in precomputed_thresholds:
                    pre_alpha, pre_cluster = precomputed_thresholds[metric_col]

                # --- Modèle OLS polynomial ---
                ols_results = pointwise_ols_model(
                    long_df_clean, var_col, confonds_to_use,
                    poly_degree=POLY_DEGREE,
                    precomputed_alphaFWE=pre_alpha,
                    precomputed_clusterFWE=pre_cluster,
                    nperm=AFQ_NPERM)

                # --- Modèle linéaire simple (deg 1 + confonds) ---
                linear_results = pd.DataFrame()
                if POLY_DEGREE > 1:
                    linear_results = pointwise_ols_model(
                        long_df_clean, var_col, confonds_to_use,
                        poly_degree=1,
                        precomputed_alphaFWE=pre_alpha,
                        precomputed_clusterFWE=pre_cluster,
                        nperm=AFQ_NPERM)
                    if not linear_results.empty:
                        linear_results['model_type'] = 'linear'

                # --- Modèle interaction (slopes par groupe) si test_by_group ---
                inter_results = pd.DataFrame()
                final_results = ols_results
                model_label = f'poly_deg{POLY_DEGREE}'
                has_group = (test_by_group is not None
                             and test_by_group in long_df_clean.columns
                             and long_df_clean[test_by_group].dropna().nunique() >= 2)

                if has_group:
                    inter_results = pointwise_ols_interaction_model(
                        long_df_clean, var_col, test_by_group, confonds_to_use,
                        precomputed_alphaFWE=pre_alpha,
                        precomputed_clusterFWE=pre_cluster,
                        nperm=AFQ_NPERM)

                # Construire le dict des modèles disponibles
                all_model_dfs = {}
                if not ols_results.empty:
                    all_model_dfs['polynomial'] = ols_results
                if not linear_results.empty:
                    all_model_dfs['linear'] = linear_results
                if not inter_results.empty:
                    all_model_dfs['interaction'] = inter_results

                # Harmoniser les points communs à tous les modèles
                if len(all_model_dfs) >= 2:
                    all_model_dfs = _harmonize_model_points(all_model_dfs)
                    ols_results = all_model_dfs.get('polynomial', ols_results)
                    linear_results = all_model_dfs.get('linear', linear_results)
                    inter_results = all_model_dfs.get('interaction', inter_results)

                # Sélection du meilleur modèle si au moins 2 candidats
                if len(all_model_dfs) >= 2:
                    final_results = select_best_model_pointwise(
                        ols_results, inter_results,
                        linear_df=linear_results if not linear_results.empty else None)
                    model_label = 'best_of_models'
                    counts = {}
                    if 'best_model' in final_results.columns:
                        for m in ['linear', 'polynomial', 'interaction']:
                            counts[m] = int((final_results['best_model'] == m).sum())
                    parts = [f"{m}={c}" for m, c in counts.items() if c > 0]
                    print(f"  {bundle_label}/{metric_col}/{var_col}: {', '.join(parts)} pts")

                out_plot = opj(figure_dir,
                               f"{bundle_label}_{metric_col}_{var_col}_ols_poly{POLY_DEGREE}.png")

                plot_ols_results(final_results, metric_col, bundle_label, var_col, out_plot,
                                 f"(OLS deg{POLY_DEGREE} + confonds)" +
                                 (" [multi-tract]" if CORRECT_MULTI_TRACT else "") +
                                 (" [model selection]" if 'best_model' in final_results.columns else ""),
                                 removed_subjects, removed_points,
                                 export_csv=True, canonical_bundle=bundle_name, centroid_id=cid,
                                 poly_degree=POLY_DEGREE,
                                 model_dfs=all_model_dfs if len(all_model_dfs) >= 2 else None)

                # --- Per-model cluster stats pour le summary ---
                per_model_cluster = compute_per_model_cluster_summary(all_model_dfs)

                n_subjects_used = int(long_df_clean['subject'].dropna().nunique())
                n_subjects_removed = len(removed_subjects)
                summary_entry = {
                    'bundle': bundle_name,
                    'centroid_id': cid if cid is not None else np.nan,
                    'metric': metric_col,
                    'type': f"ols_{var_col}",
                    'n_sig': (_count_clusters(final_results['sig_afq']) if FWE_METHOD == 'clusterFWE'
                              else int(final_results['sig_afq'].sum())) if not final_results.empty else 0,
                    'min_p': float(final_results['p_raw'].min()) if not final_results.empty else np.nan,
                    'max_r2_partial': (float(final_results['r2_partial'].max())
                                      if not final_results.empty else np.nan),
                    'removed_points': len(removed_points),
                    'subjects_used': n_subjects_used,
                    'removed_subjects': n_subjects_removed,
                    'confonds_used': confonds_str,
                    'model': model_label,
                    sig_key: float(final_results[sig_key].iloc[0]) if not final_results.empty else np.nan
                }
                if 'best_model' in final_results.columns and not final_results.empty:
                    for m in ['linear', 'polynomial', 'interaction']:
                        summary_entry[f'pct_{m}'] = float((final_results['best_model'] == m).mean() * 100)

                # Per-model cluster stats
                for m_name, m_stats in per_model_cluster.items():
                    prefix = f'{m_name}_'
                    summary_entry[prefix + 'n_clusters'] = m_stats['n_clusters']
                    summary_entry[prefix + 'total_sig_pts'] = m_stats['total_sig_points']
                    summary_entry[prefix + 'mean_aic_sig'] = m_stats['overall_mean_aic_sig']
                    summary_entry[prefix + 'mean_p_sig'] = m_stats['overall_mean_p_sig']
                    summary_entry[prefix + 'mean_r2_sig'] = m_stats['overall_mean_r2_sig']
                    # Tailles des clusters
                    cluster_sizes = [c['size'] for c in m_stats['clusters']]
                    summary_entry[prefix + 'cluster_sizes'] = str(cluster_sizes) if cluster_sizes else ''

                # --- p_raw et f_stat par modèle ---
                for _m_name, _m_df in [('poly', ols_results), ('lin', linear_results), ('int', inter_results)]:
                    if not _m_df.empty and 'p_raw' in _m_df.columns:
                        summary_entry[f'{_m_name}_min_p'] = float(_m_df['p_raw'].min())
                    else:
                        summary_entry[f'{_m_name}_min_p'] = np.nan
                    if not _m_df.empty and 'f_stat' in _m_df.columns:
                        summary_entry[f'{_m_name}_max_f_stat'] = float(_m_df['f_stat'].max())
                    else:
                        summary_entry[f'{_m_name}_max_f_stat'] = np.nan

                # --- Betas moyens sur les points significatifs (3 décimales) ---
                # Linéaire : beta_deg1 (effet de var)
                _lin_sig = (linear_results[linear_results['sig_afq']]
                            if (not linear_results.empty and 'sig_afq' in linear_results.columns)
                            else pd.DataFrame())
                summary_entry['linear_mean_beta_var'] = (
                    round(float(_lin_sig['beta_deg1'].mean()), 3)
                    if (not _lin_sig.empty and 'beta_deg1' in _lin_sig.columns)
                    else np.nan)

                # Polynomial : beta_deg2 (effet de var²)
                _poly_sig = (ols_results[ols_results['sig_afq']]
                             if (not ols_results.empty and 'sig_afq' in ols_results.columns)
                             else pd.DataFrame())
                summary_entry['poly_mean_beta_var2'] = (
                    round(float(_poly_sig['beta_deg2'].mean()), 3)
                    if (not _poly_sig.empty and 'beta_deg2' in _poly_sig.columns)
                    else np.nan)

                # Interaction : beta_var (var effet global) + beta_interaction_<g> (var×groupe)
                _inter_sig = (inter_results[inter_results['sig_afq']]
                              if (not inter_results.empty and 'sig_afq' in inter_results.columns)
                              else pd.DataFrame())
                summary_entry['inter_mean_beta_var'] = (
                    round(float(_inter_sig['beta_var'].mean()), 3)
                    if (not _inter_sig.empty and 'beta_var' in _inter_sig.columns)
                    else np.nan)
                for _icol in [c for c in (_inter_sig.columns if not _inter_sig.empty else [])
                              if c.startswith('beta_interaction_')]:
                    summary_entry[f'inter_mean_{_icol}'] = round(float(_inter_sig[_icol].mean()), 3)

                rows.append(summary_entry)
                del long_df_clean, ols_results, final_results, linear_results, inter_results, all_model_dfs
                loop_counter += 1
                if loop_counter % 10 == 0:
                    gc.collect()
    gc.collect()
    return {'summary': rows,
            'included_participants': included_participants,
            'removed_participants': removed_participants}


# =====================================================================
#                     RAPPORT & MAIN
# =====================================================================

def generate_report(bundle_names, csv_files, report_dir="report_output", n_jobs=1):
    """
    Génère le rapport complet (v3 – modèle OLS non-linéaire).
    """
    global actimetry_columns

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    bundle_data = {}
    bundle_names = list(bundle_names)
    print(f"[v3] Analyse des bundles: {', '.join(bundle_names)}")
    print(f"Modèle OLS polynomial de degré {POLY_DEGREE}")
    print(f"Correction multi-tracts: {'ACTIVÉE' if CORRECT_MULTI_TRACT else 'désactivée'}")
    print(f"FWE method: {FWE_METHOD}")
    print(f"Types de statistiques (STAT_TYPES): {STAT_TYPES if STAT_TYPES else 'toutes'}")
    print(f"Rééchantillonnage: {f'{RESAMPLE_N_POINTS} points' if RESAMPLE_N_POINTS else 'désactivé'}")

    for bundle_name in tqdm([bn.replace("_", "") for bn in bundle_names], desc="Chargement des bundles"):
        bundle_csvs = [f for f in csv_files if f.get_entities()['bundle'] == bundle_name]
        if not bundle_csvs:
            continue
        df, _ = load_and_merge_bundle_csvs(bundle_name, bundle_csvs)
        if RESAMPLE_N_POINTS:
            df = resample_bundle_data(df, RESAMPLE_N_POINTS)
        bundle_data[bundle_name] = df
        if len(bundle_data) % 5 == 0:
            gc.collect()
    if not bundle_data:
        print("Aucun bundle à analyser.")
        return

    print(f"Features d'actimétrie disponibles: {len(actimetry_columns)}")
    avg_features = [c for c in actimetry_columns if c.endswith('_avg')]
    if avg_features:
        print(f"  dont {len(avg_features)} features moyennées 12h: "
              f"{', '.join(avg_features[:5])}{'...' if len(avg_features) > 5 else ''}")

    metric_set = sorted({m for df in bundle_data.values() for m in detect_metric_columns(df)})
    print(f"Métriques détectées: {', '.join(metric_set) if metric_set else '(aucune)'}")

    var_jobs = ([('classif', c) for c in classif_variables.keys()]
                + [('corr', c) for c in corr_variables])

    summary_rows = []
    included_participants = set()
    removed_participants = set()

    if n_jobs > 1 and var_jobs:
        from joblib import Parallel, delayed

        def _dispatch(job):
            kind, var = job
            if kind == 'classif':
                return _worker_classif_variable(var, bundle_data, metric_set, report_dir,
                                                hyperparams=_HYPERPARAMS)
            else:
                return _worker_corr_variable(var, bundle_data, metric_set, report_dir,
                                             hyperparams=_HYPERPARAMS)

        if MULTIPROCESSING:
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_dispatch)(job) for job in var_jobs)
        else:
            print(var_jobs[:1])
            results = [_dispatch(job) for job in var_jobs[:1]]

        for r in results:
            if not r:
                continue
            summary_rows.extend(r.get('summary', []))
            included_participants.update(r.get('included_participants', set()))
            removed_participants.update(r.get('removed_participants', set()))
        del results
        gc.collect()
    else:
        for kind, var in tqdm(var_jobs, desc="Variables"):
            if kind == 'classif':
                res = _worker_classif_variable(var, bundle_data, metric_set, report_dir,
                                               hyperparams=_HYPERPARAMS)
            else:
                res = _worker_corr_variable(var, bundle_data, metric_set, report_dir,
                                            hyperparams=_HYPERPARAMS)
            if res:
                summary_rows.extend(res.get('summary', []))
                included_participants.update(res.get('included_participants', set()))
                removed_participants.update(res.get('removed_participants', set()))
            gc.collect()

    if not summary_rows:
        print("Aucune donnée pour le rapport.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = opj(report_dir, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)

    reference_participants = _get_reference_participants()
    if reference_participants:
        final_removed = sorted(reference_participants - included_participants)
    else:
        final_removed = sorted(removed_participants - included_participants)
    removed_csv = opj(report_dir, "removed_participants.csv")
    pd.DataFrame({'participant_id': final_removed}).to_csv(removed_csv, index=False)
    print(f"Sujets retirés uniques: {len(final_removed)} (liste dans {removed_csv})")
    print(f"Résumé CSV: {summary_csv}")


def main():
    global corr_variables, classif_variables

    pipeline = config.get("pipeline", "default")  # Exemple de pipeline pour ce rapport
    classif_variables.clear()

    if 'actimetry' in pipeline or 'actimetry' in corr_variables:
        corr_variables = [v for v in corr_variables if v != 'actimetry']
        if csv_files:
            first_bundle = list(bundle_names)[0].replace("_", "")
            test_csvs = [f for f in csv_files if f.get_entities()['bundle'] == first_bundle]
            if test_csvs:
                _ = load_and_merge_bundle_csvs(first_bundle, test_csvs[:1])
        corr_variables.extend(actimetry_columns)
        print(f"Variables de corrélation: {len(corr_variables)} features")

    hostname = os.uname().nodename
    print(f"Hostname: {hostname}")

    n_jobs = (os.cpu_count()) // 2
    out_base = '/home/ndecaux/Reports'

    if hostname == 'calcarine':
        out_base = '/data/ndecaux/Reports'
        n_jobs = 24

    output_dir = (f'{out_base}/report_{dataset}_{hcp_asso_pipeline}_{pipeline}'
                  f'_{FWE_METHOD}_olsdeg{POLY_DEGREE}')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Rapport dans {output_dir}")

    # Sauvegarder la configuration utilisée dans le dossier de sortie
    save_config(config, opj(output_dir, "config.json"))

    generate_report(bundle_names, csv_files, report_dir=output_dir, n_jobs=n_jobs)


if __name__ == "__main__":
    main()
