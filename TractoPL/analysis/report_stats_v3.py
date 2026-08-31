"""
report_stats_v3.py
==================
Fonctions de statistiques, modèles OLS, tests de groupe, visualisation
et constantes algorithmiques pour le rapport tractométrique v3.

Séparé de generate_report_v3.py qui ne conserve que la configuration,
le chargement des données et l'orchestration du rapport.
"""

import os
import re
import gc
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from scipy.stats import ttest_ind, f_oneway
from scipy.stats import f as f_dist

from tractseg.libs.AFQ_MultiCompCorrection import get_significant_areas
from TractoPL.analysis.afq_optimized import AFQ_MultiCompCorrection

# =====================================================================
#               CONSTANTES ALGORITHMIQUES
# =====================================================================

METRIC_COLUMNS_CANDIDATES = ['FA', 'IFW']
STAT_TYPES = ['mean', 'median']

# --- Paramètres spécifiques v3 (modèle non-linéaire) ---
POLY_DEGREE = 2          # degré du polynôme pour la variable d'intérêt
                          # 2 → metric ~ var + var² + confounds

AFQ_ALPHA = 0.05
AFQ_NPERM = 1000
FWE_METHOD = "mixed"
if FWE_METHOD not in {"alphaFWE", "clusterFWE", "mixed"}:
    FWE_METHOD = "alphaFWE"

CORRECT_MULTI_TRACT = False
MISSING_SUBJECTS_TOLERANCE = 1


def configure_hyperparams(afq_alpha=None, afq_nperm=None, fwe_method=None,
                          poly_degree=None, correct_multi_tract=None,
                          metric_columns_candidates=None, stat_types=None,
                          missing_subjects_tolerance=None):
    """
    Met à jour les constantes algorithmiques du module à partir de la config chargée.
    À appeler depuis generate_report_v3.py après load_config().
    """
    global AFQ_ALPHA, AFQ_NPERM, FWE_METHOD, POLY_DEGREE, CORRECT_MULTI_TRACT
    global METRIC_COLUMNS_CANDIDATES, STAT_TYPES, MISSING_SUBJECTS_TOLERANCE
    if afq_alpha is not None:
        AFQ_ALPHA = afq_alpha
    if afq_nperm is not None:
        AFQ_NPERM = afq_nperm
    if fwe_method is not None:
        FWE_METHOD = fwe_method if fwe_method in {"alphaFWE", "clusterFWE", "mixed"} else "alphaFWE"
    if poly_degree is not None:
        POLY_DEGREE = poly_degree
    if correct_multi_tract is not None:
        CORRECT_MULTI_TRACT = correct_multi_tract
    if metric_columns_candidates is not None:
        METRIC_COLUMNS_CANDIDATES = metric_columns_candidates
    if stat_types is not None:
        STAT_TYPES = stat_types
    if missing_subjects_tolerance is not None:
        MISSING_SUBJECTS_TOLERANCE = missing_subjects_tolerance


# --- Constantes de visualisation ---
_MODEL_COLORS = {'linear': 'tab:green', 'polynomial': 'tab:orange', 'interaction': 'tab:cyan'}
_MODEL_BG = {'linear': 'lightgreen', 'polynomial': 'lightyellow', 'interaction': 'lightcyan'}
_MODEL_LABELS = {
    'linear': 'metric = β0 + β1·var + conf + ε',
    'polynomial': 'metric = β0 + β1·var + β2·var² + conf + ε',
    'interaction': 'metric = β0 + β·var + Σβ_g·grp + Σγ_g·var×grp + conf + ε'
}


# =====================================================================
#                        UTILITAIRES COMMUNS
# =====================================================================

def compute_12h_averages(df, feature_cols):
    df_avg = df.copy()
    cols_12h = [col for col in df_avg.columns if '_12h_' in col]
    cols_day = [col for col in cols_12h if col.endswith(('1', '3', '5'))]
    cols_night = [col for col in cols_12h if col.endswith(('2', '4', '6'))]
    cols_12h_base = ['_'.join(col.split('_')[:-1]) for col in cols_12h]

    averaged_features = []
    for col_base in set(cols_12h_base):
        cur_cols_12h = [col for col in cols_12h if col.startswith(col_base)]
        cur_cols_day = [col for col in cur_cols_12h if col in cols_day]
        cur_cols_night = [col for col in cur_cols_12h if col in cols_night]
        df_avg[col_base + '_avg'] = df_avg[cur_cols_12h].mean(axis=1)
        df_avg[col_base + '_day'] = df_avg[cur_cols_day].mean(axis=1)
        df_avg[col_base + '_night'] = df_avg[cur_cols_night].mean(axis=1)
        averaged_features.extend([col_base + '_avg', col_base + '_day', col_base + '_night'])

    df.drop(columns=cols_12h, inplace=True)
    return df_avg, averaged_features


def apply_exclusion_filter(df: pd.DataFrame, exclude_dict: Dict[str, List]) -> pd.DataFrame:
    """
    Filter out rows based on categorical values.

    Args:
        df: DataFrame to filter
        exclude_dict: Dictionary where keys are column names and values are lists of values to exclude
                     Example: {'group': ['hc'], 'apathy': [1, 2]}

    Returns:
        Filtered DataFrame with excluded rows removed
    """
    if not exclude_dict:
        return df

    df_filtered = df.copy()
    for col, values_to_exclude in exclude_dict.items():
        if col not in df_filtered.columns:
            continue
        mask = ~df_filtered[col].isin(values_to_exclude)
        df_filtered = df_filtered[mask]

    return df_filtered


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _longest_true_run(mask) -> int:
    """Longueur de la plus longue séquence consécutive de True dans `mask`."""
    longest = current = 0
    for flag in mask:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _coerce_confounds(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convertit les colonnes de confonds en numérique.

    Les colonnes catégorielles/texte sont encodées en codes entiers (modalités
    manquantes → NaN) ; les autres sont coercées en float.
    """
    out = df[cols].copy()
    for c in out.columns:
        if out[c].dtype == 'object' or str(out[c].dtype).startswith('category'):
            out[c] = out[c].astype('category').cat.codes.replace(-1, np.nan)
        else:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out


def _significance_mask(pvalues: np.ndarray, alphaFWE: float, clusterFWE: float,
                       alpha: float = None) -> np.ndarray:
    """Masque de significativité selon la méthode FWE active (FWE_METHOD).

    - alphaFWE   : points avec p < alphaFWE
    - mixed      : clusters d'au moins clusterFWE points contenant un p < alphaFWE
    - clusterFWE : clusters d'au moins clusterFWE points (get_significant_areas)
    """
    if alpha is None:
        alpha = AFQ_ALPHA
    pvalues = np.asarray(pvalues, dtype=float)
    if FWE_METHOD == "alphaFWE":
        return (pvalues < alphaFWE) if not np.isnan(alphaFWE) else np.zeros_like(pvalues, dtype=bool)
    if FWE_METHOD == "mixed":
        return get_significant_areas_mixed(
            pvalues, int(clusterFWE) if not np.isnan(clusterFWE) else 1, alphaFWE, alpha=alpha)
    # clusterFWE
    if np.isnan(clusterFWE) or clusterFWE < 1:
        return np.zeros_like(pvalues, dtype=bool)
    return get_significant_areas(pvalues, int(clusterFWE), alpha=alpha).astype(bool)


# =====================================================================
#        FONCTIONS PARTAGÉES – détection, nettoyage …
# =====================================================================

def _count_clusters(sig_iterable):
    cnt = 0
    in_cluster = False
    for v in list(sig_iterable):
        if v and not in_cluster:
            cnt += 1
            in_cluster = True
        elif not v and in_cluster:
            in_cluster = False
    return cnt


def get_significant_areas_mixed(pvalues: np.ndarray, cluster_threshold: int,
                                 alphaFWE: float, alpha: float = 0.05) -> np.ndarray:
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
            if (end - start) >= cluster_threshold:
                if np.any(pvalues[start:end] < alphaFWE):
                    sig_mask[start:end] = True
        else:
            i += 1
    return sig_mask


def detect_point_column(df):
    if 'point_id' in df.columns:
        return 'point_id'
    if 'point' in df.columns:
        return 'point'
    for c in df.columns:
        if re.fullmatch(r'point(_?id)?', c.lower()):
            return c
    raise ValueError("Colonne des points introuvable")


def detect_metric_columns(df):
    detected_cols = []
    if STAT_TYPES:
        for metric in METRIC_COLUMNS_CANDIDATES:
            for stat_type in STAT_TYPES:
                col_name = f"{metric}_{stat_type}"
                if col_name in df.columns:
                    detected_cols.append(col_name)
    if not detected_cols:
        detected_cols = [c for c in METRIC_COLUMNS_CANDIDATES if c in df.columns]
    return detected_cols


def get_all_metric_related_columns(df) -> List[str]:
    metric_cols = []
    for col in df.columns:
        if col in METRIC_COLUMNS_CANDIDATES:
            metric_cols.append(col)
            continue
        for metric in METRIC_COLUMNS_CANDIDATES:
            if col.startswith(f"{metric}_"):
                metric_cols.append(col)
                break
    return metric_cols


def ols_residualize(y, X):
    df_tmp = pd.concat([y, X], axis=1).dropna()
    if df_tmp.empty:
        return pd.Series(index=y.index, data=np.nan)
    y_clean = df_tmp.iloc[:, 0]
    X_clean = df_tmp.iloc[:, 1:]
    if X_clean.empty:
        return y
    Xc = sm.add_constant(X_clean, has_constant='add')
    try:
        model = sm.OLS(y_clean, Xc).fit()
        resid = model.resid + model.params.get('const', 0.0)
        out = pd.Series(index=y.index, data=np.nan)
        out.loc[resid.index] = resid.astype(float)
        return out
    except Exception:
        return y


def residualize_on_confond(df_subject_level, target_col, confond):
    cols = [c for c in confond if c in df_subject_level.columns]
    if not cols:
        return df_subject_level[target_col]
    X = _coerce_confounds(df_subject_level, cols)
    return ols_residualize(df_subject_level[target_col], X)


def prepare_long(df_bundle, metric_col, point_col):
    needed = ['subject', point_col, metric_col]
    if not all(c in df_bundle.columns for c in needed):
        raise ValueError(f"Colonnes manquantes pour {metric_col}")
    all_metric_cols = get_all_metric_related_columns(df_bundle)
    meta_cols = [c for c in df_bundle.columns if c not in all_metric_cols]
    long_df = df_bundle[meta_cols + [metric_col]].rename(columns={metric_col: 'value', point_col: 'point'})
    return long_df


def clean_missing_data(long_df, threshold=0.15, missing_subjects_tolerance=0):
    """
    Stratégie de gestion des NaN:
      Sélectionne la plus longue séquence de points consécutifs telle que
      au plus `missing_subjects_tolerance` sujets ont une valeur NaN sur
      chacun des points de la séquence.
      Le paramètre threshold n'est pas utilisé mais conservé pour compatibilité.
    """
    if long_df.empty:
        return long_df, [], []
    pivot = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    if pivot.shape[0] == 0:
        return long_df, [], []

    all_points = sorted(pivot.columns.tolist())
    # Pour chaque point, True si le nombre de sujets avec NaN est dans la tolérance
    if missing_subjects_tolerance is None:
        missing_subjects_tolerance = MISSING_SUBJECTS_TOLERANCE
    complete = [pivot[p].isna().sum() <= missing_subjects_tolerance for p in all_points]

    # Trouver la plus longue séquence consécutive de points complets
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
        # Aucun point sans données manquantes : on retourne vide
        return long_df.iloc[0:0].copy(), [], all_points

    kept_points = set(all_points[best_start: best_start + best_len])
    points_to_remove = [p for p in all_points if p not in kept_points]

    mask_points = long_df['point'].isin(kept_points)
    long_df_clean = long_df[mask_points].copy()
    return long_df_clean, [], points_to_remove


def _normalize_participant_id(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    val = str(value).strip()
    if not val:
        return None
    return val if val.startswith('sub-') else f"sub-{val}"


def _build_subject_participant_map(df):
    """Associe chaque `subject` à son `participant_id` normalisé."""
    mapping = {}
    if df is None or df.empty or 'subject' not in df.columns:
        return mapping
    if 'participant_id' in df.columns:
        tmp = (df[['subject', 'participant_id']]
               .dropna(subset=['subject'])
               .drop_duplicates(subset='subject'))
        for _, row in tmp.iterrows():
            subj_key = str(row['subject'])
            mapping[subj_key] = (_normalize_participant_id(row['participant_id'])
                                 or _normalize_participant_id(subj_key))
    else:
        for subj in df['subject'].dropna().unique():
            subj_key = str(subj)
            mapping[subj_key] = _normalize_participant_id(subj_key)
    return mapping


def _subjects_to_participant_ids(subjects, subject_pid_map):
    """Traduit un itérable de `subject` en l'ensemble des participant_id normalisés."""
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


# =====================================================================
#                   ANNOTATIONS DE VISUALISATION
# =====================================================================

def _annotate_missing(ax, removed_subjects, removed_points):
    lines = []
    if removed_subjects:
        ex = ", ".join(map(str, removed_subjects[:8]))
        if len(removed_subjects) > 8:
            ex += "..."
        lines.append(f"Sujets retirés ({len(removed_subjects)}): {ex}")
    if removed_points:
        exp = ", ".join(map(str, removed_points[:12]))
        if len(removed_points) > 12:
            exp += "..."
        lines.append(f"Points retirés ({len(removed_points)}) (≥15% NaN): {exp}")
    if lines:
        ax.text(0.01, 0.99, "\n".join(lines),
                ha='left', va='top', transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, edgecolor='gray'))


def _annotate_pvalues(ax, res_df, sig_col, y_offset_factor=0.02, max_labels=30):
    if res_df is None or res_df.empty:
        return
    sig_df = res_df[res_df[sig_col]] if sig_col in res_df.columns else pd.DataFrame()
    if sig_df.empty:
        return
    sig_df = sig_df.sort_values('p_raw').head(max_labels)
    ylim = ax.get_ylim()
    y_span = ylim[1] - ylim[0]
    for _, row in sig_df.iterrows():
        pval = row.get('p_raw', np.nan)
        if np.isnan(pval):
            continue
        x = row['point']
        y = ylim[1] - y_span * 0.05
        label = f"p={'{:.2e}'.format(pval) if pval < 0.001 else '{:.3f}'.format(pval)}"
        ax.text(x, y, label, rotation=90, ha='center', va='top', fontsize=7, color='red')


def _highlight_clusters(ax, points, sig_mask, color='red', alpha=0.18):
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


# =====================================================================
#       TESTS DE GROUPE – classification
# =====================================================================

def _estimate_cluster_threshold_group(pivot_values: np.ndarray, y: np.ndarray,
                                       alpha=0.05, nperm=500, random_state=None):
    rng = np.random.default_rng(random_state)
    n_points = pivot_values.shape[1]
    unique_groups = np.unique(y[~np.isnan(y)])
    max_clusters = []
    min_pvals = []

    for _ in range(min(nperm, 2000)):
        perm = rng.permutation(y)
        groups_data = [pivot_values[perm == g] for g in unique_groups]

        pvals = []
        for j in range(n_points):
            point_data = [gd[:, j] for gd in groups_data]
            valid_counts = [np.sum(~np.isnan(pd)) for pd in point_data]
            if all(vc >= 2 for vc in valid_counts):
                try:
                    if len(unique_groups) == 2:
                        _, p = ttest_ind(point_data[0], point_data[1], equal_var=False, nan_policy='omit')
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
        max_clusters.append(_longest_true_run(pvals_arr < alpha))

    if not max_clusters:
        return np.nan, np.nan
    cluster_thr = int(np.nanpercentile(max_clusters, 95))
    alpha_thr = np.nanpercentile(min_pvals, alpha * 100)
    return max(1, cluster_thr), alpha_thr


def compute_multi_tract_thresholds_group(bundle_data: Dict, metric_col: str, classif_col: str,
                                          alpha=AFQ_ALPHA, nperm=AFQ_NPERM) -> Tuple[float, float]:
    all_values = []
    subject_groups = {}
    for bundle_name, df in bundle_data.items():
        if metric_col not in df.columns or classif_col not in df.columns:
            continue
        try:
            point_col = detect_point_column(df)
        except Exception:
            continue
        pivot = df.pivot_table(index='subject', columns=point_col, values=metric_col, aggfunc='mean')
        if pivot.empty:
            continue
        subj_groups = (df[['subject', classif_col]]
                       .drop_duplicates()
                       .set_index('subject')
                       .reindex(pivot.index))
        for subj in pivot.index:
            if subj not in subject_groups:
                subject_groups[subj] = subj_groups.loc[subj, classif_col]
            values_subj = pivot.loc[subj].values.tolist()
            found = False
            for entry in all_values:
                if entry['subject'] == subj:
                    entry['values'].extend(values_subj)
                    found = True
                    break
            if not found:
                all_values.append({'subject': subj, 'values': values_subj})

    if not all_values:
        return np.nan, np.nan
    valid_subjects = [e for e in all_values if e['subject'] in subject_groups
                      and pd.notna(subject_groups[e['subject']])]
    if len(valid_subjects) < 4:
        return np.nan, np.nan
    subjects = [e['subject'] for e in valid_subjects]
    n_points = min(len(e['values']) for e in valid_subjects)
    values_matrix = np.array([e['values'][:n_points] for e in valid_subjects])
    groups = [subject_groups[s] for s in subjects]
    sorted_groups = sorted(set(g for g in groups if pd.notna(g)), key=lambda x: str(x))
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
        alphaFWE, _, clusterFWE, _ = AFQ_MultiCompCorrection(values_matrix, y, alpha, nperm=int(nperm))
        return alphaFWE, clusterFWE
    except Exception:
        return np.nan, np.nan


def group_test_afq(long_df, classif_col, alpha=AFQ_ALPHA, nperm=AFQ_NPERM,
                   precomputed_alphaFWE=None, precomputed_clusterFWE=None):
    if classif_col not in long_df.columns or long_df.empty:
        return pd.DataFrame()
    long_df = long_df[long_df[classif_col].notna()]
    if long_df.empty:
        return pd.DataFrame()
    groups = long_df[classif_col].dropna().unique()
    if len(groups) < 2:
        return pd.DataFrame()

    sorted_groups = sorted(groups, key=lambda x: str(x))
    pivot = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    subj_groups = (long_df[['subject', classif_col]]
                   .drop_duplicates()
                   .set_index('subject')
                   .loc[pivot.index])
    mask_valid_label = subj_groups[classif_col].notna()
    pivot = pivot.loc[mask_valid_label]
    subj_groups = subj_groups.loc[mask_valid_label]
    if pivot.shape[0] < 4:
        return pd.DataFrame()
    pivot = pivot.dropna(axis=0, how='any')
    subj_groups = subj_groups.loc[pivot.index]
    if pivot.shape[0] < 4:
        return pd.DataFrame()

    group_map = {g: i for i, g in enumerate(sorted_groups)}
    y_series = subj_groups[classif_col].map(group_map)
    mask_y_valid = y_series.notna()
    pivot = pivot.loc[mask_y_valid]
    y_series = y_series.loc[mask_y_valid]
    if pivot.shape[0] < 4:
        return pd.DataFrame()
    y = y_series.values.astype(int)

    alphaFWE = precomputed_alphaFWE if precomputed_alphaFWE is not None and not np.isnan(precomputed_alphaFWE) else np.nan
    clusterFWE = precomputed_clusterFWE if precomputed_clusterFWE is not None and not np.isnan(precomputed_clusterFWE) else np.nan

    if np.isnan(alphaFWE) or np.isnan(clusterFWE):
        if len(np.unique(y)) == 2:
            try:
                nperm_local = 10000
                computed_alpha, _, computed_cluster, _ = AFQ_MultiCompCorrection(
                    pivot.values, y, alpha, nperm=int(nperm_local))
                if np.isnan(alphaFWE):
                    alphaFWE = computed_alpha
                if np.isnan(clusterFWE):
                    clusterFWE = computed_cluster
            except Exception:
                pass
        if (np.isnan(clusterFWE) or clusterFWE < 1) or np.isnan(alphaFWE):
            est_cluster, est_alpha = _estimate_cluster_threshold_group(
                pivot.values, y, alpha=alpha, nperm=int(nperm / 2))
            if np.isnan(clusterFWE) or clusterFWE < 1:
                clusterFWE = est_cluster
            if np.isnan(alphaFWE):
                alphaFWE = est_alpha

    pvalues = []
    means_list = [[] for _ in sorted_groups]
    diffs = []
    n_list = [[] for _ in sorted_groups]

    for point in pivot.columns:
        group_vals = [pivot.loc[y == i, point].astype(float) for i in range(len(sorted_groups))]
        if all(v.notna().sum() >= 2 for v in group_vals):
            try:
                if len(sorted_groups) == 2:
                    t, p = ttest_ind(group_vals[0], group_vals[1], equal_var=False, nan_policy='omit')
                    diff = group_vals[1].mean() - group_vals[0].mean()
                else:
                    clean_vals = [v.dropna() for v in group_vals]
                    if all(len(cv) >= 2 for cv in clean_vals):
                        t, p = f_oneway(*clean_vals)
                    else:
                        p = np.nan
                    diff = np.nan
            except Exception:
                p = np.nan
                diff = np.nan
        else:
            p = np.nan
            diff = np.nan
        pvalues.append(p)
        for i in range(len(sorted_groups)):
            means_list[i].append(group_vals[i].mean())
            n_list[i].append(group_vals[i].notna().sum())
        diffs.append(diff)

    pvalues = np.array(pvalues, dtype=float)
    sig_mask = _significance_mask(pvalues, alphaFWE, clusterFWE, alpha=alpha)

    out_data = {
        'point': pivot.columns,
        'p_raw': pvalues,
        'diff': diffs,
        'sig_afq': sig_mask,
        'alphaFWE': alphaFWE,
        'clusterFWE': clusterFWE
    }
    for i in range(len(sorted_groups)):
        out_data[f'mean_g{i}'] = means_list[i]
        out_data[f'n{i}'] = n_list[i]

    out_df = pd.DataFrame(out_data)
    del pivot, subj_groups, y_series
    gc.collect()
    return out_df


# =====================================================================
#       GROUP PLOT & CSV EXPORT
# =====================================================================

def _export_group_csv(long_df, ttest_df, classif_col, metric_name, bundle_name,
                      centroid_id, out_csv, analysis_label):
    agg = (long_df.groupby(['point', classif_col])['value']
                  .agg(['mean', 'std', 'count'])
                  .reset_index())
    pivot_mean = agg.pivot(index='point', columns=classif_col, values='mean')
    pivot_std = agg.pivot(index='point', columns=classif_col, values='std')
    pivot_n = agg.pivot(index='point', columns=classif_col, values='count')
    df_out = pd.DataFrame({'point': pivot_mean.index}).set_index('point')
    for g in pivot_mean.columns:
        df_out[f'mean_{g}'] = pivot_mean[g]
        df_out[f'std_{g}'] = pivot_std[g]
        df_out[f'n_{g}'] = pivot_n[g]
    if ttest_df is not None and not ttest_df.empty:
        for col in ['p_raw', 'p_fdr', 'mean_g0', 'mean_g1', 'diff', 'n0', 'n1',
                     'sig_afq', 'sig_fdr', 'alphaFWE', 'clusterFWE']:
            if col in ttest_df.columns:
                df_out[col] = ttest_df.set_index('point')[col]
    df_out['bundle'] = bundle_name
    df_out['centroid_id'] = centroid_id
    df_out['metric'] = metric_name
    df_out['classif'] = classif_col
    df_out['analysis'] = analysis_label
    df_out.reset_index().to_csv(out_csv, index=False)


def plot_group_diff(long_df, ttest_df, classif_col, metric_name, bundle_name, out_path,
                    title_suffix, removed_subjects=None, removed_points=None,
                    export_csv=True, canonical_bundle=None, centroid_id=None):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=long_df, x='point', y='value', hue=classif_col,
                 estimator='mean', errorbar=('ci', 95))
    sig_col = 'sig_afq' if (ttest_df is not None and not ttest_df.empty
                            and 'sig_afq' in ttest_df.columns) else 'sig_fdr'
    if (FWE_METHOD in ('clusterFWE', 'mixed') and ttest_df is not None
            and not ttest_df.empty and 'sig_afq' in ttest_df.columns):
        _highlight_clusters(plt.gca(), ttest_df['point'], ttest_df['sig_afq'])
    if ttest_df is not None and not ttest_df.empty and ttest_df[sig_col].any():
        all_vals = pd.to_numeric(long_df['value'], errors='coerce')
        if all_vals.notna().any():
            low = np.nanquantile(all_vals, 0.02)
            high = np.nanquantile(all_vals, 0.98)
        else:
            low, high = 0, 1
        sig_line = ttest_df[['point']].copy()
        sig_line['sig_line'] = np.where(ttest_df[sig_col], high, low)
        label_map = {'alphaFWE': 'Significatif (alphaFWE)',
                     'mixed': 'Significatif (mixed: cluster+alpha)',
                     'clusterFWE': 'Significatif (clusterFWE)'}
        plt.plot(sig_line['point'], sig_line['sig_line'], color='red',
                 linestyle='--', linewidth=1.8,
                 label=label_map.get(FWE_METHOD, 'Significatif'))
    ax = plt.gca()
    if ttest_df is not None and not ttest_df.empty and 'p_raw' in ttest_df.columns:
        ttest_df = ttest_df.copy()
        ttest_df['p_raw'] = ttest_df['p_raw'].clip(lower=0, upper=1)
        ax2 = ax.twinx()
        ax2.bar(ttest_df['point'], ttest_df['p_raw'], color='gray', alpha=0.18, width=1.0, label='p-value')
        pmax = float(ttest_df['p_raw'].max()) if ttest_df['p_raw'].notna().any() else 0.05
        ymax = min(1.0, max(0.06, pmax * 1.15))
        if 'alphaFWE' in ttest_df.columns and not ttest_df['alphaFWE'].isna().all():
            alphaFWE_val = ttest_df['alphaFWE'].iloc[0]
            if not np.isnan(alphaFWE_val) and FWE_METHOD in ('alphaFWE', 'mixed'):
                ax2.axhline(alphaFWE_val, color='green', linestyle=':', linewidth=1.2,
                            label=f'alphaFWE={alphaFWE_val:.3g}')
                ymax = max(ymax, alphaFWE_val * 1.15)
        if 'clusterFWE' in ttest_df.columns and not ttest_df['clusterFWE'].isna().all():
            cluster_val = ttest_df['clusterFWE'].iloc[0]
            if not np.isnan(cluster_val) and FWE_METHOD in ('clusterFWE', 'mixed'):
                ax2.text(0.01, 0.95, f"clusterFWE≥{int(cluster_val)} (alpha={AFQ_ALPHA})",
                         transform=ax2.transAxes, ha='left', va='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray'))
        ax2.set_ylabel("p-value")
        ax2.set_ylim(0, ymax)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
    plt.title(f"{bundle_name} - {metric_name} - {classif_col} {title_suffix} ({FWE_METHOD})")
    plt.xlabel("Point")
    plt.ylabel(metric_name)
    _annotate_missing(ax, removed_subjects or [], removed_points or [])
    _annotate_pvalues(ax, ttest_df, sig_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    if export_csv:
        out_csv = os.path.splitext(out_path)[0] + '.csv'
        analysis_lbl = 'raw' if 'raw' in out_path else 'corrected'
        bundle_for_csv = canonical_bundle if canonical_bundle else bundle_name
        if centroid_id is None:
            centroid_id = (long_df['centroid_id'].iloc[0]
                           if 'centroid_id' in long_df.columns and not long_df['centroid_id'].isna().all()
                           else np.nan)
        _export_group_csv(long_df, ttest_df, classif_col, metric_name,
                          bundle_for_csv, centroid_id, out_csv, analysis_lbl)


# =====================================================================
#          MODÈLE OLS NON-LINÉAIRE (v3)
# =====================================================================
#
#  Modèle à chaque point le long du faisceau :
#
#     metric = β0 + β1·var + β2·var² + β3·confond1 + … + ε
#
#  Test de significativité : test F partiel
#     H0 : β1 = β2 = 0    (modèle réduit = confonds seuls)
#     H1 : au moins un βk ≠ 0 pour les termes polynomiaux
#
#  Correction FWE : permutations Freedman-Lane vectorisées.
# =====================================================================


def _batch_partial_ftest(Y: np.ndarray, X_full: np.ndarray, X_reduced: np.ndarray,
                         n_var_terms: int) -> np.ndarray:
    """
    Test F partiel vectorisé sur toutes les colonnes de Y simultanément.

    Y : (n_subjects, n_points)
    X_full : (n_subjects, p_full)
    X_reduced : (n_subjects, p_reduced)
    n_var_terms : nombre de régresseurs supplémentaires dans le modèle complet

    Returns : p_values (n_points,)
    """
    n = Y.shape[0]
    df_num = n_var_terms
    df_denom = n - X_full.shape[1]

    if df_denom <= 0 or df_num <= 0:
        return np.ones(Y.shape[1])

    # Modèle réduit
    try:
        beta_red, _, _, _ = np.linalg.lstsq(X_reduced, Y, rcond=None)
        ssr_red = np.sum((Y - X_reduced @ beta_red) ** 2, axis=0)
    except np.linalg.LinAlgError:
        return np.ones(Y.shape[1])

    # Modèle complet
    try:
        beta_full, _, _, _ = np.linalg.lstsq(X_full, Y, rcond=None)
        ssr_full = np.sum((Y - X_full @ beta_full) ** 2, axis=0)
    except np.linalg.LinAlgError:
        return np.ones(Y.shape[1])

    # F-stat
    with np.errstate(divide='ignore', invalid='ignore'):
        f_stat = ((ssr_red - ssr_full) / df_num) / (ssr_full / df_denom)
    f_stat = np.maximum(np.nan_to_num(f_stat, nan=0.0), 0.0)

    p_values = 1.0 - f_dist.cdf(f_stat, df_num, df_denom)
    return np.nan_to_num(p_values, nan=1.0)


def _ols_permutation_fwe(Y: np.ndarray, var_values: np.ndarray,
                          conf_values: Optional[np.ndarray],
                          poly_degree: int = 2,
                          alpha: float = 0.05,
                          nperm: int = 1000,
                          random_state: int = 42) -> Tuple[float, float]:
    """
    Correction FWE par permutations Freedman-Lane pour le modèle OLS polynomial.

    Procédure :
      1. Ajuster le modèle réduit (confonds seuls) sur var → résidus de var
      2. Pour chaque permutation :
         a. Permuter les résidus de var
         b. Reconstruire var permuté = fitted(var|confonds) + résidus permutés
         c. Construire les termes polynomiaux à partir de var permuté
         d. Calculer les p-values du test F partiel à chaque point (vectorisé)
         e. Collecter min(p) et max(longueur cluster)
      3. alphaFWE = percentile(min_p, alpha*100)
         clusterFWE = percentile(max_cluster, 95)

    Returns : (alphaFWE, clusterFWE)
    """
    rng = np.random.default_rng(random_state)
    n_sub = Y.shape[0]

    # Construire X_reduced
    if conf_values is not None and conf_values.ndim == 2 and conf_values.shape[1] > 0:
        X_reduced = np.column_stack([np.ones(n_sub), conf_values])
    else:
        X_reduced = np.ones((n_sub, 1))

    # Résidus de var sous H0 (var ~ confonds)
    try:
        beta_var_red, _, _, _ = np.linalg.lstsq(X_reduced, var_values, rcond=None)
        var_fitted = X_reduced @ beta_var_red
        var_resid = var_values - var_fitted
    except np.linalg.LinAlgError:
        var_fitted = np.full_like(var_values, np.mean(var_values))
        var_resid = var_values - var_fitted

    min_pvals = []
    max_clusters = []

    for _ in range(nperm):
        # Permuter les résidus de var
        perm_idx = rng.permutation(n_sub)
        var_perm = var_fitted + var_resid[perm_idx]

        # Construire X_full avec var permuté
        poly_cols = [var_perm ** d for d in range(1, poly_degree + 1)]
        X_full_perm = np.column_stack([np.ones(n_sub)] + poly_cols +
                                       ([conf_values] if conf_values is not None
                                        and conf_values.ndim == 2 and conf_values.shape[1] > 0 else []))

        # Test F partiel vectorisé
        p_perm = _batch_partial_ftest(Y, X_full_perm, X_reduced, poly_degree)
        min_pvals.append(np.nanmin(p_perm))
        max_clusters.append(_longest_true_run(p_perm < alpha))

    if not min_pvals:
        return np.nan, np.nan

    alphaFWE = float(np.nanpercentile(min_pvals, alpha * 100))
    clusterFWE = max(1, int(np.nanpercentile(max_clusters, 95)))
    return alphaFWE, clusterFWE


def pointwise_ols_model(long_df: pd.DataFrame, var_col: str, confonds: List[str],
                        poly_degree: int = POLY_DEGREE,
                        precomputed_alphaFWE=None, precomputed_clusterFWE=None,
                        nperm: int = AFQ_NPERM) -> pd.DataFrame:
    """
    À chaque point le long du faisceau, ajuste :

        metric = β0 + β1·var + β2·var² + … + βk·confond_k + ε

    et évalue la significativité des termes polynomiaux via un test F partiel.

    Returns:
        DataFrame avec colonnes :
            point, f_stat, p_raw, r2, r2_partial,
            beta_deg1, beta_deg2, …, p_deg1, p_deg2, …,
            n, sig_afq, alphaFWE, clusterFWE
    """
    if var_col not in long_df.columns:
        return pd.DataFrame()

    pivot_val = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    meta_cols = ['subject', var_col] + [c for c in confonds if c in long_df.columns]
    subj_meta = long_df[meta_cols].drop_duplicates().set_index('subject')
    pivot_val = pivot_val.reindex(subj_meta.index)

    # Préparer la variable d'intérêt
    var_series = pd.to_numeric(subj_meta[var_col], errors='coerce')

    # Préparer les confonds
    conf_cols = [c for c in confonds if c in subj_meta.columns]
    conf_df = (_coerce_confounds(subj_meta, conf_cols) if conf_cols
               else pd.DataFrame(index=subj_meta.index))

    # Masque de validité (pas de NaN dans var ni confonds)
    mask_valid = var_series.notna()
    if conf_df.shape[1] > 0:
        mask_valid = mask_valid & conf_df.notna().all(axis=1)

    # Retirer les sujets avec NaN dans le pivot
    pivot_val = pivot_val.loc[mask_valid].dropna(axis=0, how='any')
    var_clean = var_series.reindex(pivot_val.index)
    mask_final = var_clean.notna()
    pivot_val = pivot_val.loc[mask_final]
    var_clean = var_clean.loc[mask_final]
    if conf_df.shape[1] > 0:
        conf_clean = conf_df.reindex(pivot_val.index)
    else:
        conf_clean = None

    n_sub = pivot_val.shape[0]
    min_obs = poly_degree + (conf_df.shape[1] if conf_df.shape[1] > 0 else 0) + 2
    if n_sub < min_obs:
        return pd.DataFrame()

    var_arr = var_clean.values.astype(float)
    conf_arr = conf_clean.values.astype(float) if conf_clean is not None and conf_clean.shape[1] > 0 else None

    # --- Construire les DataFrames statsmodels (partagés entre tous les points) ---
    full_data = {}
    for d in range(1, poly_degree + 1):
        full_data[f'var_pow{d}'] = var_arr ** d
    red_data = {}
    if conf_clean is not None:
        for c in conf_clean.columns:
            full_data[c] = conf_clean[c].values.astype(float)
            red_data[c] = conf_clean[c].values.astype(float)

    X_full_df = sm.add_constant(pd.DataFrame(full_data), has_constant='add')
    X_red_df = (sm.add_constant(pd.DataFrame(red_data), has_constant='add')
                if red_data
                else sm.add_constant(pd.Series(np.ones(n_sub), name='const'), has_constant='add'))

    # --- OLS par point avec statsmodels (compare_f_test) ---
    all_points = sorted(pivot_val.columns.tolist())
    res_by_point = {}
    for point_id in all_points:
        y = pivot_val[point_id].values.astype(float)
        try:
            model_full = sm.OLS(y, X_full_df).fit()
            model_reduced = sm.OLS(y, X_red_df).fit()
            # Cas dégénéré : fit parfait (ssr ≈ 0) → F = ∞, p = 0
            if np.isclose(model_full.ssr, 0.0):
                fstat, fpval = np.inf, 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ftest = model_full.compare_f_test(model_reduced)
                fstat, fpval, _ = ftest
            r2_partial = max(0.0, model_full.rsquared - model_reduced.rsquared)
            betas = {f'beta_deg{d}': float(model_full.params[f'var_pow{d}'])
                     for d in range(1, poly_degree + 1)}
            pvals = {f'p_deg{d}': float(model_full.pvalues[f'var_pow{d}'])
                    for d in range(1, poly_degree + 1)}
            # Intercept
            betas['beta_const'] = float(model_full.params.get('const', np.nan))
            pvals['p_const'] = float(model_full.pvalues.get('const', np.nan))
            # Confondants
            if conf_clean is not None:
                for c in conf_clean.columns:
                    if c in model_full.params.index:
                        betas[f'beta_{c}'] = float(model_full.params[c])
                        pvals[f'p_{c}'] = float(model_full.pvalues[c])
            res_by_point[point_id] = {
                'f_stat': float(fstat),
                'p_raw': float(fpval),
                'r2': float(model_full.rsquared),
                'r2_partial': float(r2_partial),
                'n': n_sub,
                'aic': float(model_full.aic),
                **betas,
                **pvals,
            }
        except Exception as e:
            res_by_point[point_id] = {'error': str(e)}

    valid_points = [p for p in all_points if 'error' not in res_by_point[p]]
    if not valid_points:
        return pd.DataFrame()

    p_values = np.array([res_by_point[p]['p_raw'] for p in valid_points])

    # --- Correction FWE (permutations Freedman-Lane vectorisées pour la performance) ---
    Y = pivot_val[valid_points].values
    alphaFWE = np.nan
    clusterFWE = np.nan

    if precomputed_alphaFWE is not None and not np.isnan(precomputed_alphaFWE):
        alphaFWE = precomputed_alphaFWE
    if precomputed_clusterFWE is not None and not np.isnan(precomputed_clusterFWE):
        clusterFWE = precomputed_clusterFWE

    if np.isnan(alphaFWE) or np.isnan(clusterFWE):
        try:
            perm_nperm = min(nperm, 5000)
            computed_alpha, computed_cluster = _ols_permutation_fwe(
                Y, var_arr, conf_arr, poly_degree=poly_degree,
                alpha=AFQ_ALPHA, nperm=perm_nperm)
            if np.isnan(alphaFWE):
                alphaFWE = computed_alpha
            if np.isnan(clusterFWE):
                clusterFWE = computed_cluster
        except Exception as e:
            print(f"Erreur permutation FWE OLS : {e}")

    # --- Masque de significativité ---
    sig_mask = _significance_mask(p_values, alphaFWE, clusterFWE)

    # --- Construire le DataFrame de sortie ---
    out_rows = []
    for i, point_id in enumerate(valid_points):
        r = res_by_point[point_id]
        row = {
            'point': point_id,
            'f_stat': r['f_stat'],
            'p_raw': r['p_raw'],
            'r2': r['r2'],
            'r2_partial': r['r2_partial'],
            'n': r['n'],
            'sig_afq': sig_mask[i],
            'alphaFWE': float(alphaFWE),
            'clusterFWE': float(clusterFWE) if not np.isnan(clusterFWE) else np.nan,
            'aic': r['aic'],
            'model_type': 'polynomial',
        }
        for key, val in r.items():
            if key.startswith('beta_') or (key.startswith('p_') and key != 'p_raw'):
                row[key] = val
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    gc.collect()
    return df_out


# =====================================================================
#   MODÈLE LINÉAIRE AVEC SLOPE PAR GROUPE (interaction var × group)
# =====================================================================

def _ols_permutation_fwe_interaction(Y: np.ndarray, var_values: np.ndarray,
                                     group_values: np.ndarray,
                                     conf_values: Optional[np.ndarray],
                                     alpha: float = 0.05,
                                     nperm: int = 1000,
                                     random_state: int = 42) -> Tuple[float, float]:
    """
    Correction FWE par permutations Freedman-Lane pour le modèle interaction.

    Procédure :
      1. Ajuster modèle réduit (group + confonds) sur var → résidus de var
      2. Pour chaque permutation :
         a. Permuter les résidus de var
         b. Reconstruire var permuté
         c. Recalculer les interactions
         d. Test F partiel vectorisé
         e. Collecter min(p) et max(longueur cluster)
    """
    rng = np.random.default_rng(random_state)
    n_sub = Y.shape[0]

    unique_groups = sorted(np.unique(group_values[~np.isnan(group_values)]))
    if len(unique_groups) < 2:
        return np.nan, np.nan

    group_dummies = [(group_values == g).astype(float) for g in unique_groups[1:]]

    # X_reduced pour le modèle réduit
    parts_red = [np.ones(n_sub)] + group_dummies
    if conf_values is not None and conf_values.ndim == 2 and conf_values.shape[1] > 0:
        parts_red.append(conf_values)
    X_reduced = np.column_stack(parts_red)

    n_var_terms = 1 + len(group_dummies)  # var + interactions

    # Résidus de var sous H0 (var ~ group + confonds)
    try:
        beta_var_red, _, _, _ = np.linalg.lstsq(X_reduced, var_values, rcond=None)
        var_fitted = X_reduced @ beta_var_red
        var_resid = var_values - var_fitted
    except np.linalg.LinAlgError:
        var_fitted = np.full_like(var_values, np.mean(var_values))
        var_resid = var_values - var_fitted

    min_pvals = []
    max_clusters = []

    for _ in range(nperm):
        perm_idx = rng.permutation(n_sub)
        var_perm = var_fitted + var_resid[perm_idx]

        # Reconstruire X_full avec var permuté et ses interactions
        interactions_perm = [var_perm * gd for gd in group_dummies]
        parts_full_perm = [np.ones(n_sub), var_perm] + group_dummies + interactions_perm
        if conf_values is not None and conf_values.ndim == 2 and conf_values.shape[1] > 0:
            parts_full_perm.append(conf_values)
        X_full_perm = np.column_stack(parts_full_perm)

        p_perm = _batch_partial_ftest(Y, X_full_perm, X_reduced, n_var_terms)
        min_pvals.append(np.nanmin(p_perm))
        max_clusters.append(_longest_true_run(p_perm < alpha))

    if not min_pvals:
        return np.nan, np.nan

    alphaFWE = float(np.nanpercentile(min_pvals, alpha * 100))
    clusterFWE = max(1, int(np.nanpercentile(max_clusters, 95)))
    return alphaFWE, clusterFWE


def pointwise_ols_interaction_model(long_df: pd.DataFrame, var_col: str,
                                     group_col: str, confonds: List[str],
                                     precomputed_alphaFWE=None,
                                     precomputed_clusterFWE=None,
                                     nperm: int = AFQ_NPERM) -> pd.DataFrame:
    """
    À chaque point, ajuste un modèle linéaire avec un slope par groupe :

        metric = β0 + β_var·var + Σ β_g·group_g + Σ γ_g·var·group_g + confonds + ε

    Le test F partiel évalue si les termes en var (β_var et γ_g) sont significatifs.

    Returns:
        DataFrame avec colonnes :
            point, f_stat, p_raw, r2, r2_partial,
            beta_var, beta_interaction_g1, ...,
            slope_group_<g>, ...,
            n, sig_afq, alphaFWE, clusterFWE, aic, model_type
    """
    if var_col not in long_df.columns or group_col not in long_df.columns:
        return pd.DataFrame()

    pivot_val = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    meta_cols = ['subject', var_col, group_col] + [c for c in confonds if c in long_df.columns]
    subj_meta = long_df[meta_cols].drop_duplicates().set_index('subject')
    pivot_val = pivot_val.reindex(subj_meta.index)

    var_series = pd.to_numeric(subj_meta[var_col], errors='coerce')
    group_series = pd.to_numeric(subj_meta[group_col], errors='coerce')

    # Confonds
    conf_cols = [c for c in confonds if c in subj_meta.columns]
    conf_df = (_coerce_confounds(subj_meta, conf_cols) if conf_cols
               else pd.DataFrame(index=subj_meta.index))

    # Masque de validité
    mask_valid = var_series.notna() & group_series.notna()
    if conf_df.shape[1] > 0:
        mask_valid = mask_valid & conf_df.notna().all(axis=1)

    pivot_val = pivot_val.loc[mask_valid].dropna(axis=0, how='any')
    var_clean = var_series.reindex(pivot_val.index)
    group_clean = group_series.reindex(pivot_val.index)
    mask_final = var_clean.notna() & group_clean.notna()
    pivot_val = pivot_val.loc[mask_final]
    var_clean = var_clean.loc[mask_final]
    group_clean = group_clean.loc[mask_final]
    if conf_df.shape[1] > 0:
        conf_clean = conf_df.reindex(pivot_val.index)
    else:
        conf_clean = None

    n_sub = pivot_val.shape[0]
    unique_groups = sorted(group_clean.dropna().unique())
    n_groups = len(unique_groups)

    if n_groups < 2:
        return pd.DataFrame()

    # Minimum d'observations : 1(intercept) + 1(var) + (n_groups-1)(dummies) + (n_groups-1)(interactions) + confonds + 2
    min_obs = 1 + 1 + (n_groups - 1) + (n_groups - 1) + (conf_df.shape[1] if conf_df.shape[1] > 0 else 0) + 2
    if n_sub < min_obs:
        return pd.DataFrame()

    var_arr = var_clean.values.astype(float)
    group_arr = group_clean.values.astype(float)
    conf_arr = conf_clean.values.astype(float) if conf_clean is not None and conf_clean.shape[1] > 0 else None

    sorted_groups = unique_groups  # déjà trié plus haut
    ref_group = sorted_groups[0]

    def _g_str(g):
        return str(int(g)) if isinstance(g, (int, float)) and int(g) == g else str(g)

    # --- Construire les DataFrames statsmodels (partagés entre tous les points) ---
    full_data_dict = {var_col: var_arr}
    red_data_dict = {}
    for g in sorted_groups[1:]:
        gs = _g_str(g)
        dummy = (group_arr == g).astype(float)
        full_data_dict[f'group_{gs}'] = dummy
        full_data_dict[f'{var_col}_x_group_{gs}'] = var_arr * dummy
        red_data_dict[f'group_{gs}'] = dummy

    if conf_clean is not None:
        for c in conf_clean.columns:
            full_data_dict[c] = conf_clean[c].values.astype(float)
            red_data_dict[c] = conf_clean[c].values.astype(float)

    X_full_df = sm.add_constant(pd.DataFrame(full_data_dict), has_constant='add')
    X_red_df = (sm.add_constant(pd.DataFrame(red_data_dict), has_constant='add')
                if red_data_dict
                else sm.add_constant(pd.Series(np.ones(n_sub), name='const'), has_constant='add'))

    # --- OLS par point avec statsmodels (compare_f_test) ---
    all_points = sorted(pivot_val.columns.tolist())
    res_by_point = {}
    ref_str = _g_str(ref_group)
    for point_id in all_points:
        y = pivot_val[point_id].values.astype(float)
        try:
            model_full = sm.OLS(y, X_full_df).fit()
            model_reduced = sm.OLS(y, X_red_df).fit()
            # Cas dégénéré : fit parfait (ssr ≈ 0) → F = ∞, p = 0
            if np.isclose(model_full.ssr, 0.0):
                fstat, fpval = np.inf, 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ftest = model_full.compare_f_test(model_reduced)
                fstat, fpval, _ = ftest
            r2_partial = max(0.0, model_full.rsquared - model_reduced.rsquared)

            betas_out = {'beta_var': float(model_full.params[var_col])}
            for g in sorted_groups[1:]:
                gs = _g_str(g)
                betas_out[f'beta_group_{gs}'] = float(model_full.params[f'group_{gs}'])
                betas_out[f'beta_interaction_{gs}'] = float(model_full.params[f'{var_col}_x_group_{gs}'])

            # Slopes effectifs par groupe
            betas_out[f'slope_group_{ref_str}'] = float(model_full.params[var_col])
            for g in sorted_groups[1:]:
                gs = _g_str(g)
                betas_out[f'slope_group_{gs}'] = (float(model_full.params[var_col])
                                                  + float(model_full.params[f'{var_col}_x_group_{gs}']))
            # P-values des termes var et interaction
            betas_out['p_var'] = float(model_full.pvalues.get(var_col, np.nan))
            for g in sorted_groups[1:]:
                gs = _g_str(g)
                betas_out[f'p_group_{gs}'] = float(model_full.pvalues.get(f'group_{gs}', np.nan))
                betas_out[f'p_interaction_{gs}'] = float(model_full.pvalues.get(f'{var_col}_x_group_{gs}', np.nan))
            # Intercept
            betas_out['beta_const'] = float(model_full.params.get('const', np.nan))
            betas_out['p_const'] = float(model_full.pvalues.get('const', np.nan))
            # Confondants
            if conf_clean is not None:
                for c in conf_clean.columns:
                    if c in model_full.params.index:
                        betas_out[f'beta_{c}'] = float(model_full.params[c])
                        betas_out[f'p_{c}'] = float(model_full.pvalues[c])

            res_by_point[point_id] = {
                'f_stat': float(fstat),
                'p_raw': float(fpval),
                'r2': float(model_full.rsquared),
                'r2_partial': float(r2_partial),
                'n': n_sub,
                'aic': float(model_full.aic),
                **betas_out,
            }
        except Exception as e:
            res_by_point[point_id] = {'error': str(e)}

    valid_points = [p for p in all_points if 'error' not in res_by_point[p]]
    if not valid_points:
        return pd.DataFrame()

    p_values = np.array([res_by_point[p]['p_raw'] for p in valid_points])

    # --- Correction FWE (permutations Freedman-Lane vectorisées pour la performance) ---
    Y_valid = pivot_val[valid_points].values
    alphaFWE = np.nan
    clusterFWE = np.nan

    if precomputed_alphaFWE is not None and not np.isnan(precomputed_alphaFWE):
        alphaFWE = precomputed_alphaFWE
    if precomputed_clusterFWE is not None and not np.isnan(precomputed_clusterFWE):
        clusterFWE = precomputed_clusterFWE

    if np.isnan(alphaFWE) or np.isnan(clusterFWE):
        try:
            perm_nperm = min(nperm, 5000)
            computed_alpha, computed_cluster = _ols_permutation_fwe_interaction(
                Y_valid, var_arr, group_arr, conf_arr,
                alpha=AFQ_ALPHA, nperm=perm_nperm)
            if np.isnan(alphaFWE):
                alphaFWE = computed_alpha
            if np.isnan(clusterFWE):
                clusterFWE = computed_cluster
        except Exception as e:
            print(f"Erreur permutation FWE interaction : {e}")

    # --- Masque de significativité ---
    sig_mask = _significance_mask(p_values, alphaFWE, clusterFWE)

    # --- Construire le DataFrame de sortie ---
    beta_keys = sorted(k for k in res_by_point[valid_points[0]]
                       if k not in {'f_stat', 'p_raw', 'r2', 'r2_partial', 'n', 'aic'})
    out_rows = []
    for i, point_id in enumerate(valid_points):
        r = res_by_point[point_id]
        row = {
            'point': point_id,
            'f_stat': r['f_stat'],
            'p_raw': r['p_raw'],
            'r2': r['r2'],
            'r2_partial': r['r2_partial'],
            'n': r['n'],
            'sig_afq': sig_mask[i],
            'alphaFWE': float(alphaFWE),
            'clusterFWE': float(clusterFWE) if not np.isnan(clusterFWE) else np.nan,
            'aic': r['aic'],
            'model_type': 'interaction',
        }
        for bk in beta_keys:
            if bk in r:
                row[bk] = r[bk]
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    gc.collect()
    return df_out


def select_best_model_pointwise(poly_df: pd.DataFrame,
                                 inter_df: pd.DataFrame,
                                 linear_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compare jusqu'à 3 modèles point par point via l'AIC :
      - linear      : metric = β0 + β1·var + confonds + ε
      - polynomial  : metric = β0 + β1·var + β2·var² + confonds + ε
      - interaction : metric = β0 + β·var + Σβ_g·group + Σγ_g·var×group + conf + ε

    Retourne un DataFrame consolidé où chaque point garde les résultats
    du meilleur modèle.
    """
    models = {}
    if linear_df is not None and not linear_df.empty:
        models['linear'] = linear_df
    if not poly_df.empty:
        models['polynomial'] = poly_df
    if not inter_df.empty:
        models['interaction'] = inter_df

    if not models:
        return pd.DataFrame()
    if len(models) == 1:
        name, df = next(iter(models.items()))
        df = df.copy()
        df['best_model'] = name
        df[f'aic_{name}'] = df['aic']
        return df

    # Fusionner tous les modèles sur 'point'
    model_names = list(models.keys())
    suffixes = {'linear': '_lin', 'polynomial': '_poly', 'interaction': '_inter'}

    merged = None
    for name in model_names:
        df_m = models[name].copy()
        rename_map = {}
        for col in df_m.columns:
            if col == 'point':
                continue
            rename_map[col] = f'{col}{suffixes[name]}'
        df_m = df_m.rename(columns=rename_map)
        if merged is None:
            merged = df_m
        else:
            merged = merged.merge(df_m, on='point', how='outer')

    # Déterminer le meilleur modèle par point via AIC
    aic_cols = {n: f'aic{suffixes[n]}' for n in model_names}
    aic_arrays = {}
    for n in model_names:
        col = aic_cols[n]
        if col in merged.columns:
            aic_arrays[n] = merged[col].values.astype(float)
        else:
            aic_arrays[n] = np.full(len(merged), np.nan)

    aic_matrix = np.column_stack([aic_arrays[n] for n in model_names])
    aic_for_argmin = np.where(np.isnan(aic_matrix), np.inf, aic_matrix)
    best_idx = np.argmin(aic_for_argmin, axis=1)
    best = np.array([model_names[i] for i in best_idx])
    all_nan = np.all(np.isnan(aic_matrix), axis=1)
    best[all_nan] = model_names[0]

    # Construire le DataFrame final
    common_cols = ['f_stat', 'p_raw', 'r2', 'r2_partial', 'n',
                   'sig_afq', 'alphaFWE', 'clusterFWE']
    result_rows = []

    for idx, row in merged.iterrows():
        out_row = {'point': row['point']}
        model = best[idx]
        suffix = suffixes[model]

        for col in common_cols:
            col_src = f'{col}{suffix}'
            out_row[col] = row.get(col_src, np.nan)

        out_row['best_model'] = model
        out_row['model_type'] = model

        # AIC de chaque modèle
        for n in model_names:
            out_row[f'aic_{n}'] = row.get(f'aic{suffixes[n]}', np.nan)

        # Copier beta/slope/p_deg du meilleur modèle
        for col_name in merged.columns:
            if col_name.startswith(('beta_', 'slope_', 'p_deg')):
                if col_name.endswith(suffix):
                    clean_name = col_name[:-len(suffix)]
                    out_row[clean_name] = row[col_name]

        result_rows.append(out_row)

    result_df = pd.DataFrame(result_rows)
    return result_df


# =====================================================================
#    CORRECTION MULTI-TRACTS POUR OLS (adaptation v3)
# =====================================================================

def compute_multi_tract_thresholds_ols(bundle_data: Dict, metric_col: str, var_col: str,
                                       confonds: List[str], poly_degree: int = POLY_DEGREE,
                                       alpha=AFQ_ALPHA, nperm=AFQ_NPERM) -> Tuple[float, float]:
    """
    Calcule les seuils alphaFWE / clusterFWE sur l'ensemble des bundles concaténés
    (correction multi-tracts) en utilisant le modèle OLS polynomial.
    """
    all_Y_rows = []
    subject_vars = {}
    subject_confs = {}

    for bundle_name, df in bundle_data.items():
        if metric_col not in df.columns or var_col not in df.columns:
            continue
        try:
            point_col = detect_point_column(df)
        except Exception:
            continue

        pivot = df.pivot_table(index='subject', columns=point_col, values=metric_col, aggfunc='mean')
        if pivot.empty:
            continue

        subj_meta = (df[['subject', var_col] + [c for c in confonds if c in df.columns]]
                     .drop_duplicates()
                     .set_index('subject')
                     .reindex(pivot.index))

        for subj in pivot.index:
            if subj not in subject_vars:
                val = subj_meta.loc[subj, var_col]
                subject_vars[subj] = pd.to_numeric(val, errors='coerce') if not isinstance(val, (int, float)) else val
                # Confonds
                conf_vals = {}
                for c in confonds:
                    if c in subj_meta.columns:
                        cv = subj_meta.loc[subj, c]
                        conf_vals[c] = pd.to_numeric(cv, errors='coerce') if not isinstance(cv, (int, float)) else cv
                subject_confs[subj] = conf_vals

            values_subj = pivot.loc[subj].values.tolist()
            found = False
            for entry in all_Y_rows:
                if entry['subject'] == subj:
                    entry['values'].extend(values_subj)
                    found = True
                    break
            if not found:
                all_Y_rows.append({'subject': subj, 'values': values_subj})

    if not all_Y_rows:
        return np.nan, np.nan

    valid = [e for e in all_Y_rows if e['subject'] in subject_vars
             and pd.notna(subject_vars[e['subject']])]
    if len(valid) < poly_degree + len(confonds) + 3:
        return np.nan, np.nan

    subjects = [e['subject'] for e in valid]
    n_points = min(len(e['values']) for e in valid)
    Y = np.array([e['values'][:n_points] for e in valid])
    var_arr = np.array([subject_vars[s] for s in subjects], dtype=float)

    # Confonds
    if confonds:
        conf_arr = np.array([[subject_confs[s].get(c, np.nan) for c in confonds] for s in subjects])
    else:
        conf_arr = None

    mask_valid = ~np.isnan(var_arr)
    if conf_arr is not None:
        mask_valid = mask_valid & ~np.any(np.isnan(conf_arr), axis=1)
    Y = Y[mask_valid]
    var_arr = var_arr[mask_valid]
    if conf_arr is not None:
        conf_arr = conf_arr[mask_valid]

    Y = np.nan_to_num(Y, nan=np.nanmean(Y))

    if Y.shape[0] < poly_degree + (conf_arr.shape[1] if conf_arr is not None else 0) + 3:
        return np.nan, np.nan

    try:
        alphaFWE, clusterFWE = _ols_permutation_fwe(
            Y, var_arr, conf_arr, poly_degree=poly_degree, alpha=alpha, nperm=nperm)
        return alphaFWE, clusterFWE
    except Exception:
        return np.nan, np.nan


# =====================================================================
#         PLOT & EXPORT POUR OLS NON-LINÉAIRE (v3)
# =====================================================================

def _harmonize_model_points(model_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Restreint tous les DataFrames de modèles à leur intersection de points.
    Garantit que plot et CSV ont exactement le même nombre de points.
    """
    non_empty = {k: v for k, v in model_dfs.items() if v is not None and not v.empty}
    if len(non_empty) <= 1:
        return dict(model_dfs)
    common = set(list(non_empty.values())[0]['point'])
    for df in list(non_empty.values())[1:]:
        common &= set(df['point'])
    return {
        k: (v[v['point'].isin(common)].reset_index(drop=True) if (v is not None and not v.empty) else v)
        for k, v in model_dfs.items()
    }


def _extract_significant_clusters(df: pd.DataFrame) -> List[Dict]:
    """
    Extrait la liste des clusters significatifs d'un DataFrame de modèle.
    """
    if df is None or df.empty or 'sig_afq' not in df.columns:
        return []
    sig = df['sig_afq'].values.astype(bool)
    points = df['point'].values
    clusters = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i < len(sig) and sig[i]:
                i += 1
            end = i  # exclusive
            cl = df.iloc[start:end]
            cluster_info = {
                'start_point': points[start],
                'end_point': points[end - 1],
                'size': end - start,
                'mean_p': float(cl['p_raw'].mean()),
                'min_p': float(cl['p_raw'].min()),
                'mean_r2_partial': float(cl['r2_partial'].mean()) if 'r2_partial' in cl.columns else np.nan,
                'mean_aic': float(cl['aic'].mean()) if 'aic' in cl.columns else np.nan,
                'mean_neg_log10p': float((-np.log10(cl['p_raw'].clip(lower=1e-300))).mean()),
            }
            clusters.append(cluster_info)
        else:
            i += 1
    return clusters


def compute_per_model_cluster_summary(model_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Pour chaque modèle, calcule des statistiques résumées sur ses clusters
    significatifs.
    """
    summary = {}
    for name, df in model_dfs.items():
        if df is None or df.empty:
            continue
        clusters = _extract_significant_clusters(df)
        total_sig = sum(c['size'] for c in clusters)
        overall_mean_aic = np.nan
        overall_mean_p = np.nan
        overall_mean_r2 = np.nan
        if clusters:
            sizes = np.array([c['size'] for c in clusters], dtype=float)
            weights = sizes / sizes.sum()
            overall_mean_aic = float(np.nansum([c['mean_aic'] * w for c, w in zip(clusters, weights)]))
            overall_mean_p = float(np.nansum([c['mean_p'] * w for c, w in zip(clusters, weights)]))
            overall_mean_r2 = float(np.nansum([c['mean_r2_partial'] * w for c, w in zip(clusters, weights)]))

        summary[name] = {
            'n_clusters': len(clusters),
            'total_sig_points': total_sig,
            'clusters': clusters,
            'overall_mean_aic_sig': overall_mean_aic,
            'overall_mean_p_sig': overall_mean_p,
            'overall_mean_r2_sig': overall_mean_r2,
        }
    return summary


_MODEL_PREFIX_MAP = {'linear': 'lin', 'polynomial': 'poly', 'interaction': 'int'}


def _add_model_betas_to_df(base_df: pd.DataFrame,
                           model_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pour chaque modèle dans model_dfs, ajoute dans base_df les colonnes de betas,
    pvals et slopes préfixées par le nom du modèle :
        beta_lin_*,  beta_poly_*,  beta_int_*
        pval_lin_*,  pval_poly_*,  pval_int_*
        slope_lin_*, slope_poly_*, slope_int_*
    Les lignes sont alignées sur la colonne 'point'.
    Toutes les lignes (significatives ou non) sont incluses.
    """
    result = base_df.copy()
    for model_name, model_df in model_dfs.items():
        if model_df is None or model_df.empty or 'point' not in model_df.columns:
            continue
        prefix = _MODEL_PREFIX_MAP.get(model_name, model_name[:3])
        model_indexed = model_df.set_index('point')
        for col in model_df.columns:
            if col == 'point':
                continue
            if col.startswith('beta_'):
                new_col = f'beta_{prefix}_{col[5:]}'
            elif col.startswith('slope_'):
                new_col = f'slope_{prefix}_{col[6:]}'
            elif col.startswith('p_') and col != 'p_raw':
                new_col = f'pval_{prefix}_{col[2:]}'
            elif col == 'p_raw':
                new_col = f'p_raw_{prefix}'
            elif col == 'f_stat':
                new_col = f'f_stat_{prefix}'
            else:
                continue
            result[new_col] = result['point'].map(model_indexed[col])
    return result


def _export_ols_csv(ols_df, metric_name, bundle_name, centroid_id, var_name,
                    out_csv, analysis_label, poly_degree=POLY_DEGREE):
    if ols_df is None or ols_df.empty:
        pd.DataFrame({'info': ['aucune donnée'],
                      'bundle': [bundle_name],
                      'centroid_id': [centroid_id],
                      'metric': [metric_name],
                      'variable': [var_name],
                      'analysis': [analysis_label],
                      'model': [f'poly_deg{poly_degree}']}).to_csv(out_csv, index=False)
        return
    df_out = ols_df.copy()
    df_out['bundle'] = bundle_name
    df_out['centroid_id'] = centroid_id
    df_out['metric'] = metric_name
    df_out['variable'] = var_name
    df_out['analysis'] = analysis_label
    if 'best_model' not in df_out.columns:
        df_out['model'] = f'poly_deg{poly_degree}'
    df_out.to_csv(out_csv, index=False)


def plot_ols_results(ols_df: pd.DataFrame, metric_name: str, bundle_name: str,
                     var_name: str, out_path: str, title_suffix: str,
                     removed_subjects=None, removed_points=None, export_csv=True,
                     canonical_bundle=None, centroid_id=None, poly_degree=POLY_DEGREE,
                     model_dfs: Dict[str, pd.DataFrame] = None):
    """
    Trace les résultats OLS le long du faisceau.

    Si model_dfs est fourni (dict nom→DataFrame pour chaque modèle candidat),
    génère une grille de subfigures : une colonne par modèle montrant
    R² partiel + -log10(p) + clusters, plus une ligne finale AIC comparatif.

    Sinon (modèle unique), trace 2 panneaux classiques.
    """
    if model_dfs is None or len(model_dfs) == 0:
        model_dfs = {}

    has_multi = len(model_dfs) >= 2 and not ols_df.empty

    if not has_multi:
        _plot_single_model(ols_df, metric_name, bundle_name, var_name, out_path,
                           title_suffix, removed_subjects, removed_points,
                           poly_degree=poly_degree)
    else:
        _plot_multi_model(ols_df, model_dfs, metric_name, bundle_name, var_name,
                          out_path, title_suffix, removed_subjects, removed_points,
                          poly_degree=poly_degree)


    if export_csv:
        out_csv = os.path.splitext(out_path)[0] + '.csv'
        analysis_lbl = 'ols_nonlinear'
        bundle_for_csv = canonical_bundle if canonical_bundle else bundle_name
        if centroid_id is None:
            centroid_id = (ols_df.get('centroid_id', pd.Series([np.nan])).iloc[0]
                           if isinstance(ols_df, pd.DataFrame) and 'centroid_id' in ols_df.columns
                           and not ols_df['centroid_id'].isna().all()
                           else np.nan)
        # Enrichir le DataFrame avec les betas/pvals préfixés de chaque modèle
        export_df = ols_df
        if not ols_df.empty and 'point' in ols_df.columns:
            if has_multi and model_dfs:
                # Multi-modèle : ajouter beta_lin_*, beta_poly_*, beta_int_*, pval_*, slope_*
                export_df = _add_model_betas_to_df(ols_df, model_dfs)
            elif not ols_df.empty and 'model_type' in ols_df.columns:
                # Mono-modèle : préfixer selon le type
                mt = ols_df['model_type'].iloc[0]
                if mt in _MODEL_PREFIX_MAP:
                    export_df = _add_model_betas_to_df(ols_df, {mt: ols_df})
        _export_ols_csv(export_df, metric_name, bundle_for_csv, centroid_id, var_name,
                        out_csv, analysis_lbl, poly_degree)


def _plot_single_model(df, metric_name, bundle_name, var_name, out_path,
                       title_suffix, removed_subjects, removed_points,
                       poly_degree=POLY_DEGREE):
    """Plot pour un seul modèle (2 panneaux : R² partiel + -log10(p))."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
    if df.empty:
        axes[0].text(0.5, 0.5, "Aucune donnée", ha='center', va='center',
                     transform=axes[0].transAxes)
        axes[0].set_title(f"{bundle_name} - {metric_name} - OLS({var_name}) {title_suffix}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    ax_r2, ax_p = axes

    # R² partiel
    if FWE_METHOD in ('clusterFWE', 'mixed') and 'sig_afq' in df.columns:
        _highlight_clusters(ax_r2, df['point'], df['sig_afq'])
    ax_r2.fill_between(df['point'], 0, df['r2_partial'], alpha=0.25, color='tab:blue',
                       label='R² partiel')
    ax_r2.plot(df['point'], df['r2_partial'], color='tab:blue', linewidth=1.5)
    ax_r2.set_ylabel("R² partiel")
    ax_r2.axhline(0, color='black', lw=0.5)
    if 'sig_afq' in df.columns and df['sig_afq'].any():
        sig = df[df['sig_afq']]
        ax_r2.scatter(sig['point'], sig['r2_partial'], color='red', s=25,
                      zorder=5, label='significatif')
    ax_r2.legend(loc='upper right', fontsize=7)
    terms = ' + '.join([f'β{d}·var{"²" if d == 2 else "³" if d == 3 else f"^{d}" if d > 3 else ""}'
                        for d in range(1, poly_degree + 1)])
    model_str = f"metric = β0 + {terms} + confonds + ε"
    ax_r2.text(0.01, 0.95, model_str, transform=ax_r2.transAxes, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'),
               verticalalignment='top')
    ax_r2.set_title(f"{bundle_name} - {metric_name} - OLS({var_name}) {title_suffix} ({FWE_METHOD})")
    _annotate_missing(ax_r2, removed_subjects or [], removed_points or [])

    # -log10(p)
    _plot_neg_log10_pval(ax_p, df)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_neg_log10_pval(ax, df, show_xlabel=True):
    """Trace -log10(p) sur un axe avec seuils FWE."""
    pvals = df['p_raw'].clip(lower=1e-300).values
    neg_log10 = -np.log10(pvals)

    ax.bar(df['point'], neg_log10, color='gray', alpha=0.4, width=1.0, label='-log₁₀(p)')
    ax.set_ylabel("-log₁₀(p)")
    if show_xlabel:
        ax.set_xlabel("Point")

    # Seuil α = 0.05
    ax.axhline(-np.log10(0.05), color='orange', linestyle='--', linewidth=0.8,
               alpha=0.5, label='α=0.05')

    # Seuil alphaFWE
    if 'alphaFWE' in df.columns and not df['alphaFWE'].isna().all():
        aFWE = df['alphaFWE'].iloc[0]
        if not np.isnan(aFWE) and aFWE > 0 and FWE_METHOD in ('alphaFWE', 'mixed'):
            ax.axhline(-np.log10(aFWE), color='red', linestyle=':', linewidth=1.2,
                       label=f'alphaFWE={aFWE:.3g}')

    if 'clusterFWE' in df.columns and not df['clusterFWE'].isna().all():
        cFWE = df['clusterFWE'].iloc[0]
        if not np.isnan(cFWE) and FWE_METHOD in ('clusterFWE', 'mixed'):
            ax.text(0.01, 0.90, f"clusterFWE≥{int(cFWE)}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray'))

    if FWE_METHOD in ('clusterFWE', 'mixed') and 'sig_afq' in df.columns:
        _highlight_clusters(ax, df['point'], df['sig_afq'])

    ax.legend(loc='upper right', fontsize=6)


def _plot_multi_model(best_df, model_dfs, metric_name, bundle_name, var_name,
                      out_path, title_suffix, removed_subjects, removed_points,
                      poly_degree=POLY_DEGREE):
    """
    Grille de subfigures : une colonne par modèle.
    Chaque colonne a 2 lignes (R² partiel, -log10(p)).
    Dernière ligne commune : AIC comparatif.
    """
    model_names = [n for n in ['linear', 'polynomial', 'interaction'] if n in model_dfs]
    n_models = len(model_names)

    if n_models == 0:
        return

    fig = plt.figure(figsize=(6 * n_models, 12))
    gs = fig.add_gridspec(3, n_models, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)

    cluster_summaries = compute_per_model_cluster_summary(model_dfs)

    for col_idx, model_name in enumerate(model_names):
        df_m = model_dfs[model_name]
        if df_m is None or df_m.empty:
            continue

        color = _MODEL_COLORS.get(model_name, 'tab:gray')
        label = _MODEL_LABELS.get(model_name, model_name)

        # --- R² partiel ---
        ax_r2 = fig.add_subplot(gs[0, col_idx])
        if FWE_METHOD in ('clusterFWE', 'mixed') and 'sig_afq' in df_m.columns:
            _highlight_clusters(ax_r2, df_m['point'], df_m['sig_afq'])
        ax_r2.fill_between(df_m['point'], 0, df_m['r2_partial'], alpha=0.25,
                           color=color, label='R² partiel')
        ax_r2.plot(df_m['point'], df_m['r2_partial'], color=color, linewidth=1.5)
        ax_r2.set_ylabel("R² partiel" if col_idx == 0 else "")
        ax_r2.axhline(0, color='black', lw=0.5)
        if 'sig_afq' in df_m.columns and df_m['sig_afq'].any():
            sig = df_m[df_m['sig_afq']]
            ax_r2.scatter(sig['point'], sig['r2_partial'], color='red', s=20,
                          zorder=5, label='sig.')

        # Annotation cluster summary
        cs = cluster_summaries.get(model_name)
        if cs and cs['n_clusters'] > 0:
            info_lines = [f"{cs['n_clusters']} cluster(s), {cs['total_sig_points']} pts sig."]
            for i, cl in enumerate(cs['clusters'][:4]):
                info_lines.append(
                    f"  C{i+1}: pts {cl['start_point']}-{cl['end_point']} "
                    f"(n={cl['size']}, R²={cl['mean_r2_partial']:.3f}, "
                    f"-log₁₀p={cl['mean_neg_log10p']:.1f})")
            if len(cs['clusters']) > 4:
                info_lines.append(f"  ... +{len(cs['clusters'])-4} autres")
            ax_r2.text(0.02, 0.98, '\n'.join(info_lines), transform=ax_r2.transAxes,
                       fontsize=6, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax_r2.set_title(f"{model_name}\n{label}", fontsize=8)
        ax_r2.legend(loc='upper right', fontsize=6)

        # --- -log10(p) ---
        ax_p = fig.add_subplot(gs[1, col_idx], sharex=ax_r2)
        _plot_neg_log10_pval(ax_p, df_m, show_xlabel=True)

    # --- Ligne AIC (couvre toutes les colonnes) ---
    ax_aic = fig.add_subplot(gs[2, :])
    for model_name in model_names:
        df_m = model_dfs[model_name]
        if df_m is not None and not df_m.empty and 'aic' in df_m.columns:
            ax_aic.plot(df_m['point'], df_m['aic'], color=_MODEL_COLORS[model_name],
                        linewidth=1.2, alpha=0.8, label=f'AIC {model_name}')

    # Colorer le fond selon le best_model
    if 'best_model' in best_df.columns and not best_df.empty:
        for i in range(len(best_df) - 1):
            row_i = best_df.iloc[i]
            x0 = row_i['point']
            x1 = best_df.iloc[i + 1]['point']
            bg = _MODEL_BG.get(row_i['best_model'], 'white')
            ax_aic.axvspan(x0, x1, color=bg, alpha=0.22, linewidth=0)

        # Compter les modèles
        counts = {m: int((best_df['best_model'] == m).sum()) for m in model_names}
        counts_str = ' | '.join([f'{m}: {c}' for m, c in counts.items() if c > 0])
        ax_aic.text(0.01, 0.95, f"Meilleur modèle (AIC) — {counts_str} pts",
                    transform=ax_aic.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    ax_aic.set_ylabel("AIC")
    ax_aic.set_xlabel("Point")
    ax_aic.legend(loc='upper right', fontsize=7)

    fig.suptitle(f"{bundle_name} - {metric_name} - OLS({var_name}) {title_suffix} ({FWE_METHOD})",
                 fontsize=10, y=0.99)
    _annotate_missing(fig.axes[0], removed_subjects or [], removed_points or [])

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
