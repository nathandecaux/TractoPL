"""
Dashboard Streamlit pour la visualisation des statistiques de faisceaux.
Remplace le dashboard Panel/HoloViews par une interface plus simple et robuste.

Usage:
    streamlit run actiDep/ui/streamlit_dashboard.py
"""

import os
from os.path import join as opj
import glob
import json
import re
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import f as f_dist
from scipy.interpolate import interp1d
import statsmodels.api as sm
from TractoPL.analysis.AFQ_analysis import interpolate_missing_points, resample_bundle_data

def get_dataset_paths(dataset_path: str) -> dict:
    """Return conventional optional metadata paths for a BIDS dataset root."""
    base = os.path.abspath(os.path.expanduser(dataset_path))
    return {
        'dataset_path': base,
        'subjects_file': os.path.join(base, 'subjects.txt'),
        'participants_file': f"{base}/participants_full_info.xlsx",
        'actimetry_file': f"{base}/actimetry_features.xlsx",
    }


# =============================================================================
# Utility Functions
# =============================================================================

def color_to_rgba(color: str, alpha: float = 1.0) -> str:
    """
    Convertit une couleur (hex ou rgb) en format rgba.
    Gère les formats: '#RRGGBB', 'rgb(r, g, b)', 'rgba(r, g, b, a)'
    """
    if color.startswith('rgba'):
        # Déjà en rgba, juste remplacer l'alpha
        match = re.match(r'rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)', color)
        if match:
            r, g, b = match.groups()
            return f'rgba({r}, {g}, {b}, {alpha})'
        return color
    elif color.startswith('rgb'):
        # Format rgb(r, g, b)
        match = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color)
        if match:
            r, g, b = match.groups()
            return f'rgba({r}, {g}, {b}, {alpha})'
        return color
    elif color.startswith('#'):
        # Format hex
        try:
            rgb = px.colors.hex_to_rgb(color)
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
        except Exception:
            return f'rgba(128, 128, 128, {alpha})'
    else:
        # Couleur nommée ou autre - fallback
        return f'rgba(128, 128, 128, {alpha})'


# =============================================================================
# Configuration
# =============================================================================

MODEL = 'MCM'
DEFAULT_PIPELINE = 'tractometry'
TRACTOMETRY_MARKER = '.tag_tractometry'

METRICS = ['FA', 'MD', 'RD', 'AD', 'IFW', 'IRF']
STAT_TYPES = ['mean', 'std', 'median', 'gmm_mean', 'gmm_std', 'gmm_prec']
INTERPOLATION_METHODS = ['linear', 'nearest', 'cubic']
CORRELATION_METHODS = ['pearson', 'spearman']
CORRECTION_METHODS = ['unconfound', 'ols']
MULTIPLE_CORRECTION_METHODS = ['none', 'bonferroni', 'fdr_bh']
OLS_MODEL_DEGREES = [1, 2, 3]  # Degrés polynomiaux disponibles pour le modèle OLS

# Styles de lignes pour le mode comparaison
PIPELINE_DASH_STYLES = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
PIPELINE_COLORS = px.colors.qualitative.Set2


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(ttl=3600)
def get_available_pipelines(dataset_path: str) -> List[str]:
    """Detect tractometry derivatives from explicit metadata or metric outputs."""
    derivatives_path = opj(dataset_path, 'derivatives')
    if not os.path.exists(derivatives_path):
        return []
    
    pipelines = []
    for pipeline_dir in glob.glob(opj(derivatives_path, '*')):
        if not os.path.isdir(pipeline_dir):
            continue
        pipeline_name = os.path.basename(pipeline_dir)
        description_path = opj(pipeline_dir, 'dataset_description.json')
        is_tractometry = os.path.isfile(opj(pipeline_dir, TRACTOMETRY_MARKER))
        try:
            with open(description_path, encoding='utf-8') as description_file:
                description = json.load(description_file)
            pipeline_description = description.get('PipelineDescription', {})
            description_name = pipeline_description.get('Name', description.get('Name', ''))
            is_tractometry = is_tractometry or 'tractometry' in description_name.lower()
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        metric_pattern = opj(
            pipeline_dir,
            'sub-*',
            'metric',
            '*_bundle-*_model-*_*.csv',
        )
        is_tractometry = is_tractometry or bool(glob.glob(metric_pattern))
        is_tractometry = is_tractometry or pipeline_name.startswith('hcp_association')
        if is_tractometry:
            pipelines.append(pipeline_name)
    return sorted(pipelines)

@st.cache_data(ttl=3600)
def load_subjects_from_file(filepath: str) -> set:
    """Charge la liste des sujets depuis le fichier subjects.txt"""
    valid_subjects = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('subject_id'):
                    parts = line.split()
                    if len(parts) >= 1:
                        subject_id = parts[0].replace('sub-', '')
                        valid_subjects.add(subject_id)
    except Exception as e:
        st.warning(f"Erreur lors de la lecture de {filepath}: {e}")
    return valid_subjects


@st.cache_data(ttl=3600)
def load_participants_info(filepath: str) -> Optional[pd.DataFrame]:
    """Charge les informations des participants depuis le fichier Excel."""
    try:
        df = pd.read_excel(filepath)
        if 'participant_id' in df.columns:
            df['subject_id'] = df['participant_id'].str.replace('sub-', '', regex=False)
        elif 'subject_id' not in df.columns:
            st.warning("Aucune colonne 'participant_id' ou 'subject_id' trouvée")
            return None
        return df
    except FileNotFoundError:
        st.warning(f"Fichier participants non trouvé: {filepath}")
        return None
    except Exception as e:
        st.warning(f"Erreur lors du chargement des participants: {e}")
        return None


@st.cache_data(ttl=3600)
def load_actimetry_data(filepath: str) -> Optional[pd.DataFrame]:
    """Charge les données d'actimétrie depuis le fichier Excel."""
    try:
        df = pd.read_excel(filepath)
        if 'participant_id' in df.columns:
            df['subject_id'] = df['participant_id'].str.replace('sub-', '', regex=False)
        elif 'subject_id' not in df.columns:
            st.warning("Aucune colonne 'participant_id' ou 'subject_id' trouvée dans actimétrie")
            return None
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Erreur lors du chargement de l'actimétrie: {e}")
        return None


@st.cache_data(ttl=3600)
def load_bundle_data(
    dataset_path: str,
    pipeline: str,
    model: str,
    valid_subjects: Optional[set] = None,
    filter_by_subjects: bool = False
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Charge les données CSV _stats.csv pour tous les faisceaux.
    Retourne (DataFrame, liste des faisceaux, liste des sujets).
    """
    pattern = opj(dataset_path, 'derivatives', pipeline, 'sub-*', 'metric',
                  f'*_bundle-*_model-{model}*_stats.csv')
    csv_files = glob.glob(pattern)

    #if csv_files is empty, try with mean.csv
    if not csv_files:
        pattern = opj(dataset_path, 'derivatives', pipeline, 'sub-*', 'metric',
                      f'*_bundle-*_model-{model}*_mean.csv')
        csv_files = glob.glob(pattern)
    print(f"Found {len(csv_files)} CSV files for bundles. Pattern: {pattern}")
    all_data = []
    bundle_names = set()
    subjects = set()
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        subject_match = re.search(r'sub-([a-zA-Z0-9]+)', filename)
        bundle_match = re.search(r'_bundle-([a-zA-Z0-9]+)_', filename)
        
        if not (subject_match and bundle_match):
            continue
            
        subject_id = subject_match.group(1)
        bundle_name = bundle_match.group(1)
        
        # Filtrage par sujets valides
        if filter_by_subjects and valid_subjects and subject_id not in valid_subjects:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            df['subject'] = subject_id
            df['bundle'] = bundle_name
            all_data.append(df)
            bundle_names.add(bundle_name)
            subjects.add(subject_id)
        except Exception:
            continue
    
    if not all_data:
        return pd.DataFrame(), [], []
    
    df_all = pd.concat(all_data, ignore_index=True)
    return df_all, sorted(list(bundle_names)), sorted(list(subjects))


def merge_participant_data(
    df: pd.DataFrame,
    participants_info: Optional[pd.DataFrame],
    actimetry_data: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Fusionne les données de faisceaux avec les infos participants et actimétrie."""
    if participants_info is not None and not df.empty:
        df = df.merge(participants_info, left_on='subject', right_on='subject_id', how='left')
    if actimetry_data is not None and not df.empty:
        df = df.merge(actimetry_data, left_on='subject', right_on='subject_id', how='left')
    return df


# =============================================================================
# Data Processing Functions
# =============================================================================

def resample_curve(
    x: np.ndarray,
    y: np.ndarray,
    n_points: int,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """Rééchantillonne une courbe à n_points points."""
    if len(x) < 2:
        return x, y
    
    # Supprimer les doublons
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    
    x_new = np.linspace(xu.min(), xu.max(), n_points)
    
    if len(xu) == 1:
        return x_new, np.full_like(x_new, yu[0])
    
    if method == 'cubic' and len(xu) >= 4:
        try:
            f = interp1d(xu, yu, kind='cubic', bounds_error=False, fill_value='extrapolate')
            y_new = f(x_new)
        except Exception:
            y_new = np.interp(x_new, xu, yu)
    elif method == 'nearest':
        idxs = np.searchsorted(xu, x_new, side='left')
        idxs = np.clip(idxs, 0, len(xu) - 1)
        left_idx = np.clip(idxs - 1, 0, len(xu) - 1)
        right_idx = idxs
        choose_left = np.abs(x_new - xu[left_idx]) <= np.abs(xu[right_idx] - x_new)
        nearest_idx = np.where(choose_left, left_idx, right_idx)
        y_new = yu[nearest_idx]
    else:  # linear
        y_new = np.interp(x_new, xu, yu)
    
    return x_new, y_new


def resample_bundle_df(
    df: pd.DataFrame,
    n_points: int,
    method: str = 'linear'
) -> pd.DataFrame:
    """Rééchantillonne toutes les courbes d'un DataFrame de bundle."""
    # df['subject'] = df['subject_id_x']
    print(df.head(2),"1")
    print(df.shape)
    print(sorted(df.columns),"1")
    cols2keep= ['point_id', 'subject', 'value','centroid_id']
    other_df=resample_bundle_data(df[cols2keep], n_points=n_points)
    #other_df = other_df.merge(df.drop(columns=[c for c in cols2keep if c not in ['subject']]), on=['subject'], how='left')
    new_df=other_df
    # new_df=[]
    # for b in df['bundle'].unique():
    #     _df=df[df['bundle']==b][cols2keep]
    #     other_df=resample_bundle_data(_df, n_points=n_points)
    #     other_df['bundle']=b
    #     other_df = other_df.merge(df.drop(columns=[c for c in cols2keep if c not in ['subject','bundle']]), on=['subject','bundle'], how='left')
    #     new_df.append(other_df)
    # new_df=pd.concat(new_df, ignore_index=True)
    print(new_df.head(2),"qz2")
    print(n_points,new_df['point_id'].nunique())
    #Drop value == 0
    # new_df=new_df[new_df['value']!=0]
    return new_df

    # if n_points < 2:
    #     return df
    
    # resampled = []
    # cols_to_keep = [c for c in df.columns if c not in ['point_id', 'value']]
    
    # for subject_id, g in df.groupby('subject'):
    #     g = g.sort_values('point_id')
    #     x = g['point_id'].to_numpy(dtype=float)
    #     y = g['value'].to_numpy(dtype=float)
        
    #     if len(x) == 0:
    #         continue
        
    #     x_new, y_new = resample_curve(x, y, n_points, method)
        
    #     base_vals = {col: g.iloc[0][col] for col in cols_to_keep if col in g.columns}
    #     new_df = pd.DataFrame({
    #         **base_vals,
    #         'point_id': x_new,
    #         'value': y_new
    #     })
    #     resampled.append(new_df)
    
    # return pd.concat(resampled, ignore_index=True) if resampled else df


def apply_nuisance_correction_ols(
    df: pd.DataFrame,
    nuisance_cols: List[str],
    group_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Applique la correction de nuisances par OLS point par point."""
    df = df.copy()
    
    # Préparer encodage
    enc = {}
    for c in nuisance_cols:
        if c == group_col:
            continue
        s = df[c]
        if s.dtype == 'object' or str(s.dtype).startswith('category'):
            enc[c] = s.astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            enc[c] = pd.to_numeric(s, errors='coerce')
    
    if not enc:
        return df, {'status': 'skipped', 'reason': 'Aucune colonne de nuisance utilisable'}
    
    enc_df = pd.DataFrame(enc, index=df.index)
    
    # Identifier sujets valides (sans NaN dans nuisances)
    subjects_meta = pd.DataFrame(enc).groupby(df['subject']).first()
    valid_subjects_mask = subjects_meta.notna().all(axis=1)
    valid_subjects = valid_subjects_mask[valid_subjects_mask].index.tolist()
    n_excluded = len(valid_subjects_mask) - len(valid_subjects)
    
    if len(valid_subjects) < 3:
        return df, {
            'status': 'failed',
            'reason': f'Pas assez de sujets avec nuisances valides ({len(valid_subjects)} < 3)',
            'n_excluded': n_excluded
        }
    
    # Filtrer
    df_valid = df[df['subject'].isin(valid_subjects)].copy()
    enc_df_valid = enc_df.loc[df_valid.index]
    
    corrected_vals = {}
    n_points_total = 0
    n_points_corrected = 0
    
    for pid, idx in df_valid.groupby('point_id').groups.items():
        n_points_total += 1
        sub_idx = pd.Index(idx)
        y = pd.to_numeric(df_valid.loc[sub_idx, 'value'], errors='coerce')
        X = enc_df_valid.loc[sub_idx]
        
        valid_cols = [c for c in X.columns if X[c].notna().sum() >= 3 and X[c].nunique(dropna=True) > 1]
        if not valid_cols:
            continue
        
        Xi = X[valid_cols].apply(lambda col: col.fillna(col.mean()), axis=0)
        
        try:
            Xi_const = sm.add_constant(Xi, has_constant='add')
            model = sm.OLS(y, Xi_const, missing='drop').fit()
            resid = model.resid
            const_val = float(model.params.get('const', 0.0))
            y_corr = resid + const_val
            corrected_vals[(pid,)] = (y_corr.index, y_corr.values)
            n_points_corrected += 1
        except Exception:
            continue
    
    if not corrected_vals:
        return df_valid, {
            'status': 'failed',
            'reason': f'Aucun point corrigé sur {n_points_total} points',
            'n_points_total': n_points_total,
            'n_excluded': n_excluded
        }
    
    for (pid,), (indices, vals) in corrected_vals.items():
        df_valid.loc[indices, 'value'] = vals
    
    return df_valid, {
        'status': 'success',
        'n_points_total': n_points_total,
        'n_points_corrected': n_points_corrected,
        'n_subjects_total': len(valid_subjects) + n_excluded,
        'n_subjects_used': len(valid_subjects),
        'n_excluded': n_excluded,
        'nuisance_cols_used': list(enc.keys())
    }


def _ols_residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """
    Retourne résidus + intercept pour conserver le niveau moyen.
    Approche identique à generate_report_v2.py.
    """
    df_combined = pd.concat([y, X], axis=1)
    df_combined = df_combined.dropna()
    if df_combined.empty:
        return pd.Series(index=y.index, data=np.nan)
    y_clean = df_combined.iloc[:, 0]
    X_clean = df_combined.iloc[:, 1:]
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


def _residualize_on_confond(df_subject_level: pd.DataFrame, target_col: str, confond: List[str]) -> pd.Series:
    """
    Résidualise une colonne cible sur les confondants.
    Approche identique à generate_report_v2.py.
    """
    cols = [c for c in confond if c in df_subject_level.columns]
    if not cols:
        return df_subject_level[target_col]
    X = df_subject_level[cols].copy()
    # Encodage catégoriel
    for c in cols:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return _ols_residualize(df_subject_level[target_col], X)


def apply_nuisance_correction_unconfound(
    df: pd.DataFrame,
    nuisance_cols: List[str],
    group_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applique la correction de nuisances via OLS point par point.
    Approche identique à generate_report_v2.py: résidualisation avec conservation du niveau moyen.
    """
    if not nuisance_cols:
        return df, {'status': 'skipped', 'reason': 'Aucune colonne de nuisance'}

    df = df.copy()
    
    # Confondants à utiliser (exclure la colonne de groupe si spécifiée)
    conf_list = [c for c in nuisance_cols if c != group_col]
    
    if not conf_list:
        return df, {'status': 'skipped', 'reason': 'Aucune colonne de nuisance après exclusion du groupe'}
    
    # Vérifier que les colonnes existent
    available_cols = [c for c in conf_list if c in df.columns]
    if not available_cols:
        return df, {'status': 'failed', 'reason': f'Colonnes de nuisance non trouvées: {conf_list}'}
    
    n_points_total = 0
    n_points_corrected = 0
    n_subjects_total = df['subject'].nunique()
    subjects_used = set()
    
    # Boucle point par point (comme dans generate_report_v2.py)
    corrected_segments = []
    
    for pid, gpt in df.groupby('point_id'):
        n_points_total += 1
        
        # Construire DataFrame au niveau sujet pour ce point
        cols_to_keep = ['subject', 'value'] + [c for c in available_cols if c in gpt.columns]
        subj_level = (gpt[cols_to_keep]
                      .drop_duplicates(subset='subject')
                      .set_index('subject'))
        
        # Vérifier qu'on a assez de sujets avec données valides
        valid_mask = subj_level['value'].notna()
        for c in available_cols:
            if c in subj_level.columns:
                valid_mask &= subj_level[c].notna()
        
        n_valid = valid_mask.sum()
        if n_valid < 3:
            # Pas assez de sujets pour corriger ce point, conserver les valeurs originales
            corrected_segments.append(gpt.copy())
            continue
        
        try:
            # Résidualisation OLS (comme generate_report_v2.py)
            y_res = _residualize_on_confond(
                subj_level.assign(target=subj_level['value']),
                'target',
                available_cols
            )
            
            # Appliquer les valeurs corrigées
            g_corr = gpt.copy()
            g_corr['value'] = g_corr['subject'].map(y_res)
            corrected_segments.append(g_corr)
            
            # Comptabiliser les sujets utilisés
            subjects_used.update(y_res.dropna().index.tolist())
            n_points_corrected += 1
            
        except Exception:
            # En cas d'erreur, conserver les valeurs originales
            corrected_segments.append(gpt.copy())
            continue
    
    if not corrected_segments:
        return df, {
            'status': 'failed',
            'reason': f'Aucun point corrigé sur {n_points_total} points',
            'n_points_total': n_points_total
        }
    
    df_corrected = pd.concat(corrected_segments, ignore_index=True)
    
    return df_corrected, {
        'status': 'success',
        'n_points_total': n_points_total,
        'n_points_corrected': n_points_corrected,
        'n_subjects_total': n_subjects_total,
        'n_subjects_used': len(subjects_used),
        'n_excluded': n_subjects_total - len(subjects_used),
        'nuisance_cols_used': available_cols,
        'method': 'ols_residualize_pointwise'
    }


def compute_pointwise_correlation(
    df: pd.DataFrame,
    column: str,
    method: str = 'pearson'
) -> pd.DataFrame:
    """Calcule r et p pour chaque point entre 'value' et la colonne spécifiée."""
    if column not in df.columns:
        return pd.DataFrame(columns=['point_id', 'r', 'p', 'n'])
    
    # Encoder si catégoriel
    s = df[column]
    if s.dtype == 'object' or str(s.dtype).startswith('category'):
        x_full = s.astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        x_full = pd.to_numeric(s, errors='coerce')
    
    rows = []
    for pid, idx in df.groupby('point_id').groups.items():
        idx = pd.Index(idx)
        y = pd.to_numeric(df.loc[idx, 'value'], errors='coerce')
        x = x_full.loc[idx]
        
        mask = x.notna() & y.notna()
        xv = x[mask].values
        yv = y[mask].values
        n = int(mask.sum())
        
        if n < 3:
            rows.append({'point_id': pid, 'r': np.nan, 'p': np.nan, 'n': n})
            continue
        
        try:
            if method == 'spearman':
                res = stats.spearmanr(xv, yv, nan_policy='omit')
                r, p = res.correlation, res.pvalue
            else:
                r, p = stats.pearsonr(xv, yv)
        except Exception:
            r, p = np.nan, np.nan
        
        rows.append({'point_id': pid, 'r': r, 'p': p, 'n': n})
    
    return pd.DataFrame(rows).sort_values('point_id')


def compute_pointwise_ols_model(
    df: pd.DataFrame,
    var_col: str,
    confound_cols: List[str],
    poly_degree: int = 2
) -> pd.DataFrame:
    """
    À chaque point le long du faisceau, ajuste un modèle OLS polynomial :

        metric = β0 + β1·var + β2·var² + … + βk·confond_k + ε

    et évalue la significativité des termes polynomiaux via un test F partiel
    (modèle complet vs modèle réduit = confonds seuls).

    Returns:
        DataFrame avec colonnes :
            point_id, f_stat, p, r2, r2_partial,
            beta_deg1, beta_deg2, …, n
    """
    if var_col not in df.columns:
        return pd.DataFrame()

    # Encoder la variable d'intérêt
    s = df[var_col]
    if s.dtype == 'object' or str(s.dtype).startswith('category'):
        var_full = s.astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        var_full = pd.to_numeric(s, errors='coerce')

    # Encoder les confonds
    conf_encoded = {}
    available_confs = [c for c in confound_cols if c in df.columns and c != var_col]
    for c in available_confs:
        sc = df[c]
        if sc.dtype == 'object' or str(sc.dtype).startswith('category'):
            conf_encoded[c] = sc.astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            conf_encoded[c] = pd.to_numeric(sc, errors='coerce')

    rows = []
    for pid, idx in df.groupby('point_id').groups.items():
        idx = pd.Index(idx)
        y = pd.to_numeric(df.loc[idx, 'value'], errors='coerce')
        x_var = var_full.loc[idx]

        # Masque de validité
        mask = y.notna() & x_var.notna()
        for c in available_confs:
            mask = mask & conf_encoded[c].loc[idx].notna()

        n = int(mask.sum())
        min_obs = poly_degree + len(available_confs) + 2
        if n < min_obs:
            row = {'point_id': pid, 'f_stat': np.nan, 'p': np.nan,
                   'r2': np.nan, 'r2_partial': np.nan, 'n': n}
            for d in range(1, poly_degree + 1):
                row[f'beta_deg{d}'] = np.nan
            rows.append(row)
            continue

        yv = y[mask].values
        xv = x_var[mask].values

        # Termes polynomiaux
        poly_cols_arr = [xv ** d for d in range(1, poly_degree + 1)]

        # Confonds
        conf_arr_list = [conf_encoded[c].loc[idx][mask].values for c in available_confs]

        # Matrices de design
        X_reduced = np.column_stack([np.ones(n)] + conf_arr_list) if conf_arr_list else np.ones((n, 1))
        X_full = np.column_stack([np.ones(n)] + poly_cols_arr + conf_arr_list) if conf_arr_list else np.column_stack([np.ones(n)] + poly_cols_arr)

        df_num = poly_degree
        df_denom = n - X_full.shape[1]

        if df_denom <= 0:
            row = {'point_id': pid, 'f_stat': np.nan, 'p': np.nan,
                   'r2': np.nan, 'r2_partial': np.nan, 'n': n}
            for d in range(1, poly_degree + 1):
                row[f'beta_deg{d}'] = np.nan
            rows.append(row)
            continue

        try:
            # Modèle réduit
            beta_red, _, _, _ = np.linalg.lstsq(X_reduced, yv, rcond=None)
            ssr_red = np.sum((yv - X_reduced @ beta_red) ** 2)

            # Modèle complet
            beta_full, _, _, _ = np.linalg.lstsq(X_full, yv, rcond=None)
            ssr_full = np.sum((yv - X_full @ beta_full) ** 2)

            # F-stat
            f_stat = ((ssr_red - ssr_full) / df_num) / (ssr_full / df_denom) if ssr_full > 0 else 0.0
            f_stat = max(0.0, f_stat)
            p_val = 1.0 - f_dist.cdf(f_stat, df_num, df_denom)

            # R²
            ss_tot = np.sum((yv - yv.mean()) ** 2)
            r2_full = 1.0 - ssr_full / ss_tot if ss_tot > 0 else 0.0
            r2_red = 1.0 - ssr_red / ss_tot if ss_tot > 0 else 0.0
            r2_partial = max(0.0, r2_full - r2_red)

            row = {'point_id': pid, 'f_stat': f_stat, 'p': p_val,
                   'r2': r2_full, 'r2_partial': r2_partial, 'n': n}
            for d in range(1, poly_degree + 1):
                row[f'beta_deg{d}'] = beta_full[d]  # col 0 = intercept, col 1..deg = var

        except Exception:
            row = {'point_id': pid, 'f_stat': np.nan, 'p': np.nan,
                   'r2': np.nan, 'r2_partial': np.nan, 'n': n}
            for d in range(1, poly_degree + 1):
                row[f'beta_deg{d}'] = np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('point_id')


def compute_pointwise_interaction_model(
    df: pd.DataFrame,
    var_col: str,
    group_col: str,
    confound_cols: List[str],
) -> pd.DataFrame:
    """
    À chaque point le long du faisceau, ajuste un modèle linéaire avec
    un slope différent par groupe (interaction var × group) :

        metric = β0 + β_var·var + Σ β_g·group_g + Σ γ_g·var·group_g + confonds + ε

    Le test F partiel évalue si les termes liés à var (β_var et γ_g)
    sont significatifs par rapport au modèle réduit (group + confonds seuls).

    Returns:
        DataFrame avec colonnes :
            point_id, f_stat, p, r2, r2_partial,
            beta_var, slope_<group>, ..., n, groups
    """
    if var_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    # Encoder la variable d'intérêt
    s = df[var_col]
    if s.dtype == 'object' or str(s.dtype).startswith('category'):
        var_full = s.astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        var_full = pd.to_numeric(s, errors='coerce')

    # Encoder le groupe
    sg = df[group_col]
    if sg.dtype == 'object' or str(sg.dtype).startswith('category'):
        group_full = sg.astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        group_full = pd.to_numeric(sg, errors='coerce')

    unique_groups = sorted(group_full.dropna().unique())
    if len(unique_groups) < 2:
        return pd.DataFrame()

    # Mapping inversé pour les noms de groupe
    # Tenter de retrouver les labels originaux
    group_labels = {}
    for gval in unique_groups:
        original_vals = df.loc[group_full == gval, group_col].dropna().unique()
        group_labels[gval] = str(original_vals[0]) if len(original_vals) > 0 else str(int(gval))

    # Encoder les confonds
    conf_encoded = {}
    available_confs = [c for c in confound_cols if c in df.columns and c != var_col and c != group_col]
    for c in available_confs:
        sc = df[c]
        if sc.dtype == 'object' or str(sc.dtype).startswith('category'):
            conf_encoded[c] = sc.astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            conf_encoded[c] = pd.to_numeric(sc, errors='coerce')

    n_groups = len(unique_groups)
    n_dummies = n_groups - 1  # reference coding
    # var_terms = 1 (var) + n_dummies (interactions)
    n_var_terms = 1 + n_dummies

    rows = []
    for pid, idx in df.groupby('point_id').groups.items():
        idx = pd.Index(idx)
        y = pd.to_numeric(df.loc[idx, 'value'], errors='coerce')
        x_var = var_full.loc[idx]
        x_group = group_full.loc[idx]

        mask = y.notna() & x_var.notna() & x_group.notna()
        for c in available_confs:
            mask = mask & conf_encoded[c].loc[idx].notna()

        n = int(mask.sum())
        # Minimum: intercept + var + dummies + interactions + confonds + 2
        min_obs = 1 + 1 + n_dummies + n_dummies + len(available_confs) + 2
        if n < min_obs:
            row = {'point_id': pid, 'f_stat': np.nan, 'p': np.nan,
                   'r2': np.nan, 'r2_partial': np.nan, 'n': n, 'beta_var': np.nan}
            for g in unique_groups:
                row[f'slope_{group_labels[g]}'] = np.nan
            rows.append(row)
            continue

        yv = y[mask].values
        xv = x_var[mask].values
        gv = x_group[mask].values

        # Dummy variables (reference coding: drop first group)
        group_dummies = [(gv == g).astype(float) for g in unique_groups[1:]]
        # Interaction terms
        interactions = [xv * gd for gd in group_dummies]
        # Confounds
        conf_arr_list = [conf_encoded[c].loc[idx][mask].values for c in available_confs]

        # X_reduced: intercept + dummies + confounds
        parts_red = [np.ones(n)] + conf_arr_list
        X_reduced = np.column_stack(parts_red) if len(parts_red) > 1 else np.ones((n, 1))

        # X_full: intercept + var + dummies + interactions + confounds
        parts_full = [np.ones(n), xv] + group_dummies + interactions + conf_arr_list
        X_full = np.column_stack(parts_full)

        df_num = n_var_terms
        df_denom = n - X_full.shape[1]

        if df_denom <= 0:
            row = {'point_id': pid, 'f_stat': np.nan, 'p': np.nan,
                   'r2': np.nan, 'r2_partial': np.nan, 'n': n, 'beta_var': np.nan}
            for g in unique_groups:
                row[f'slope_{group_labels[g]}'] = np.nan
            rows.append(row)
            continue

        try:
            beta_red, _, _, _ = np.linalg.lstsq(X_reduced, yv, rcond=None)
            ssr_red = np.sum((yv - X_reduced @ beta_red) ** 2)

            beta_full, _, _, _ = np.linalg.lstsq(X_full, yv, rcond=None)
            ssr_full = np.sum((yv - X_full @ beta_full) ** 2)

            f_stat = ((ssr_red - ssr_full) / df_num) / (ssr_full / df_denom) if ssr_full > 0 else 0.0
            f_stat = max(0.0, f_stat)
            p_val = 1.0 - f_dist.cdf(f_stat, df_num, df_denom)

            ss_tot = np.sum((yv - yv.mean()) ** 2)
            r2_full = 1.0 - ssr_full / ss_tot if ss_tot > 0 else 0.0
            r2_red = 1.0 - ssr_red / ss_tot if ss_tot > 0 else 0.0
            r2_partial = max(0.0, r2_full - r2_red)

            # X_full layout: [intercept, var, dummy_1, ..., dummy_{G-1}, var*dummy_1, ..., var*dummy_{G-1}, conf...]
            # beta[0] = intercept
            # beta[1] = slope de var (groupe de référence)
            # beta[2..2+n_dummies-1] = différence d'intercept
            # beta[2+n_dummies..2+2*n_dummies-1] = différence de slope
            row = {'point_id': pid, 'f_stat': f_stat, 'p': p_val,
                   'r2': r2_full, 'r2_partial': r2_partial, 'n': n,
                   'beta_var': beta_full[1]}
            # Slope effectif par groupe
            # slope(ref) = beta_var
            row[f'slope_{group_labels[unique_groups[0]]}'] = beta_full[1]
            for i, g in enumerate(unique_groups[1:]):
                row[f'slope_{group_labels[g]}'] = beta_full[1] + beta_full[2 + n_dummies + i]

        except Exception:
            row = {'point_id': pid, 'f_stat': np.nan, 'p': np.nan,
                   'r2': np.nan, 'r2_partial': np.nan, 'n': n, 'beta_var': np.nan}
            for g in unique_groups:
                row[f'slope_{group_labels[g]}'] = np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values('point_id')
    result.attrs['group_labels'] = group_labels
    result.attrs['unique_groups'] = unique_groups
    return result


def plot_interaction_along_bundle(
    inter_df: pd.DataFrame,
    metric: str,
    var_name: str,
    group_col: str,
    sig_mask: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    confound_cols: Optional[List[str]] = None
) -> go.Figure:
    """
    Trace les résultats du modèle interaction le long du faisceau.
    Panneau 1 : Slopes par groupe + R² partiel
    Panneau 2 : -10·log₁₀(p) du test F partiel
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=[
            f"Slopes par groupe & R² partiel",
            "-10·log₁₀(p)  (test F partiel)"
        ]
    )

    if inter_df.empty:
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        return fig

    points = inter_df['point_id'].values

    # --- Panneau haut : R² partiel + slopes ---
    fig.add_trace(go.Scatter(
        x=points, y=inter_df['r2_partial'],
        mode='lines', name='R² partiel',
        fill='tozeroy', fillcolor='rgba(31,119,180,0.10)',
        line=dict(width=2, color='steelblue'),
        hovertemplate='Point: %{x:.1f}<br>R² partiel: %{y:.4f}<extra></extra>'
    ), row=1, col=1)

    # Slopes par groupe
    slope_cols = [c for c in inter_df.columns if c.startswith('slope_')]
    colors_slopes = px.colors.qualitative.Set1
    for i, sc in enumerate(slope_cols):
        group_label = sc.replace('slope_', '')
        color = colors_slopes[i % len(colors_slopes)]
        fig.add_trace(go.Scatter(
            x=points, y=inter_df[sc],
            mode='lines', name=f'slope({group_label})',
            line=dict(width=2, color=color, dash='dash'),
            opacity=0.8,
            yaxis='y3',
            hovertemplate=f'Point: %{{x:.1f}}<br>slope({group_label}): %{{y:.4f}}<extra></extra>'
        ), row=1, col=1)

    # Points significatifs
    if sig_mask is not None and sig_mask.any():
        sig_pts = inter_df[sig_mask]
        fig.add_trace(go.Scatter(
            x=sig_pts['point_id'], y=sig_pts['r2_partial'],
            mode='markers', name='Significatif',
            marker=dict(size=8, color='red', symbol='star'),
            hoverinfo='skip'
        ), row=1, col=1)

    # --- Panneau bas : -10·log10(p) ---
    p_safe = inter_df['p'].clip(lower=1e-300, upper=1)
    neg10logp = -10 * np.log10(p_safe)
    neg10logp_alpha = -10 * np.log10(alpha)

    fig.add_trace(go.Scatter(
        x=points, y=neg10logp,
        mode='lines+markers', name='-10·log₁₀(p)',
        line=dict(width=2, color='gray'),
        marker=dict(size=4),
        customdata=np.column_stack([inter_df['p'].values]),
        hovertemplate='Point: %{x:.1f}<br>-10·log₁₀(p): %{y:.2f}<br>p: %{customdata[0]:.2e}<extra></extra>'
    ), row=2, col=1)

    fig.add_hline(y=neg10logp_alpha, line_dash='dash', line_color='orange',
                  annotation_text=f'α={alpha} (-10·log₁₀={neg10logp_alpha:.1f})', row=2, col=1)

    if sig_mask is not None and sig_mask.any():
        sig_points = points[sig_mask]
        fig.add_trace(go.Scatter(
            x=sig_points, y=neg10logp.values[sig_mask],
            mode='markers', name=f'p < {alpha}',
            marker=dict(size=8, color='red', symbol='star'),
            showlegend=False
        ), row=2, col=1)

    # Formule
    conf_str = ' + '.join(confound_cols) if confound_cols else '∅'
    model_str = f"metric = β₀ + β·var + Σβ_g·{group_col} + Σγ_g·var×{group_col} + [{conf_str}] + ε"

    fig.update_layout(
        title=f"Modèle interaction : {metric} ~ {var_name} × {group_col}<br>"
              f"<span style='font-size:11px; color:gray'>{model_str}</span>",
        height=550,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02),
        margin=dict(r=150)
    )
    fig.update_yaxes(title_text='R² partiel / Slopes', row=1, col=1)
    fig.update_yaxes(title_text='-10·log₁₀(p)', row=2, col=1)
    fig.update_xaxes(title_text='Position le long du faisceau', row=2, col=1)

    return fig


def plot_interaction_scatter_at_point(
    df: pd.DataFrame,
    point_id: float,
    var_col: str,
    group_col: str,
    metric: str,
    confound_cols: List[str],
    color_mapping: Optional[Dict[Any, str]] = None
) -> go.Figure:
    """
    Scatter plot à un point donné pour le modèle interaction.
    Affiche les données colorées par groupe + une droite de régression par groupe.
    """
    fig = go.Figure()

    points = df['point_id'].unique()
    nearest_point = points[np.argmin(np.abs(points - point_id))]
    df_point = df[df['point_id'] == nearest_point].copy()

    if df_point.empty or var_col not in df_point.columns or group_col not in df_point.columns:
        fig.add_annotation(text="Données indisponibles", x=0.5, y=0.5, showarrow=False)
        return fig

    # Encoder var
    if df_point[var_col].dtype == 'object' or str(df_point[var_col].dtype).startswith('category'):
        df_point['_x'] = df_point[var_col].astype('category').cat.codes.replace(-1, np.nan)
        x_label = f"{var_col} (encodé)"
    else:
        df_point['_x'] = pd.to_numeric(df_point[var_col], errors='coerce')
        x_label = var_col

    # Encoder confonds
    conf_encoded = {}
    available_confs = [c for c in confound_cols if c in df_point.columns and c != var_col and c != group_col]
    for c in available_confs:
        sc = df_point[c]
        if sc.dtype == 'object' or str(sc.dtype).startswith('category'):
            conf_encoded[c] = sc.astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            conf_encoded[c] = pd.to_numeric(sc, errors='coerce')

    mask_base = df_point['_x'].notna() & df_point['value'].notna() & df_point[group_col].notna()
    for c in available_confs:
        mask_base = mask_base & conf_encoded[c].notna()

    groups = sorted(df_point.loc[mask_base, group_col].dropna().unique())
    colors = px.colors.qualitative.Set1

    if color_mapping is None:
        color_mapping = {g: colors[i % len(colors)] for i, g in enumerate(groups)}

    # Scatter + droite par groupe
    for i, g in enumerate(groups):
        mask_g = mask_base & (df_point[group_col] == g)
        sub = df_point[mask_g]
        color = color_mapping.get(g, colors[i % len(colors)])

        fig.add_trace(go.Scatter(
            x=sub['_x'], y=sub['value'], mode='markers',
            name=str(g),
            marker=dict(size=8, color=color),
            text=sub['subject'],
            hovertemplate=f'Subject: %{{text}}<br>{x_label}: %{{x}}<br>Value: %{{y:.4f}}<br>{group_col}: {g}<extra></extra>'
        ))

        # Droite de régression par groupe
        xv = sub['_x'].values
        yv = sub['value'].values
        if len(xv) >= 2:
            try:
                slope, intercept, r_val, p_val, _ = stats.linregress(xv, yv)
                x_line = np.linspace(xv.min(), xv.max(), 100)
                y_line = intercept + slope * x_line
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode='lines',
                    name=f'{g} (slope={slope:.3f}, r={r_val:.3f})',
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=True
                ))
            except Exception:
                pass

    conf_info = f" | confonds: {', '.join(available_confs)}" if available_confs else ""
    fig.update_layout(
        title=f"Point {nearest_point:.1f} : {metric} ~ {var_col} × {group_col}{conf_info}",
        xaxis_title=x_label,
        yaxis_title=metric,
        hovermode='closest'
    )

    return fig


def plot_ols_along_bundle(
    ols_df: pd.DataFrame,
    metric: str,
    var_name: str,
    poly_degree: int = 2,
    sig_mask: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    confound_cols: Optional[List[str]] = None
) -> go.Figure:
    """
    Trace les résultats OLS le long du faisceau avec Plotly.
    Panneau 1 : R² partiel + coefficients β
    Panneau 2 : p-values du test F partiel
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=[
            f"R² partiel & coefficients β",
            "-10·log₁₀(p)  (test F partiel)"
        ]
    )

    if ols_df.empty:
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        return fig

    points = ols_df['point_id'].values

    # --- Panneau haut : R² partiel ---
    fig.add_trace(go.Scatter(
        x=points, y=ols_df['r2_partial'],
        mode='lines', name='R² partiel',
        fill='tozeroy', fillcolor='rgba(31,119,180,0.15)',
        line=dict(width=2, color='steelblue'),
        hovertemplate='Point: %{x:.1f}<br>R² partiel: %{y:.4f}<extra></extra>'
    ), row=1, col=1)

    # Coefficients β sur axe secondaire (via traces avec yaxis2)
    colors_beta = ['darkorange', 'green', 'purple', 'brown']
    labels_beta = {1: 'β₁ (var)', 2: 'β₂ (var²)', 3: 'β₃ (var³)'}
    for d in range(1, poly_degree + 1):
        col_name = f'beta_deg{d}'
        if col_name in ols_df.columns:
            label = labels_beta.get(d, f'β{d} (var^{d})')
            fig.add_trace(go.Scatter(
                x=points, y=ols_df[col_name],
                mode='lines', name=label,
                line=dict(width=1.5, color=colors_beta[(d-1) % len(colors_beta)], dash='dash'),
                opacity=0.7,
                yaxis='y3',
                hovertemplate=f'Point: %{{x:.1f}}<br>{label}: %{{y:.4f}}<extra></extra>'
            ), row=1, col=1)

    # Points significatifs
    if sig_mask is not None and sig_mask.any():
        sig_pts = ols_df[sig_mask]
        fig.add_trace(go.Scatter(
            x=sig_pts['point_id'], y=sig_pts['r2_partial'],
            mode='markers', name='Significatif',
            marker=dict(size=8, color='red', symbol='star'),
            hoverinfo='skip'
        ), row=1, col=1)

    # --- Panneau bas : -10·log10(p) ---
    p_safe = ols_df['p'].clip(lower=1e-300, upper=1)
    neg10logp = -10 * np.log10(p_safe)
    neg10logp_alpha = -10 * np.log10(alpha)

    fig.add_trace(go.Scatter(
        x=points, y=neg10logp,
        mode='lines+markers', name='-10·log₁₀(p)',
        line=dict(width=2, color='gray'),
        marker=dict(size=4),
        customdata=np.column_stack([ols_df['p'].values]),
        hovertemplate='Point: %{x:.1f}<br>-10·log₁₀(p): %{y:.2f}<br>p: %{customdata[0]:.2e}<extra></extra>'
    ), row=2, col=1)

    # Ligne seuil alpha
    fig.add_hline(y=neg10logp_alpha, line_dash='dash', line_color='orange',
                  annotation_text=f'α={alpha} (-10·log₁₀={neg10logp_alpha:.1f})', row=2, col=1)

    # Points significatifs
    if sig_mask is not None and sig_mask.any():
        sig_points = points[sig_mask]
        fig.add_trace(go.Scatter(
            x=sig_points, y=neg10logp.values[sig_mask],
            mode='markers', name=f'p < {alpha}',
            marker=dict(size=8, color='red', symbol='star'),
            showlegend=False
        ), row=2, col=1)

    # Formule du modèle
    terms = ' + '.join([f'β{d}·var{"²" if d == 2 else "³" if d == 3 else f"^{d}" if d > 3 else ""}'
                        for d in range(1, poly_degree + 1)])
    conf_str = ' + '.join(confound_cols) if confound_cols else '∅'
    model_str = f"metric = β₀ + {terms} + [{conf_str}] + ε"

    fig.update_layout(
        title=f"OLS polynomial deg{poly_degree} : {metric} ~ {var_name}<br>"
              f"<span style='font-size:11px; color:gray'>{model_str}</span>",
        height=550,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02),
        margin=dict(r=150)
    )
    fig.update_yaxes(title_text='R² partiel', row=1, col=1)
    fig.update_yaxes(title_text='-10·log₁₀(p)', row=2, col=1)
    fig.update_xaxes(title_text='Position le long du faisceau', row=2, col=1)

    return fig


def plot_ols_scatter_at_point(
    df: pd.DataFrame,
    point_id: float,
    var_col: str,
    metric: str,
    confound_cols: List[str],
    poly_degree: int = 2,
    color_col: Optional[str] = None,
    color_mapping: Optional[Dict[Any, str]] = None
) -> go.Figure:
    """
    Scatter plot à un point donné avec la courbe ajustée du modèle polynomial.
    Affiche les données brutes + la courbe de prédiction metric ~ var + var² (+ confonds à leur moyenne).
    """
    fig = go.Figure()

    # Point le plus proche
    points = df['point_id'].unique()
    nearest_point = points[np.argmin(np.abs(points - point_id))]
    df_point = df[df['point_id'] == nearest_point].copy()

    if df_point.empty or var_col not in df_point.columns:
        fig.add_annotation(text="Données indisponibles", x=0.5, y=0.5, showarrow=False)
        return fig

    # Encoder var
    if df_point[var_col].dtype == 'object' or str(df_point[var_col].dtype).startswith('category'):
        df_point['_x'] = df_point[var_col].astype('category').cat.codes.replace(-1, np.nan)
        x_label = f"{var_col} (encodé)"
    else:
        df_point['_x'] = pd.to_numeric(df_point[var_col], errors='coerce')
        x_label = var_col

    # Encoder confonds
    conf_encoded = {}
    available_confs = [c for c in confound_cols if c in df_point.columns and c != var_col]
    for c in available_confs:
        sc = df_point[c]
        if sc.dtype == 'object' or str(sc.dtype).startswith('category'):
            conf_encoded[c] = sc.astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            conf_encoded[c] = pd.to_numeric(sc, errors='coerce')

    mask = df_point['_x'].notna() & df_point['value'].notna()
    for c in available_confs:
        mask = mask & conf_encoded[c].notna()

    # Scatter points
    if color_col and color_col in df_point.columns:
        color_vals_unique = df_point[color_col].dropna().unique()
        is_numeric_color = df_point[color_col].dtype in ['float64', 'float32', 'int64', 'int32']
        use_continuous = len(color_vals_unique) > 4

        if use_continuous and is_numeric_color:
            # Palette continue (viridis) pour variables numériques avec >4 valeurs
            color_numeric = pd.to_numeric(df_point[color_col], errors='coerce')
            fig.add_trace(go.Scatter(
                x=df_point['_x'], y=df_point['value'], mode='markers',
                name='Données',
                marker=dict(
                    size=8,
                    color=color_numeric,
                    colorscale='Viridis',
                    colorbar=dict(title=color_col, thickness=15, len=0.6),
                    showscale=True
                ),
                text=df_point['subject'],
                customdata=color_numeric,
                hovertemplate=f'Subject: %{{text}}<br>{x_label}: %{{x}}<br>Value: %{{y:.4f}}<br>{color_col}: %{{customdata:.3g}}<extra></extra>'
            ))
        elif not use_continuous:
            if color_mapping is None:
                colors = px.colors.qualitative.Plotly
                color_mapping = {v: colors[i % len(colors)] for i, v in enumerate(sorted(color_vals_unique))}
            for val in sorted(color_vals_unique):
                sub = df_point[df_point[color_col] == val]
                fig.add_trace(go.Scatter(
                    x=sub['_x'], y=sub['value'], mode='markers',
                    name=str(val),
                    marker=dict(size=8, color=color_mapping.get(val, 'gray')),
                    text=sub['subject'],
                    hovertemplate=f'Subject: %{{text}}<br>{x_label}: %{{x}}<br>Value: %{{y:.4f}}<br>{color_col}: {val}<extra></extra>'
                ))
        else:
            # >4 catégories non numériques : viridis sur indices
            cat_codes = df_point[color_col].astype('category').cat.codes
            fig.add_trace(go.Scatter(
                x=df_point['_x'], y=df_point['value'], mode='markers',
                name='Données',
                marker=dict(
                    size=8,
                    color=cat_codes,
                    colorscale='Viridis',
                    colorbar=dict(title=color_col, thickness=15, len=0.6),
                    showscale=True
                ),
                text=df_point['subject'],
                customdata=df_point[color_col],
                hovertemplate=f'Subject: %{{text}}<br>{x_label}: %{{x}}<br>Value: %{{y:.4f}}<br>{color_col}: %{{customdata}}<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df_point['_x'], y=df_point['value'], mode='markers',
            name='Données', marker=dict(size=8, color='steelblue'),
            text=df_point['subject'],
            hovertemplate=f'Subject: %{{text}}<br>{x_label}: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
        ))

    # Ajuster et tracer le modèle polynomial
    if mask.sum() >= poly_degree + len(available_confs) + 2:
        xv = df_point.loc[mask, '_x'].values
        yv = df_point.loc[mask, 'value'].values

        poly_cols_arr = [xv ** d for d in range(1, poly_degree + 1)]
        conf_arr_list = [conf_encoded[c][mask].values for c in available_confs]

        X_full = np.column_stack([np.ones(len(xv))] + poly_cols_arr + conf_arr_list) if conf_arr_list else np.column_stack([np.ones(len(xv))] + poly_cols_arr)

        try:
            beta, _, _, _ = np.linalg.lstsq(X_full, yv, rcond=None)

            # Courbe de prédiction : varier var, fixer confonds à leur moyenne
            x_line = np.linspace(xv.min(), xv.max(), 200)
            # Construire X pour la prédiction
            X_pred = [np.ones(200)]
            for d in range(1, poly_degree + 1):
                X_pred.append(x_line ** d)
            for c in available_confs:
                X_pred.append(np.full(200, conf_encoded[c][mask].mean()))
            X_pred = np.column_stack(X_pred)

            y_pred = X_pred @ beta

            # Formule lisible
            terms = [f'{beta[0]:.4f}']
            for d in range(1, poly_degree + 1):
                sign = '+' if beta[d] >= 0 else ''
                if d == 1:
                    terms.append(f'{sign}{beta[d]:.4f}·{var_col}')
                elif d == 2:
                    terms.append(f'{sign}{beta[d]:.4f}·{var_col}²')
                elif d == 3:
                    terms.append(f'{sign}{beta[d]:.4f}·{var_col}³')
                else:
                    terms.append(f'{sign}{beta[d]:.4f}·{var_col}^{d}')

            # F-test
            X_reduced = np.column_stack([np.ones(len(xv))] + conf_arr_list) if conf_arr_list else np.ones((len(xv), 1))
            beta_red, _, _, _ = np.linalg.lstsq(X_reduced, yv, rcond=None)
            ssr_full = np.sum((yv - X_full @ beta) ** 2)
            ssr_red = np.sum((yv - X_reduced @ beta_red) ** 2)
            df_num = poly_degree
            df_denom = len(xv) - X_full.shape[1]
            if df_denom > 0 and ssr_full > 0:
                f_stat = ((ssr_red - ssr_full) / df_num) / (ssr_full / df_denom)
                p_val = 1.0 - f_dist.cdf(max(0, f_stat), df_num, df_denom)
            else:
                f_stat, p_val = np.nan, np.nan

            ss_tot = np.sum((yv - yv.mean()) ** 2)
            r2 = 1.0 - ssr_full / ss_tot if ss_tot > 0 else 0.0
            r2_partial = max(0, r2 - (1.0 - ssr_red / ss_tot)) if ss_tot > 0 else 0.0

            formula = ' '.join(terms)
            r2_str = f'R²={r2:.3f}'
            p_str = f'p={p_val:.2e}' if p_val < 0.001 else f'p={p_val:.3f}'

            fig.add_trace(go.Scatter(
                x=x_line, y=y_pred, mode='lines',
                name=f'OLS deg{poly_degree} ({r2_str}, {p_str})',
                line=dict(color='red', width=2.5),
                hovertemplate=f'{x_label}: %{{x:.2f}}<br>Prédiction: %{{y:.4f}}<extra></extra>'
            ))

            # Ajouter régression linéaire simple pour comparaison
            if poly_degree > 1:
                slope, intercept, r_lin, p_lin, _ = stats.linregress(xv, yv)
                y_lin = intercept + slope * x_line
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_lin, mode='lines',
                    name=f'Linéaire (r={r_lin:.3f})',
                    line=dict(color='gray', width=1.5, dash='dot'),
                    opacity=0.6
                ))

        except Exception:
            pass

    # Confonds info
    conf_info = f" | confonds: {', '.join(available_confs)}" if available_confs else ""
    fig.update_layout(
        title=f"Point {nearest_point:.1f} : {metric} ~ poly({var_col}, deg={poly_degree}){conf_info}",
        xaxis_title=x_label,
        yaxis_title=metric,
        hovermode='closest'
    )

    return fig


def invert_points_order(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse l'ordre des points le long du faisceau pour chaque sujet."""
    if df.empty or 'point_id' not in df.columns:
        return df
    
    df = df.copy()
    
    # Calculer le max des point_id pour chaque sujet
    for subject in df['subject'].unique():
        mask = df['subject'] == subject
        point_ids = df.loc[mask, 'point_id'].values
        max_point = point_ids.max()
        min_point = point_ids.min()
        # Inverser: nouveau_point = max + min - ancien_point
        df.loc[mask, 'point_id'] = max_point + min_point - point_ids
    
    return df


def normalize_points_to_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les point_id en pourcentage (0-100) pour chaque sujet.
    Permet de comparer des pipelines avec des nombres de points différents.
    """
    if df.empty or 'point_id' not in df.columns:
        return df
    
    df = df.copy()
    
    for subject in df['subject'].unique():
        mask = df['subject'] == subject
        point_ids = df.loc[mask, 'point_id'].values
        min_pt = point_ids.min()
        max_pt = point_ids.max()
        
        if max_pt > min_pt:
            # Normaliser en pourcentage (0-100)
            df.loc[mask, 'point_id'] = (point_ids - min_pt) / (max_pt - min_pt) * 100
        else:
            df.loc[mask, 'point_id'] = 50.0  # Point unique -> milieu
    
    return df


def exclude_extreme_points(df: pd.DataFrame) -> pd.DataFrame:
    """Exclut le premier et le dernier point de chaque sujet."""
    if df.empty or 'point_id' not in df.columns:
        return df
    
    df = df.copy()
    
    # S'assurer que point_id est numérique pour un tri correct min/max
    try:
        df['point_id'] = pd.to_numeric(df['point_id'])
    except Exception:
        pass

    filtered_dfs = []
    
    for subject, sub_df in df.groupby('subject'):
        points = sorted(sub_df['point_id'].unique())
        if len(points) > 2:
            min_pt, max_pt = points[0], points[-1]
            sub_filtered = sub_df[(sub_df['point_id'] != min_pt) & (sub_df['point_id'] != max_pt)]
            filtered_dfs.append(sub_filtered)
        else:
            # Si moins de 3 points, on garde tout (ou rien ?) - ici on garde tout
            filtered_dfs.append(sub_df)
    
    if not filtered_dfs:
        return pd.DataFrame(columns=df.columns)
        
    return pd.concat(filtered_dfs, ignore_index=True)


def plot_bundle_curves_comparison(
    pipeline_dfs: Dict[str, pd.DataFrame],
    metric: str,
    stat_type: str,
    show_individual: bool = True,
    show_group_avg: bool = True,
    group_col: Optional[str] = None,
    title: str = ""
) -> go.Figure:
    """
    Trace les courbes de plusieurs pipelines sur le même graphique.
    Utilise des styles de lignes différents pour chaque pipeline.
    """
    fig = go.Figure()
    
    if not pipeline_dfs:
        fig.add_annotation(text="Aucune donnée disponible", x=0.5, y=0.5, showarrow=False)
        return fig
    
    pipeline_names = list(pipeline_dfs.keys())
    
    for p_idx, (pipeline_name, df) in enumerate(pipeline_dfs.items()):
        if df.empty:
            continue
        
        # Déterminer la colonne de valeur
        value_col = f"{metric}_{stat_type}"
        if value_col not in df.columns:
            if metric in df.columns:
                value_col = metric
            else:
                continue
        
        df = df.copy()
        df['value'] = df[value_col]
        
        # Style pour cette pipeline
        dash_style = PIPELINE_DASH_STYLES[p_idx % len(PIPELINE_DASH_STYLES)]
        pipeline_color = PIPELINE_COLORS[p_idx % len(PIPELINE_COLORS)]
        
        # Courbes individuelles
        if show_individual:
            subjects = df['subject'].unique()
            for subject in subjects:
                sub_df = df[df['subject'] == subject].sort_values('point_id')
                fig.add_trace(go.Scatter(
                    x=sub_df['point_id'],
                    y=sub_df['value'],
                    mode='lines',
                    name=f'{pipeline_name}',
                    legendgroup=pipeline_name,
                    line=dict(width=1, color=pipeline_color, dash=dash_style),
                    opacity=0.3,
                    hovertemplate=f'Pipeline: {pipeline_name}<br>Subject: {subject}<br>Position: %{{x:.1f}}%<br>Value: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ))
        
        # Moyenne globale par pipeline (toujours affichée en mode comparaison)
        if show_group_avg or not show_individual:
            avg = df.groupby('point_id')['value'].agg(['mean', 'std']).reset_index()
            
            fig.add_trace(go.Scatter(
                x=avg['point_id'],
                y=avg['mean'],
                mode='lines',
                name=f'{pipeline_name}',
                legendgroup=pipeline_name,
                line=dict(width=3, color=pipeline_color, dash=dash_style),
                hovertemplate=f'Pipeline: {pipeline_name}<br>Position: %{{x:.1f}}%<br>Mean: %{{y:.4f}}<extra></extra>'
            ))
            
            # Bande d'erreur
            fig.add_trace(go.Scatter(
                x=pd.concat([avg['point_id'], avg['point_id'][::-1]]),
                y=pd.concat([avg['mean'] + avg['std'], (avg['mean'] - avg['std'])[::-1]]),
                fill='toself',
                fillcolor=color_to_rgba(pipeline_color, 0.15),
                line=dict(color='rgba(255,255,255,0)'),
                legendgroup=pipeline_name,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Position le long du faisceau (%)",
        yaxis_title=f"{metric} ({stat_type})",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150)
    )
    
    return fig


def plot_correlation_curves_comparison(
    pipeline_corr_dfs: Dict[str, pd.DataFrame],
    column: str,
    metric: str,
    method: str,
    pipeline_sig_masks: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.05
) -> go.Figure:
    """
    Trace les courbes de corrélation de plusieurs pipelines sur le même graphique.
    """
    fig = go.Figure()
    
    if not pipeline_corr_dfs:
        fig.add_annotation(text="Aucun résultat de corrélation", x=0.5, y=0.5, showarrow=False)
        return fig
    
    for p_idx, (pipeline_name, corr_df) in enumerate(pipeline_corr_dfs.items()):
        if corr_df.empty:
            continue
        
        dash_style = PIPELINE_DASH_STYLES[p_idx % len(PIPELINE_DASH_STYLES)]
        pipeline_color = PIPELINE_COLORS[p_idx % len(PIPELINE_COLORS)]
        
        # Courbe principale
        fig.add_trace(go.Scatter(
            x=corr_df['point_id'],
            y=corr_df['r'],
            mode='lines+markers',
            name=f'{pipeline_name}',
            legendgroup=pipeline_name,
            line=dict(width=2, color=pipeline_color, dash=dash_style),
            marker=dict(size=4, color=pipeline_color),
            hovertemplate=f'Pipeline: {pipeline_name}<br>Position: %{{x:.1f}}%<br>r: %{{y:.3f}}<br>p: %{{customdata[0]:.4f}}<extra></extra>',
            customdata=corr_df[['p', 'n']].values
        ))
        
        # Points significatifs
        if pipeline_sig_masks and pipeline_name in pipeline_sig_masks:
            sig_mask = pipeline_sig_masks[pipeline_name]
            if sig_mask is not None and sig_mask.any():
                sig_df = corr_df[sig_mask]
                fig.add_trace(go.Scatter(
                    x=sig_df['point_id'],
                    y=sig_df['r'],
                    mode='markers',
                    name=f'{pipeline_name} (p<{alpha})',
                    legendgroup=pipeline_name,
                    marker=dict(size=10, color=pipeline_color, symbol='star'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Ligne zéro
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Corrélation ({method}) : {metric} vs {column} - Comparaison",
        xaxis_title="Position le long du faisceau (%)",
        yaxis_title=f"Coefficient de corrélation (r)",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150)
    )
    
    return fig


def apply_centroid_selection(df: pd.DataFrame, centroid_val) -> pd.DataFrame:
    """
    Filtre ou agrège les données selon le centroïde sélectionné.
    - centroid_val is None  → moyenne de toutes les métriques numériques par (subject, point_id)
    - centroid_val = valeur → filtre sur centroid_id == centroid_val
    Si la colonne 'centroid_id' est absente, retourne df tel quel.
    """
    if 'centroid_id' not in df.columns:
        return df
    if centroid_val is None:
        # Moyenne par (subject, point_id) — conserver les colonnes non-numériques (première valeur)
        non_num_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype).startswith('category')]
        non_num_cols = [c for c in non_num_cols if c not in ('centroid_id',)]
        num_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]


        group_keys = ['subject', 'point_id']
        agg_dict = {c: 'mean' for c in num_cols if c in df.columns}
        for c in non_num_cols:
            if c in df.columns and c not in group_keys:
                agg_dict[c] = 'first'
        if not agg_dict:
            return df.drop(columns=['centroid_id'], errors='ignore')
        result = df.groupby(group_keys, as_index=False).agg(agg_dict)
        return result
    else:
        return df[df['centroid_id'] == centroid_val].copy()


def apply_interpolate_missing_points(df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """
    Wrapper autour de interpolate_missing_points (generate_report_v2).
    Adapte la colonne 'point_id' en 'point' pour l'appel, puis remerge
    les valeurs interpolées dans le DataFrame d'origine.
    Gère deux cas :
      - NaN dans les lignes existantes (remplacé par la valeur interpolée)
      - Points entièrement absents pour un sujet (nouvelles lignes ajoutées)
    """
    if df.empty or 'point_id' not in df.columns or 'value' not in df.columns:
        return df

    original_dtype = df['point_id'].dtype
    df_in = df[['subject', 'point_id', 'value']].rename(columns={'point_id': 'point'})
    df_out = interpolate_missing_points(df_in, max_gap=max_gap)
    df_out = df_out.rename(columns={'point': 'point_id'})
    # Forcer le même type que l'original pour éviter les échecs de merge silencieux
    try:
        df_out['point_id'] = df_out['point_id'].astype(original_dtype)
    except (ValueError, TypeError):
        df_out['point_id'] = pd.to_numeric(df_out['point_id'], errors='coerce')

    # Ne garder que les valeurs valides (les NaN restants n'ont pas pu être interpolés)
    df_out = df_out[df_out['value'].notna()].copy()

    # Cas 1 : lignes existantes → mettre à jour la valeur (NaN remplacé par interpolé)
    df_no_val = df.drop(columns=['value'])
    df_updated = df_no_val.merge(
        df_out[['subject', 'point_id', 'value']],
        on=['subject', 'point_id'],
        how='left'
    )

    # Cas 2 : points absents pour certains sujets → ajouter de nouvelles lignes
    # Comparaison par tuples pour éviter les problèmes de dtype avec MultiIndex.isin
    existing_pairs = set(zip(df_updated['subject'].tolist(), df_updated['point_id'].tolist()))
    new_mask = [
        (s, p) not in existing_pairs
        for s, p in zip(df_out['subject'].tolist(), df_out['point_id'].tolist())
    ]
    new_points = df_out[new_mask].copy()

    if not new_points.empty:
        # Métadonnées sujet-niveau (première valeur par sujet pour les colonnes non-point)
        meta_cols = [c for c in df_no_val.columns if c != 'point_id']
        subject_meta = df_no_val[meta_cols].drop_duplicates(subset='subject').set_index('subject')
        # Remplir les métadonnées pour les nouvelles lignes
        new_points = new_points.join(subject_meta, on='subject', how='left')
        df_updated = pd.concat([df_updated, new_points[df_updated.columns]], ignore_index=True)

    return df_updated


def plot_subject_count_per_point(
    df: pd.DataFrame,
    value_col: str = 'value',
    title: str = ""
) -> go.Figure:
    """
    Affiche un graphique en barres empilées du nombre de sujets avec/sans données valides par point.
    Le hover affiche les IDs des sujets présents et manquants.
    """
    fig = go.Figure()

    if df.empty or 'point_id' not in df.columns or 'subject' not in df.columns:
        fig.add_annotation(text="Pas de données", x=0.5, y=0.5, showarrow=False)
        return fig

    all_subjects = sorted(df['subject'].unique())
    all_points = sorted(df['point_id'].unique())
    n_total = len(all_subjects)

    points_list = []
    n_present_list = []
    n_missing_list = []
    present_str_list = []
    missing_str_list = []

    for pid in all_points:
        df_pt = df[df['point_id'] == pid]
        if value_col in df_pt.columns:
            df_valid = df_pt[df_pt[value_col].notna()]
        else:
            df_valid = df_pt
        present = sorted(df_valid['subject'].unique())
        missing = sorted(set(all_subjects) - set(present))

        points_list.append(pid)
        n_present_list.append(len(present))
        n_missing_list.append(len(missing))
        present_str_list.append(', '.join(str(s) for s in present) if present else '—')
        missing_str_list.append(', '.join(str(s) for s in missing) if missing else '—')

    customdata_present = [
        [n_missing_list[i], present_str_list[i], missing_str_list[i]]
        for i in range(len(points_list))
    ]
    fig.add_trace(go.Bar(
        x=points_list,
        y=n_present_list,
        name='Présents',
        marker_color='steelblue',
        customdata=customdata_present,
        hovertemplate=(
            '<b>Point %{x:.1f}</b><br>'
            'Présents : %{y}<br>'
            'Manquants : %{customdata[0]}<br>'
            '<b>IDs présents :</b> %{customdata[1]}'
            '<extra>Présents</extra>'
        )
    ))

    customdata_missing = [
        [n_present_list[i], missing_str_list[i], present_str_list[i]]
        for i in range(len(points_list))
    ]
    fig.add_trace(go.Bar(
        x=points_list,
        y=n_missing_list,
        name='Manquants',
        marker_color='salmon',
        customdata=customdata_missing,
        hovertemplate=(
            '<b>Point %{x:.1f}</b><br>'
            'Manquants : %{y}<br>'
            'Présents : %{customdata[0]}<br>'
            '<b>IDs manquants :</b> %{customdata[1]}'
            '<extra>Manquants</extra>'
        )
    ))

    fig.update_layout(
        barmode='stack',
        title=title or f"Sujets par point (total : {n_total})",
        xaxis_title="Position le long du faisceau",
        yaxis_title="Nombre de sujets",
        height=250,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150, t=40)
    )

    return fig


def apply_multiple_correction(p_values: np.ndarray, method: str, alpha: float = 0.05) -> np.ndarray:
    """Applique une correction pour tests multiples."""
    if method == 'none':
        return p_values < alpha
    elif method == 'bonferroni':
        return p_values < (alpha / len(p_values))
    elif method == 'fdr_bh':
        from scipy.stats import false_discovery_control
        try:
            # scipy >= 1.11
            return false_discovery_control(p_values, method='bh') < alpha
        except Exception:
            # Fallback manuel
            n = len(p_values)
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            threshold = np.arange(1, n + 1) / n * alpha
            sig = sorted_p <= threshold
            # Propagate significance
            for i in range(n - 1, -1, -1):
                if sig[i]:
                    sig[:i + 1] = True
                    break
            result = np.zeros(n, dtype=bool)
            result[sorted_idx] = sig
            return result
    return np.zeros_like(p_values, dtype=bool)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_bundle_curves(
    df: pd.DataFrame,
    metric: str,
    stat_type: str,
    show_individual: bool = True,
    show_group_avg: bool = False,
    group_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: str = ""
) -> go.Figure:
    """Trace les courbes de métriques le long d'un faisceau."""
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(text="Aucune donnée disponible", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Déterminer la colonne de valeur (format {metric}_{stat_type} ou {metric} seul)
    value_col = f"{metric}_{stat_type}"
    if value_col not in df.columns:
        # Fallback: utiliser juste {metric} si disponible (équivalent à mean)
        if metric in df.columns:
            value_col = metric
        else:
            fig.add_annotation(text=f"Colonne {value_col} ou {metric} non trouvée", x=0.5, y=0.5, showarrow=False)
            return fig
    
    df = df.copy()
    df['value'] = df[value_col]
    
    # Courbes individuelles
    if show_individual:
        subjects = df['subject'].unique()
        
        # Déterminer le mode de coloration
        use_continuous_color = False
        color_values = None
        color_map = None
        colorscale = 'Viridis'
        
        if color_col and color_col in df.columns:
            # Obtenir la valeur de couleur par sujet (première valeur rencontrée)
            subject_colors = df.groupby('subject')[color_col].first()
            
            # Vérifier si numérique ou catégoriel
            if df[color_col].dtype in ['float64', 'float32', 'int64', 'int32']:
                unique_vals = df[color_col].dropna().unique()
                if len(unique_vals) > 4:
                    use_continuous_color = True
                    color_values = subject_colors
                else:
                    # Peu de valeurs numériques -> traiter comme catégoriel
                    use_continuous_color = False
                    color_values = subject_colors
                    unique_sorted = sorted(unique_vals)
                    colors_discrete = px.colors.qualitative.Plotly
                    color_map = {v: colors_discrete[i % len(colors_discrete)] for i, v in enumerate(unique_sorted)}
            else:
                # Catégoriel
                unique_vals = df[color_col].dropna().unique()
                if len(unique_vals) > 4:
                    # Encoder en numérique pour échelle continue
                    use_continuous_color = True
                    cat_codes = df[color_col].astype('category').cat.codes
                    subject_codes = df.assign(_code=cat_codes).groupby('subject')['_code'].first()
                    color_values = subject_codes
                else:
                    use_continuous_color = False
                    color_values = subject_colors
                    colors_discrete = px.colors.qualitative.Plotly
                    color_map = {v: colors_discrete[i % len(colors_discrete)] for i, v in enumerate(sorted(unique_vals))}
        
        if use_continuous_color and color_values is not None:
            # Échelle continue (viridis)
            import plotly.colors as pc
            vmin = color_values.min()
            vmax = color_values.max()
            
            for subject in subjects:
                sub_df = df[df['subject'] == subject].sort_values('point_id')
                cval = color_values.get(subject, np.nan)
                
                # Normaliser pour obtenir la couleur
                if pd.notna(cval) and vmax > vmin:
                    norm_val = (cval - vmin) / (vmax - vmin)
                    # Obtenir la couleur de l'échelle viridis
                    color = px.colors.sample_colorscale('Viridis', [norm_val])[0]
                else:
                    color = 'gray'
                
                # Info couleur pour hover
                color_info = f"<br>{color_col}: {cval}" if pd.notna(cval) else ""
                
                fig.add_trace(go.Scatter(
                    x=sub_df['point_id'],
                    y=sub_df['value'],
                    mode='lines',
                    name=f'Sub {subject}',
                    line=dict(width=1.5, color=color),
                    opacity=0.7,
                    hovertemplate=f'Subject: {subject}{color_info}<br>Point: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ))
            
            # Ajouter une colorbar factice
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    colorscale='Viridis',
                    cmin=vmin,
                    cmax=vmax,
                    colorbar=dict(title=color_col, thickness=15),
                    showscale=True
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        elif color_col and color_map is not None:
            # Coloration discrète (<=4 valeurs)
            for subject in subjects:
                sub_df = df[df['subject'] == subject].sort_values('point_id')
                cval = color_values.get(subject) if color_values is not None else None
                color = color_map.get(cval, 'gray') if cval is not None else 'gray'
                
                color_info = f"<br>{color_col}: {cval}" if cval is not None else ""
                
                fig.add_trace(go.Scatter(
                    x=sub_df['point_id'],
                    y=sub_df['value'],
                    mode='lines',
                    name=f'{cval}' if cval is not None else f'Sub {subject}',
                    legendgroup=str(cval) if cval is not None else subject,
                    line=dict(width=1.5, color=color),
                    opacity=0.7,
                    hovertemplate=f'Subject: {subject}{color_info}<br>Point: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ))
            
            # Ajouter des traces pour la légende (une par groupe)
            for val, color in color_map.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    name=str(val),
                    line=dict(width=3, color=color),
                    showlegend=True
                ))
        
        else:
            # Pas de coloration spéciale -> coloration par sujet
            colors = px.colors.qualitative.Plotly
            for i, subject in enumerate(subjects):
                sub_df = df[df['subject'] == subject].sort_values('point_id')
                fig.add_trace(go.Scatter(
                    x=sub_df['point_id'],
                    y=sub_df['value'],
                    mode='lines',
                    name=f'Sub {subject}',
                    line=dict(width=1, color=colors[i % len(colors)]),
                    opacity=0.5,
                    hovertemplate=f'Subject: {subject}<br>Point: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
                ))
    
    # Moyennes de groupe
    if show_group_avg and group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        colors = px.colors.qualitative.Set1
        
        for i, group in enumerate(sorted(groups)):
            group_df = df[df[group_col] == group]
            avg = group_df.groupby('point_id')['value'].agg(['mean', 'std', 'count']).reset_index()
            import scipy.stats as stats
            avg['ci95'] = avg.apply(
                lambda r: stats.t.ppf(0.975, df=r['count'] - 1) * r['std'] / (r['count'] ** 0.5) if r['count'] > 1 else 0,
                axis=1
            )

            fig.add_trace(go.Scatter(
                x=avg['point_id'],
                y=avg['mean'],
                mode='lines',
                name=f'Group {group}',
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate=f'Group: {group}<br>Point: %{{x}}<br>Mean: %{{y:.4f}}<extra></extra>'
            ))

            # Bande IC 95%
            fig.add_trace(go.Scatter(
                x=pd.concat([avg['point_id'], avg['point_id'][::-1]]),
                y=pd.concat([avg['mean'] + avg['ci95'], (avg['mean'] - avg['ci95'])[::-1]]),
                fill='toself',
                fillcolor=color_to_rgba(colors[i % len(colors)], 0.2),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Position le long du faisceau",
        yaxis_title=f"{metric} ({stat_type})",
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150)
    )
    
    return fig


def plot_correlation_curve(
    corr_df: pd.DataFrame,
    column: str,
    metric: str,
    method: str,
    sig_mask: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> go.Figure:
    """Trace la courbe de corrélation point par point."""
    fig = go.Figure()
    
    if corr_df.empty:
        fig.add_annotation(text="Aucun résultat de corrélation", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Courbe principale
    fig.add_trace(go.Scatter(
        x=corr_df['point_id'],
        y=corr_df['r'],
        mode='lines+markers',
        name='r',
        line=dict(width=2, color='steelblue'),
        marker=dict(size=4),
        hovertemplate='Point: %{x}<br>r: %{y:.3f}<br>p: %{customdata[0]:.4f}<br>n: %{customdata[1]}<extra></extra>',
        customdata=corr_df[['p', 'n']].values
    ))
    
    # Points significatifs
    if sig_mask is not None and sig_mask.any():
        sig_df = corr_df[sig_mask]
        fig.add_trace(go.Scatter(
            x=sig_df['point_id'],
            y=sig_df['r'],
            mode='markers',
            name=f'Significatif (p<{alpha})',
            marker=dict(size=10, color='red', symbol='star'),
            hoverinfo='skip'
        ))
    
    # Ligne zéro
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Corrélation ({method}) : {metric} vs {column}",
        xaxis_title="Position le long du faisceau",
        yaxis_title=f"Coefficient de corrélation (r)",
        hovermode='x unified'
    )
    
    return fig


def plot_scatter_at_point(
    df: pd.DataFrame,
    point_id: float,
    x_col: str,
    metric: str,
    color_col: Optional[str] = None,
    color_mapping: Optional[Dict[Any, str]] = None
) -> go.Figure:
    """Trace un scatter plot pour un point spécifique."""
    fig = go.Figure()
    
    # Filtrer au point le plus proche
    points = df['point_id'].unique()
    nearest_point = points[np.argmin(np.abs(points - point_id))]
    df_point = df[df['point_id'] == nearest_point].copy()
    
    if df_point.empty or x_col not in df_point.columns:
        fig.add_annotation(text="Données indisponibles", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Encoder x si catégoriel
    if df_point[x_col].dtype == 'object' or str(df_point[x_col].dtype).startswith('category'):
        df_point['_x'] = df_point[x_col].astype('category').cat.codes.replace(-1, np.nan)
        x_label = f"{x_col} (encodé)"
    else:
        df_point['_x'] = pd.to_numeric(df_point[x_col], errors='coerce')
        x_label = x_col
    
    # Scatter avec ou sans couleur - utiliser go.Scatter pour éviter FutureWarning
    if color_col and color_col in df_point.columns:
        color_values_unique = df_point[color_col].dropna().unique()
        n_unique = len(color_values_unique)
        
        # Déterminer si on utilise une échelle continue ou discrète
        is_numeric = df_point[color_col].dtype in ['float64', 'float32', 'int64', 'int32']
        use_continuous = n_unique > 4
        
        if use_continuous:
            # Échelle continue (Viridis)
            if is_numeric:
                color_vals = pd.to_numeric(df_point[color_col], errors='coerce')
            else:
                # Encoder catégoriel en numérique
                color_vals = df_point[color_col].astype('category').cat.codes.replace(-1, np.nan).astype(float)
            
            fig.add_trace(go.Scatter(
                x=df_point['_x'],
                y=df_point['value'],
                mode='markers',
                marker=dict(
                    color=color_vals,
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title=color_col, thickness=15),
                    showscale=True
                ),
                text=df_point['subject'],
                customdata=df_point[color_col],
                hovertemplate=f'Subject: %{{text}}<br>{x_col}: %{{x}}<br>Value: %{{y:.4f}}<br>{color_col}: %{{customdata}}<extra></extra>',
                showlegend=False
            ))
        else:
            # Échelle discrète (<=4 valeurs)
            color_values_sorted = sorted(color_values_unique)
            
            # Utiliser le mapping de couleurs si fourni, sinon utiliser les couleurs par défaut
            if color_mapping is None:
                colors = px.colors.qualitative.Plotly
                color_mapping = {val: colors[i % len(colors)] for i, val in enumerate(color_values_sorted)}
            
            for val in color_values_sorted:
                mask = df_point[color_col] == val
                sub_df = df_point[mask]
                fig.add_trace(go.Scatter(
                    x=sub_df['_x'],
                    y=sub_df['value'],
                    mode='markers',
                    name=str(val),
                    marker=dict(color=color_mapping.get(val, 'gray'), size=8),
                    text=sub_df['subject'],
                    hovertemplate=f'Subject: %{{text}}<br>{x_col}: %{{x}}<br>Value: %{{y:.4f}}<extra>{val}</extra>'
                ))
    else:
        fig.add_trace(go.Scatter(
            x=df_point['_x'],
            y=df_point['value'],
            mode='markers',
            name='Données',
            marker=dict(color='steelblue', size=8),
            text=df_point['subject'],
            hovertemplate=f'Subject: %{{text}}<br>{x_col}: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
        ))
    
    # Régression linéaire
    mask = df_point['_x'].notna() & df_point['value'].notna()
    if mask.sum() >= 2:
        x_valid = df_point.loc[mask, '_x'].values
        y_valid = df_point.loc[mask, 'value'].values
        
        try:
            slope, intercept, r, p, _ = stats.linregress(x_valid, y_valid)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
            y_line = intercept + slope * x_line
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f'Régression (r={r:.3f}, p={p:.3g})',
                line=dict(color='red', dash='dash')
            ))
        except Exception:
            pass
    
    fig.update_layout(
        title=f"Point {nearest_point:.1f} : {metric} vs {x_col}",
        xaxis_title=x_label,
        yaxis_title=metric
    )
    
    return fig


# =============================================================================
# Streamlit App
# =============================================================================

def preprocess_bundle_df(
    df_all: pd.DataFrame,
    *,
    selected_bundle: str,
    selected_subjects: List[str],
    selected_centroid,
    metric: str,
    stat_type: str,
    interpolate_missing: bool,
    interpolate_max_gap: int,
    force_nan_zero: bool,
    exclude_extremes: bool,
    enable_nuisance: bool,
    nuisance_cols: List[str],
    nuisance_method: str,
    group_col: Optional[str],
    enable_correlation: bool,
    corr_col: Optional[str],
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Dict[str, Any]]]:
    """Filtre un faisceau puis prépare la colonne 'value' et applique les corrections.

    Étapes communes au mode normal et au mode comparaison : filtrage
    faisceau/sujets → centroïde → colonne value → interpolation des points
    manquants → NaN→0 → exclusion des points extrêmes → correction de nuisances
    (sur la métrique et sur la variable de corrélation).

    Retourne (df, value_col, nuisance_diag). df vaut None si aucune donnée n'est
    exploitable (faisceau absent, sujets filtrés, ou métrique introuvable).
    """
    df = df_all[df_all['bundle'] == selected_bundle].copy()
    if selected_subjects:
        df = df[df['subject'].isin(selected_subjects)]

    df = apply_centroid_selection(df, selected_centroid)
    if df.empty:
        return None, None, None

    # Colonne value : format {metric}_{stat_type}, ou {metric} seul (≈ mean)
    value_col = f"{metric}_{stat_type}"
    if value_col not in df.columns:
        if metric in df.columns:
            value_col = metric
        else:
            return None, None, None
    df['value'] = df[value_col]

    # Interpolation des points manquants (prioritaire, avant toute correction)
    if interpolate_missing:
        df = apply_interpolate_missing_points(df, max_gap=interpolate_max_gap)
        df[value_col] = df['value']

    if force_nan_zero:
        df['value'] = df['value'].fillna(0)

    if exclude_extremes:
        df = exclude_extreme_points(df)

    nuisance_diag = None
    if enable_nuisance and nuisance_cols:
        valid_nuisance_cols = [c for c in nuisance_cols if c in df.columns]
        if valid_nuisance_cols:
            correct = (apply_nuisance_correction_ols if nuisance_method == 'ols'
                       else apply_nuisance_correction_unconfound)
            df, nuisance_diag = correct(df, valid_nuisance_cols, group_col)
            df[value_col] = df['value']  # pour que les courbes utilisent la valeur corrigée

            # Corriger aussi la variable de corrélation si elle n'est pas déjà une nuisance
            if enable_correlation and corr_col and corr_col not in valid_nuisance_cols:
                df_temp = df.copy()
                df_temp['value'] = pd.to_numeric(df_temp[corr_col], errors='coerce')
                df_corr_var, _ = correct(df_temp, valid_nuisance_cols, group_col)
                df_corr_only = (df_corr_var[['subject', 'point_id', 'value']]
                                .rename(columns={'value': corr_col}))
                if corr_col in df.columns:
                    df = df.drop(columns=[corr_col])
                df = df.merge(df_corr_only, on=['subject', 'point_id'], how='inner')

    return df, value_col, nuisance_diag


def compute_metric_stat_availability(
    pipeline_data: Dict[str, dict],
) -> Tuple[set, set, set, set]:
    """Détermine les métriques et stats disponibles à travers les pipelines.

    Retourne (common_metrics, common_stats, all_metrics, all_stats) : l'intersection
    (présentes dans TOUTES les pipelines) et l'union (au moins une). Une colonne
    'METRIC_STAT' active (metric, stat) ; une colonne 'METRIC' seule compte comme
    (metric, 'mean').
    """
    def available(df: pd.DataFrame) -> Tuple[set, set]:
        metrics, stats_ = set(), set()
        if df.empty:
            return metrics, stats_
        cols = df.columns
        for metric in METRICS:
            for stat in STAT_TYPES:
                if f"{metric}_{stat}" in cols:
                    metrics.add(metric)
                    stats_.add(stat)
            if metric in cols:
                metrics.add(metric)
                stats_.add('mean')
        return metrics, stats_

    per_pipeline = [available(pdata.get('df', pd.DataFrame())) for pdata in pipeline_data.values()]
    if not per_pipeline:
        return set(), set(), set(), set()
    metric_sets = [m for m, _ in per_pipeline]
    stat_sets = [s for _, s in per_pipeline]
    return (
        set.intersection(*metric_sets),
        set.intersection(*stat_sets),
        set.union(*metric_sets),
        set.union(*stat_sets),
    )


def build_subject_table(
    subjects: List[str],
    participants: Optional[pd.DataFrame],
    actimetry: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Construit une table sujet x métadonnées (participants + actimétrie).

    Une ligne par sujet chargé ; les colonnes proviennent des fichiers
    participants et actimétrie (fusion sur subject_id, doublons évités).
    """
    table = pd.DataFrame({'subject_id': sorted(subjects)})
    for extra in (participants, actimetry):
        if extra is None or extra.empty:
            continue
        cols = [c for c in extra.columns if c == 'subject_id' or c not in table.columns]
        table = table.merge(extra[cols], on='subject_id', how='left')
    return table


def render_subject_table_tab(
    subjects: List[str],
    participants: Optional[pd.DataFrame],
    actimetry: Optional[pd.DataFrame],
) -> List[str]:
    """Affiche le tableau des données avec filtres par colonne.

    Retourne la liste des subject_id retenus par les filtres. Cette liste
    sert ensuite à restreindre les sujets dans toutes les analyses : le
    tableau filtré (ou non) est le tableau utilisé partout.
    """
    table = build_subject_table(subjects, participants, actimetry)
    if table.empty:
        st.warning("Aucune donnée de sujet disponible")
        return list(subjects)
    print()
    filterable = [c for c in table.columns if c not in ('subject_id')]
    chosen = st.multiselect(
        "Colonnes de filtrage",
        options=filterable,
        key="table_filter_cols",
        help="Sélectionnez les colonnes sur lesquelles filtrer les sujets",
    )

    # Un sujet n'est exclu que s'il possède une valeur hors filtre (les NaN
    # ne sont jamais filtrés, pour ne pas écarter de sujet involontairement).
    mask = pd.Series(True, index=table.index)
    if chosen:
        widget_cols = st.columns(min(3, len(chosen)))
        for i, col in enumerate(chosen):
            with widget_cols[i % len(widget_cols)]:
                s = table[col]
                if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 2:
                    lo, hi = float(s.min()), float(s.max())
                    rng = st.slider(col, lo, hi, (lo, hi), key=f"filt_{col}")
                    mask &= s.between(rng[0], rng[1]) | s.isna()
                else:
                    opts = sorted(s.dropna().astype(str).unique())
                    sel = st.multiselect(col, opts, default=opts, key=f"filt_{col}")
                    mask &= s.astype(str).isin(sel) | s.isna()

    filtered = table[mask]
    st.caption(f"{len(filtered)} / {len(table)} sujets retenus")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    return filtered['subject_id'].tolist()


def main():
    st.set_page_config(
        page_title="Visualiseur de Statistiques TractoPL",
        page_icon="chart_with_upwards_trend",
        layout="wide"
    )
    
    # CSS pour améliorer l'affichage des pipelines dans le multiselect
    st.markdown("""
        <style>
        /* Augmenter la largeur du dropdown multiselect */
        div[data-baseweb="select"] > div {
            min-width: 100%;
        }
        /* Afficher le texte complet des options */
        div[data-baseweb="select"] span {
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        /* Augmenter la largeur des tags sélectionnés */
        div[data-baseweb="tag"] {
            max-width: 100% !important;
        }
        div[data-baseweb="tag"] span {
            max-width: 100% !important;
            overflow: visible !important;
            white-space: normal !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialiser le session state avec tous les paramètres
    defaults = {
        'data_loaded': False,
        'invert_points': {},  # Dict pipeline_name -> bool
        'param_dataset_path': os.environ.get('TRACTOPL_DATASET_ROOT', ''),
        # Paramètres de visualisation
        'param_metric': 'FA',
        'param_stat_type': 'mean',
        'param_selected_bundle': None,
        'param_selected_subjects': [],
        'param_show_individual': True,
        'param_show_group_avg': False,
        'param_group_col': None,
        'param_curve_color_col': None,
        'param_resample': False,
        'param_resample_points': 100,
        'param_interp_method': 'linear',
        'param_enable_nuisance': False,
        'param_nuisance_cols': [],
        'param_nuisance_method': 'unconfound',
        'param_enable_correlation': False,
        'param_corr_col': None,
        'param_corr_method': 'pearson',
        'param_multiple_corr': 'none',
        'param_sig_alpha': 0.05,
        'param_filter_subjects': False,
        'param_common_metrics_only': True,  # Filtrer métriques/stats communes
        'param_comparison_mode': False,  # Mode comparaison des pipelines
        'param_comparison_n_points': 100,  # Nombre de points pour l'interpolation en mode comparaison
        'param_force_nan_zero': False, # Forcer NaN à 0
        'param_exclude_extremes': False, # Exclure les points extrêmes (premier et dernier)
        'param_show_subject_count': False,  # Afficher le nombre de sujets par point
        'param_centroid': None,  # None = moyenne par point_id, sinon centroid_id sélectionné
        'param_interpolate_missing': False,  # Interpoler les points manquants
        'param_interpolate_max_gap': 2,  # Nombre max de points consécutifs à interpoler
        # OLS polynomial model
        'param_enable_ols': False,
        'param_ols_degree': 2,
        'param_ols_confounds': [],
        # Interaction model (slope par groupe)
        'param_enable_interaction': False,
        'param_interaction_group_col': None,
    }
    
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val
    
    # ==========================================================================
    # Sidebar - Configuration
    # ==========================================================================
    
    with st.sidebar:
        st.header("Configuration des données")

        dataset_path = st.text_input(
            "Racine du dataset BIDS",
            value=st.session_state.param_dataset_path,
            key="dataset_path_input",
        )
        if dataset_path != st.session_state.param_dataset_path:
            st.session_state.param_dataset_path = dataset_path
            st.session_state.data_loaded = False  # Forcer le rechargement
            st.rerun()

        _paths = get_dataset_paths(st.session_state.param_dataset_path) if st.session_state.param_dataset_path else None
        if _paths is None:
            st.info("Saisissez la racine d'un dataset BIDS pour détecter les derivatives.")
            return
        _dataset_path = _paths['dataset_path']
        _subjects_file = _paths['subjects_file']
        _participants_file = _paths['participants_file']
        _actimetry_file = _paths['actimetry_file']  # None si pas actidep

        # Sélection de la pipeline (multi-sélection)
        available_pipelines = get_available_pipelines(_dataset_path)
        if not available_pipelines:
            st.warning("Aucune pipeline trouvée dans derivatives/")
            available_pipelines = [DEFAULT_PIPELINE]
        
        # Déterminer la valeur par défaut
        default_pipelines = [DEFAULT_PIPELINE] if DEFAULT_PIPELINE in available_pipelines else [available_pipelines[0]]
        
        selected_pipelines = st.multiselect(
            "Pipelines",
            options=available_pipelines,
            default=default_pipelines,
            help="Sélectionnez une ou plusieurs pipelines de tractométrie détectées dans derivatives/."
        )
        
        if not selected_pipelines:
            st.warning("Veuillez sélectionner au moins une pipeline")
        
        # Filtrage par subjects.txt
        filter_subjects = st.checkbox(
            "Filtrer par subjects.txt", 
            value=st.session_state.param_filter_subjects,
            key="filter_subjects_cb"
        )
        st.session_state.param_filter_subjects = filter_subjects
        
        # Bouton de chargement
        if st.button("Charger les données", type="primary") and selected_pipelines:
            with st.spinner("Chargement en cours..."):
                # Charger sujets valides
                valid_subjects = load_subjects_from_file(_subjects_file) if filter_subjects else None
                
                # Charger infos participants et actimétrie (commun à toutes les pipelines)
                participants = load_participants_info(_participants_file)
                actimetry = load_actimetry_data(_actimetry_file) if _actimetry_file is not None else None
                
                # Charger données pour chaque pipeline
                pipeline_data = {}
                all_bundles = set()
                all_subjects = set()
                
                for pipeline in selected_pipelines:
                    df_pipeline, bundles, subjects = load_bundle_data(
                        _dataset_path, pipeline, MODEL,
                        valid_subjects=valid_subjects,
                        filter_by_subjects=filter_subjects
                    )
                    
                    # Fusionner avec participants/actimétrie
                    if not df_pipeline.empty:
                        df_pipeline = merge_participant_data(df_pipeline, participants, actimetry)
                    
                    pipeline_data[pipeline] = {
                        'df': df_pipeline,
                        'bundles': bundles,
                        'subjects': subjects
                    }
                    all_bundles.update(bundles)
                    all_subjects.update(subjects)
                
                # Calculer les bundles communs à toutes les pipelines
                pipeline_bundles_sets = [set(pdata['bundles']) for pdata in pipeline_data.values()]
                common_bundles = set.intersection(*pipeline_bundles_sets) if pipeline_bundles_sets else set()
                
                # Calculer le nombre max de points par pipeline (pour le mode comparaison)
                max_points_per_pipeline = {}
                for pname, pdata in pipeline_data.items():
                    df = pdata.get('df', pd.DataFrame())
                    if not df.empty and 'point_id' in df.columns:
                        # Nombre de points uniques par sujet (prendre le max)
                        points_per_subject = df.groupby('subject')['point_id'].nunique()
                        max_points_per_pipeline[pname] = int(points_per_subject.max()) if len(points_per_subject) > 0 else 0
                    else:
                        max_points_per_pipeline[pname] = 0
                
                # Stocker dans session state
                st.session_state.pipeline_data = pipeline_data
                st.session_state.selected_pipelines = selected_pipelines
                st.session_state.bundles = sorted(list(all_bundles))
                st.session_state.common_bundles = sorted(list(common_bundles))
                st.session_state.max_points_per_pipeline = max_points_per_pipeline
                st.session_state.subjects = sorted(list(all_subjects))
                st.session_state.participants = participants
                st.session_state.actimetry = actimetry
                st.session_state.data_loaded = True
                
                # Mettre à jour la valeur par défaut du nombre de points d'interpolation
                global_max_points = max(max_points_per_pipeline.values()) if max_points_per_pipeline else 100
                st.session_state.param_comparison_n_points = global_max_points
                
                st.success(f"{len(selected_pipelines)} pipeline(s), {len(all_bundles)} faisceaux ({len(common_bundles)} communs), {len(all_subjects)} sujets chargés")
    
    if not st.session_state.data_loaded:
        st.info("Cliquez sur 'Charger les données' pour commencer")
        return
    
    # Récupérer les données
    pipeline_data = st.session_state.pipeline_data
    loaded_pipelines = st.session_state.selected_pipelines
    bundles = st.session_state.bundles
    common_bundles = st.session_state.get('common_bundles', bundles)
    subjects = st.session_state.subjects
    participants = st.session_state.participants
    actimetry = st.session_state.actimetry
    
    if not pipeline_data:
        st.error("Aucune donnée chargée")
        return
    
    # Colonnes disponibles
    participant_cols = []
    if participants is not None:
        participant_cols = [c for c in participants.columns 
                          if c not in ['participant_id', 'subject_id'] 
                          and participants[c].notna().sum() > 0]
    
    actimetry_cols = []
    if actimetry is not None:
        actimetry_cols = [c for c in actimetry.columns 
                        if c not in ['participant_id', 'subject_id'] 
                        and actimetry[c].notna().sum() > 0]
    
    all_extra_cols = sorted(set(participant_cols + actimetry_cols))
    
    # Métriques et stats disponibles (intersection = communes, union = au moins une)
    common_metrics, common_stats, all_available_metrics, all_available_stats = \
        compute_metric_stat_availability(pipeline_data)

    # ==========================================================================
    # Sidebar - Paramètres de visualisation
    # ==========================================================================
    
    with st.sidebar:
        st.header("Paramètres")
        
        # Option pour filtrer les métriques communes
        common_metrics_only = st.checkbox(
            "Communs uniquement (métriques, stats, bundles)",
            value=st.session_state.param_common_metrics_only,
            key="common_metrics_cb",
            help="Si activé, n'affiche que les métriques, statistiques et bundles disponibles dans TOUTES les pipelines sélectionnées"
        )
        st.session_state.param_common_metrics_only = common_metrics_only
        
        # Déterminer les métriques, stats et bundles à afficher
        if common_metrics_only:
            display_metrics = sorted([m for m in METRICS if m in common_metrics]) if common_metrics else METRICS
            display_stats = sorted([s for s in STAT_TYPES if s in common_stats]) if common_stats else STAT_TYPES
            display_bundles = common_bundles if common_bundles else bundles
            
            # Afficher un warning si certaines métriques/stats/bundles sont exclues
            excluded_metrics = all_available_metrics - common_metrics
            excluded_stats = all_available_stats - common_stats
            excluded_bundles = set(bundles) - set(common_bundles)
            
            if excluded_metrics or excluded_stats or excluded_bundles:
                with st.expander("⚠️ Éléments exclus (non communs)", expanded=False):
                    if excluded_metrics:
                        st.caption(f"Métriques: {', '.join(sorted(excluded_metrics))}")
                    if excluded_stats:
                        st.caption(f"Stats: {', '.join(sorted(excluded_stats))}")
                    if excluded_bundles:
                        st.caption(f"Bundles: {len(excluded_bundles)} exclus")
        else:
            # Utiliser toutes les métriques disponibles (union)
            display_metrics = sorted([m for m in METRICS if m in all_available_metrics]) if all_available_metrics else METRICS
            display_stats = sorted([s for s in STAT_TYPES if s in all_available_stats]) if all_available_stats else STAT_TYPES
            display_bundles = bundles
        
        # S'assurer qu'il y a au moins une option
        if not display_metrics:
            display_metrics = METRICS
        if not display_stats:
            display_stats = STAT_TYPES
        
        # Sélection métrique et stats
        col1, col2 = st.columns(2)
        with col1:
            # Vérifier que la métrique précédente est toujours disponible
            if st.session_state.param_metric in display_metrics:
                metric_idx = display_metrics.index(st.session_state.param_metric)
            else:
                metric_idx = 0
            metric = st.selectbox("Métrique", display_metrics, index=metric_idx, key="metric_select")
            st.session_state.param_metric = metric
        with col2:
            # Vérifier que la stat précédente est toujours disponible
            if st.session_state.param_stat_type in display_stats:
                stat_idx = display_stats.index(st.session_state.param_stat_type)
            else:
                stat_idx = 0
            stat_type = st.selectbox("Statistique", display_stats, index=stat_idx, key="stat_select")
            st.session_state.param_stat_type = stat_type
        
        # Sélection faisceau (avec restriction si communs uniquement)
        bundle_idx = 0
        if st.session_state.param_selected_bundle and st.session_state.param_selected_bundle in display_bundles:
            bundle_idx = display_bundles.index(st.session_state.param_selected_bundle)
        selected_bundle = st.selectbox("Faisceau", display_bundles, index=bundle_idx if display_bundles else None, key="bundle_select")
        st.session_state.param_selected_bundle = selected_bundle

        # Sélection du centroïde
        _centroid_options_raw = set()
        for _pdata in pipeline_data.values():
            _df_tmp = _pdata.get('df', pd.DataFrame())
            if not _df_tmp.empty and 'centroid_id' in _df_tmp.columns and selected_bundle:
                _centroid_options_raw.update(
                    _df_tmp[_df_tmp['bundle'] == selected_bundle]['centroid_id'].dropna().unique()
                )
        _MEAN_LABEL = "Moyenne par point_id"
        if _centroid_options_raw:
            _centroid_sorted = sorted(_centroid_options_raw)
            _centroid_display = [_MEAN_LABEL] + [str(v) for v in _centroid_sorted]
            _current = st.session_state.param_centroid
            _current_str = _MEAN_LABEL if _current is None else str(_current)
            _centroid_idx = _centroid_display.index(_current_str) if _current_str in _centroid_display else 0
            _centroid_selected_str = st.selectbox(
                "Centroïde",
                options=_centroid_display,
                index=_centroid_idx,
                help="Sélectionnez un centroïde ou choisissez la moyenne de tous les centroïdes par point",
                key="centroid_select"
            )
            if _centroid_selected_str == _MEAN_LABEL:
                selected_centroid = None
            else:
                # Retrouver le type d'origine
                _match = [v for v in _centroid_sorted if str(v) == _centroid_selected_str]
                selected_centroid = _match[0] if _match else None
            st.session_state.param_centroid = selected_centroid
        else:
            selected_centroid = None
            st.session_state.param_centroid = None

        
        # Options d'affichage
        st.subheader("Affichage")
        show_individual = st.checkbox("Courbes individuelles", value=st.session_state.param_show_individual, key="show_indiv_cb")
        st.session_state.param_show_individual = show_individual
        
        show_group_avg = st.checkbox("Moyennes de groupe", value=st.session_state.param_show_group_avg, key="show_group_cb")
        st.session_state.param_show_group_avg = show_group_avg
        
        force_nan_zero = st.checkbox(
            "Forcer NaN à 0", 
            value=st.session_state.param_force_nan_zero, 
            key="nan_zero_cb",
            help="Remplace toutes les valeurs manquantes (NaN) par 0 dans les courbes"
        )
        st.session_state.param_force_nan_zero = force_nan_zero
        
        exclude_extremes = st.checkbox(
            "Exclure points extrêmes",
            value=st.session_state.param_exclude_extremes,
            key="exclude_extremes_cb",
            help="Exclut le premier et le dernier point de chaque faisceau (souvent bruités ou nuls)"
        )
        st.session_state.param_exclude_extremes = exclude_extremes

        interpolate_missing = st.checkbox(
            "Interpoler les points manquants",
            value=st.session_state.param_interpolate_missing,
            key="interpolate_missing_cb",
            help="Interpolation linéaire des NaN par sujet. Appliquée en priorité, avant toute autre correction."
        )
        st.session_state.param_interpolate_missing = interpolate_missing

        if interpolate_missing:
            interpolate_max_gap = st.number_input(
                "Gap max (points)",
                min_value=1, max_value=20,
                value=st.session_state.param_interpolate_max_gap,
                step=1,
                key="interpolate_max_gap_input",
                help="Nombre maximum de points consécutifs manquants à interpoler"
            )
            st.session_state.param_interpolate_max_gap = int(interpolate_max_gap)
        else:
            interpolate_max_gap = st.session_state.param_interpolate_max_gap

        show_subject_count = st.checkbox(
            "Nombre de sujets par point",
            value=st.session_state.param_show_subject_count,
            key="show_subject_count_cb",
            help="Affiche le nombre de sujets avec données valides (et manquants) par point. Le hover révèle les IDs."
        )
        st.session_state.param_show_subject_count = show_subject_count
        
        group_col = None
        if show_group_avg:
            group_options = [None] + participant_cols
            group_idx = 0
            if st.session_state.param_group_col in group_options:
                group_idx = group_options.index(st.session_state.param_group_col)
            group_col = st.selectbox(
                "Grouper par",
                options=group_options,
                index=group_idx,
                format_func=lambda x: "Aucun" if x is None else x,
                key="group_col_select"
            )
            st.session_state.param_group_col = group_col
        
        # Coloration des courbes individuelles
        curve_color_col = None
        if show_individual:
            color_options = [None] + all_extra_cols
            color_idx = 0
            if st.session_state.param_curve_color_col in color_options:
                color_idx = color_options.index(st.session_state.param_curve_color_col)
            curve_color_col = st.selectbox(
                "Colorer par",
                options=color_options,
                index=color_idx,
                format_func=lambda x: "Aucune" if x is None else x,
                help="Si >4 valeurs uniques, une échelle continue (Viridis) sera utilisée",
                key="curve_color_select"
            )
            st.session_state.param_curve_color_col = curve_color_col
        
        # Rééchantillonnage
        with st.expander("Rééchantillonnage"):
            resample = st.checkbox("Activer le rééchantillonnage", value=st.session_state.param_resample, key="resample_cb")
            st.session_state.param_resample = resample
            
            resample_points = st.slider("Nombre de points", 10, 200, st.session_state.param_resample_points, disabled=not resample, key="resample_pts")
            st.session_state.param_resample_points = resample_points
            
            interp_idx = INTERPOLATION_METHODS.index(st.session_state.param_interp_method) if st.session_state.param_interp_method in INTERPOLATION_METHODS else 0
            interp_method = st.selectbox("Interpolation", INTERPOLATION_METHODS, index=interp_idx, disabled=not resample, key="interp_select")
            st.session_state.param_interp_method = interp_method
        
        # Correction de nuisances
        with st.expander("Correction de nuisances"):
            enable_nuisance = st.checkbox("Activer la correction", value=st.session_state.param_enable_nuisance, key="nuisance_cb")
            st.session_state.param_enable_nuisance = enable_nuisance
            
            # Déterminer les valeurs par défaut pour nuisance_cols
            valid_nuisance_default = [c for c in st.session_state.param_nuisance_cols if c in participant_cols]
            if not valid_nuisance_default and not st.session_state.param_nuisance_cols:
                # Première fois: utiliser age/sex si disponibles
                valid_nuisance_default = [c for c in ['age', 'sex'] if c in participant_cols]
            
            nuisance_cols = st.multiselect(
                "Variables de nuisance",
                options=participant_cols,
                default=valid_nuisance_default,
                disabled=not enable_nuisance,
                key="nuisance_cols_select"
            )
            st.session_state.param_nuisance_cols = nuisance_cols
            
            nuisance_method_idx = CORRECTION_METHODS.index(st.session_state.param_nuisance_method) if st.session_state.param_nuisance_method in CORRECTION_METHODS else 0
            nuisance_method = st.selectbox(
                "Méthode",
                CORRECTION_METHODS,
                index=nuisance_method_idx,
                disabled=not enable_nuisance,
                key="nuisance_method_select"
            )
            st.session_state.param_nuisance_method = nuisance_method
        
        # Mode corrélation
        with st.expander("Corrélation"):
            enable_correlation = st.checkbox("Mode corrélation", value=st.session_state.param_enable_correlation, key="corr_cb")
            st.session_state.param_enable_correlation = enable_correlation
            
            corr_options = [None] + all_extra_cols
            corr_col_idx = 0
            if st.session_state.param_corr_col in corr_options:
                corr_col_idx = corr_options.index(st.session_state.param_corr_col)
            corr_col = st.selectbox(
                "Colonne de corrélation",
                options=corr_options,
                index=corr_col_idx,
                format_func=lambda x: "Aucune" if x is None else x,
                disabled=not enable_correlation,
                key="corr_col_select"
            )
            st.session_state.param_corr_col = corr_col
            
            corr_method_idx = CORRELATION_METHODS.index(st.session_state.param_corr_method) if st.session_state.param_corr_method in CORRELATION_METHODS else 0
            corr_method = st.selectbox(
                "Méthode",
                CORRELATION_METHODS,
                index=corr_method_idx,
                disabled=not enable_correlation,
                key="corr_method_select"
            )
            st.session_state.param_corr_method = corr_method
            
            # Correction multiple
            if enable_correlation:
                st.markdown("---")
                mult_corr_idx = MULTIPLE_CORRECTION_METHODS.index(st.session_state.param_multiple_corr) if st.session_state.param_multiple_corr in MULTIPLE_CORRECTION_METHODS else 0
                multiple_corr = st.selectbox("Correction multi-tests", MULTIPLE_CORRECTION_METHODS, index=mult_corr_idx, key="mult_corr_select")
                st.session_state.param_multiple_corr = multiple_corr
                
                sig_alpha = st.number_input("Seuil α", 0.001, 0.5, st.session_state.param_sig_alpha, 0.005, key="sig_alpha_input")
                st.session_state.param_sig_alpha = sig_alpha

                st.markdown("---")
                st.markdown("**Modèle OLS polynomial**")
                enable_ols = st.checkbox(
                    "Activer le modèle OLS",
                    value=st.session_state.param_enable_ols,
                    key="ols_cb",
                    help="Ajuste un modèle polynomial OLS (metric ~ var + var² + confonds) au lieu d'une simple corrélation"
                )
                st.session_state.param_enable_ols = enable_ols

                if enable_ols:
                    ols_degree = st.selectbox(
                        "Degré polynomial",
                        OLS_MODEL_DEGREES,
                        index=OLS_MODEL_DEGREES.index(st.session_state.param_ols_degree) if st.session_state.param_ols_degree in OLS_MODEL_DEGREES else 1,
                        key="ols_degree_select",
                        help="Degré du polynôme (1=linéaire, 2=quadratique, 3=cubique)"
                    )
                    st.session_state.param_ols_degree = ols_degree

                    ols_confounds = st.multiselect(
                        "Confonds OLS",
                        options=[c for c in all_extra_cols if c != corr_col],
                        default=[c for c in st.session_state.param_ols_confounds if c in all_extra_cols and c != corr_col],
                        key="ols_confounds_select",
                        help="Variables de confusion à inclure dans le modèle OLS"
                    )
                    st.session_state.param_ols_confounds = ols_confounds

                    st.markdown("---")
                    st.markdown("**Modèle interaction (slope par groupe)**")
                    enable_interaction = st.checkbox(
                        "Activer le modèle interaction",
                        value=st.session_state.param_enable_interaction,
                        key="interaction_cb",
                        help="Ajuste un modèle linéaire avec un slope différent par catégorie : metric ~ var + group + var×group + confonds"
                    )
                    st.session_state.param_enable_interaction = enable_interaction

                    if enable_interaction:
                        # Sélection colonne de groupe (catégorielle)
                        # Proposer les colonnes catégorielles des participants
                        cat_cols = [c for c in all_extra_cols if c != corr_col]
                        interaction_group_options = [None] + cat_cols
                        interaction_group_idx = 0
                        if st.session_state.param_interaction_group_col in interaction_group_options:
                            interaction_group_idx = interaction_group_options.index(st.session_state.param_interaction_group_col)
                        interaction_group_col = st.selectbox(
                            "Colonne de groupe (interaction)",
                            options=interaction_group_options,
                            index=interaction_group_idx,
                            format_func=lambda x: "Aucune" if x is None else x,
                            key="interaction_group_select",
                            help="Colonne catégorielle définissant les groupes pour les slopes différents"
                        )
                        st.session_state.param_interaction_group_col = interaction_group_col
        
        # Mode comparaison (si plusieurs pipelines)
        comparison_mode = False
        comparison_n_points = 100
        if len(loaded_pipelines) > 1:
            # Récupérer le nombre max de points par pipeline
            max_points_per_pipeline = st.session_state.get('max_points_per_pipeline', {})
            global_max_points = max(max_points_per_pipeline.values()) if max_points_per_pipeline else 100
            
            with st.expander("Mode Comparaison"):
                comparison_mode = st.checkbox(
                    "Activer le mode comparaison",
                    value=st.session_state.param_comparison_mode,
                    key="comparison_mode_cb",
                    help="Affiche toutes les pipelines sur le même graphique avec des styles différents"
                )
                st.session_state.param_comparison_mode = comparison_mode
                
                if comparison_mode:
                    # Afficher le nombre de points par pipeline
                    if max_points_per_pipeline:
                        pts_info = ", ".join([f"{p}: {n}" for p, n in max_points_per_pipeline.items()])
                        st.caption(f"📏 Points par pipeline: {pts_info}")
                    
                    # Slider avec max dynamique basé sur les données
                    slider_max = max(200, global_max_points + 50)
                    current_value = min(st.session_state.param_comparison_n_points, slider_max)
                    
                    comparison_n_points = st.slider(
                        "Points d'interpolation",
                        min_value=20,
                        max_value=slider_max,
                        value=current_value,
                        step=10,
                        key="comparison_n_points_slider",
                        help=f"Nombre de points pour l'interpolation (défaut: {global_max_points} = max des pipelines)"
                    )
                    st.session_state.param_comparison_n_points = comparison_n_points
                    
                    st.caption("💡 Les positions sont normalisées en % pour comparer des pipelines avec des nombres de points différents.")
    
    # ==========================================================================
    # Onglets : Données (table + filtres) et Analyses
    # ==========================================================================
    tab_data, tab_viz = st.tabs(["📋 Données", "📈 Analyses"])

    with tab_data:
        selected_subjects = render_subject_table_tab(subjects, participants, actimetry)
        st.session_state.param_selected_subjects = selected_subjects

    with tab_viz:
        # ==========================================================================
        # Main Content - Boucle sur chaque pipeline
        # ==========================================================================
    
        # Initialiser les états pour la corrélation (partagés entre pipelines)
        if 'selected_point_idx' not in st.session_state:
            st.session_state.selected_point_idx = 0
        if 'scatter_color_by' not in st.session_state:
            st.session_state.scatter_color_by = None
    
        # Sélection de la colonne de couleur (avant la boucle pour éviter les problèmes de state)
        color_options = [None] + participant_cols
        current_color_idx = 0
        if st.session_state.scatter_color_by in color_options:
            current_color_idx = color_options.index(st.session_state.scatter_color_by)
    
        # Widget pour la couleur du scatter (en dehors de la boucle)
        if enable_correlation and corr_col:
            color_by = st.selectbox(
                "Colorer les scatter plots par",
                options=color_options,
                index=current_color_idx,
                format_func=lambda x: "Aucune" if x is None else x,
                key="scatter_color_global"
            )
            st.session_state.scatter_color_by = color_by
        else:
            color_by = None
    
        # Créer un mapping de couleurs global pour garantir la cohérence entre pipelines
        color_mapping = None
        if enable_correlation and corr_col and color_by:
            # Collecter toutes les valeurs uniques de la colonne de couleur à travers toutes les pipelines
            all_color_values = set()
            for pname in loaded_pipelines:
                pdata = pipeline_data.get(pname, {})
                df_temp = pdata.get('df', pd.DataFrame())
                if not df_temp.empty and color_by in df_temp.columns:
                    all_color_values.update(df_temp[color_by].dropna().unique())
        
            # Créer le mapping avec des couleurs fixes
            if all_color_values:
                all_color_values = sorted(list(all_color_values))
                colors = px.colors.qualitative.Plotly
                color_mapping = {val: colors[i % len(colors)] for i, val in enumerate(all_color_values)}
    
        # Variable pour stocker les points uniques (pour la sélection)
        global_unique_points = []
        global_selected_point = None
    
        # ==========================================================================
        # Mode Comparaison - Préparer les données de toutes les pipelines
        # ==========================================================================
    
        if comparison_mode and len(loaded_pipelines) > 1:
            st.subheader("🔄 Mode Comparaison")
        
            # Préparer les données de toutes les pipelines
            pipeline_dfs_for_comparison = {}
            pipeline_corr_dfs = {}
            pipeline_sig_masks = {}
        
            for pipeline_name in loaded_pipelines:
                pdata = pipeline_data.get(pipeline_name, {})
                df_all = pdata.get('df', pd.DataFrame())
            
                if df_all.empty:
                    continue
            
                df_filtered, value_col, _ = preprocess_bundle_df(
                    df_all,
                    selected_bundle=selected_bundle,
                    selected_subjects=selected_subjects,
                    selected_centroid=selected_centroid,
                    metric=metric, stat_type=stat_type,
                    interpolate_missing=interpolate_missing,
                    interpolate_max_gap=interpolate_max_gap,
                    force_nan_zero=force_nan_zero,
                    exclude_extremes=exclude_extremes,
                    enable_nuisance=enable_nuisance,
                    nuisance_cols=nuisance_cols,
                    nuisance_method=nuisance_method,
                    group_col=group_col,
                    enable_correlation=enable_correlation,
                    corr_col=corr_col,
                )
                if df_filtered is None:
                    continue

                # Inversion par pipeline (mode comparaison)
                if st.session_state.invert_points.get(pipeline_name, False):
                    df_filtered = invert_points_order(df_filtered)

                # Positions normalisées en % puis rééchantillonnage commun à toutes les pipelines
                df_filtered = normalize_points_to_percentage(df_filtered)
            
                df_filtered = resample_bundle_df(df_filtered, comparison_n_points, 'linear')
                df_filtered[value_col] = df_filtered['value']

                pipeline_dfs_for_comparison[pipeline_name] = df_filtered
            
                # Calculer les corrélations si nécessaire
                if enable_correlation and corr_col:
                    corr_df = compute_pointwise_correlation(df_filtered, corr_col, corr_method)
                    if not corr_df.empty:
                        pipeline_corr_dfs[pipeline_name] = corr_df
                        # Correction multiple
                        if 'p' in corr_df.columns:
                            p_vals = corr_df['p'].values
                            pipeline_sig_masks[pipeline_name] = apply_multiple_correction(p_vals, multiple_corr, sig_alpha)
        
            # Options d'inversion par pipeline
            with st.expander("⚙️ Options par pipeline"):
                cols = st.columns(len(loaded_pipelines))
                for i, pipeline_name in enumerate(loaded_pipelines):
                    with cols[i]:
                        invert_key = f"invert_cmp_{pipeline_name}"
                        current_invert = st.session_state.invert_points.get(pipeline_name, False)
                        new_invert = st.checkbox(
                            f"Inverser {pipeline_name[:20]}...",
                            value=current_invert,
                            key=invert_key
                        )
                        if new_invert != current_invert:
                            st.session_state.invert_points[pipeline_name] = new_invert
                            st.rerun()
        
            # Afficher les graphiques comparatifs
            if enable_correlation and corr_col and pipeline_corr_dfs:
                # Mode corrélation comparative
                col_left, col_right = st.columns([1, 1])
            
                with col_left:
                    fig_corr = plot_correlation_curves_comparison(
                        pipeline_corr_dfs, corr_col, metric, corr_method,
                        pipeline_sig_masks=pipeline_sig_masks, alpha=sig_alpha
                    )
                    st.plotly_chart(fig_corr, use_container_width=True, key="corr_comparison_chart")
            
                with col_right:
                    # Afficher les statistiques moyennes par pipeline
                    st.markdown("**Statistiques par pipeline**")
                    for pname, corr_df in pipeline_corr_dfs.items():
                        r_mean = corr_df['r'].mean()
                        r_max = corr_df['r'].max()
                        r_min = corr_df['r'].min()
                        n_sig = pipeline_sig_masks.get(pname, np.array([])).sum() if pname in pipeline_sig_masks else 0
                        st.caption(f"**{pname}**: r̄={r_mean:.3f}, r∈[{r_min:.3f}, {r_max:.3f}], {n_sig} pts significatifs")
        
            elif pipeline_dfs_for_comparison:
                # Mode courbes comparatives
                fig = plot_bundle_curves_comparison(
                    pipeline_dfs_for_comparison, metric, stat_type,
                    show_individual=show_individual,
                    show_group_avg=show_group_avg,
                    group_col=group_col,
                    title=f"Faisceau: {selected_bundle} - Comparaison des pipelines"
                )
                st.plotly_chart(fig, use_container_width=True, key="bundle_comparison_chart")
        
            # Infos sur les données
            st.caption(f"📊 {len(pipeline_dfs_for_comparison)} pipelines comparées, {comparison_n_points} points interpolés")

            # Graphiques nombre de sujets par point (mode comparaison)
            if show_subject_count and pipeline_dfs_for_comparison:
                with st.expander("Nombre de sujets par point (comparaison)", expanded=True):
                    for pname, df_cmp in pipeline_dfs_for_comparison.items():
                        fig_count = plot_subject_count_per_point(
                            df_cmp,
                            title=f"Sujets par point — {pname}"
                        )
                        st.plotly_chart(fig_count, use_container_width=True, key=f"subject_count_cmp_{pname}")
    
        else:
            # ==========================================================================
            # Mode Normal - Boucle sur chaque pipeline chargée
            # ==========================================================================
        
            for pipeline_idx, pipeline_name in enumerate(loaded_pipelines):
                pdata = pipeline_data.get(pipeline_name, {})
                df_all = pdata.get('df', pd.DataFrame())
            
                if df_all.empty:
                    with st.container(border=True):
                        st.subheader(f"📁 Pipeline: {pipeline_name}")
                        st.warning("Aucune donnée pour cette pipeline")
                    continue
            
                df_filtered, value_col, nuisance_diag = preprocess_bundle_df(
                    df_all,
                    selected_bundle=selected_bundle,
                    selected_subjects=selected_subjects,
                    selected_centroid=selected_centroid,
                    metric=metric, stat_type=stat_type,
                    interpolate_missing=interpolate_missing,
                    interpolate_max_gap=interpolate_max_gap,
                    force_nan_zero=force_nan_zero,
                    exclude_extremes=exclude_extremes,
                    enable_nuisance=enable_nuisance,
                    nuisance_cols=nuisance_cols,
                    nuisance_method=nuisance_method,
                    group_col=group_col,
                    enable_correlation=enable_correlation,
                    corr_col=corr_col,
                )
                if df_filtered is None:
                    with st.container(border=True):
                        st.subheader(f"📁 Pipeline: {pipeline_name}")
                        st.warning(f"Aucune donnée exploitable pour le faisceau '{selected_bundle}' "
                                   f"(sujets filtrés ou métrique {metric}_{stat_type} absente)")
                    continue

                # Rééchantillonnage
                if resample:
                    df_filtered = resample_bundle_df(df_filtered, resample_points, interp_method)
                    # Mise à jour de la colonne spécifique après rééchantillonnage
                    df_filtered[value_col] = df_filtered['value']
            
                # ==========================================================================
                # Affichage dans une card pour cette pipeline
                # ==========================================================================
            
                with st.container(border=True):
                    # En-tête avec option d'inversion
                    col_header, col_invert = st.columns([4, 1])
                    with col_header:
                        st.subheader(f"📁 Pipeline: {pipeline_name}")
                    with col_invert:
                        # Checkbox pour inverser l'ordre des points
                        invert_key = f"invert_{pipeline_name}"
                        current_invert = st.session_state.invert_points.get(pipeline_name, False)
                        invert_points_flag = st.checkbox(
                            "Inverser points",
                            value=current_invert,
                            key=invert_key,
                            help="Inverse l'ordre des points le long du faisceau"
                        )
                        st.session_state.invert_points[pipeline_name] = invert_points_flag
                
                    # Appliquer l'inversion si activée
                    if invert_points_flag:
                        df_filtered = invert_points_order(df_filtered)
                
                    # Afficher info correction nuisance
                    if nuisance_diag:
                        with st.expander("Diagnostic correction de nuisances", expanded=False):
                            if nuisance_diag.get('status') == 'success':
                                st.success(f"Correction appliquée: {nuisance_diag.get('n_subjects_used')}/{nuisance_diag.get('n_subjects_total')} sujets")
                                st.json(nuisance_diag)
                            else:
                                st.error(f"{nuisance_diag.get('reason', 'Échec')}")
                                st.json(nuisance_diag)
                
                    # ==========================================================================
                    # Graphiques
                    # ==========================================================================
                
                    if enable_correlation and corr_col:
                        # Mode corrélation
                        col_left, col_right = st.columns([1, 1])
                    
                        # Calculer d'abord les données de corrélation pour avoir les points uniques
                        corr_df = compute_pointwise_correlation(df_filtered, corr_col, corr_method)
                    
                        # Définir les points uniques et le point sélectionné
                        unique_points = []
                        selected_point = None
                        selected_point_relative = None  # Position relative (0.0 à 1.0)
                    
                        if not corr_df.empty:
                            unique_points = sorted(corr_df['point_id'].unique())
                        
                            # Pour la première pipeline, initialiser les points globaux
                            if pipeline_idx == 0 and unique_points:
                                global_unique_points = unique_points
                                # S'assurer que l'index est valide (entre 0 et len-1)
                                if len(unique_points) > 0:
                                    st.session_state.selected_point_idx = max(0, min(
                                        int(st.session_state.selected_point_idx), 
                                        len(unique_points) - 1
                                    ))
                                    global_selected_point = unique_points[st.session_state.selected_point_idx]
                                    # Calculer la position relative
                                    selected_point_relative = st.session_state.selected_point_idx / max(1, len(unique_points) - 1)
                        
                            # Pour les autres pipelines, utiliser la position relative
                            if pipeline_idx > 0 and global_unique_points and unique_points:
                                # Récupérer la position relative de la première pipeline
                                global_idx = st.session_state.selected_point_idx
                                global_relative = global_idx / max(1, len(global_unique_points) - 1)
                                # Calculer l'index correspondant dans cette pipeline
                                local_idx = int(round(global_relative * max(0, len(unique_points) - 1)))
                                local_idx = max(0, min(local_idx, len(unique_points) - 1))
                                selected_point = unique_points[local_idx]
                            # Pour la première pipeline, utiliser le point sélectionné directement
                            elif global_unique_points:
                                selected_point = global_selected_point
                            elif unique_points:
                                selected_point = unique_points[0]
                    
                        with col_left:
                            # Correction multiple
                            sig_mask = None
                            if not corr_df.empty and 'p' in corr_df.columns:
                                p_vals = corr_df['p'].values
                                sig_mask = apply_multiple_correction(p_vals, multiple_corr, sig_alpha)
                        
                            fig_corr = plot_correlation_curve(
                                corr_df, corr_col, metric, corr_method,
                                sig_mask=sig_mask, alpha=sig_alpha
                            )
                        
                            # Ajouter un marqueur pour le point sélectionné
                            if not corr_df.empty and selected_point is not None:
                                point_row = corr_df[corr_df['point_id'] == selected_point]
                                if not point_row.empty:
                                    fig_corr.add_trace(go.Scatter(
                                        x=[selected_point],
                                        y=[point_row['r'].iloc[0]],
                                        mode='markers',
                                        marker=dict(size=15, color='green', symbol='circle-open', line=dict(width=3)),
                                        name='Point sélectionné',
                                        hoverinfo='skip'
                                    ))
                        
                            # Afficher avec événement de sélection (uniquement pour la première pipeline)
                            if pipeline_idx == 0 and not corr_df.empty:
                                event = st.plotly_chart(
                                    fig_corr, 
                                    use_container_width=True,
                                    on_select="rerun",
                                    selection_mode="points",
                                    key=f"corr_chart_{pipeline_idx}"
                                )
                            
                                # Gérer le clic sur le graphique
                                if event is not None:
                                    selection = getattr(event, 'selection', None)
                                    if selection is not None and not callable(selection):
                                        points = selection.get('points', []) if isinstance(selection, dict) else getattr(selection, 'points', [])
                                        if points:
                                            clicked_point = points[0]
                                            clicked_x = clicked_point.get('x') if isinstance(clicked_point, dict) else getattr(clicked_point, 'x', None)
                                            if clicked_x is not None and unique_points:
                                                # Trouver l'index du point le plus proche
                                                nearest_idx = int(np.argmin(np.abs(np.array(unique_points) - clicked_x)))
                                                if nearest_idx != st.session_state.selected_point_idx:
                                                    st.session_state.selected_point_idx = nearest_idx
                                                    st.rerun()
                            else:
                                # Pour les autres pipelines, juste afficher
                                st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_chart_{pipeline_idx}")
                        
                            # Sélection du point uniquement pour la première pipeline
                            if pipeline_idx == 0 and unique_points:
                                selected_point_idx = st.selectbox(
                                    "Point sélectionné",
                                    options=list(range(len(unique_points))),
                                    index=int(st.session_state.selected_point_idx),
                                    format_func=lambda i: f"Point {unique_points[i]:.1f}",
                                    key=f"point_selector_{pipeline_idx}"
                                )
                            
                                # Mettre à jour si changement via selectbox
                                if selected_point_idx != st.session_state.selected_point_idx:
                                    st.session_state.selected_point_idx = selected_point_idx
                                    st.rerun()
                    
                        with col_right:
                            # Scatter au point sélectionné
                            if not corr_df.empty and selected_point is not None:
                                # Si OLS activé, afficher le scatter avec courbe polynomiale ajustée
                                if st.session_state.param_enable_ols:
                                    ols_confounds = [c for c in st.session_state.param_ols_confounds if c != corr_col]
                                    ols_degree = st.session_state.param_ols_degree
                                    fig_scatter = plot_ols_scatter_at_point(
                                        df_filtered, selected_point, corr_col, metric,
                                        confound_cols=ols_confounds,
                                        poly_degree=ols_degree,
                                        color_col=color_by,
                                        color_mapping=color_mapping
                                    )
                                else:
                                    fig_scatter = plot_scatter_at_point(
                                        df_filtered, selected_point, corr_col, metric, color_by, color_mapping
                                    )
                                st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_chart_{pipeline_idx}")
                            
                                # Stats au point sélectionné
                                points = corr_df['point_id'].values
                                nearest_idx = np.argmin(np.abs(points - selected_point))
                                row = corr_df.iloc[nearest_idx]
                            
                                col_r, col_p, col_n = st.columns(3)
                                with col_r:
                                    st.metric("r", f"{row['r']:.3f}" if pd.notna(row['r']) else "N/A")
                                with col_p:
                                    st.metric("p-value", f"{row['p']:.4f}" if pd.notna(row['p']) else "N/A")
                                with col_n:
                                    st.metric("n", int(row['n']) if pd.notna(row['n']) else "N/A")

                                # --- Scatter interaction ---
                                if (st.session_state.param_enable_interaction
                                        and st.session_state.param_interaction_group_col):
                                    inter_group_col = st.session_state.param_interaction_group_col
                                    inter_confounds = [c for c in st.session_state.get('param_ols_confounds', []) if c != corr_col and c != inter_group_col]
                                    fig_inter_scatter = plot_interaction_scatter_at_point(
                                        df_filtered, selected_point, corr_col,
                                        inter_group_col, metric,
                                        confound_cols=inter_confounds,
                                        color_mapping=color_mapping
                                    )
                                    st.plotly_chart(fig_inter_scatter, use_container_width=True,
                                                    key=f"inter_scatter_{pipeline_idx}")
                    
                        # OLS along-bundle plot (pleine largeur, sous les deux colonnes)
                        if st.session_state.param_enable_ols and corr_col:
                            ols_confounds = [c for c in st.session_state.param_ols_confounds if c != corr_col]
                            ols_degree = st.session_state.param_ols_degree
                        
                            ols_df = compute_pointwise_ols_model(
                                df_filtered, corr_col, ols_confounds, ols_degree
                            )
                        
                            if not ols_df.empty:
                                # Correction multiple sur les p-values OLS
                                ols_p_vals = ols_df['p'].values
                                ols_sig_mask = apply_multiple_correction(ols_p_vals, multiple_corr, sig_alpha)
                            
                                fig_ols = plot_ols_along_bundle(
                                    ols_df, metric, corr_col,
                                    poly_degree=ols_degree,
                                    sig_mask=ols_sig_mask,
                                    alpha=sig_alpha,
                                    confound_cols=ols_confounds
                                )

                                # Afficher avec événement de sélection (clic sur beta/pval -> sélection de point)
                                if pipeline_idx == 0 and not ols_df.empty:
                                    ols_event = st.plotly_chart(
                                        fig_ols,
                                        use_container_width=True,
                                        on_select="rerun",
                                        selection_mode="points",
                                        key=f"ols_chart_{pipeline_idx}"
                                    )

                                    # Gérer le clic sur le graphique OLS
                                    if ols_event is not None:
                                        ols_selection = getattr(ols_event, 'selection', None)
                                        if ols_selection is not None and not callable(ols_selection):
                                            ols_pts = ols_selection.get('points', []) if isinstance(ols_selection, dict) else getattr(ols_selection, 'points', [])
                                            if ols_pts:
                                                ols_clicked = ols_pts[0]
                                                ols_clicked_x = ols_clicked.get('x') if isinstance(ols_clicked, dict) else getattr(ols_clicked, 'x', None)
                                                if ols_clicked_x is not None and unique_points:
                                                    nearest_ols_idx = int(np.argmin(np.abs(np.array(unique_points) - ols_clicked_x)))
                                                    if nearest_ols_idx != st.session_state.selected_point_idx:
                                                        st.session_state.selected_point_idx = nearest_ols_idx
                                                        st.rerun()
                                else:
                                    st.plotly_chart(fig_ols, use_container_width=True, key=f"ols_chart_{pipeline_idx}")
                            
                                # Stats OLS résumées
                                n_sig_ols = int(ols_sig_mask.sum()) if ols_sig_mask is not None else 0
                                mean_r2 = ols_df['r2_partial'].mean()
                                max_r2 = ols_df['r2_partial'].max()
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("R² partiel moyen", f"{mean_r2:.4f}")
                                with col_b:
                                    st.metric("R² partiel max", f"{max_r2:.4f}")
                                with col_c:
                                    st.metric("Points significatifs (OLS)", n_sig_ols)

                        # --- Modèle interaction along-bundle ---
                        if (st.session_state.param_enable_interaction
                                and st.session_state.param_interaction_group_col
                                and corr_col):
                            inter_group_col = st.session_state.param_interaction_group_col
                            inter_confounds = [c for c in st.session_state.get('param_ols_confounds', []) if c != corr_col and c != inter_group_col]

                            inter_df = compute_pointwise_interaction_model(
                                df_filtered, corr_col, inter_group_col, inter_confounds
                            )

                            if not inter_df.empty:
                                inter_p_vals = inter_df['p'].values
                                inter_sig_mask = apply_multiple_correction(inter_p_vals, multiple_corr, sig_alpha)

                                fig_inter = plot_interaction_along_bundle(
                                    inter_df, metric, corr_col, inter_group_col,
                                    sig_mask=inter_sig_mask,
                                    alpha=sig_alpha,
                                    confound_cols=inter_confounds
                                )

                                if pipeline_idx == 0 and not inter_df.empty:
                                    inter_event = st.plotly_chart(
                                        fig_inter,
                                        use_container_width=True,
                                        on_select="rerun",
                                        selection_mode="points",
                                        key=f"inter_chart_{pipeline_idx}"
                                    )
                                    if inter_event is not None:
                                        inter_selection = getattr(inter_event, 'selection', None)
                                        if inter_selection is not None and not callable(inter_selection):
                                            inter_pts = inter_selection.get('points', []) if isinstance(inter_selection, dict) else getattr(inter_selection, 'points', [])
                                            if inter_pts:
                                                inter_clicked = inter_pts[0]
                                                inter_clicked_x = inter_clicked.get('x') if isinstance(inter_clicked, dict) else getattr(inter_clicked, 'x', None)
                                                if inter_clicked_x is not None and unique_points:
                                                    nearest_inter_idx = int(np.argmin(np.abs(np.array(unique_points) - inter_clicked_x)))
                                                    if nearest_inter_idx != st.session_state.selected_point_idx:
                                                        st.session_state.selected_point_idx = nearest_inter_idx
                                                        st.rerun()
                                else:
                                    st.plotly_chart(fig_inter, use_container_width=True, key=f"inter_chart_{pipeline_idx}")

                                # Stats interaction résumées
                                n_sig_inter = int(inter_sig_mask.sum()) if inter_sig_mask is not None else 0
                                mean_r2_inter = inter_df['r2_partial'].mean()
                                max_r2_inter = inter_df['r2_partial'].max()
                                col_ia, col_ib, col_ic = st.columns(3)
                                with col_ia:
                                    st.metric("R² partiel moyen (inter.)", f"{mean_r2_inter:.4f}")
                                with col_ib:
                                    st.metric("R² partiel max (inter.)", f"{max_r2_inter:.4f}")
                                with col_ic:
                                    st.metric("Points significatifs (inter.)", n_sig_inter)
                
                    else:
                        # Mode courbes standard
                        fig = plot_bundle_curves(
                            df_filtered, metric, stat_type,
                            show_individual=show_individual,
                            show_group_avg=show_group_avg,
                            group_col=group_col,
                            color_col=curve_color_col,
                            title=f"Faisceau: {selected_bundle}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"bundle_chart_{pipeline_idx}")
                
                    # Graphique nombre de sujets par point
                    if show_subject_count:
                        fig_count = plot_subject_count_per_point(
                            df_filtered,
                            title=f"Sujets par point — {pipeline_name}"
                        )
                        st.plotly_chart(fig_count, use_container_width=True, key=f"subject_count_{pipeline_idx}")

                    # Info sur cette pipeline
                    st.caption(f"📊 {len(df_filtered['subject'].unique())} sujets, {len(df_filtered)} points de données")
    
        # ==========================================================================
        # Infos supplémentaires
        # ==========================================================================
    
        with st.expander("Informations sur les données"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pipelines", len(loaded_pipelines))
            with col2:
                st.metric("Faisceaux", len(bundles))
            with col3:
                st.metric("Sujets (total)", len(subjects))
            with col4:
                st.metric("Faisceau sélectionné", selected_bundle)
        
            st.markdown("### Colonnes disponibles")
            st.write(f"**Participants:** {', '.join(participant_cols[:10])}{'...' if len(participant_cols) > 10 else ''}")
            st.write(f"**Actimétrie:** {', '.join(actimetry_cols[:10])}{'...' if len(actimetry_cols) > 10 else ''}")


if __name__ == "__main__":
    main()
