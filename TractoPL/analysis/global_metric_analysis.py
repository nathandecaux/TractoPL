#!/usr/bin/env python3
"""
Analyse statistique des métriques moyennes par bundle.

Ce script calcule les valeurs moyennes de FA et ICVF (IFW) pour chaque bundle
et effectue des tests statistiques (corrélations de Pearson ou t-tests) avec 
les différentes features cliniques et d'actimétrie.

Usage:
    python global_metric_analysis.py
"""

import os
import sys
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Tuple
from scipy.stats import pearsonr, ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

from TractoPL.data.loader import Dataset
from TractoPL.set_config import get_HCP_bundle_names

# Configuration
dataset = 'actidep'
db_root = f'/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids'
hcp_asso_pipeline = 'hcp_association_50pts'

# Chemins des données externes
_ADDITIONAL_INFO_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/participants_full_info.xlsx"
_ACTIMETRY_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/actimetry_features.xlsx"

# Métriques à analyser
METRIC_COLUMNS_CANDIDATES = ['FA', 'IFW']

# Variables d'analyse
corr_variables = ['ami', 'aes']
classif_variables = {'group': 'with_controls', 'apathy': 'no_controls'}

confond_variables_with_control = ['age', 'sex', 'city']
confond_variables_without_control = confond_variables_with_control

CLASSIF_CONFOUND_MAP = {
    'group': confond_variables_with_control,
    'apathy': confond_variables_without_control
}

# Seuils de significativité
ALPHA = 0.05

# Seuils de significativité
ALPHA = 0.05


@lru_cache(maxsize=1)
def _load_external_tables(additional_info_path: str, actimetry_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les tables externes (participants, actimétrie) avec gestion d'erreurs et mise en cache."""
    try:
        add_df = pd.read_excel(additional_info_path)
        print(f"Loaded participants info: {len(add_df)} participants")
    except Exception as e:
        print(f"Warning: Cannot load participants_full_info.xlsx: {e}")
        add_df = pd.DataFrame()
    try:
        act_df = pd.read_excel(actimetry_path)
        print(f"Loaded actimetry features: {len(act_df)} participants")
    except Exception as e:
        print(f"Warning: Cannot load actimetry_features.xlsx: {e}")
        act_df = pd.DataFrame()
    return add_df, act_df


def compute_12h_averages(df, feature_cols):
    """
    Calcule la moyenne des 6 périodes temporelles pour chaque feature 12h.
    Ex: acti_activity_mean_12h_0, ..., acti_activity_mean_12h_5 -> acti_activity_mean_12h_avg
    """
    df_avg = df.copy()
    feature_groups = {}
    
    # Identifier les groupes de features 12h
    for col in feature_cols:
        if '12h_' in col:
            last_part = col.split('_')[-1]
            if last_part.isdigit() and int(last_part) in range(7):
                base_name = '_'.join(col.split('_')[:-1])
                if base_name not in feature_groups:
                    feature_groups[base_name] = []
                feature_groups[base_name].append(col)
    
    # Calculer les moyennes
    averaged_features = []
    for base_name, group_cols in feature_groups.items():
        if len(group_cols) == 6:
            avg_name = f"{base_name}_avg"
            df_avg[avg_name] = df_avg[group_cols].mean(axis=1)
            averaged_features.append(avg_name)
    
    return df_avg, averaged_features


def load_and_merge_bundle_csvs(bundle_name, bundle_csvs, additional_info_df, actimetry_df):
    """Charge et fusionne les CSVs pour un bundle donné."""
    metric_files = [pd.read_csv(f.path) for f in bundle_csvs]
    for df, f in zip(metric_files, bundle_csvs):
        df['subject'] = f.get_full_entities()['subject']
        df['participant_id'] = 'sub-' + df["subject"].astype(str)
    
    metrics_df = pd.concat(metric_files, ignore_index=True)
    
    # Ajout des infos externes
    if not additional_info_df.empty:
        metrics_df = metrics_df.merge(additional_info_df, on='participant_id', how='left')
    
    avg_features = []
    if not actimetry_df.empty:
        metrics_df = metrics_df.merge(actimetry_df, on='participant_id', how='left')
        
        # Calculer les moyennes 12h pour les features d'actimétrie
        acti_cols = [c for c in metrics_df.columns if '12h_' in c and c.endswith(tuple(str(i) for i in range(7)))]
        if acti_cols:
            print(f"  Calcul des moyennes 12h pour {len(set('_'.join(c.split('_')[:-1]) for c in acti_cols))} features d'actimétrie")
            metrics_df, avg_features = compute_12h_averages(metrics_df, acti_cols)
            
            # Vérifier que les features avg ont été ajoutées
            if not avg_features:
                raise ValueError(f"ERREUR: Aucune feature 12h_avg n'a été créée pour le bundle {bundle_name}")
            
            missing_avg = [f for f in avg_features if f not in metrics_df.columns]
            if missing_avg:
                raise ValueError(f"ERREUR: Features 12h_avg manquantes dans metrics_df: {missing_avg}")
            
            print(f"  ✓ {len(avg_features)} features 12h_avg créées avec succès")
    
    del metric_files
    gc.collect()
    return metrics_df


def detect_metric_columns(df):
    """Détecte les colonnes de métriques dans le DataFrame."""
    return [c for c in METRIC_COLUMNS_CANDIDATES if c in df.columns]


def ols_residualize(y, X):
    """
    Retourne residus + intercept pour conserver le niveau moyen.
    y: Series
    X: DataFrame (peut contenir NaN, gérés par drop)
    """
    df = pd.concat([y, X], axis=1)
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float)
    
    y_clean = df.iloc[:, 0]
    X_clean = df.iloc[:, 1:]
    if X_clean.empty:
        return y_clean
    
    Xc = sm.add_constant(X_clean, has_constant='add')
    try:
        model = sm.OLS(y_clean, Xc, missing='drop').fit()
        residuals = model.resid
        return residuals + model.params['const']
    except Exception:
        return y_clean


def residualize_on_confond(df_subject_level, target_col, confond):
    """Applique la correction des confondants."""
    cols = [c for c in confond if c in df_subject_level.columns]
    if not cols:
        return df_subject_level[target_col]
    
    X = df_subject_level[cols].copy()
    # Encodage catégoriel
    for c in cols:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            X[c] = X[c].astype('category').cat.codes
    
    return ols_residualize(df_subject_level[target_col], X)


def compute_bundle_means(bundle_df, metric_cols):
    """
    Calcule la moyenne de chaque métrique pour chaque sujet sur l'ensemble du bundle.
    
    Returns:
        DataFrame avec colonnes: subject, participant_id, metric_name, mean_value, + autres colonnes
    """
    # Colonnes à conserver (métadonnées sujet)
    meta_cols = [c for c in bundle_df.columns if c not in metric_cols and c != 'point_id']
    
    # Grouper par sujet et calculer la moyenne de chaque métrique
    subject_means = []
    for subject, group in bundle_df.groupby('subject'):
        row_data = {'subject': subject}
        
        # Ajouter les métadonnées (prendre la première valeur, elles sont identiques pour un sujet)
        for col in meta_cols:
            if col in group.columns:
                row_data[col] = group[col].iloc[0]
        
        # Calculer les moyennes des métriques
        for metric in metric_cols:
            if metric in group.columns:
                mean_val = group[metric].mean()
                row_data[f'{metric}_mean'] = mean_val
        
        subject_means.append(row_data)
    
    return pd.DataFrame(subject_means)


def test_correlation_global(df_means, metric_name, var_name, confounds=None, filter_group=None):
    """
    Teste la corrélation entre la moyenne d'une métrique et une variable.
    
    Parameters:
        filter_group: Si fourni, filtre les sujets par groupe (ex: 'dep')
    
    Returns:
        dict avec r, p, n, r_partial (si confounds)
    """
    metric_col = f'{metric_name}_mean'
    
    if metric_col not in df_means.columns or var_name not in df_means.columns:
        return None
    
    # Filtrer par groupe si demandé
    df_filtered = df_means
    if filter_group and 'group' in df_means.columns:
        df_filtered = df_means[df_means['group'] == filter_group]
    
    # Filtrer les valeurs manquantes
    data = df_filtered[[metric_col, var_name]].dropna()
    
    if len(data) < 3:
        return None
    
    # Corrélation brute
    try:
        r_raw, p_raw = pearsonr(data[metric_col], data[var_name])
    except Exception as e:
        print(f"Error in correlation: {e}")
        return None
    
    result = {
        'r_raw': r_raw,
        'p_raw': p_raw,
        'n': len(data)
    }
    
    # Corrélation partielle si confounds
    if confounds and len(confounds) > 0:
        try:
            # Résidualiser la métrique
            metric_res = residualize_on_confond(df_filtered, metric_col, confounds)
            # Résidualiser la variable
            var_res = residualize_on_confond(df_filtered, var_name, confounds)
            
            # Aligner les indices
            common_idx = metric_res.index.intersection(var_res.index)
            metric_res = metric_res.loc[common_idx]
            var_res = var_res.loc[common_idx]
            
            if len(metric_res) >= 3:
                r_partial, p_partial = pearsonr(metric_res, var_res)
                result['r_partial'] = r_partial
                result['p_partial'] = p_partial
        except Exception as e:
            print(f"Error in partial correlation: {e}")
    
    return result


def test_ttest_global(df_means, metric_name, classif_var, confounds=None):
    """
    Teste la différence entre groupes pour la moyenne d'une métrique.
    
    Returns:
        dict avec t, p, n_group0, n_group1, mean_group0, mean_group1, cohens_d, + versions corrigées
    """
    metric_col = f'{metric_name}_mean'
    
    if metric_col not in df_means.columns or classif_var not in df_means.columns:
        return None
    
    data = df_means[[metric_col, classif_var]].dropna()
    groups = data[classif_var].unique()
    
    if len(groups) != 2:
        return None
    
    g0, g1 = sorted(groups, key=lambda x: str(x))
    
    # Test brut
    group0_data = data[data[classif_var] == g0][metric_col]
    group1_data = data[data[classif_var] == g1][metric_col]
    
    if len(group0_data) < 2 or len(group1_data) < 2:
        return None
    
    try:
        t_raw, p_raw = ttest_ind(group0_data, group1_data)
        
        # Cohen's d
        pooled_std = np.sqrt(
            ((len(group0_data) - 1) * group0_data.std() ** 2 + 
             (len(group1_data) - 1) * group1_data.std() ** 2) / 
            (len(group0_data) + len(group1_data) - 2)
        )
        cohens_d_raw = (group0_data.mean() - group1_data.mean()) / pooled_std if pooled_std > 0 else 0
    except Exception as e:
        print(f"Error in t-test: {e}")
        return None
    
    result = {
        't_raw': t_raw,
        'p_raw': p_raw,
        'n_group0': len(group0_data),
        'n_group1': len(group1_data),
        'mean_group0': group0_data.mean(),
        'mean_group1': group1_data.mean(),
        'group0_name': str(g0),
        'group1_name': str(g1),
        'cohens_d_raw': cohens_d_raw
    }
    
    # Test corrigé si confounds
    if confounds and len(confounds) > 0:
        try:
            metric_res = residualize_on_confond(df_means, metric_col, confounds)
            
            # Récupérer les groupes après résidualisation
            df_res = pd.DataFrame({
                'metric_res': metric_res,
                'group': df_means.loc[metric_res.index, classif_var]
            }).dropna()
            
            if len(df_res) >= 4:
                group0_res = df_res[df_res['group'] == g0]['metric_res']
                group1_res = df_res[df_res['group'] == g1]['metric_res']
                
                if len(group0_res) >= 2 and len(group1_res) >= 2:
                    t_corr, p_corr = ttest_ind(group0_res, group1_res)
                    
                    pooled_std_corr = np.sqrt(
                        ((len(group0_res) - 1) * group0_res.std() ** 2 + 
                         (len(group1_res) - 1) * group1_res.std() ** 2) / 
                        (len(group0_res) + len(group1_res) - 2)
                    )
                    cohens_d_corr = (group0_res.mean() - group1_res.mean()) / pooled_std_corr if pooled_std_corr > 0 else 0
                    
                    result['t_corrected'] = t_corr
                    result['p_corrected'] = p_corr
                    result['cohens_d_corrected'] = cohens_d_corr
        except Exception as e:
            print(f"Error in corrected t-test: {e}")
    
    return result


def analyze_bundle(bundle_name, bundle_df, additional_info_df, actimetry_df, 
                   corr_vars, classif_vars, confound_map):
    """Analyse complète d'un bundle."""
    print(f"\nAnalyzing bundle: {bundle_name}")
    
    # Détecter les métriques
    metric_cols = detect_metric_columns(bundle_df)
    if not metric_cols:
        print(f"  No metrics found for {bundle_name}")
        return {'correlations': [], 'ttests': []}
    
    print(f"  Metrics: {', '.join(metric_cols)}")
    
    # Calculer les moyennes par sujet
    df_means = compute_bundle_means(bundle_df, metric_cols)
    print(f"  Computed means for {len(df_means)} subjects")
    
    results = {'correlations': [], 'ttests': []}
    
    # Tests de corrélation
    for var in corr_vars:
        if var not in df_means.columns:
            continue
        
        confounds = confound_map.get('correlation', confond_variables_with_control)
        
        # Déterminer si c'est une variable d'actimétrie (contient 12h_avg)
        is_actimetry = '12h_avg' in var
        filter_group = 'dep' if is_actimetry else None
        
        for metric in metric_cols:
            # Test brut
            result_raw = test_correlation_global(df_means, metric, var, confounds=None, filter_group=filter_group)
            if result_raw:
                results['correlations'].append({
                    'bundle': bundle_name,
                    'metric': metric,
                    'variable': var,
                    'analysis': 'raw',
                    **result_raw
                })
            
            # Test corrigé
            result_corr = test_correlation_global(df_means, metric, var, confounds=confounds, filter_group=filter_group)
            if result_corr and 'r_partial' in result_corr:
                results['correlations'].append({
                    'bundle': bundle_name,
                    'metric': metric,
                    'variable': var,
                    'analysis': 'corrected',
                    'r_raw': result_corr.get('r_partial'),
                    'p_raw': result_corr.get('p_partial'),
                    'n': result_corr['n']
                })
    
    # T-tests
    for classif_var in classif_vars.keys():
        if classif_var not in df_means.columns:
            continue
        
        confounds = confound_map.get(classif_var, confond_variables_with_control)
        
        for metric in metric_cols:
            result = test_ttest_global(df_means, metric, classif_var, confounds=confounds)
            if result:
                # Résultat brut
                results['ttests'].append({
                    'bundle': bundle_name,
                    'metric': metric,
                    'variable': classif_var,
                    'analysis': 'raw',
                    't': result['t_raw'],
                    'p': result['p_raw'],
                    'cohens_d': result['cohens_d_raw'],
                    **{k: v for k, v in result.items() if k not in ['t_raw', 'p_raw', 'cohens_d_raw', 't_corrected', 'p_corrected', 'cohens_d_corrected']}
                })
                
                # Résultat corrigé si disponible
                if 't_corrected' in result:
                    results['ttests'].append({
                        'bundle': bundle_name,
                        'metric': metric,
                        'variable': classif_var,
                        'analysis': 'corrected',
                        't': result['t_corrected'],
                        'p': result['p_corrected'],
                        'cohens_d': result['cohens_d_corrected'],
                        **{k: v for k, v in result.items() if k not in ['t_raw', 'p_raw', 'cohens_d_raw', 't_corrected', 'p_corrected', 'cohens_d_corrected']}
                    })
    
    return results


def apply_fdr_correction(df_results, p_col='p_raw'):
    """Applique une correction FDR sur les p-values."""
    if len(df_results) == 0 or p_col not in df_results.columns:
        return df_results
    
    _, p_corrected, _, _ = multipletests(
        df_results[p_col], 
        alpha=ALPHA, 
        method='fdr_bh'
    )
    
    df_results['p_fdr'] = p_corrected
    df_results['significant_fdr'] = p_corrected < ALPHA
    
    return df_results


def save_results(all_results, output_dir):
    """Sauvegarde les résultats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Corrélations
    if all_results['correlations']:
        df_corr = pd.DataFrame(all_results['correlations'])
        df_corr = apply_fdr_correction(df_corr, 'p_raw')
        
        corr_path = output_path / 'global_mean_correlations.csv'
        df_corr.to_csv(corr_path, index=False)
        print(f"\nSaved correlations: {corr_path}")
        
        # Significatifs uniquement
        df_sig = df_corr[df_corr['significant_fdr']]
        if len(df_sig) > 0:
            sig_path = output_path / 'global_mean_correlations_significant.csv'
            df_sig.to_csv(sig_path, index=False)
            print(f"Saved {len(df_sig)} significant correlations: {sig_path}")
    
    # T-tests
    if all_results['ttests']:
        df_ttest = pd.DataFrame(all_results['ttests'])
        df_ttest = apply_fdr_correction(df_ttest, 'p')
        
        ttest_path = output_path / 'global_mean_ttests.csv'
        df_ttest.to_csv(ttest_path, index=False)
        print(f"Saved t-tests: {ttest_path}")
        
        # Significatifs uniquement
        df_sig = df_ttest[df_ttest['significant_fdr']]
        if len(df_sig) > 0:
            sig_path = output_path / 'global_mean_ttests_significant.csv'
            df_sig.to_csv(sig_path, index=False)
            print(f"Saved {len(df_sig)} significant t-tests: {sig_path}")


def create_summary_plots(all_results, output_dir):
    """Crée des graphiques de synthèse combinant FA et IFW."""
    output_path = Path(output_dir)
    
    # 1. Matrices de corrélation COMBINÉES FA+IFW par test
    if all_results['correlations']:
        df_corr = pd.DataFrame(all_results['correlations'])
        df_corr = apply_fdr_correction(df_corr, 'p_raw')
        
        # Filtrer pour ne garder que les features 12h_avg pour l'actimétrie
        clinical_tests = ['ami', 'aes', 'age', 'duration_dep', 'sex', 'group']
        is_clinical = df_corr['variable'].isin(clinical_tests)
        has_12h_avg = df_corr['variable'].str.contains('12h_avg', na=False)
        
        print(f"\nGénération des matrices de corrélation:")
        print(f"  Total variables: {len(df_corr['variable'].unique())}")
        print(f"  Variables cliniques: {sum(is_clinical.unique())}")
        print(f"  Features 12h_avg: {sum(has_12h_avg.unique())}")
        
        df_corr = df_corr[is_clinical | has_12h_avg].copy()
        print(f"  Après filtrage: {len(df_corr['variable'].unique())} variables")
        
        # Créer une colonne de groupe pour les features d'actimétrie
        def get_feature_group(var):
            if var in clinical_tests:
                return var
            parts = var.split('_')
            return parts[0] if len(parts) >= 1 else var
        
        df_corr['feature_group'] = df_corr['variable'].apply(get_feature_group)
        
        # Afficher les groupes trouvés
        acti_groups = df_corr[~df_corr['variable'].isin(clinical_tests)]['feature_group'].unique()
        print(f"  Groupes d'actimétrie trouvés: {sorted(acti_groups)}")
        
        for analysis in ['raw', 'corrected']:
            df_ana = df_corr[df_corr['analysis'] == analysis]
            
            if len(df_ana) == 0:
                continue
            
            # Test 1: Variables cliniques (ami, aes, etc.)
            df_clinical = df_ana[df_ana['variable'].isin(clinical_tests)]
            
            if len(df_clinical) > 0:
                # Créer un graphique combiné FA+IFW
                fig, axes = plt.subplots(1, 2, figsize=(18, max(10, len(df_clinical['bundle'].unique()) * 0.4)))
                
                for idx, metric in enumerate(['FA', 'IFW']):
                    df_metric = df_clinical[df_clinical['metric'] == metric]
                    
                    if len(df_metric) == 0:
                        axes[idx].text(0.5, 0.5, f'Pas de données pour {metric}', 
                                      ha='center', va='center', transform=axes[idx].transAxes)
                        axes[idx].set_title(f'{metric}')
                        continue
                    
                    pivot_r = df_metric.pivot_table(
                        index='bundle',
                        columns='variable',
                        values='r_raw',
                        aggfunc='first'
                    )
                    
                    pivot_sig = df_metric.pivot_table(
                        index='bundle',
                        columns='variable',
                        values='significant_fdr',
                        aggfunc='first'
                    ).fillna(False)
                    
                    if not pivot_r.empty:
                        # Créer annotations avec étoiles
                        annot_matrix = pivot_r.copy()
                        for i in range(len(pivot_r)):
                            for j in range(len(pivot_r.columns)):
                                val = pivot_r.iloc[i, j]
                                is_sig = pivot_sig.iloc[i, j]
                                if pd.notna(val):
                                    annot_matrix.iloc[i, j] = f"{val:.2f}{'*' if is_sig else ''}"
                                else:
                                    annot_matrix.iloc[i, j] = ""
                        
                        mask = pivot_r.isna()
                        
                        sns.heatmap(
                            pivot_r,
                            mask=mask,
                            cmap='RdBu_r',
                            center=0,
                            vmin=-1,
                            vmax=1,
                            annot=annot_matrix,
                            fmt='',
                            cbar_kws={'label': 'Corrélation (r)'},
                            linewidths=0.5,
                            linecolor='gray',
                            ax=axes[idx],
                            annot_kws={'size': 8}
                        )
                        
                        axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
                        axes[idx].set_xlabel('Variable clinique', fontsize=10)
                        axes[idx].set_ylabel('Bundle' if idx == 0 else '', fontsize=10)
                
                fig.suptitle(f'Corrélations avec variables cliniques ({analysis})\n(* = FDR < {ALPHA})',
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                plot_path = output_path / f'matrix_correlations_clinical_{analysis}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved clinical correlation matrix: {plot_path}")
            
            # Test 2: Features d'actimétrie par groupe
            df_acti = df_ana[~df_ana['variable'].isin(clinical_tests)]
            
            print(f"\n  Features d'actimétrie pour {analysis}:")
            print(f"    Total lignes: {len(df_acti)}")
            print(f"    Variables uniques: {len(df_acti['variable'].unique())}")
            
            if len(df_acti) > 0:
                # Obtenir les groupes uniques
                feature_groups = sorted(df_acti['feature_group'].unique())
                print(f"    Groupes: {feature_groups}")
                
                for group in feature_groups:
                    df_group = df_acti[df_acti['feature_group'] == group]
                    
                    print(f"\n    Groupe '{group}':")
                    print(f"      Lignes: {len(df_group)}")
                    print(f"      Variables: {sorted(df_group['variable'].unique())}")
                    print(f"      Bundles: {len(df_group['bundle'].unique())}")
                    
                    if len(df_group) == 0:
                        continue
                    
                    # Créer un graphique combiné FA+IFW
                    fig, axes = plt.subplots(1, 2, figsize=(16, max(10, len(df_group['bundle'].unique()) * 0.4)))
                    
                    for idx, metric in enumerate(['FA', 'IFW']):
                        df_metric = df_group[df_group['metric'] == metric]
                        
                        if len(df_metric) == 0:
                            axes[idx].text(0.5, 0.5, f'Pas de données pour {metric}', 
                                          ha='center', va='center', transform=axes[idx].transAxes)
                            axes[idx].set_title(f'{metric}')
                            continue
                        
                        pivot_r = df_metric.pivot_table(
                            index='bundle',
                            columns='variable',
                            values='r_raw',
                            aggfunc='first'
                        )
                        
                        print(f"      {metric}: pivot shape = {pivot_r.shape}, empty = {pivot_r.empty}")
                        if not pivot_r.empty:
                            print(f"        Columns: {list(pivot_r.columns[:3])}...")
                            print(f"        Index: {list(pivot_r.index[:3])}...")
                        
                        pivot_sig = df_metric.pivot_table(
                            index='bundle',
                            columns='variable',
                            values='significant_fdr',
                            aggfunc='first'
                        ).fillna(False)
                        
                        if not pivot_r.empty:
                            # Créer annotations
                            annot_matrix = pivot_r.copy()
                            for i in range(len(pivot_r)):
                                for j in range(len(pivot_r.columns)):
                                    val = pivot_r.iloc[i, j]
                                    is_sig = pivot_sig.iloc[i, j]
                                    if pd.notna(val):
                                        annot_matrix.iloc[i, j] = f"{val:.2f}{'*' if is_sig else ''}"
                                    else:
                                        annot_matrix.iloc[i, j] = ""
                            
                            mask = pivot_r.isna()
                            
                            sns.heatmap(
                                pivot_r,
                                mask=mask,
                                cmap='RdBu_r',
                                center=0,
                                vmin=-1,
                                vmax=1,
                                annot=annot_matrix,
                                fmt='',
                                cbar_kws={'label': 'Corrélation (r)'},
                                linewidths=0.5,
                                linecolor='gray',
                                ax=axes[idx],
                                annot_kws={'size': 7}
                            )
                            
                            axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
                            axes[idx].set_xlabel('Feature actimétrie', fontsize=10)
                            axes[idx].set_ylabel('Bundle' if idx == 0 else '', fontsize=10)
                            axes[idx].tick_params(axis='x', rotation=45, labelsize=8)
                    
                    fig.suptitle(f'Corrélations avec features actimétrie - Groupe: {group} ({analysis})\n(* = FDR < {ALPHA})',
                               fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    plot_path = output_path / f'matrix_correlations_acti_{group}_{analysis}.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved actimetry correlation matrix: {plot_path}")
    
    # 2. Matrices de t-tests COMBINÉES FA+IFW par variable
    if all_results['ttests']:
        df_ttest = pd.DataFrame(all_results['ttests'])
        df_ttest = apply_fdr_correction(df_ttest, 'p')
        
        for analysis in ['raw', 'corrected']:
            df_ana = df_ttest[df_ttest['analysis'] == analysis]
            
            if len(df_ana) == 0:
                continue
            
            # Créer un graphique par variable de classification
            for variable in sorted(df_ana['variable'].unique()):
                df_var = df_ana[df_ana['variable'] == variable]
                
                # Créer un graphique combiné FA+IFW
                fig, axes = plt.subplots(1, 2, figsize=(16, max(10, len(df_var['bundle'].unique()) * 0.4)))
                
                for idx, metric in enumerate(['FA', 'IFW']):
                    df_metric = df_var[df_var['metric'] == metric]
                    
                    if len(df_metric) == 0:
                        axes[idx].text(0.5, 0.5, f'Pas de données pour {metric}', 
                                      ha='center', va='center', transform=axes[idx].transAxes)
                        axes[idx].set_title(f'{metric}')
                        continue
                    
                    # Matrice des Cohen's d
                    pivot_d = df_metric.pivot_table(
                        index='bundle',
                        columns='variable',
                        values='cohens_d',
                        aggfunc='first'
                    )
                    
                    pivot_sig = df_metric.pivot_table(
                        index='bundle',
                        columns='variable',
                        values='significant_fdr',
                        aggfunc='first'
                    ).fillna(False)
                    
                    if not pivot_d.empty:
                        # Créer annotations personnalisées
                        annot_matrix = pivot_d.copy()
                        for i in range(len(pivot_d)):
                            for j in range(len(pivot_d.columns)):
                                val = pivot_d.iloc[i, j]
                                is_sig = pivot_sig.iloc[i, j]
                                if pd.notna(val):
                                    annot_matrix.iloc[i, j] = f"{val:.2f}{'*' if is_sig else ''}"
                                else:
                                    annot_matrix.iloc[i, j] = ""
                        
                        # Masquer NaN
                        mask = pivot_d.isna()
                        
                        # Déterminer la plage pour une échelle symétrique
                        max_abs = pivot_d.abs().max().max()
                        if pd.notna(max_abs):
                            vmax = min(2.0, max_abs * 1.1)
                            vmin = -vmax
                        else:
                            vmax = 1.0
                            vmin = -1.0
                        
                        sns.heatmap(
                            pivot_d,
                            mask=mask,
                            cmap='RdBu_r',
                            center=0,
                            vmin=vmin,
                            vmax=vmax,
                            annot=annot_matrix,
                            fmt='',
                            cbar_kws={'label': "Cohen's d"},
                            linewidths=0.5,
                            linecolor='gray',
                            ax=axes[idx],
                            annot_kws={'size': 8}
                        )
                        
                        axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
                        axes[idx].set_xlabel('')
                        axes[idx].set_ylabel('Bundle' if idx == 0 else '', fontsize=10)
                
                fig.suptitle(f"Tailles d'effet (Cohen's d) - {variable} ({analysis})\n(* = FDR < {ALPHA})",
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                plot_path = output_path / f'matrix_ttests_{variable}_{analysis}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved t-test matrix: {plot_path}")
    
    # 3. Heatmap des corrélations SIGNIFICATIVES UNIQUEMENT (ancien comportement)
    if all_results['correlations']:
        df_corr = pd.DataFrame(all_results['correlations'])
        df_corr = apply_fdr_correction(df_corr, 'p_raw')
        df_sig = df_corr[df_corr['significant_fdr'] & (df_corr['analysis'] == 'corrected')]
        
        if len(df_sig) > 0:
            for metric in df_sig['metric'].unique():
                df_metric = df_sig[df_sig['metric'] == metric]
                
                pivot = df_metric.pivot_table(
                    index='bundle',
                    columns='variable',
                    values='r_raw',
                    aggfunc='first'
                )
                
                if not pivot.empty:
                    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.3)))
                    sns.heatmap(
                        pivot,
                        cmap='RdBu_r',
                        center=0,
                        vmin=-1,
                        vmax=1,
                        annot=True,
                        fmt='.2f',
                        cbar_kws={'label': 'Corrélation (r)'},
                        ax=ax
                    )
                    ax.set_title(f'Corrélations significatives (moyennes bundle) - {metric}\n(FDR < {ALPHA})')
                    plt.tight_layout()
                    
                    plot_path = output_path / f'heatmap_correlations_significant_{metric}.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved significant correlations heatmap: {plot_path}")
    
    # 4. Bar plot des Cohen's d SIGNIFICATIFS (ancien comportement)
    if all_results['ttests']:
        df_ttest = pd.DataFrame(all_results['ttests'])
        df_ttest = apply_fdr_correction(df_ttest, 'p')
        df_sig = df_ttest[df_ttest['significant_fdr'] & (df_ttest['analysis'] == 'corrected')]
        
        if len(df_sig) > 0:
            for metric in df_sig['metric'].unique():
                df_metric = df_sig[df_sig['metric'] == metric].copy()
                df_metric['abs_d'] = df_metric['cohens_d'].abs()
                df_metric = df_metric.sort_values('abs_d', ascending=True)
                
                fig, ax = plt.subplots(figsize=(10, max(6, len(df_metric) * 0.3)))
                
                colors = ['red' if d < 0 else 'blue' for d in df_metric['cohens_d']]
                ax.barh(range(len(df_metric)), df_metric['cohens_d'], color=colors, alpha=0.7)
                
                labels = [f"{row['bundle']} ({row['variable']})" for _, row in df_metric.iterrows()]
                ax.set_yticks(range(len(df_metric)))
                ax.set_yticklabels(labels)
                ax.set_xlabel("Cohen's d")
                ax.set_title(f"Tailles d'effet significatives (moyennes bundle) - {metric}\n(FDR < {ALPHA})")
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                plot_path = output_path / f'barplot_cohens_d_{metric}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved Cohen's d bar plot: {plot_path}")


def generate_summary_report(all_results, output_dir):
    """Génère un rapport textuel."""
    output_path = Path(output_dir)
    report_path = output_path / 'global_mean_analysis_summary.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANALYSE DES MÉTRIQUES MOYENNES PAR BUNDLE\n")
        f.write("="*80 + "\n\n")
        
        # Corrélations
        f.write("CORRÉLATIONS\n")
        f.write("-"*80 + "\n\n")
        
        if all_results['correlations']:
            df_corr = pd.DataFrame(all_results['correlations'])
            df_corr = apply_fdr_correction(df_corr, 'p_raw')
            
            n_total = len(df_corr)
            n_sig = df_corr['significant_fdr'].sum()
            
            f.write(f"Total tests: {n_total}\n")
            f.write(f"Significatifs (FDR < {ALPHA}): {n_sig}\n\n")
            
            for analysis in ['raw', 'corrected']:
                df_ana = df_corr[df_corr['analysis'] == analysis]
                df_sig = df_ana[df_ana['significant_fdr']]
                
                f.write(f"\n{analysis.upper()}:\n")
                f.write(f"  Tests: {len(df_ana)}\n")
                f.write(f"  Significatifs: {len(df_sig)}\n")
                
                if len(df_sig) > 0:
                    f.write(f"\n  Top 10 (par |r|):\n")
                    df_top = df_sig.copy()
                    df_top['abs_r'] = df_top['r_raw'].abs()
                    df_top = df_top.sort_values('abs_r', ascending=False).head(10)
                    
                    for _, row in df_top.iterrows():
                        f.write(f"    {row['bundle']:25s} - {row['metric']:4s} - {row['variable']:10s}: "
                               f"r={row['r_raw']:6.3f}, p={row['p_raw']:.2e}, n={row['n']}\n")
        
        # T-tests
        f.write("\n\n" + "="*80 + "\n")
        f.write("T-TESTS\n")
        f.write("-"*80 + "\n\n")
        
        if all_results['ttests']:
            df_ttest = pd.DataFrame(all_results['ttests'])
            df_ttest = apply_fdr_correction(df_ttest, 'p')
            
            n_total = len(df_ttest)
            n_sig = df_ttest['significant_fdr'].sum()
            
            f.write(f"Total tests: {n_total}\n")
            f.write(f"Significatifs (FDR < {ALPHA}): {n_sig}\n\n")
            
            for analysis in ['raw', 'corrected']:
                df_ana = df_ttest[df_ttest['analysis'] == analysis]
                df_sig = df_ana[df_ana['significant_fdr']]
                
                f.write(f"\n{analysis.upper()}:\n")
                f.write(f"  Tests: {len(df_ana)}\n")
                f.write(f"  Significatifs: {len(df_sig)}\n")
                
                if len(df_sig) > 0:
                    f.write(f"\n  Top 10 (par |Cohen's d|):\n")
                    df_top = df_sig.copy()
                    df_top['abs_d'] = df_top['cohens_d'].abs()
                    df_top = df_top.sort_values('abs_d', ascending=False).head(10)
                    
                    for _, row in df_top.iterrows():
                        f.write(f"    {row['bundle']:25s} - {row['metric']:4s} - {row['variable']:10s}: "
                               f"d={row['cohens_d']:6.3f}, p={row['p']:.2e}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nGenerated summary report: {report_path}")


def main():
    global corr_variables, classif_variables
    
    print("="*80)
    print("ANALYSE DES MÉTRIQUES MOYENNES PAR BUNDLE")
    print("="*80)
    
    # Configuration du pipeline
    pipeline = "actimetry_base_confounds"
    
    # Charger les données externes
    print("\nLoading external data...")
    additional_info_df, actimetry_df = _load_external_tables(_ADDITIONAL_INFO_PATH, _ACTIMETRY_PATH)
    
    # Ajouter les features d'actimétrie aux variables de corrélation si applicable
    if 'actimetry' in pipeline and not actimetry_df.empty:
        actimetry_columns = [c for c in actimetry_df.columns if c not in ['subject_id', 'participant_id']]
        print(f"Adding {len(actimetry_columns)} actimetry features to correlation variables")

        acti_3d = [c for c in actimetry_columns if c.endswith('_3d')]
        acti_12h = ['_'.join(c.split('_')[:-1])+'_avg' for c in actimetry_columns if '_12h_' in c]
        # actimetry_columns = list(set(acti_3d) + set(acti_12h))
        actimetry_columns = list(set(acti_12h))+ list(set(acti_3d))

        corr_variables = corr_variables + actimetry_columns
        print(actimetry_columns)

    # Initialiser le dataset
    print("\nInitializing dataset...")
    ds = Dataset(db_root)
    
    # Récupérer les fichiers CSV
    print("Loading CSV files...")
    csv_files = ds.get_global(pipeline=hcp_asso_pipeline, extension='csv', datatype='metric', suffix='mean')
    print(f"Found {len(csv_files)} CSV files")
    
    # Récupérer les noms de bundles
    bundle_names = list(get_HCP_bundle_names().keys())
    print(f"Analyzing {len(bundle_names)} bundles")
    
    # Configuration de la sortie
    hostname = os.uname().nodename
    out_base = '/home/ndecaux' if hostname != 'calcarine' else '/data/ndecaux'
    output_dir = f'{out_base}/report_{dataset}_{hcp_asso_pipeline}_{pipeline}_global_means'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Charger et analyser tous les bundles
    all_results = {'correlations': [], 'ttests': []}
    
    for bundle_name in tqdm(bundle_names, desc="Analyzing bundles"):
        bundle_name_clean = bundle_name.replace("_", "")
        bundle_csvs = [f for f in csv_files if f.get_entities()['bundle'] == bundle_name_clean]
        
        if not bundle_csvs:
            continue
        
        try:
            # Charger les données du bundle
            bundle_df = load_and_merge_bundle_csvs(bundle_name_clean, bundle_csvs, 
                                                   additional_info_df, actimetry_df)
            
            # Analyser le bundle
            bundle_results = analyze_bundle(bundle_name_clean, bundle_df, additional_info_df, 
                                           actimetry_df, corr_variables, classif_variables, 
                                           CLASSIF_CONFOUND_MAP)
            
            # Accumuler les résultats
            all_results['correlations'].extend(bundle_results['correlations'])
            all_results['ttests'].extend(bundle_results['ttests'])
            
            del bundle_df
            gc.collect()
            
        except Exception as e:
            print(f"Error analyzing {bundle_name}: {e}")
            continue
    
    # Sauvegarder les résultats
    print("\n" + "="*80)
    print("Saving results...")
    save_results(all_results, output_dir)
    
    # Créer les graphiques
    print("\nGenerating plots...")
    create_summary_plots(all_results, output_dir)
    
    # Générer le rapport
    print("\nGenerating summary report...")
    generate_summary_report(all_results, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
