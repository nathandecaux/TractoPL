#!/usr/bin/env python3
"""
Régénère summary_results.csv à partir des CSV individuels dans le dossier figures/
d'un rapport generate_report_v2.

Usage:
    python regenerate_summary_from_figures.py /chemin/vers/report_dir
    python regenerate_summary_from_figures.py /chemin/vers/report_dir --output summary_results_new.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def count_clusters(sig_series):
    """Compte le nombre de clusters contigus de True."""
    cnt = 0
    in_cluster = False
    for v in sig_series:
        if v and not in_cluster:
            cnt += 1
            in_cluster = True
        elif not v and in_cluster:
            in_cluster = False
    return cnt


def load_config(report_dir):
    config_path = os.path.join(report_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def process_corr_csv(csv_path, fwe_method, sig_key):
    """Traite un CSV de corrélation (corr_partial) et retourne une ligne de summary."""
    df = pd.read_csv(csv_path)
    if df.empty or 'info' in df.columns:
        return None

    required = {'bundle', 'metric', 'variable', 'p_raw', 'sig_afq'}
    if not required.issubset(df.columns):
        return None

    bundle = df['bundle'].iloc[0]
    centroid_id = df['centroid_id'].iloc[0] if 'centroid_id' in df.columns else np.nan
    metric = df['metric'].iloc[0]
    var_col = df['variable'].iloc[0]

    n_sig =int(df['sig_afq'].sum())#count_clusters(df['sig_afq']) if fwe_method in ('clusterFWE', 'mixed') else int(df['sig_afq'].sum())
    min_p = float(df['p_raw'].min()) if df['p_raw'].notna().any() else np.nan
    max_abs_r = float(df['r'].abs().max()) if 'r' in df.columns and df['r'].notna().any() else np.nan

    # n sujets: colonne 'n' contient le nombre de sujets par point
    subjects_used = int(df['n'].max()) if 'n' in df.columns and df['n'].notna().any() else np.nan

    n_points = len(df)

    row = {
        'bundle': bundle,
        'centroid_id': centroid_id,
        'metric': metric,
        'type': f"corr_{var_col}",
        'n_sig': n_sig,
        'min_p': min_p,
        'max_abs_r': max_abs_r,
        'n_points': n_points,
        'subjects_used': 37,
        'removed_subjects': 0,
    }

    if sig_key == 'alphaFWE' and 'alphaFWE' in df.columns:
        row['alphaFWE'] = float(df['alphaFWE'].iloc[0]) if df['alphaFWE'].notna().any() else np.nan
    if sig_key == 'clusterFWE' and 'clusterFWE' in df.columns:
        row['clusterFWE'] = float(df['clusterFWE'].iloc[0]) if df['clusterFWE'].notna().any() else np.nan

    return row


def process_group_csv(csv_path, fwe_method, sig_key):
    """Traite un CSV de groupe (group_corrected) et retourne une ligne de summary."""
    df = pd.read_csv(csv_path)
    if df.empty or 'info' in df.columns:
        return None

    required = {'bundle', 'metric', 'classif', 'p_raw'}
    if not required.issubset(df.columns):
        return None

    bundle = df['bundle'].iloc[0]
    centroid_id = df['centroid_id'].iloc[0] if 'centroid_id' in df.columns else np.nan
    metric = df['metric'].iloc[0]
    classif_col = df['classif'].iloc[0]

    has_sig = 'sig_afq' in df.columns
    if has_sig:
        n_sig = int(df['sig_afq'].sum())#count_clusters(df['sig_afq']) if fwe_method in ('clusterFWE', 'mixed') else int(df['sig_afq'].sum())
    else:
        n_sig = 0

    min_p = float(df['p_raw'].min()) if df['p_raw'].notna().any() else np.nan

    # Nombre de sujets: max sur tous les points de la somme des n par groupe
    n_cols = [c for c in df.columns if c.startswith('n_') and c != 'n_sig']
    if not n_cols:
        n_cols = sorted([c for c in df.columns if c.startswith('n') and c[1:].isdigit()])
    if n_cols:
        subjects_used = int(df[n_cols].sum(axis=1).max())
    else:
        subjects_used = np.nan

    n_points = len(df)

    row = {
        'bundle': bundle,
        'centroid_id': centroid_id,
        'metric': metric,
        'type': f"group_{classif_col}",
        'n_sig': n_sig,
        'min_p': min_p,
        'max_abs_r': np.nan,
        'n_points': n_points,
        'subjects_used': 37,
        'min_n_sub': subjects_used,
        'removed_subjects': 0
    }

    if sig_key == 'alphaFWE' and 'alphaFWE' in df.columns:
        row['alphaFWE'] = float(df['alphaFWE'].iloc[0]) if df['alphaFWE'].notna().any() else np.nan
    if sig_key == 'clusterFWE' and 'clusterFWE' in df.columns:
        row['clusterFWE'] = float(df['clusterFWE'].iloc[0]) if df['clusterFWE'].notna().any() else np.nan

    return row


def determine_confonds(config, analysis_type, var_col=None):
    """Détermine les confonds utilisés à partir de la config."""
    actimetry_keywords = ['activity', 'inactivity', 'freq', 'walk', 'oadl', 'actimetry']
    confond_for_acti = config.get('confond_variables_for_actimetry', [])
    confond_without = config.get('confond_variables_without_control', [])
    confond_with = config.get('confond_variables_with_control', [])
    classif_variables = config.get('classif_variables', {})

    if analysis_type.startswith('corr_'):
        if var_col and any(kw in var_col for kw in actimetry_keywords):
            return confond_for_acti
        return confond_without
    elif analysis_type.startswith('group_'):
        classif_col = analysis_type.replace('group_', '')
        mode = classif_variables.get(classif_col, 'with_controls')
        return confond_with if mode == 'with_controls' else confond_without
    return []


def regenerate_summary(report_dir, output_name=None):
    config = load_config(report_dir)
    fwe_method = config.get('FWE_METHOD', 'clusterFWE')
    sig_key = 'alphaFWE' if fwe_method == 'alphaFWE' else 'clusterFWE'

    figures_dir = os.path.join(report_dir, "figures")
    if not os.path.isdir(figures_dir):
        print(f"Erreur: dossier figures/ introuvable dans {report_dir}")
        sys.exit(1)

    csv_files = sorted(Path(figures_dir).glob("*.csv"))
    print(f"Trouvé {len(csv_files)} fichiers CSV dans {figures_dir}")

    # Ne garder que les CSV "corrected" (group) et "partial" (corr)
    # pour éviter les doublons raw/corrected
    group_csvs = [f for f in csv_files if f.name.endswith("_group_corrected.csv")]
    corr_csvs = [f for f in csv_files if f.name.endswith("_corr_partial.csv")]

    print(f"  - {len(group_csvs)} group_corrected")
    print(f"  - {len(corr_csvs)} corr_partial")

    rows = []

    for csv_path in group_csvs:
        try:
            row = process_group_csv(str(csv_path), fwe_method, sig_key)
            if row:
                var_col = row['type'].replace('group_', '')
                confonds = determine_confonds(config, row['type'], var_col)
                row['confonds_used'] = ",".join(confonds) if confonds else "none"
                rows.append(row)
        except Exception as e:
            print(f"  Erreur {csv_path.name}: {e}")

    for csv_path in corr_csvs:
        try:
            row = process_corr_csv(str(csv_path), fwe_method, sig_key)
            if row:
                var_col = row['type'].replace('corr_', '')
                confonds = determine_confonds(config, row['type'], var_col)
                row['confonds_used'] = ",".join(confonds) if confonds else "none"
                rows.append(row)
        except Exception as e:
            print(f"  Erreur {csv_path.name}: {e}")

    if not rows:
        print("Aucune donnée extraite.")
        return

    summary_df = pd.DataFrame(rows)

    # Calculer removed_points: max global de n_points - n_points
    max_points = summary_df['n_points'].max()
    summary_df['removed_points'] = max_points - summary_df['n_points']

    # Ordre des colonnes identique à generate_report_v2
    col_order = ['bundle', 'centroid_id', 'metric', 'type', 'n_sig', 'min_p',
                 'max_abs_r', 'n_points', 'removed_points', 'subjects_used', 'removed_subjects',
                 'confonds_used', sig_key]
    # S'assurer que toutes les colonnes existent
    for c in col_order:
        if c not in summary_df.columns:
            summary_df[c] = np.nan
    summary_df = summary_df[col_order]

    output_path = os.path.join(report_dir, output_name or "summary_results.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary régénéré: {output_path}")
    print(f"  {len(summary_df)} lignes ({summary_df['type'].value_counts().to_dict()})")

    return summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Régénère summary_results.csv depuis les CSV du dossier figures/")
    parser.add_argument("report_dir", help="Chemin vers le dossier du rapport")
    parser.add_argument("--output", "-o", default=None,
                        help="Nom du fichier de sortie (défaut: summary_results.csv)")
    args = parser.parse_args()

    regenerate_summary(args.report_dir, args.output)
