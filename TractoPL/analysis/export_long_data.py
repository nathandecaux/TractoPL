"""
Export des données au format long pour analyses statistiques.

Ce script génère un DataFrame unique contenant toutes les données
de tractométrie au format long, nettoyées et prêtes pour l'analyse.

Format de sortie:
- Une ligne par (sujet, bundle, métrique, point)
- Colonnes: subject, bundle, metric, point, value + confounds + variables cliniques
- Données nettoyées (gestion des NaN selon le seuil défini)

Auteur: Adapté de mixed_effect.py
"""

import gc
import warnings
import pandas as pd
import numpy as np
from os.path import join as opj
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from TractoPL.set_config import get_HCP_bundle_names
from TractoPL.data.loader import Dataset

# =============================================================================
# Configuration
# =============================================================================

dataset = 'actidep'
db_root = f'/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids'

# Variables à inclure dans l'export
CONFOUND_VARIABLES = ['age', 'sex', 'city', 'duration_dep']
CLINICAL_VARIABLES = ['group', 'apathy', 'aes', 'ami']

# Colonnes métriques
METRIC_COLUMNS = ['FA_median', 'IFW_median','FA_mean', 'IFW_mean']

# Seuil pour le nettoyage des NaN
NAN_THRESHOLD = 0.15

# Chemins externes
_ADDITIONAL_INFO_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/participants_full_info.xlsx"
_ACTIMETRY_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/actimetry_features.xlsx"


# =============================================================================
# Fonctions utilitaires (réutilisées de mixed_effect.py)
# =============================================================================

def ensure_dir(p: Path) -> None:
    """Crée le répertoire s'il n'existe pas."""
    Path(p).mkdir(parents=True, exist_ok=True)


def load_external_tables(additional_info_path: str, actimetry_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les tables externes (participants, actimétrie)."""
    try:
        add_df = pd.read_excel(additional_info_path)
        print(f"✓ Chargé {len(add_df)} participants depuis participants_full_info.xlsx")
    except Exception as e:
        print(f"⚠ Impossible de charger participants_full_info.xlsx: {e}")
        add_df = pd.DataFrame()
    
    try:
        act_df = pd.read_excel(actimetry_path)
        print(f"✓ Chargé {len(act_df)} participants depuis actimetry_features.xlsx")
    except Exception as e:
        print(f"⚠ Impossible de charger actimetry_features.xlsx: {e}")
        act_df = pd.DataFrame()
    
    return add_df, act_df


def load_and_merge_bundle_csvs(bundle_name: str, bundle_csvs: List, 
                               add_df: pd.DataFrame, act_df: pd.DataFrame) -> pd.DataFrame:
    """Charge et fusionne les CSV d'un bundle avec les informations externes."""
    metric_files = [pd.read_csv(f.path) for f in bundle_csvs]
    
    for df, f in zip(metric_files, bundle_csvs):
        df['subject'] = f.get_full_entities()['subject']
        df['participant_id'] = 'sub-' + df["subject"].astype(str)
    
    metrics_df = pd.concat(metric_files, ignore_index=True)

    # Ajout des infos externes
    if not add_df.empty:
        metrics_df = metrics_df.merge(add_df, on='participant_id', how='left')
    if not act_df.empty:
        metrics_df = metrics_df.merge(act_df, on='participant_id', how='left')

    del metric_files
    gc.collect()
    return metrics_df


def detect_point_column(df: pd.DataFrame) -> str:
    """Détecte la colonne des points dans le DataFrame."""
    if 'point_id' in df.columns:
        return 'point_id'
    if 'point' in df.columns:
        return 'point'
    for c in df.columns:
        if 'point' in c.lower():
            return c
    raise ValueError("Colonne des points introuvable (point_id / point).")


def prepare_long(df_bundle: pd.DataFrame, metric_col: str, point_col: str, 
                bundle_name: str) -> pd.DataFrame:
    """
    Prépare le DataFrame au format long pour les modèles mixtes.
    
    Returns:
        DataFrame avec colonnes: subject, bundle, metric, point, value + colonnes meta
    """
    needed = ['subject', point_col, metric_col]
    if not all(c in df_bundle.columns for c in needed):
        raise ValueError(f"Colonnes manquantes pour {metric_col}")
    
    meta_cols = [c for c in df_bundle.columns if c not in METRIC_COLUMNS]
    long_df = df_bundle[meta_cols + [metric_col]].copy()
    long_df = long_df.rename(columns={metric_col: 'value', point_col: 'point'})
    
    # Ajouter les identifiants
    long_df['bundle'] = bundle_name
    long_df['metric'] = metric_col
    long_df['subject'] = long_df['subject'].astype(str)
    
    return long_df


def clean_missing_data(long_df: pd.DataFrame, threshold: float = NAN_THRESHOLD) -> Tuple[pd.DataFrame, List, List]:
    """
    Gestion des NaN:
      - Si un point a >= threshold de NaN, on le supprime
      - Sinon, on supprime les sujets avec des NaN
    
    Returns:
        Tuple[DataFrame, list, list]: données nettoyées, sujets retirés, points retirés
    """
    if long_df.empty:
        return long_df, [], []
    
    pivot = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    n_subjects = pivot.shape[0]
    
    if n_subjects == 0:
        return long_df, [], []
    
    # Proportion de NaN par point
    prop_nan_point = pivot.isna().sum(axis=0) / n_subjects
    points_to_remove = sorted([p for p, prop in prop_nan_point.items() if prop >= threshold])
    
    # Ajouter points extrêmes
    min_point = pivot.columns.min()
    max_point = pivot.columns.max()
    if min_point not in points_to_remove:
        points_to_remove.insert(0, min_point)
    if max_point not in points_to_remove:
        points_to_remove.append(max_point)
    
    # Retirer ces points
    pivot_reduced = pivot.drop(columns=points_to_remove, errors='ignore')
    
    # Sujets à retirer
    subjects_to_remove = sorted(pivot_reduced.index[pivot_reduced.isna().any(axis=1)].tolist())
    
    # Filtrer long_df
    mask_points = ~long_df['point'].isin(points_to_remove)
    mask_subjects = ~long_df['subject'].isin(subjects_to_remove)
    long_df_clean = long_df[mask_points & mask_subjects].copy()
    
    return long_df_clean, subjects_to_remove, points_to_remove


# =============================================================================
# Export des données longues
# =============================================================================

def export_long_data(bundle_names: List[str],
                    csv_files: List,
                    output_path: str,
                    columns_to_keep: Optional[List[str]] = None,
                    include_all_subjects: bool = False) -> None:
    """
    Génère un DataFrame long unique avec toutes les données nettoyées.
    
    Args:
        bundle_names: Liste des bundles à inclure
        csv_files: Liste des fichiers CSV (BIDSFile)
        output_path: Chemin du fichier CSV de sortie
        columns_to_keep: Liste des colonnes à garder (en plus de subject, bundle, metric, point, value)
                        Si None, garde les confounds + variables cliniques par défaut
        include_all_subjects: Si True, garde tous les sujets même avec NaN dans certaines variables
    """
    if columns_to_keep is None:
        columns_to_keep = CONFOUND_VARIABLES + CLINICAL_VARIABLES
    
    print("="*80)
    print("EXPORT DES DONNÉES AU FORMAT LONG")
    print("="*80)
    print(f"Bundles à traiter: {len(bundle_names)}")
    print(f"Métriques: {METRIC_COLUMNS}")
    print(f"Colonnes à conserver: {columns_to_keep}")
    print(f"Seuil NaN: {NAN_THRESHOLD}")
    print(f"Sortie: {output_path}")
    print()
    
    # Charger les tables externes
    print("[1/3] Chargement des tables externes...")
    add_df, act_df = load_external_tables(_ADDITIONAL_INFO_PATH, _ACTIMETRY_PATH)
    
    # Préparer les données par bundle
    print("\n[2/3] Traitement des bundles...")
    all_data = []
    stats = {
        'n_bundles': 0,
        'n_rows_before': 0,
        'n_rows_after': 0,
        'subjects_removed_by_bundle': {},
        'points_removed_by_bundle': {}
    }
    
    for bundle_name in tqdm(bundle_names, desc="Bundles"):
        bundle_csvs = [f for f in csv_files 
                       if f.get_entities().get('bundle') == bundle_name]
        print(f"\nTraitement bundle: {bundle_name} ({len(bundle_csvs)} fichiers)")
        if not bundle_csvs:
            continue
        
        if True:#try:
            # Charger le bundle
            df_bundle = load_and_merge_bundle_csvs(bundle_name, bundle_csvs, add_df, act_df)
            point_col = detect_point_column(df_bundle)
            
            # Traiter chaque métrique
            for metric_col in METRIC_COLUMNS:
                if metric_col not in df_bundle.columns:
                    continue
                
                # Format long
                long_df = prepare_long(df_bundle, metric_col, point_col, bundle_name)
                stats['n_rows_before'] += len(long_df)
                
                # Nettoyage
                # long_df_clean, removed_subj, removed_pts = clean_missing_data(long_df,
                long_df_clean, removed_subj, removed_pts = long_df, [], []  # Désactiver le nettoyage pour inclure tous les sujets
                
                if not long_df_clean.empty:
                    # Filtrer les colonnes
                    base_cols = ['subject', 'bundle', 'metric', 'point', 'value']
                    available_cols = [c for c in columns_to_keep if c in long_df_clean.columns]
                    cols_to_export = base_cols + available_cols
                    
                    long_df_export = long_df_clean[cols_to_export].copy()
                    
                    # # Optionnel: retirer les lignes avec NaN dans les variables d'intérêt
                    if not include_all_subjects:
                        # Retirer les sujets avec NaN dans au moins une des colonnes d'intérêt
                        long_df_export = long_df_export.dropna(subset=available_cols)
                    
                    all_data.append(long_df_export)
                    stats['n_rows_after'] += len(long_df_export)
                    
                    # Statistiques
                    key = f"{bundle_name}_{metric_col}"
                    stats['subjects_removed_by_bundle'][key] = removed_subj
                    stats['points_removed_by_bundle'][key] = removed_pts
            
            stats['n_bundles'] += 1
            
        # except Exception as e:
        #     print(f"⚠ Erreur bundle {bundle_name}: {e}")
        #     continue
    
    # Combiner toutes les données
    print("\n[3/3] Fusion et sauvegarde...")
    if not all_data:
        print("❌ Aucune donnée à exporter!")
        return
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Statistiques finales
    print("\n" + "="*80)
    print("STATISTIQUES")
    print("="*80)
    print(f"Bundles traités: {stats['n_bundles']}")
    print(f"Lignes avant nettoyage: {stats['n_rows_before']:,}")
    print(f"Lignes après nettoyage: {stats['n_rows_after']:,}")
    print(f"Taux de conservation: {100*stats['n_rows_after']/stats['n_rows_before']:.1f}%")
    print(f"\nSujets uniques: {final_df['subject'].nunique()}")
    print(f"Bundles: {final_df['bundle'].nunique()}")
    print(f"Métriques: {final_df['metric'].nunique()}")
    print(f"Points moyens par bundle: {final_df.groupby(['bundle', 'metric'])['point'].nunique().mean():.1f}")
    print(f"\nTaille du DataFrame: {len(final_df):,} lignes × {len(final_df.columns)} colonnes")
    print(f"Mémoire: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Aperçu des NaN par colonne
    print(f"\nTaux de NaN par colonne:")
    for col in final_df.columns:
        nan_pct = 100 * final_df[col].isna().sum() / len(final_df)
        if nan_pct > 0:
            print(f"  {col}: {nan_pct:.1f}%")
    
    # Sauvegarder
    ensure_dir(Path(output_path).parent)
    final_df.to_csv(output_path, index=False)
    print(f"\n✓ Données exportées: {output_path}")
    
    # Sauvegarder aussi les statistiques
    stats_path = output_path.replace('.csv', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("STATISTIQUES D'EXPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Bundles traités: {stats['n_bundles']}\n")
        f.write(f"Lignes avant nettoyage: {stats['n_rows_before']:,}\n")
        f.write(f"Lignes après nettoyage: {stats['n_rows_after']:,}\n")
        f.write(f"Taux de conservation: {100*stats['n_rows_after']/stats['n_rows_before']:.1f}%\n\n")
        
        f.write("SUJETS UNIQUES PAR BUNDLE/MÉTRIQUE\n")
        f.write("-"*80 + "\n")
        for bundle_metric in sorted(stats['subjects_removed_by_bundle'].keys()):
            removed = stats['subjects_removed_by_bundle'][bundle_metric]
            f.write(f"{bundle_metric}: {len(removed)} sujets retirés\n")
            if removed:
                f.write(f"  {removed}\n")
        
        f.write("\nPOINTS RETIRÉS PAR BUNDLE/MÉTRIQUE\n")
        f.write("-"*80 + "\n")
        for bundle_metric in sorted(stats['points_removed_by_bundle'].keys()):
            removed = stats['points_removed_by_bundle'][bundle_metric]
            f.write(f"{bundle_metric}: {len(removed)} points retirés\n")
            if removed:
                f.write(f"  {removed}\n")
    
    print(f"✓ Statistiques sauvegardées: {stats_path}")
    
    # Sauvegarder un échantillon pour vérification
    sample_path = output_path.replace('.csv', '_sample.csv')
    final_df.sample(min(1000, len(final_df))).to_csv(sample_path, index=False)
    print(f"✓ Échantillon (1000 lignes): {sample_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Point d'entrée principal."""
    
    # Charger le dataset
    print("Chargement du dataset...")
    ds = Dataset(db_root)
    hcp_asso_pipeline = 'hcp_association_new_100pts_mcm_tensors_staniz_longcentral'
    csv_files = ds.get_global(
        pipeline=hcp_asso_pipeline, 
        extension='csv', 
        datatype='metric', 
        suffix='mean'
    )
    bundle_names = list(get_HCP_bundle_names().keys())
    
    print(f"✓ {len(csv_files)} fichiers CSV trouvés")
    print(f"✓ {len(bundle_names)} bundles HCP")
    
    # Répertoire de sortie
    output_dir = Path(f'/home/ndecaux/NAS_EMPENN/share/projects/actidep/Results/{dataset}_{hcp_asso_pipeline}_long_data')
    ensure_dir(output_dir)
    
    # Chemins de sortie
    output_path = output_dir / 'tractometry_long_data_all.csv'
    
    # Export complet (toutes les variables disponibles)
    export_long_data(
        bundle_names=bundle_names,
        csv_files=csv_files,
        output_path=str(output_path),
        columns_to_keep=CONFOUND_VARIABLES + CLINICAL_VARIABLES,
        include_all_subjects=True  # True = garde les sujets même avec des NaN
    )
    
    # # Export minimal (seulement les variables essentielles)
    # output_path_minimal = output_dir / 'tractometry_long_data_minimal.csv'
    # export_long_data(
    #     bundle_names=bundle_names,
    #     csv_files=csv_files,
    #     output_path=str(output_path_minimal),
    #     columns_to_keep=['group', 'aes', 'age', 'sex'],  # Variables minimales
    #     include_all_subjects=False
    # )
    
    print("\n" + "="*80)
    print("✓ EXPORT TERMINÉ")
    print("="*80)


if __name__ == "__main__":
    main()
