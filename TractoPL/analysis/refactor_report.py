#!/usr/bin/env python
import os
from os.path import join as opj
import re
import gc
import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import ttest_ind, f_oneway, spearmanr, pearsonr

from tractseg.libs.AFQ_MultiCompCorrection import get_significant_areas
from TractoPL.analysis.afq_optimized import AFQ_MultiCompCorrection
from TractoPL.set_config import get_HCP_bundle_names
from TractoPL.data.loader import Dataset
import argparse


config_path = "/home/ndecaux/Code/actiDep/actiDep/analysis/config_actidep_test_clusterFWE.json" 

parser = argparse.ArgumentParser(description="Génération de rapport d'analyse AFQ")
parser.add_argument('-c', type=str, default=config_path, help="Chemin vers le fichier de configuration JSON")
parser.add_argument('--subjects-table', type=str, default=None,
                    help="Chemin vers un CSV/XLSX avec une colonne 'participant_id' listant les sujets à traiter. "
                         "Les colonnes 'confond_variables', 'corr_variables' et 'classif_variables' (séparées par ';') "
                         "servent à vérifier que ces variables existent dans le tableau.")
args = parser.parse_args()
config_path = args.c

DEFAULT_CONFIG = {
    "pipeline":"default",
    "dataset": "actidep",
    "hcp_asso_pipeline": "hcp_association_100pts_mcm_tensors_staniz",
    "confond_variables_with_control": ["age"],
    "confond_variables_without_control": ["age"],
    "confond_variables_for_actimetry": ["age"],
    "corr_variables": ["aes"],
    "FWE_METHOD": "mixed",
    "CORRELATION_TEST": "pearson",
    "AFQ_ALPHA": 0.05,
    "AFQ_NPERM": 1000,
    "CORRECT_MULTI_TRACT": False,
    "KEEP_12H_INDIVIDUAL": False,
    "MISSING_SUBJECTS_TOLERANCE": 0,
    "INTERPOLATE_MISSING_POINTS" : 2,
    "classif_variables": {
        "apathy": "no_controls",
    },
    "METRIC_COLUMNS_CANDIDATES": ['FA', 'IFW'],
    "STAT_TYPES": ['mean'],
    "EXCLUDE": {
        "group": [
            "hc"
        ]
    },
    "RESAMPLE_N_POINTS": None,
    "CORRECT_METRIC_CONFOND_ONLY": False,
    "KEEP_LARGEST_CONTIGUOUS": True
}

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Charge la configuration depuis un fichier JSON, ou retourne la config par défaut.
    Si config_path est fourni et existe, le fichier est chargé.
    Sinon, la config par défaut est utilisée.
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Configuration chargée depuis: {config_path}")
            return config
        except Exception as e:
            print(f"Erreur lors du chargement de la config ({config_path}): {e}")
            print("Utilisation de la configuration par défaut.")
            return DEFAULT_CONFIG.copy()
    else:
        if config_path:
            print(f"Fichier de config non trouvé ({config_path}). Utilisation de la configuration par défaut.")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict, output_path: str) -> None:
    """
    Sauvegarde la configuration dans un fichier JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration sauvegardée dans: {output_path}")


#Initialisation des variables globales pour l'analyse
global corr_variables, classif_variables, config, dataset, hcp_asso_pipeline
global confond_variables_with_control, confond_variables_without_control, confond_variables_for_actimetry
global db_root, csv_files, bundle_names, ds
global _SUBJECTS_FILTER, _SUBJECTS_TABLE_DF

_SUBJECTS_FILTER = None      # set de participant_id à traiter (None = tous)
_SUBJECTS_TABLE_DF = None    # DataFrame complet du tableau des sujets (pour merge)

config = load_config(config_path)

AFQ_ALPHA = config.get("AFQ_ALPHA", DEFAULT_CONFIG["AFQ_ALPHA"])
AFQ_NPERM = config.get("AFQ_NPERM", DEFAULT_CONFIG["AFQ_NPERM"])
FWE_METHOD = config.get("FWE_METHOD", DEFAULT_CONFIG["FWE_METHOD"])


CORRECT_MULTI_TRACT = config.get("CORRECT_MULTI_TRACT", DEFAULT_CONFIG["CORRECT_MULTI_TRACT"])
RESAMPLE_N_POINTS = config.get("RESAMPLE_N_POINTS", DEFAULT_CONFIG["RESAMPLE_N_POINTS"])
CORRECT_METRIC_CONFOND_ONLY = config.get("CORRECT_METRIC_CONFOND_ONLY", DEFAULT_CONFIG["CORRECT_METRIC_CONFOND_ONLY"])

hcp_asso_pipeline = config["hcp_asso_pipeline"]
test_by_group = config.get("test_by_group", None)

# Les variables de corrélation et de classification sont définies plus tard à partir de la config
corr_variables = config.get("corr_variables", [])
confond_variables_with_control = config.get("confond_variables_with_control", DEFAULT_CONFIG['confond_variables_with_control'])
confond_variables_without_control = config.get("confond_variables_without_control", DEFAULT_CONFIG['confond_variables_without_control'])
confond_variables_for_actimetry = config.get("confond_variables_for_actimetry", DEFAULT_CONFIG['confond_variables_for_actimetry'])

classif_variables = config.get("classif_variables", DEFAULT_CONFIG['classif_variables'])


METRIC_COLUMNS_CANDIDATES = config.get("METRIC_COLUMNS_CANDIDATES", DEFAULT_CONFIG['METRIC_COLUMNS_CANDIDATES'])
STAT_TYPES = config.get("STAT_TYPES", DEFAULT_CONFIG['STAT_TYPES'])
CORRELATION_TEST= config.get("CORRELATION_TEST", DEFAULT_CONFIG["CORRELATION_TEST"])

CLASSIF_CONFOUND_MAP = {k:confond_list for k,v in classif_variables.items() for confond_list in [confond_variables_with_control if v == "with_controls" else confond_variables_without_control]}
pipeline = config.get("pipeline", DEFAULT_CONFIG["pipeline"])


# Chargement des données
dataset = config["dataset"]

_ADDITIONAL_INFO_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/participants_full_info.xlsx"
_ACTIMETRY_PATH = f"/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids/actimetry_features.xlsx"
_ADDITIONAL_INFO_DF, _ACTIMETRY_DF = pd.read_excel(_ADDITIONAL_INFO_PATH), pd.read_excel(_ACTIMETRY_PATH)
actimetry_columns = [c for c in _ACTIMETRY_DF.columns if c not in ['subject_id', 'participant_id']] if not _ACTIMETRY_DF.empty else []

db_root = f'/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids'
ds = Dataset(db_root,restore="test")


def apply_exclude_filters(df):
    """
    Applique les filtres de sujets et d'exclusion sur le DataFrame.
    """
    for var, values in config.get("EXCLUDE", {}).items():
        if var in df.columns:
            df = df[~df[var].isin(values)]
    return df



csv_files = ds.get_global(pipeline=hcp_asso_pipeline, extension='csv', datatype='metric', suffix='mean')
csv_files= pd.concat([f.df for f in csv_files])
csv_files['participant_id'] = "sub-" + csv_files['subject']
csv_files = csv_files.merge(_ADDITIONAL_INFO_DF, on='participant_id', how='left')
csv_files = csv_files.merge(_ACTIMETRY_DF, on='participant_id', how='left')
csv_files = apply_exclude_filters(csv_files)

def load_and_merge_csv_files(csv_files: List[str]) -> pd.DataFrame:
    """
    Charge et fusionne les fichiers CSV en un seul DataFrame.
    Filtre les sujets si subjects_filter est fourni.
    """
    dataframes = []
    for csv_file in tqdm(csv_files.itertuples(), desc="Chargement des CSV"):
        sub, bun =csv_file.participant_id, csv_file.bundle
        df = pd.read_csv(csv_file.path)
        df['subject'] = sub
        df['bundle'] = bun
        dataframes.append(df)
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


print("N fichiers CSV à traiter:", len(csv_files))
long_df=load_and_merge_csv_files(csv_files)
print("DataFrame fusionné shape:", long_df.shape)






















# hostname = os.uname().nodename
# print(f"Hostname: {hostname}")

# n_jobs = os.cpu_count() // 3
# out_base = f'/home/ndecaux/Reports'

# if hostname == 'calcarine':
#     out_base='/data/ndecaux/Reports'
#     n_jobs = 32

# output_dir = f'{out_base}/report_{dataset}_{hcp_asso_pipeline}_{pipeline}_{FWE_METHOD}_{CORRELATION_TEST}'

# os.makedirs(output_dir, exist_ok=True)
# print(f"Rapport dans {output_dir}")

# # Sauvegarder la configuration utilisée dans le dossier de sortie
# config_output_path = opj(output_dir, "config.json")
# save_config(config, config_output_path)
