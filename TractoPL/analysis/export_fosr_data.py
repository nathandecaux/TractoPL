"""
Export des profils de tractométrie au format « tidy » lisible par R (refund).

L'analyse FDA / function-on-scalar regression (BUAN 2.0, section 6.2.2.7) est
implémentée côté R dans `notebooks/fda_chandio.Rmd` avec le paquet `refund`.
Or l'accès aux données (arborescence BIDS + participants_full_info.xlsx) passe par
la machinerie Python `Dataset`. Ce script fait le pont : il assemble, pour chaque
(faisceau, métrique), le profil moyen par sujet (une valeur par segment `point`)
déjà fusionné avec les covariables (group, apathy, age, sex...), et écrit le tout
dans un unique CSV long que le Rmd relit directement.

Sortie : `fosr_report/data/profiles_long.csv` avec les colonnes
    bundle, metric, subject, participant_id, point, value, <covariables...>

À lancer quand le NAS (données BIDS) est monté :
    python -m actiDep.analysis.export_fosr_data
"""

import os
import pandas as pd

from TractoPL.data.loader import Dataset
from TractoPL.set_config import get_HCP_bundle_names

# --- Configuration (alignée sur actiDep/analysis/fosr.py) ------------------
PIPELINE = 'hcp_association_new_50pts_mcm_tensors_staniz_longcentral'
SUB_INFOS = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx"
OUTDIR = 'fosr_report'
DATADIR = os.path.join(OUTDIR, 'data')

METRICS = ['FA', 'IFW']

# Covariables candidates à propager vers R si présentes dans le fichier participants
COVARIATES = ['group', 'apathy', 'age', 'sex', 'aes', 'ami', 'activity_rate_3d']


def export_profiles(restore='pouet', bundles=None, metrics=None):
    os.makedirs(DATADIR, exist_ok=True)

    metrics = metrics or METRICS
    bundles = bundles or list(get_HCP_bundle_names().keys())

    ds = Dataset(restore=restore)
    sub_info_df = pd.read_excel(SUB_INFOS)
    cov_cols = [c for c in COVARIATES if c in sub_info_df.columns]

    all_rows = []
    for bundle in bundles:
        bundle_paths = ds.get_global(pipeline=PIPELINE, bundle=bundle,
                                     suffix='mean', extension='csv')
        if not bundle_paths:
            continue
        for metric in metrics:
            col = metric + '_mean'
            rows = []
            for bp in bundle_paths:
                csv = pd.read_csv(bp.path)
                if col not in csv.columns:
                    break
                sub = csv[['point_id', col]].rename(
                    columns={col: 'value', 'point_id': 'point'})
                sub['subject'] = bp.subject
                sub['participant_id'] = 'sub-' + str(bp.subject)
                rows.append(sub)
            if not rows:
                continue
            long_df = pd.concat(rows, ignore_index=True)
            long_df = long_df.merge(
                sub_info_df[['participant_id'] + cov_cols],
                on='participant_id', how='left')
            long_df.insert(0, 'metric', metric)
            long_df.insert(0, 'bundle', bundle)
            all_rows.append(long_df)

    if not all_rows:
        raise RuntimeError("Aucun profil trouvé — le NAS est-il monté ?")

    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(DATADIR, 'profiles_long.csv')
    out.to_csv(out_path, index=False)
    print(f"Écrit {out_path} : {len(out)} lignes, "
          f"{out['bundle'].nunique()} faisceaux, {out['metric'].nunique()} métriques, "
          f"{out['subject'].nunique()} sujets.")
    print(f"Covariables exportées : {cov_cols}")
    return out_path


if __name__ == '__main__':
    export_profiles()
