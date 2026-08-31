import os
from os.path import join as opj
import pandas as pd
import glob
import re
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import panel as pn
# pn.extension('bokeh') # Already called by hv.extension('bokeh')
import param
import random  # Ajouter cette importation en haut du fichier
import numpy as np
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression 
from tractseg.libs.AFQ_MultiCompCorrection import AFQ_MultiCompCorrection, get_significant_areas 
from tractseg.libs import metric_utils
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage.interpolation import map_coordinates
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa

from tractseg.libs import fiber_utils


#### Custom imports ####
from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config,get_HCP_bundle_names
from TractoPL.data.loader import Subject, parse_filename, BIDSFile, Dataset
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# db_root, ds, csv_files, bundle_names, corr_variables, classif_variables, confond lists

db_root = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'

cache = False
#If ds_association.pkl already exists, load it instead of re-running the whole pipeline

ds = Dataset(db_root)

csv_files = ds.get_global(pipeline='hcp_association',extension='csv',datatype='metric')

# from pprint import pprint

# pprint([f.path for f in csv_files if f.get_entities()['bundle']=='CSTleft'])

bundle_names = get_HCP_bundle_names().values()

#Filter for now to only look at CSTleft
bundle_names = ['CSTleft','CSTright']

corr_variables = ['ami','aes']
classif_variables = {'group':'with_controls','apathy':'no_controls'}

confond_variables_with_control = ['age','sex','city']

confond_variables_without_control = confond_variables_with_control + ['duration_dep','type_dep']


# --- NEW: utilitaires chargement / mise en forme ---------------------------------
def load_participants_info(excel_path="/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx"):
    return pd.read_excel(excel_path)

def load_metric_long_dataframe(actidep_files, participants_df=None):
    """
    Parsing simplifié inspiré du notebook:
      - Chaque fichier correspond à un bundle (entité 'bundle')
      - Colonnes: point_id (ou point), plusieurs métriques (FA, MD, ...), éventuellement métadonnées
      - Ajout subject / participant_id
      - Fusion participants
      - Passage au format long
    Retourne DataFrame: subject, participant_id, bundle, metric, point, value (+ colonnes participants si dispo)
    """
    per_file_wide = []
    for f in actidep_files:
        ent = f.get_full_entities()
        subject = ent.get('subject') or ent.get('sub')
        bundle = ent.get('bundle')
        if subject is None or bundle is None:
            continue
        # Lecture avec détection simple du séparateur
        df = None
        for sep in [';', ',']:
            try:
                tmp = pd.read_csv(f.path, sep=sep)
                # Heuristique: si une seule colonne contenant le séparateur -> mauvais split
                if tmp.shape[1] == 1 and sep in tmp.columns[0]:
                    continue
                df = tmp
                break
            except Exception:
                continue
        if df is None or df.empty:
            continue

        # Normaliser colonne point
        point_col = None
        for cand in ['point_id', 'point', 'Point', 'POINT']:
            if cand in df.columns:
                point_col = cand
                break
        if point_col is None:
            df = df.reset_index().rename(columns={'index': 'point_id'})
            point_col = 'point_id'
        df.rename(columns={point_col: 'point_id'}, inplace=True)

        # Ajout colonnes sujet / participant / bundle
        df['subject'] = str(subject)
        df['participant_id'] = 'sub-' + df['subject']
        df['bundle'] = bundle

        per_file_wide.append(df)

    if not per_file_wide:
        print("[LOAD] Aucun CSV exploitable.")
        return pd.DataFrame(columns=['subject','participant_id','bundle','metric','point','value'])

    wide_df = pd.concat(per_file_wide, ignore_index=True)

    # Fusion participants (sur participant_id)
    if participants_df is not None and 'participant_id' in participants_df.columns:
        wide_df = wide_df.merge(participants_df, on='participant_id', how='left', suffixes=('', '_part'))

    # Identification colonnes métriques: numériques hors métadonnées
    meta_cols = {'point_id', 'subject', 'participant_id', 'bundle'}
    if participants_df is not None:
        meta_cols.update([c for c in participants_df.columns if c in wide_df.columns])
    metric_cols = [
        c for c in wide_df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(wide_df[c])
    ]
    if not metric_cols:
        print("[LOAD] Aucune colonne de métrique détectée.")
        return pd.DataFrame(columns=['subject','participant_id','bundle','metric','point','value'])

    # Passage au format long
    long_df = wide_df.melt(
        id_vars=['subject','participant_id','bundle','point_id'],
        value_vars=metric_cols,
        var_name='metric',
        value_name='value'
    ).rename(columns={'point_id': 'point'})

    # Réordonner
    ordered_cols = ['subject','participant_id','bundle','metric','point','value']
    # Ajouter les colonnes participants à la fin
    extra_cols = [c for c in wide_df.columns if c not in ordered_cols and c not in metric_cols]
    long_df = long_df.merge(
        wide_df[['subject','participant_id','point_id'] + list(set(extra_cols)-{'point_id'})]
              .drop_duplicates()
              .rename(columns={'point_id':'point'}),
        on=['subject','participant_id','point'],
        how='left'
    )
    # Déplacer colonnes participants après les colonnes principales
    base = ['subject','participant_id','bundle','metric','point','value']
    other = [c for c in long_df.columns if c not in base]
    long_df = long_df[base + other]

    # Tri cohérent
    long_df = long_df.sort_values(['bundle','metric','subject','point']).reset_index(drop=True)
    return long_df

# --- NEW: ajustement confondeurs -------------------------------------------------
def residualize(values_series, design_df):
    """
    Calcule les résidus (value ~ confounds) et ajoute l'intercept moyen.
    """
    y = pd.to_numeric(values_series, errors='coerce')
    X = design_df.copy()
    # Encodage des objets/cat
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("category"):
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    # Supprimer colonnes quasi constantes
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    if not keep:
        return y
    X = X[keep].apply(lambda col: pd.to_numeric(col, errors='coerce'))
    # Imputation simple
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    X = sm.add_constant(X, has_constant='add')
    try:
        model = sm.OLS(y, X, missing='drop').fit()
        resid = model.resid + model.params.get('const', 0.0)
        # réaligner index
        y_corr = y.copy()
        y_corr.loc[resid.index] = resid
        return y_corr
    except Exception:
        return y

# --- NEW: tests de groupe --------------------------------------------------------
def group_pointwise_tests(df_long, participants, classif_var, confounds=None):
    """
    Retourne DataFrame: metric,bundle,point,n1,n2,mean1,mean2,effect,t,p_unc,p_fdr,p_bonf,sig_fdr,sig_bonf
    """
    results = []
    if participants is None or classif_var not in participants.columns:
        print(f"[WARN] Variable de classification {classif_var} absente.")
        return pd.DataFrame()
    # S'assurer commun
    merged = df_long.merge(participants[['subject', classif_var] + (confounds or [])], on='subject', how='left')
    # Catégories
    cats = merged[classif_var].dropna().unique()
    if len(cats) != 2:
        print(f"[WARN] {classif_var}: besoin de 2 catégories, trouvé {cats}")
        return pd.DataFrame()
    g0, g1 = cats[0], cats[1]
    # Boucle metric/bundle/point
    for (metric, bundle), subdf in merged.groupby(['metric','bundle']):
        for point, pdf in subdf.groupby('point'):
            pdf = pdf.dropna(subset=['value', classif_var])
            if pdf.empty: 
                continue
            # Ajuster confounds si demandé
            if confounds:
                design = pdf[confounds]
                adj_values = residualize(pdf['value'], design)
            else:
                adj_values = pd.to_numeric(pdf['value'], errors='coerce')
            grp0 = adj_values[pdf[classif_var]==g0].dropna()
            grp1 = adj_values[pdf[classif_var]==g1].dropna()
            if len(grp0) < 2 or len(grp1) < 2:
                continue
            try:
                tval, pval = stats.ttest_ind(grp0, grp1, equal_var=False)
            except Exception:
                continue
            effect = grp1.mean() - grp0.mean()  # différence (g1 - g0)
            results.append([metric,bundle,point,len(grp0),len(grp1),
                            grp0.mean(),grp1.mean(),effect,tval,pval])
    if not results:
        return pd.DataFrame()
    res = pd.DataFrame(results, columns=['metric','bundle','point','n0','n1',
                                         'mean0','mean1','effect','t','p_unc'])
    # Corrections multiples par (metric,bundle)
    res['p_fdr'] = np.nan
    res['p_bonf'] = np.nan
    for (metric,bundle), sub in res.groupby(['metric','bundle']):
        idx = sub.index
        p = sub['p_unc'].values
        if len(p):
            _, p_fdr, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
            p_bonf = np.minimum(p * len(p), 1.0)
            res.loc[idx, 'p_fdr'] = p_fdr
            res.loc[idx, 'p_bonf'] = p_bonf
    res['sig_fdr'] = res['p_fdr'] < 0.05
    res['sig_bonf'] = res['p_bonf'] < 0.05
    return res

# --- NEW: corrélations -----------------------------------------------------------
def pointwise_correlations(df_long, participants, corr_var, confounds=None):
    """
    Retourne DataFrame: metric,bundle,point,n,r,p_unc,p_fdr,p_bonf,sig_fdr,sig_bonf
    """
    if participants is None or corr_var not in participants.columns:
        print(f"[WARN] Variable de corrélation {corr_var} absente.")
        return pd.DataFrame()
    merged = df_long.merge(participants[['subject', corr_var] + (confounds or [])], on='subject', how='left')
    # Encoder corr_var numérique
    cv = merged[corr_var]
    if cv.dtype == object or str(cv.dtype).startswith('category'):
        merged['_corr_x'] = cv.astype('category').cat.codes.replace(-1,np.nan).astype(float)
    else:
        merged['_corr_x'] = pd.to_numeric(cv, errors='coerce')
    results = []
    for (metric,bundle), sub in merged.groupby(['metric','bundle']):
        for point, pdf in sub.groupby('point'):
            pdf = pdf.dropna(subset=['value','_corr_x'])
            if len(pdf) < 3:
                continue
            # Ajuster confounds si demandé
            if confounds:
                design = pdf[confounds]
                y_adj = residualize(pdf['value'], design)
                x_adj = residualize(pdf['_corr_x'], design)
            else:
                y_adj = pd.to_numeric(pdf['value'], errors='coerce')
                x_adj = pdf['_corr_x']
            mask = y_adj.notna() & x_adj.notna()
            if mask.sum() < 3:
                continue
            try:
                r, p = stats.pearsonr(x_adj[mask], y_adj[mask])
            except Exception:
                continue
            results.append([metric,bundle,point,mask.sum(),r,p])
    if not results:
        return pd.DataFrame()
    res = pd.DataFrame(results, columns=['metric','bundle','point','n','r','p_unc'])
    res['p_fdr'] = np.nan
    res['p_bonf'] = np.nan
    for (metric,bundle), sub in res.groupby(['metric','bundle']):
        idx = sub.index
        p = sub['p_unc'].values
        if len(p):
            _, p_fdr, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
            p_bonf = np.minimum(p*len(p),1.0)
            res.loc[idx,'p_fdr'] = p_fdr
            res.loc[idx,'p_bonf'] = p_bonf
    res['sig_fdr'] = res['p_fdr'] < 0.05
    res['sig_bonf'] = res['p_bonf'] < 0.05
    return res

# --- NEW: tracés -----------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def plot_group_results(bundle_df, res_bundle, out_path, title_suffix):
    """
    bundle_df: sous-dataframe long (un metric, un bundle)
    res_bundle: résultats group tests (même metric,bundle)
    """
    if bundle_df.empty or res_bundle.empty:
        return None
    metric = bundle_df['metric'].iloc[0]
    bundle = bundle_df['bundle'].iloc[0]
    fig, ax = plt.subplots(figsize=(8,4))
    # Moyennes par group (recherche de variable groupée utilisée dans res -> on l’a perdu; on trace global + sem)
    grp = bundle_df.groupby(['point']).agg(val_mean=('value','mean'),
                                           val_sem=('value',lambda x: x.std(ddof=1)/np.sqrt(max(len(x),1))))
    ax.plot(grp.index, grp['val_mean'], color='black', lw=2, label='Moyenne (tous)')
    ax.fill_between(grp.index,
                    grp['val_mean']-grp['val_sem'],
                    grp['val_mean']+grp['val_sem'],
                    color='gray', alpha=0.25, label='SEM')
    # Points significatifs
    sig_points = res_bundle[res_bundle['sig_fdr']]['point'].values
    if len(sig_points):
        ax.scatter(sig_points,
                   np.interp(sig_points, grp.index, grp['val_mean']),
                   color='red', s=30, zorder=5, label='p<0.05 FDR')
    ax.set_xlabel("Point")
    ax.set_ylabel(metric)
    ax.set_title(f"{bundle} - {metric} {title_suffix}")
    ax.legend(loc='upper right', fontsize=8)
    out_file = os.path.join(out_path, f"group_{metric}_{bundle}_{title_suffix.replace(' ','_')}.png")
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)
    return out_file

def plot_corr_results(bundle_df, corr_res_bundle, out_path, title_suffix):
    if bundle_df.empty or corr_res_bundle.empty:
        return None
    metric = bundle_df['metric'].iloc[0]
    bundle = bundle_df['bundle'].iloc[0]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axhline(0, color='gray', lw=1)
    ax.plot(corr_res_bundle['point'], corr_res_bundle['r'], lw=2, color='blue', label='r')
    sig_points = corr_res_bundle[corr_res_bundle['sig_fdr']]['point'].values
    if len(sig_points):
        ax.scatter(sig_points,
                   np.interp(sig_points, corr_res_bundle['point'], corr_res_bundle['r']),
                   color='red', s=30, zorder=5, label='p<0.05 FDR')
    ax.set_xlabel("Point")
    ax.set_ylabel("r")
    ax.set_title(f"{bundle} - {metric} corr {title_suffix}")
    ax.legend(loc='upper right', fontsize=8)
    out_file = os.path.join(out_path, f"corr_{metric}_{bundle}_{title_suffix.replace(' ','_')}.png")
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)
    return out_file

# --- NEW: résumé -----------------------------------------------------------------
def build_summary_table(group_results_dict, corr_results_dict, alpha=0.05, min_abs_r=0.3):
    """
    group_results_dict: {(classif_var, adj_flag): DataFrame}
    corr_results_dict: {(corr_var, adj_flag): DataFrame}
    """
    summary_rows = []
    # Group differences
    for (cvar, adj), df in group_results_dict.items():
        if df is None or df.empty: 
            continue
        agg = (df[df['sig_fdr']]
               .groupby(['metric','bundle'])
               .agg(n_sig=('point','count'),
                    max_effect=('effect', lambda x: x.abs().max())))
        if agg.empty:
            continue
        for (metric,bundle), row in agg.iterrows():
            summary_rows.append({
                'type':'group_diff',
                'variable':cvar,
                'adjusted':adj,
                'metric':metric,
                'bundle':bundle,
                'n_sig_points':int(row['n_sig']),
                'magnitude':row['max_effect']
            })
    # Correlations
    for (cvar, adj), df in corr_results_dict.items():
        if df is None or df.empty:
            continue
        # FDR sig & amplitude
        cand = df[(df['sig_fdr']) | (df['r'].abs()>=min_abs_r)]
        agg = cand.groupby(['metric','bundle']).agg(
            n_sig_or_strong=('point','count'),
            max_abs_r=('r', lambda x: x.abs().max())
        )
        if agg.empty:
            continue
        for (metric,bundle), row in agg.iterrows():
            summary_rows.append({
                'type':'correlation',
                'variable':cvar,
                'adjusted':adj,
                'metric':metric,
                'bundle':bundle,
                'n_sig_points':int(row['n_sig_or_strong']),
                'magnitude':row['max_abs_r']
            })
    if not summary_rows:
        return pd.DataFrame(columns=['type','variable','adjusted','metric','bundle','n_sig_points','magnitude'])
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(['type','variable','adjusted','metric','bundle'])
    return summary

def summary_markdown_table(summary_df):
    if summary_df.empty:
        return "Aucun résultat significatif."
    cols = ['type','variable','adjusted','metric','bundle','n_sig_points','magnitude']
    lines = ["| " + " | ".join(cols) + " |", "|"+ "|".join(['---']*len(cols)) + "|"]
    for _, r in summary_df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

# --- NEW: génération rapport -----------------------------------------------------
def generate_report(
    db_root=db_root,
    pipeline='hcp_association',
    classif_variables=classif_variables,
    corr_variables=corr_variables,
    conf_with=confond_variables_with_control,
    conf_without=confond_variables_without_control,
    out_dir="report_output",
    report_name="tractometry_report.html"
):
    print("[INFO] Démarrage génération du rapport...")
    actidep_files = ds.get_global(pipeline=pipeline, extension='csv', datatype='metric')
    print(f"[INFO] {len(actidep_files)} fichiers métriques récupérés.")

    participants = load_participants_info(db_root)
    if participants is None:
        print("[WARN] Pas d'informations participants; certaines analyses seront limitées.")

    # NOUVEAU: chargement unifié inspiré du notebook
    long_df = load_metric_long_dataframe(actidep_files, participants_df=participants)
    if long_df.empty:
        print("[ERREUR] Aucune donnée de métrique après parsing unifié.")
        return None
    print(f"[INFO] Données long: {long_df.shape[0]} lignes, {long_df['bundle'].nunique()} faisceaux, {long_df['metric'].nunique()} métriques.")

    out_dir = ensure_dir(out_dir)
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    # Dictionnaires de résultats
    group_results_dict = {}
    corr_results_dict = {}
    figure_paths = []

    # Tests de group
    for cvar, mode in classif_variables.items():
        print(f"[GROUP] Variable={cvar} mode={mode}")
        use_conf = conf_with if 'with' in mode else conf_without
        # Sans ajustement
        raw_res = group_pointwise_tests(long_df, participants, cvar, confounds=None)
        group_results_dict[(cvar,'raw')] = raw_res
        # Ajusté
        adj_res = group_pointwise_tests(long_df, participants, cvar, confounds=use_conf)
        group_results_dict[(cvar,'adjusted')] = adj_res
        # Figures par bundle/metric
        for (metric,bundle), bdf in long_df.groupby(['metric','bundle']):
            r_raw = raw_res[(raw_res.metric==metric) & (raw_res.bundle==bundle)]
            if not r_raw.empty:
                p = plot_group_results(bdf[bdf.metric==metric], r_raw, fig_dir, f"{cvar}_raw")
                if p: figure_paths.append(p)
            r_adj = adj_res[(adj_res.metric==metric) & (adj_res.bundle==bundle)]
            if not r_adj.empty:
                p = plot_group_results(bdf[bdf.metric==metric], r_adj, fig_dir, f"{cvar}_adj")
                if p: figure_paths.append(p)

    # Corrélations
    for corr_var in corr_variables:
        print(f"[CORR] Variable={corr_var}")
        # Sans ajustement
        raw_corr = pointwise_correlations(long_df, participants, corr_var, confounds=None)
        corr_results_dict[(corr_var,'raw')] = raw_corr
        # Ajusté (on choisit confounds_with_control par défaut)
        adj_corr = pointwise_correlations(long_df, participants, corr_var, confounds=conf_with)
        corr_results_dict[(corr_var,'adjusted')] = adj_corr
        for (metric,bundle), bdf in long_df.groupby(['metric','bundle']):
            r_raw = raw_corr[(raw_corr.metric==metric) & (raw_corr.bundle==bundle)]
            if not r_raw.empty:
                p = plot_corr_results(bdf[bdf.metric==metric], r_raw, fig_dir, f"{corr_var}_raw")
                if p: figure_paths.append(p)
            r_adj = adj_corr[(adj_corr.metric==metric) & (adj_corr.bundle==bundle)]
            if not r_adj.empty:
                p = plot_corr_results(bdf[bdf.metric==metric], r_adj, fig_dir, f"{corr_var}_adj")
                if p: figure_paths.append(p)

    # Résumé
    summary_df = build_summary_table(group_results_dict, corr_results_dict)
    summary_md = summary_markdown_table(summary_df)

    # Sauvegarde CSV des résultats détaillés
    for (k, v) in group_results_dict.items():
        if v is not None and not v.empty:
            v.to_csv(os.path.join(out_dir, f"group_results__{k[0]}__{k[1]}.csv"), index=False)
    for (k, v) in corr_results_dict.items():
        if v is not None and not v.empty:
            v.to_csv(os.path.join(out_dir, f"corr_results__{k[0]}__{k[1]}.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    # Construction du HTML
    html_parts = []
    html_parts.append("<html><head><meta charset='utf-8'><title>Tractometry Report</title></head><body>")
    html_parts.append("<h1>Rapport Tractométrie</h1>")
    html_parts.append("<h2>Résumé</h2>")
    html_parts.append("<pre>"+summary_md+"</pre>")

    # Sections group
    html_parts.append("<h2>Différences de groupes</h2>")
    for (cvar, adj), df in group_results_dict.items():
        html_parts.append(f"<h3>{cvar} ({adj})</h3>")
        if df is None or df.empty or not df['sig_fdr'].any():
            html_parts.append("<p>Aucun point significatif.</p>")
            continue
        sig_bundles = df[df.sig_fdr][['metric','bundle']].drop_duplicates()
        for _, row in sig_bundles.iterrows():
            metric, bundle = row.metric, row.bundle
            pattern = f"group_{metric}_{bundle}_{cvar}_{'raw' if adj=='raw' else 'adj'}".replace(' ','_')
            for fp in figure_paths:
                if pattern in os.path.basename(fp):
                    html_parts.append(f"<div><b>{bundle} / {metric}</b><br><img src='figures/{os.path.basename(fp)}' width='600'></div>")
                    break

    # Sections corr
    html_parts.append("<h2>Corrélations</h2>")
    for (cvar, adj), df in corr_results_dict.items():
        html_parts.append(f"<h3>{cvar} ({adj})</h3>")
        if df is None or df.empty or not (df['sig_fdr'] | (df['r'].abs()>=0.3)).any():
            html_parts.append("<p>Aucun point significatif ou corrélation forte.</p>")
            continue
        sig_bundles = df[(df.sig_fdr) | (df.r.abs()>=0.3)][['metric','bundle']].drop_duplicates()
        for _, row in sig_bundles.iterrows():
            metric, bundle = row.metric, row.bundle
            pattern = f"corr_{metric}_{bundle}_{cvar}_{'raw' if adj=='raw' else 'adj'}".replace(' ','_')
            for fp in figure_paths:
                if pattern in os.path.basename(fp):
                    html_parts.append(f"<div><b>{bundle} / {metric}</b><br><img src='figures/{os.path.basename(fp)}' width='600'></div>")
                    break

    html_parts.append("</body></html>")
    report_path = os.path.join(out_dir, report_name)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))
    print(f"[INFO] Rapport généré: {report_path}")
    return report_path

# --- NEW: exécution directe ------------------------------------------------------
if __name__ == "__main__":
    generate_report()

