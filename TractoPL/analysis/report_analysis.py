#!/usr/bin/env python3

import pandas as pd
import panel as pn
import hvplot.pandas  # active l'accessor hvplot
import os, time
import sys
import json
import argparse
import numpy as np

try:
    from TractoPL.analysis.fwe_utils import FWE_METHODS, n_significant, point_significance
except ImportError:  # lancé comme script autonome
    from fwe_utils import FWE_METHODS, n_significant, point_significance

pn.extension('tabulator')

parser = argparse.ArgumentParser(description="ActiDep - Résultats hors actimétrie")
parser.add_argument("filepath", nargs='?', default=None, help="Chemin vers summary_results.csv")
parser.add_argument("--port", type=int, default=6767, help="Premier port essayé (incrémenté s'il est pris)")
parser.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir de navigateur")
args = parser.parse_args()
CSV_PATH = args.filepath
print(CSV_PATH)
def point_csv_path(row, base_dir):
    """CSV par point d'une ligne du résumé (dans base_dir/figures ou base_dir), ou None."""
    bundle, metric, var, type_ = (row.get(k) for k in ('bundle', 'metric', 'var', 'type'))
    if not all([bundle, metric, var, type_]):
        return None
    cid = row.get('centroid_id')
    unit = bundle
    if cid is not None and not pd.isna(cid) and float(cid) != -1:
        unit = f"{bundle}_cent{int(float(cid))}"
    kind = 'corrected' if type_ == 'group' else 'partial'
    filename = f"{unit}_{metric}_{var}_{type_}_{kind}.csv"
    for d in (os.path.join(base_dir, 'figures'), base_dir):
        if os.path.exists(os.path.join(d, filename)):
            return os.path.join(d, filename)
    return None


def read_alpha(base_dir):
    """AFQ_ALPHA et FWE_METHOD de config.json (dossier de l'analyse)."""
    cfg_path = os.path.join(base_dir, 'config.json')
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            pass
    return float(cfg.get('AFQ_ALPHA') or 0.05), cfg.get('FWE_METHOD') or 'clusterFWE'


def row_stats(row, base_dir, alpha):
    """Lit le CSV par point d'une ligne et renvoie, pour chaque méthode FWE,
    n_sig_<m> et mean_r_<m> (r moyen des points significatifs), ainsi que min_n."""
    out = {}
    path = point_csv_path(row, base_dir)
    if path is None:
        return out
    try:
        data = pd.read_csv(path)
    except Exception as e:
        print(f"Erreur lecture {os.path.basename(path)}: {e}")
        return out
    for m in FWE_METHODS:
        sig = point_significance(data, m, alpha)
        if sig is None:  # anciens résultats sans p_tfce
            continue
        out[f'n_sig_{m}'] = n_significant(sig, m)
        if 'r' in data.columns and sig.any():
            out[f'mean_r_{m}'] = data.loc[sig, 'r'].mean()
    if 'n0' in data.columns and 'n1' in data.columns:
        out['min_n'] = data['n0'].min() + data['n1'].min()
    elif 'n' in data.columns:
        out['min_n'] = data['n'].min()
    return out


METHOD_COLS = [f'{c}_{m}' for m in FWE_METHODS for c in ('n_sig', 'mean_r')]


def load_data(path=CSV_PATH):
    if path is None or not os.path.exists(path):
        print (f"Le fichier CSV spécifié est introuvable ou aucun chemin fourni: {path}")
        # Retourner un DataFrame vide avec les colonnes attendues pour éviter les erreurs
        empty_cols = ['bundle', 'metric', 'type', 'var', 'centroid_id']
        return pd.DataFrame(columns=empty_cols)
    df = pd.read_csv(path)
    df['var']  = df['type'].apply(lambda x: '_'.join(x.split('_')[1:]))
    df['type'] = df['type'].apply(lambda x: x.split('_')[0])
    df['var']  = df['var'].astype('category')
    df['type'] = df['type'].astype('category')
    
    # n_sig / mean_r de chaque méthode FWE, lus dans les CSV par point (avec cache)
    base_dir = os.path.dirname(path)
    alpha, _ = read_alpha(base_dir)
    cache_path = path.replace('.csv', '_fwe_cache.csv')
    cached = None
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(path):
        cached = pd.read_csv(cache_path)
        if len(cached) != len(df) or 'min_n' not in cached.columns:
            cached = None
    if cached is not None:
        print(f"Chargement du cache FWE depuis {cache_path}")
    else:
        print("Lecture des CSV par point (toutes les méthodes FWE)...")
        cached = pd.DataFrame([row_stats(row, base_dir, alpha) for _, row in df.iterrows()],
                              columns=METHOD_COLS + ['min_n'])
        cached.to_csv(cache_path, index=False)
        print(f"Cache sauvegardé dans {cache_path}")
    for c in METHOD_COLS + ['min_n']:
        df[c] = cached[c].to_numpy() if c in cached.columns else np.nan
    df['removed_subjects'] = df['subjects_used'] - df['min_n']
    
    cols = ['type', 'var'] + [c for c in df.columns if c not in ['type', 'var']]
    df = df[cols]
    return df

df = load_data()
if 'open' not in df.columns:
    df.insert(0, 'open', "▶")

base_dir = os.path.dirname(CSV_PATH) if CSV_PATH else os.getcwd()

table = pn.widgets.Tabulator(
    df,
    pagination='local',
    page_size=30,
    sizing_mode='stretch_both',
    show_index=False,
    selectable='row',
    header_filters=True,  # garde la ligne de filtres
    layout='fit_data',
    buttons={'print': '<i class="fa fa-print"></i>'},
    configuration={
        # Ajout headerFilter pour toutes les colonnes (persist/recréation)
        "columnDefaults": {
            "resizable": True,
            "maxWidth": 260,
            "headerFilter": "input",
            "headerFilterLiveFilter": True,  # filtrage live
        },
        "columns": [
            {"title": "", "field": "open", "hozAlign": "center", "formatter": "html",
             "headerSort": False, "width": 55, "headerFilter": False}
        ]
    },
)

detail_panel = pn.Column(pn.pane.Markdown("Sélectionnez une ligne dans l'onglet Table."))

tabs = pn.Tabs(
    ("Table", table),
    ("Détail", detail_panel),
    dynamic=True  # on conserve dynamic=True mais on gère la restauration des filtres
)

# --- Gestion persistance des filtres Tabulator quand le widget est reconstruit ---
_saved_filters = {}

def _snapshot_filters():
    global _saved_filters
    _saved_filters = {f["field"]: f for f in table.filters if f.get("value") not in (None, "", [])}

def _capture_filters(event):
    # event.new: liste de dicts [{'field':..., 'type':..., 'value':...}, ...]
    _snapshot_filters()

table.param.watch(_capture_filters, 'filters')

def _restore_filters(event):
    if event.new == 0 and _saved_filters:
        # Ne réapplique que si le widget a perdu ses filtres visibles (ex: recréation)
        if not table.filters:
            filters_list = list(_saved_filters.values())
            def _apply():
                if table.filters != filters_list:
                    table.filters = filters_list
            try:
                pn.state.curdoc.add_next_tick_callback(_apply)
            except Exception:
                _apply()

tabs.param.watch(_restore_filters, 'active')
# -------------------------------------------------------------------------------

def _build_resource(path, kind):
    if not os.path.exists(path):
        return pn.pane.Markdown(f"**Manquant:** {os.path.basename(path)}", sizing_mode='stretch_width')
    if kind == 'img':
        return pn.pane.image.Image(path, sizing_mode='stretch_width')
    if kind == 'csv':
        try:
            d = pd.read_csv(path)
            return pn.widgets.Tabulator(d, height=300, page_size=50,
                                        show_index=False,theme='site',width=500)
        except Exception as e:
            return pn.pane.Markdown(f"Erreur lecture {os.path.basename(path)}: {e}")
    return pn.pane.Markdown("Type inconnu")

def _true_runs(mask):
    """(start, end) inclusive index runs where mask is True."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _build_profile_plot(csv_path, type_, method, alpha=0.05):
    """Profil le long du faisceau : r (corr) ou moyennes par groupe (group),
    avec -log10(p) sur un second axe y ; zones significatives selon `method`."""
    from bokeh.plotting import figure
    from bokeh.models import LinearAxis, Range1d, Span, BoxAnnotation, HoverTool, ColumnDataSource
    from bokeh.palettes import Category10_10

    if not os.path.exists(csv_path):
        return None
    try:
        d = pd.read_csv(csv_path)
    except Exception as e:
        return pn.pane.Markdown(f"Erreur lecture {os.path.basename(csv_path)}: {e}")
    if 'point' not in d.columns or 'p_raw' not in d.columns:
        return None

    d = d.sort_values('point').reset_index(drop=True)
    pts = d['point'].to_numpy()
    p = pd.to_numeric(d['p_raw'], errors='coerce').clip(lower=1e-300)
    d['_mlogp'] = -np.log10(p)
    sig_arr = point_significance(d, method, alpha)
    if sig_arr is None:
        return pn.pane.Markdown(f"Résultats sans les données nécessaires à **{method}** "
                                "(relancer l'analyse avec la version actuelle).")
    sig = pd.Series(sig_arr, index=d.index)

    fig = figure(height=420, sizing_mode='stretch_width', x_axis_label='Point',
                 tools='pan,wheel_zoom,box_zoom,reset,save')

    # Zones significatives
    for s, e in _true_runs(sig.to_list()):
        fig.add_layout(BoxAnnotation(left=pts[s] - 0.5, right=pts[e] + 0.5,
                                     fill_color='red', fill_alpha=0.12, line_alpha=0))

    renderers = []
    mean_cols = [c for c in d.columns if c.startswith('mean_') and c not in ('mean_g0', 'mean_g1')]
    if type_ != 'group' and 'r' in d.columns:
        fig.yaxis.axis_label = 'r'
        src = ColumnDataSource(d)
        renderers.append(fig.line('point', 'r', source=src, color=Category10_10[0],
                                  line_width=2, legend_label='r'))
        fig.scatter('point', 'r', source=ColumnDataSource(d[sig]), color='red', size=7,
                    legend_label='significatif')
        fig.add_layout(Span(location=0, dimension='width', line_color='gray',
                            line_dash='dashed', line_width=1))
        lo, hi = np.nanmin(d['r']), np.nanmax(d['r'])
    elif mean_cols:
        fig.yaxis.axis_label = 'Moyenne'
        lo, hi = np.inf, -np.inf
        for i, mc in enumerate(mean_cols):
            g = mc[len('mean_'):]
            color = Category10_10[i % 10]
            m = d[mc].to_numpy(dtype=float)
            sc = f'std_{g}'
            if sc in d.columns:
                sd = d[sc].to_numpy(dtype=float)
                fig.varea(x=pts, y1=m - sd, y2=m + sd, fill_color=color, fill_alpha=0.15)
                lo, hi = min(lo, np.nanmin(m - sd)), max(hi, np.nanmax(m + sd))
            else:
                lo, hi = min(lo, np.nanmin(m)), max(hi, np.nanmax(m))
            src = ColumnDataSource(pd.DataFrame({'point': pts, 'value': m, 'group': g}))
            renderers.append(fig.line('point', 'value', source=src, color=color, line_width=2,
                                      legend_label=str(g)))
    else:
        return pn.pane.Markdown("Colonnes `r` / `mean_*` absentes : pas de profil à tracer.")

    pad = 0.05 * (hi - lo) if np.isfinite(hi - lo) and hi > lo else 0.1
    fig.y_range = Range1d(lo - pad, hi + pad)

    # Second axe : -log10(p)
    thr = [(-np.log10(alpha), f'p={alpha:g}', 'gray')]
    if method in ('alphaFWE', 'mixed') and 'alphaFWE' in d.columns:
        a = pd.to_numeric(d['alphaFWE'], errors='coerce').dropna()
        if len(a) and a.iloc[0] > 0:
            thr.append((-np.log10(a.iloc[0]), f'alphaFWE={a.iloc[0]:.3g}', 'green'))
    has_tfce = method == 'TFCE' and 'p_tfce' in d.columns
    if has_tfce:
        d['_mlogp_tfce'] = -np.log10(pd.to_numeric(d['p_tfce'], errors='coerce').clip(lower=1e-300))
    ymax = max([d['_mlogp'].max(skipna=True)] + [t[0] for t in thr]
               + ([d['_mlogp_tfce'].max(skipna=True)] if has_tfce else [])) * 1.1
    fig.extra_y_ranges = {'logp': Range1d(0, ymax)}
    fig.add_layout(LinearAxis(y_range_name='logp', axis_label='-log10(p)'), 'right')
    psrc = ColumnDataSource(d[['point', '_mlogp', 'p_raw']])
    fig.vbar(x='point', top='_mlogp', width=0.9, source=psrc, y_range_name='logp',
             color='black', alpha=0.15, legend_label='-log10(p)')
    for y, lbl, col in thr:
        fig.line([pts.min(), pts.max()], [y, y], y_range_name='logp', color=col,
                 line_dash='dotted', line_width=1.5, legend_label=lbl)
    if has_tfce:
        fig.line('point', '_mlogp_tfce', source=ColumnDataSource(d[['point', '_mlogp_tfce']]),
                 y_range_name='logp', color='purple', line_width=1.5,
                 legend_label='-log10(p TFCE)')
    if method == 'clusterFWE' and 'clusterFWE' in d.columns:
        c = pd.to_numeric(d['clusterFWE'], errors='coerce').dropna()
        if len(c):
            fig.title.text = f'clusterFWE ≥ {int(c.iloc[0])} points (p < {alpha:g})'

    fig.add_tools(HoverTool(renderers=renderers, mode='vline',
                            tooltips=[('point', '@point'),
                                      ('valeur', '@r{0.000}' if type_ != 'group' else '@value{0.000}')]
                                     + ([('groupe', '@group')] if type_ == 'group' else [])))
    fig.add_tools(HoverTool(renderers=[r for r in fig.renderers if r.data_source is psrc],
                            tooltips=[('point', '@point'), ('p', '@p_raw{0.0000}'),
                                      ('-log10(p)', '@_mlogp{0.00}')]))
    fig.legend.location = 'top_left'
    fig.legend.click_policy = 'hide'
    return pn.pane.Bokeh(fig, sizing_mode='stretch_width')


def _make_detail_panel(row):
    bundle = row.get('bundle')
    metric = row.get('metric')
    var    = row.get('var')
    type_  = row.get('type')
    centroid_id= row.get('centroid_id', -1)
    model= row.get('model', 'unknown')

    if not all([bundle, metric, var, type_]):
        return pn.pane.Markdown("Données insuffisantes pour afficher le détail.")
    def build_name(kind, ext): 
        if kind=='partial' and type_=='group':
            kind='corrected'
        
        if model!="unknown":
            kind='poly2'
        if centroid_id is not None and not pd.isna(centroid_id) and float(centroid_id) != -1:
            return f"{bundle}_cent{int(float(centroid_id))}_{metric}_{var}_{type_}_{kind}.{ext}"
        else:
            return f"{bundle}_{metric}_{var}_{type_}_{kind}.{ext}"
    
    #If folder figures in base_dir exists, use it
    figures_dir = os.path.join(base_dir, 'figures')
    if os.path.exists(figures_dir) and os.path.isdir(figures_dir):
        base_dir_figures = figures_dir
    else:
        base_dir_figures = base_dir
   

    partial_img_path = os.path.join(base_dir_figures, build_name('partial', 'png'))
    partial_csv_path = os.path.join(base_dir_figures, build_name('partial', 'csv'))
    header = pn.pane.Markdown(f"### Détails: {bundle} / {metric} / {var} / {type_} — FWE : {fwe_select.value}",
                              sizing_mode='stretch_width')
    alpha, _ = read_alpha(base_dir)
    profile_plot = _build_profile_plot(partial_csv_path, type_, fwe_select.value, alpha)
    partial_section = pn.Column(
        *([pn.pane.Markdown("#### Profil"), profile_plot] if profile_plot is not None else []),
        pn.pane.Markdown("#### Partial"),
        _build_resource(partial_img_path, 'img'),
        _build_resource(partial_csv_path, 'csv'),
    )
    back_btn = pn.widgets.Button(name="Retour au tableau", button_type='primary')
    def _back(e):
        tabs.active = 0
        # Efface sélection (évite réouverture)
        try:
            table.selection = []
        except Exception:
            pass
    back_btn.on_click(_back)
    return pn.Column(
        back_btn,
        header,
        partial_section,
        sizing_mode='stretch_width'
    )

def _get_row_dict(idx):
    try:
        if idx in table.value.index:
            return table.value.loc[idx].to_dict()
    except Exception:
        pass
    try:
        return table.value.iloc[int(idx)].to_dict()
    except Exception:
        return None

def _cell_click(event):
    # Capture filtres avant de quitter l'onglet
    _snapshot_filters()
    # Ouvre le détail
    row_dict = _get_row_dict(event.row)
    if not row_dict:
        return
    if not all(k in row_dict for k in ('bundle','metric','var','type')):
        return
    panel = _make_detail_panel(row_dict)
    detail_panel.objects = panel.objects
    tabs.active = 1

# Reset anciens callbacks et attache
try:
    table._click_callbacks = []
except Exception:
    pass
table.on_click(_cell_click)

# Optionnel: désactiver surbrillance de sélection
table.selectable = False

# --- Widgets pour charger un CSV différent ---
csv_path_input = pn.widgets.TextInput(
    name="CSV Path", value=CSV_PATH or "", placeholder="Chemin vers summary_results.csv", sizing_mode='stretch_width'
)
load_csv_btn = pn.widgets.Button(name="Charger CSV", button_type='primary')
load_status = pn.pane.Markdown(f"Chargé: `{CSV_PATH}`" if CSV_PATH else "Aucun fichier chargé", sizing_mode='stretch_width')
# ---------------------------------------------

# --- Méthode FWE (appliquée à la volée : n_sig, mean_r, profils) ---
fwe_select = pn.widgets.Select(name="Méthode FWE", options=list(FWE_METHODS),
                               value=read_alpha(base_dir)[1])


def _with_fwe(sub, method):
    """n_sig / mean_r de la méthode choisie, colonnes par méthode masquées."""
    sub = sub.copy()
    for c in ('n_sig', 'mean_r'):
        col = f'{c}_{method}'
        if col in sub.columns:
            sub[c] = sub[col]
    return sub.drop(columns=[c for c in METHOD_COLS if c in sub.columns])


# --- Filtres sidebar ---
metric_filter = pn.widgets.MultiChoice(name="Metric contient", placeholder="ex: FA", options=sorted(df['metric'].unique()), value=[])
bundle_filter = pn.widgets.MultiChoice(name="Bundle contient", options=sorted(df['bundle'].unique()), value=[])
type_filter   = pn.widgets.MultiChoice(name="Type", options=sorted(df['type'].unique()), value=[])
var_filter    = pn.widgets.MultiChoice(name="Var",  options=sorted(df['var'].unique()), value=[])
# Listes prédéfinies de bundles (identiques à report_analysis_actimetry.py)
fronto_limbiques = [
    "CG_left", "CG_right",
    "UF_left", "UF_right",
    "FX_left", "FX_right",
    "FPT_left", "FPT_right",
    "T_PREF_left", "T_PREF_right",
    "ST_FO_left", "ST_FO_right",
    "ST_PREF_left", "ST_PREF_right"
]
emotions = [
    "CG_left", "CG_right",
    "UF_left", "UF_right",
    "FX_left", "FX_right",
    "ATR_left", "ATR_right",
    "CA",
    "T_PREF_left", "T_PREF_right",
    "ST_FO_left", "ST_FO_right",
    "ST_PREF_left", "ST_PREF_right"
]
moteurs = [
    "CST_left", "CST_right",
    "CC_3", "CC_4",
    "T_PREM_left", "T_PREM_right",
    "T_PREC_left", "T_PREC_right",
    "T_POSTC_left", "T_POSTC_right",
    "ST_PREM_left", "ST_PREM_right",
    "ST_PREC_left", "ST_PREC_right",
    "ST_POSTC_left", "ST_POSTC_right",
    "ICP_left", "ICP_right",
    "MCP",
    "SCP_left", "SCP_right"
]

fronto_limbiques_btn = pn.widgets.Button(name="Fronto-Limbiques", button_type="primary")
emotions_btn = pn.widgets.Button(name="Émotions", button_type="success")
moteurs_btn = pn.widgets.Button(name="Moteurs", button_type="warning")

def _apply_list_to_bundle(event, bundle_list):
    # Même logique que dans report_analysis_actimetry.py
    bundle_filter.value = bundle_filter.value + [x.replace('_','') for x in bundle_list]

fronto_limbiques_btn.on_click(lambda event: _apply_list_to_bundle(event, fronto_limbiques))
emotions_btn.on_click(lambda event: _apply_list_to_bundle(event, emotions))
moteurs_btn.on_click(lambda event: _apply_list_to_bundle(event, moteurs))

# Filtres numériques (mêmes colonnes et parsing d'opérateurs)
n_sig_corrected_filter = pn.widgets.TextInput(name="n_sig_corrected (>, <)", placeholder="ex: >5")
min_p_corrected_filter = pn.widgets.TextInput(name="min_p_corrected (>, <)", placeholder="ex: <0.05")
max_abs_r_partial_filter = pn.widgets.TextInput(name="max_abs_r_partial (>, <)", placeholder="ex: >0.6")
removed_subjects_filter = pn.widgets.TextInput(name="removed_subjects (>, <)", placeholder="ex: >1")
removed_points_filter = pn.widgets.TextInput(name="removed_points (>, <)", placeholder="ex: <3")

reset_filters_btn = pn.widgets.Button(name="Réinitialiser filtres", button_type='warning')

def _apply_sidebar_filters(event=None):
    # Toujours travailler sur une copie pour éviter effets de bord
    sub = _with_fwe(df, fwe_select.value)

    # Créer des colonnes dérivées si elles n'existent pas, sans supposer l'existence des colonnes sources
    if 'n_sig_corrected' not in sub.columns:
        sub['n_sig_corrected'] = sub['n_sig'] if 'n_sig' in sub.columns else pd.Series(dtype='float')
    if 'min_p_corrected' not in sub.columns:
        sub['min_p_corrected'] = sub['min_p'] if 'min_p' in sub.columns else pd.Series(dtype='float')
    if 'max_abs_r_partial' not in sub.columns:
        sub['max_abs_r_partial'] = sub['max_abs_r'] if 'max_abs_r' in sub.columns else pd.Series(dtype='float')
        
    if metric_filter.value:
        if 'metric' in sub.columns:
            sub = sub[sub['metric'].isin(metric_filter.value)]
    if bundle_filter.value:
        if 'bundle' in sub.columns:
            sub = sub[sub['bundle'].isin(bundle_filter.value)]
    if type_filter.value:
        if 'type' in sub.columns:
            sub = sub[sub['type'].isin(type_filter.value)]
    if var_filter.value:
        if 'var' in sub.columns:
            sub = sub[sub['var'].isin(var_filter.value)]
    # Application des filtres numériques avec seuils (comme dans report_analysis_actimetry.py)
    for col, widget in [
        ('n_sig_corrected', n_sig_corrected_filter),
        ('min_p_corrected', min_p_corrected_filter),
        ('max_abs_r_partial', max_abs_r_partial_filter),
        ('removed_subjects', removed_subjects_filter),
        ('removed_points', removed_points_filter),
    ]:
        if widget.value:
            try:
                # Si la colonne n'existe pas, on l'ignore
                if col not in sub.columns:
                    continue
                if widget.value[0] in ('>', '<'):
                    op, val = widget.value[0], float(widget.value[1:])
                else:
                    op, val = '==', float(widget.value)
                if op == '>':
                    sub = sub[sub[col] > val]
                elif op == '<':
                    sub = sub[sub[col] < val]
                elif op == '==':
                    sub = sub[sub[col] == val]
            except Exception:
                pass
    # Met à jour le dataframe affiché
    table.value = sub

for w in (fwe_select, metric_filter, bundle_filter, type_filter, var_filter,
          n_sig_corrected_filter, min_p_corrected_filter, max_abs_r_partial_filter,
          removed_subjects_filter, removed_points_filter):
    w.param.watch(_apply_sidebar_filters, 'value')


def _reload_data(path=None):
    """Recharge le CSV indiqué et met à jour le tableau et les filtres."""
    global df, base_dir, CSV_PATH
    if path is None:
        path = csv_path_input.value
    try:
        df_new = load_data(path)
    except Exception as e:
        load_status.object = f"**Erreur lecture:** {e}"
        return
    df = df_new
    if 'open' not in df.columns:
        df.insert(0, 'open', "▶")
    CSV_PATH = path
    base_dir = os.path.dirname(CSV_PATH)
    fwe_select.value = read_alpha(base_dir)[1]
    # Met à jour les options des filtres si disponibles
    try:
        metric_filter.options = sorted(df['metric'].unique())
        bundle_filter.options = sorted(df['bundle'].unique())
        type_filter.options = sorted(df['type'].unique())
        var_filter.options = sorted(df['var'].unique())
    except Exception:
        pass
    # Réinitialise et applique les filtres visibles, puis met à jour le tableau
    _reset_filters(None)
    load_status.object = f"Chargé: `{CSV_PATH}`"

# Bind du bouton
load_csv_btn.on_click(lambda e: _reload_data())


def _reset_filters(event):
    metric_filter.value = []
    bundle_filter.value = []
    type_filter.value = []
    var_filter.value = []
    # Reset numériques
    n_sig_corrected_filter.value = ""
    min_p_corrected_filter.value = ""
    max_abs_r_partial_filter.value = ""
    removed_subjects_filter.value = ""
    removed_points_filter.value = ""
    _apply_sidebar_filters()

reset_filters_btn.on_click(_reset_filters)
# Initial
_apply_sidebar_filters()
# -----------------------------------------

template = pn.template.FastListTemplate(
    title="ActiDep - Résultats hors actimétrie",
    sidebar=[
        "Table interactive (filtrez dans les en-têtes).",
        "Cliquez une ligne pour voir les détails (onglet Détail).",
        "Dans l'onglet Détail, utilisez le bouton pour revenir.",
        ("Autoreload actif." if '--autoreload' in sys.argv else ""),
        "#### CSV",
        csv_path_input,
        load_csv_btn,
        load_status,
        "#### Correction FWE",
        fwe_select,
        "#### Filtres",
        metric_filter,
        bundle_filter,
        type_filter,
        var_filter,
        "#### Listes prédéfinies",
        fronto_limbiques_btn,
        emotions_btn,
        moteurs_btn,
        "#### Filtres numériques",
        n_sig_corrected_filter,
        min_p_corrected_filter,
        max_abs_r_partial_filter,
        removed_subjects_filter,
        removed_points_filter,
        reset_filters_btn,
    ],
    main=[tabs],
    accent_base_color="#1f77b4",
)

pn.config.sizing_mode = 'stretch_width'
template.servable()



def main():
        #Check port 6767, if already in use, choose, increase
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = args.port
    while True:
        try:
            sock.bind(("0.0.0.0", port))
            sock.close()
            break
        except OSError:
            port += 1
    print(f"Using port {port}")

    #Open is false if this is starting from an ssh session
    import os
    if os.environ.get("SSH_CONNECTION") or args.no_browser:
        open_browser = False
    else:
        open_browser = True
    template.show(autoreload=True, websocket_origin=["*"], port=port, open=open_browser)

if __name__ == '__main__':
    main()


    # #If hostname is calcarine, don't open
    # import socket
    # hostname = socket.gethostname()
    # if hostname != "calcarine":
    #     template.show(autoreload=True,websocket_origin=["*"], port=port)








