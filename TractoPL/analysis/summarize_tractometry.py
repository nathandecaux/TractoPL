import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 17,
    'axes.titlesize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

CSV_RES = "/data/ndecaux/Reports/report_actidep_hcp_association_new_100pts_mcm_tensors_staniz_frechetlong2_AGE_ONLY_NON_DEP_RESAMPLED50PTS_mixed_pearson/summary_results.csv"

PALETTE = "Set2"

sns.set_theme(context='paper', font_scale=1.6, palette=PALETTE, style='whitegrid')

df = pd.read_csv(CSV_RES)


def extract_var(df):
    df = df.copy()
    df['var'] = df['type'].apply(lambda x: '_'.join(x.split('_')[1:]) or 'none')
    df['type'] = df['type'].apply(lambda x: x.split('_')[0])
    return df


def get_more_stat_from_csv(row, base_dir):
    """Find and read the detail CSV for a row, extract summary stats.

    Returns a dict of new columns (or None if the row is incomplete).
    """
    bundle = row.get('bundle')
    metric = row.get('metric')
    var = row.get('var')
    type_ = row.get('type')
    centroid_id = row.get('centroid_id', -1)

    if not all([bundle, metric, var, type_]):
        warnings.warn(f"Incomplete row, skipping enrichment: {row.to_dict()}")
        return None

    # Build filename
    kind = 'partial' if type_ != 'group' else 'corrected'
    if centroid_id not in (-1, None) and not pd.isna(centroid_id):
        filename = f"{bundle}_cent{centroid_id}_{metric}_{var}_{type_}_{kind}.csv"
    else:
        filename = f"{bundle}_{metric}_{var}_{type_}_{kind}.csv"

    # Check base_dir AND base_dir/figures (the original code overwrote
    # csv_path with the figures path without ever checking base_dir)
    figures_dir = os.path.join(base_dir, 'figures')
    candidates = [os.path.join(base_dir, filename),
                  os.path.join(figures_dir, filename)]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)

    if csv_path is None:
        raise FileNotFoundError(
            'CSV file not found for row: ' + str(row.to_dict()) +
            '\nChecked paths:\n' + '\n'.join(candidates))

    data = pd.read_csv(csv_path)
    new_data = {'detail_csv': csv_path}

    # Filter significant points
    if 'sig_afq' in data.columns:
        # Robust boolean conversion (handles bools and string 'True'/'False')
        sig_mask = data['sig_afq'].astype(str).str.lower().isin(['true', '1', '1.0'])
        sig_data = data[sig_mask]

        if sig_data.empty:
            for key in ['mean_r', 'mean_stat', 'mean_p', 'max_abs_stat']:
                new_data[key] = np.nan
        else:
            if 'r' in sig_data.columns:
                new_data['mean_r'] = sig_data['r'].mean()
            elif 'stat' in sig_data.columns:
                new_data['mean_stat'] = -sig_data['stat'].mean()
            if 'p_raw' in sig_data.columns:
                new_data['mean_p'] = sig_data['p_raw'].mean()
            stat_col = 'r' if 'r' in sig_data.columns else 'stat'
            new_data['max_abs_stat'] = sig_data[stat_col].abs().max()

    return new_data


df = extract_var(df)

# Enrich the DataFrame column by column (the original applied a function
# returning a Series/None with axis=1, which produces an object column
# instead of new columns)
base_dir = os.path.dirname(CSV_RES)
new_columns = defaultdict(lambda: [np.nan] * len(df))
for idx, row in df.iterrows():
    extra = get_more_stat_from_csv(row, base_dir)
    if extra is None:
        continue
    for key, value in extra.items():
        new_columns[key][idx] = value
for key, values in new_columns.items():
    df[key] = values


def _plot_pvalue_bars(detail_df, ax2):
    """Plot -10*log10(p_raw) as bars, red where significant."""
    pval_color = 'grey'
    p = np.clip(detail_df['p_raw'].values, 1e-300, None)  # avoid log(0)
    colors = np.where(detail_df['sig_afq'].astype(bool), 'red', pval_color)
    alphas = np.where(detail_df['sig_afq'].astype(bool), 0.3, 0.15)
    for x, h, c, a in zip(detail_df['point'], -10 * np.log10(p), colors, alphas):
        ax2.bar(x, h, color=c, alpha=a, width=0.8)
    ax2.set_ylabel('$-10\\log_{10}(p)$', color=pval_color)
    ax2.tick_params(axis='y', labelcolor=pval_color)
    ax2.axhline(-10 * np.log10(0.05), color=pval_color, linestyle='--')


def plot_tract_profile(detail_df, ax=None, title=None, output_path=None):
    """Plot a tract profile onto an axis (or a new figure if ax is None).

    - Two-group data ('mean_x'/'std_x' columns): group means +- std.
    - Correlation data ('r' column): r profile with p-value bars.
    """
    if detail_df is None or detail_df.empty:
        return None

    single = ax is None
    if single:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    mean_cols = [c for c in detail_df.columns if c.startswith('mean_')]

    if 'r' not in detail_df.columns and len(mean_cols) >= 2:
        # Two-group comparison (columns like 'mean_1.0', 'std_1.0', ...)
        g0 = mean_cols[0].split('_')[-1]
        g1 = mean_cols[1].split('_')[-1]

        mean_g0, mean_g1 = detail_df[mean_cols[0]], detail_df[mean_cols[1]]
        std_g0 = detail_df.get(f'std_{g0}', 0)
        std_g1 = detail_df.get(f'std_{g1}', 0)

        ax.plot(detail_df['point'], mean_g0, label=f'mean_{g0}', color='tab:blue')
        ax.plot(detail_df['point'], mean_g1, label=f'mean_{g1}', color='tab:orange')
        ax.fill_between(detail_df['point'], mean_g0 - std_g0, mean_g0 + std_g0,
                        color='tab:blue', alpha=0.2)
        ax.fill_between(detail_df['point'], mean_g1 - std_g1, mean_g1 + std_g1,
                        color='tab:orange', alpha=0.2)
        ax.set_ylabel('Mean value')
        ax.legend()
    else:
        # Correlation profile
        ycol = 'r' if 'r' in detail_df.columns else 'stat'
        ax.plot(detail_df['point'], detail_df[ycol], label=ycol, color='tab:blue')
        ax.set_ylabel('Mean correlation (r)' if ycol == 'r' else 'Mean statistic')

    ax.set_xlabel('Point along tract')

    # Add p-value bars on a secondary axis when available
    if 'p_raw' in detail_df.columns and 'point' in detail_df.columns:
        ax2 = ax.twinx()
        _plot_pvalue_bars(detail_df, ax2)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    if single and output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    return fig

out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# Set to True only if you also want individual PNGs
SAVE_INDIVIDUAL = False

if SAVE_INDIVIDUAL:
    for _, row in df.iterrows():
        if not isinstance(row.get('detail_csv'), str):
            continue
        detail_df = pd.read_csv(row['detail_csv'])
        png_name = os.path.basename(row['detail_csv']).replace('.csv', '.png')
        plot_tract_profile(detail_df,
                           output_path=os.path.join(out_dir, png_name))

# ---------------- Fused plot: one figure per variable ----------------
unique_vars = df['var'].dropna().unique()

for var in unique_vars:
    var_df = df[df['var'] == var].reset_index(drop=True)
    n = len(var_df)
    if n == 0:
        continue

    for metric in df['metric'].dropna().unique():
        var_met_df=var_df[var_df['metric']==metric].reset_index(drop=True)
        
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(7 * ncols, 5 * nrows),
                                squeeze=False)

        for ax, (_, row) in zip(axes.flat, var_met_df.iterrows()):
            detail_df = pd.read_csv(row['detail_csv'])
            title = f"{row['bundle']} | {row['metric']} | {row['type']}"
            # ax is not None -> nothing is saved, the figure is only drawn
            plot_tract_profile(detail_df, ax=ax, title=title)
    

        for ax in axes.flat[n:]:
            ax.set_visible(False)

        fig.suptitle(f"Variable: {var} - Metric {metric}", fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fused_path = os.path.join(out_dir, f"fused_{var}_{metric}.png")
        fig.savefig(fused_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved fused plot: {fused_path}")