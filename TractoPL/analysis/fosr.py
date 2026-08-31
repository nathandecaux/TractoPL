"""
Function-on-Scalar Regression (FoSR) pour l'analyse tractométrique (BUAN 2.0, section 6.2.2.7).

Implémente en Python pur (numpy/scipy) le modèle:

    Y(t) = X * beta(t) + eps(t)

où :
    - Y est une matrice (N x m) de réponses fonctionnelles (ex: FA moyennée par segment
      le long d'un faisceau, une ligne par sujet/streamline, une colonne par segment t).
    - X est la matrice de dessin (N x q) : intercept + variable(s) de groupe + confondants.
    - beta(t) = [beta_0(t), ..., beta_{q-1}(t)]^T est le vecteur de fonctions à estimer,
      représentées par une base de B-splines et régularisées par une pénalité de rugosité
      (P-splines, Eilers & Marx 1996), avec choix du paramètre de lissage par GCV.

Ceci est l'équivalent Python de `refund::fosr` (R) utilisé dans le papier BUAN 2.0, sans
dépendance à R/rpy2. La significativité globale (contrôle du FWER le long du profil) est
obtenue par test de permutation de type Freedman-Lane avec statistique du maximum |t(t)|.

Convention d'axes : contrairement à un modèle mixte point par point (cf. mixed_effect.py),
il n'y a ici qu'une seule observation fonctionnelle par sujet (le profil complet), donc pas
besoin d'effet aléatoire sujet : c'est un modèle de régression multivarié classique, mais
la fonction beta(t) est lissée conjointement sur tous les points au lieu d'être estimée
indépendamment point par point.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.05


# =============================================================================
# Base de B-splines et pénalité de rugosité
# =============================================================================

def _bspline_design_matrix(t: np.ndarray, nbasis: int = 10, degree: int = 3) -> np.ndarray:
    """Matrice de dessin B-spline Phi (m x nbasis) évaluée aux abscisses t."""
    t = np.asarray(t, float)
    tmin, tmax = t.min(), t.max()
    n_interior = max(nbasis - degree - 1, 0)
    interior_knots = np.linspace(tmin, tmax, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
    knots = np.concatenate([
        np.repeat(tmin, degree + 1),
        interior_knots,
        np.repeat(tmax, degree + 1),
    ])
    nb = len(knots) - degree - 1
    Phi = np.zeros((len(t), nb))
    for i in range(nb):
        coef = np.zeros(nb)
        coef[i] = 1.0
        Phi[:, i] = BSpline(knots, coef, degree, extrapolate=True)(t)
    return np.nan_to_num(Phi)


def _difference_penalty(nbasis: int, order: int = 2) -> np.ndarray:
    """Matrice de pénalité de rugosité P (nbasis x nbasis) = D^T D, D = différences d'ordre `order`."""
    D = np.eye(nbasis)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D.T @ D


def _smooth_subject_coefficients(y_obs: np.ndarray, Phi_obs: np.ndarray,
                                 P: np.ndarray, lam: float) -> Optional[np.ndarray]:
    """Calcule les coefficients de B-spline pénalisés pour un sujet à partir de points observés."""
    if len(y_obs) <= Phi_obs.shape[1] // 2:
        return None
    try:
        A = Phi_obs.T @ Phi_obs + lam * P
        b = Phi_obs.T @ y_obs
        coefs = np.linalg.solve(A, b)
        return coefs
    except np.linalg.LinAlgError:
        return None


def _smooth_subject_profiles(pivot: pd.DataFrame, points: np.ndarray,
                             nbasis: int = 10, degree: int = 3,
                             penalty_order: int = 2, lambda_smooth: float = 1e-2,
                             min_obs: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lisse chaque sujet indépendamment sur les points observés en gardant la grille complète."""
    points = np.asarray(points, float)
    Phi_full = _bspline_design_matrix(points, nbasis=nbasis, degree=degree)
    P = _difference_penalty(Phi_full.shape[1], order=penalty_order)

    subject_ids = pivot.index.to_numpy()
    n_subjects = len(subject_ids)
    n_points = len(points)
    Y_smoothed = np.full((n_subjects, n_points), np.nan, float)
    n_obs = np.zeros(n_subjects, int)
    keep_mask = np.zeros(n_subjects, bool)

    for i, subject in enumerate(subject_ids):
        row = pivot.loc[subject].to_numpy(float)
        obs_mask = ~np.isnan(row)
        n_obs[i] = int(obs_mask.sum())
        if n_obs[i] < min_obs:
            continue
        Phi_obs = Phi_full[obs_mask]
        y_obs = row[obs_mask]
        coefs = _smooth_subject_coefficients(y_obs, Phi_obs, P, lambda_smooth)
        if coefs is None or np.any(np.isnan(coefs)):
            continue
        Y_smoothed[i] = Phi_full @ coefs
        keep_mask[i] = True

    if not keep_mask.any():
        return np.empty((0, n_points)), subject_ids[keep_mask], n_obs
    return Y_smoothed[keep_mask], subject_ids[keep_mask], n_obs[keep_mask]


# =============================================================================
# Ajustement du modèle FoSR (P-splines, sélection de lambda par GCV)
# =============================================================================

def fit_fosr(Y: np.ndarray, X: np.ndarray, t: Optional[np.ndarray] = None,
             nbasis: int = 10, degree: int = 3, penalty_order: int = 2,
             lam: Optional[float] = None, lam_grid: Optional[Sequence[float]] = None) -> Dict:
    """
    Ajuste Y(t) = X beta(t) + eps(t) avec beta_j(t) représentée par des B-splines pénalisées.

    Args:
        Y: (N, m) réponses fonctionnelles (une ligne par sujet).
        X: (N, q) matrice de dessin (l'appelant doit inclure la colonne d'intercept).
        t: (m,) abscisses des segments ; par défaut 0..m-1.
        nbasis: nombre de fonctions de base B-spline pour beta(t).
        penalty_order: ordre de la pénalité de différences (2 = pénalise la courbure).
        lam: paramètre de lissage fixé ; si None, sélectionné par GCV sur `lam_grid`.
        lam_grid: grille de valeurs de lambda à essayer (log-espacée par défaut).

    Returns:
        dict avec beta (q, m), se (q, m), t, Phi, Theta (q, nbasis), lam, sigma2,
        edf (degrés de liberté effectifs), fitted (N, m), resid (N, m), r2.
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    N, m = Y.shape
    Nx, q = X.shape
    if N != Nx:
        raise ValueError(f"Y et X doivent avoir le même nombre de lignes (N): {N} vs {Nx}")
    if t is None:
        t = np.arange(m)
    t = np.asarray(t, float)

    Phi = _bspline_design_matrix(t, nbasis=nbasis, degree=degree)
    nb = Phi.shape[1]
    P = _difference_penalty(nb, order=penalty_order)

    XtX = X.T @ X
    PhitPhi = Phi.T @ Phi
    rhs = (X.T @ Y @ Phi).flatten(order='F')  # vec(X^T Y Phi), (q*nb,)
    A0 = np.kron(PhitPhi, XtX)  # matrice non pénalisée (nb*q, nb*q)
    Ip = np.eye(q)

    def _fit_for_lambda(lam_):
        A = A0 + lam_ * np.kron(P, Ip)
        theta_vec = np.linalg.solve(A, rhs)
        Theta = theta_vec.reshape((q, nb), order='F')
        fitted = X @ Theta @ Phi.T
        rss = float(np.sum((Y - fitted) ** 2))
        edf = float(np.trace(np.linalg.solve(A, A0)))
        gcv = (rss / (N * m)) / max(1.0 - edf / (N * m), 1e-6) ** 2
        return Theta, fitted, rss, edf, gcv

    if lam is None:
        if lam_grid is None:
            lam_grid = np.logspace(-3, 4, 25)
        best = None
        for lam_ in lam_grid:
            Theta, fitted, rss, edf, gcv = _fit_for_lambda(lam_)
            if best is None or gcv < best[0]:
                best = (gcv, lam_, Theta, fitted, rss, edf)
        _, lam, Theta, fitted, rss, edf = best
    else:
        Theta, fitted, rss, edf = _fit_for_lambda(lam)[:4]

    A = A0 + lam * np.kron(P, Ip)
    A_inv = np.linalg.inv(A)
    sigma2 = rss / max(N * m - edf, 1.0)
    cov_theta = sigma2 * (A_inv @ A0 @ A_inv)  # covariance "sandwich" du vecteur theta

    beta = Theta @ Phi.T  # (q, m)
    se = np.zeros((q, m))
    for j in range(q):
        idx = np.arange(nb) * q + j  # indices de vec(Theta) (ordre 'F') pour la covariable j
        cov_j = cov_theta[np.ix_(idx, idx)]  # (nb, nb)
        var_t = np.einsum('ti,ij,tj->t', Phi, cov_j, Phi)
        se[j] = np.sqrt(np.clip(var_t, 0, None))

    resid = Y - fitted
    tss = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan

    return dict(beta=beta, se=se, t=t, Phi=Phi, Theta=Theta, cov_theta=cov_theta,
                lam=lam, sigma2=sigma2, edf=edf, fitted=fitted, resid=resid, r2=r2, nbasis=nb)


# =============================================================================
# Test de permutation (Freedman-Lane) pour contrôle du FWER le long du profil
# =============================================================================

def fosr_permutation_test(Y: np.ndarray, X: np.ndarray, col_idx: int, t: Optional[np.ndarray] = None,
                           nbasis: int = 10, degree: int = 3, penalty_order: int = 2,
                           lam: Optional[float] = None, n_perm: int = 499,
                           random_state: Optional[int] = 0) -> Dict:
    """
    Test de permutation de type Freedman-Lane pour la covariable X[:, col_idx].

    Principe : on ajuste le modèle réduit (sans la covariable testée), on permute ses résidus,
    on réajuste le modèle complet sur les données permutées, et on calcule la statistique du
    maximum de |beta/se| le long de t sous H0. Cela donne :
      - p_perm(t) : p-value ponctuelle corrigée pour comparaisons multiples (contrôle du FWER),
      - p_global : p-value globale (test sur tout le profil).

    Returns:
        dict avec t_stat_obs (m,), p_raw (m,) [approx. Gaussienne, non corrigée],
        p_perm (m,) [FWER, statistique du max], p_global (scalaire), null_max (n_perm,).
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    N, m = Y.shape
    q = X.shape[1]
    rng = np.random.default_rng(random_state)

    fit_full = fit_fosr(Y, X, t=t, nbasis=nbasis, degree=degree, penalty_order=penalty_order, lam=lam)
    beta_obs = fit_full['beta'][col_idx]
    se_obs = fit_full['se'][col_idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat_obs = np.where(se_obs > 0, beta_obs / se_obs, 0.0)
    p_raw = 2 * (1 - stats.norm.cdf(np.abs(t_stat_obs)))

    # Modèle réduit : toutes les colonnes sauf celle testée
    reduced_cols = [j for j in range(q) if j != col_idx]
    X_reduced = X[:, reduced_cols]
    fit_reduced = fit_fosr(Y, X_reduced, t=t, nbasis=nbasis, degree=degree, penalty_order=penalty_order,
                            lam=fit_full['lam'])
    resid_reduced = Y - fit_reduced['fitted']
    fitted_reduced = fit_reduced['fitted']

    null_max = np.empty(n_perm)
    lam_fixed = fit_full['lam']
    for p in range(n_perm):
        perm = rng.permutation(N)
        Y_perm = fitted_reduced + resid_reduced[perm]
        fit_p = fit_fosr(Y_perm, X, t=t, nbasis=nbasis, degree=degree, penalty_order=penalty_order,
                          lam=lam_fixed)
        se_p = fit_p['se'][col_idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            stat_p = np.where(se_p > 0, fit_p['beta'][col_idx] / se_p, 0.0)
        null_max[p] = np.max(np.abs(stat_p))

    obs_abs = np.abs(t_stat_obs)
    p_perm = np.array([(np.sum(null_max >= v) + 1) / (n_perm + 1) for v in obs_abs])
    p_global = (np.sum(null_max >= obs_abs.max()) + 1) / (n_perm + 1)

    return dict(t_stat_obs=t_stat_obs, p_raw=p_raw, p_perm=p_perm, p_global=p_global,
                null_max=null_max, beta=beta_obs, se=se_obs, fit_full=fit_full)


# =============================================================================
# API haut niveau, alignée sur les conventions de mixed_effect.py (long_df)
# =============================================================================

def _build_design_matrix(meta: pd.DataFrame, effect_col: str, confounds: Optional[List[str]],
                          effect_type: str = 'group') -> Tuple[np.ndarray, List[str]]:
    """Construit X (intercept + effet + confondants) à partir d'un DataFrame indexé par sujet."""
    cols = []
    names = []
    ones = np.ones(len(meta))
    cols.append(ones)
    names.append('Intercept')

    if effect_type == 'group':
        dummies = pd.get_dummies(meta[effect_col].astype('category'), drop_first=True)
        for c in dummies.columns:
            cols.append(dummies[c].to_numpy(float))
            names.append(f'{effect_col}[{c}]')
    else:
        cols.append(pd.to_numeric(meta[effect_col], errors='coerce').to_numpy(float))
        names.append(effect_col)

    for conf in (confounds or []):
        if conf not in meta.columns:
            continue
        if meta[conf].dtype == 'object' or str(meta[conf].dtype).startswith('category'):
            dummies = pd.get_dummies(meta[conf].astype('category'), drop_first=True)
            for c in dummies.columns:
                cols.append(dummies[c].to_numpy(float))
                names.append(f'{conf}[{c}]')
        else:
            cols.append(pd.to_numeric(meta[conf], errors='coerce').to_numpy(float))
            names.append(conf)

    X = np.column_stack(cols)
    return X, names


def fosr_group_test(long_df: pd.DataFrame, classif_col: str,
                     confounds: Optional[List[str]] = None,
                     effect_type: str = 'group',
                     nbasis: int = 10, n_perm: int = 499,
                     random_state: int = 0, verbose: bool = False,
                     lambda_smooth: float = 1e-2, min_obs: int = 4) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyse FoSR d'un profil de faisceau (une observation fonctionnelle par sujet).

    Args:
        long_df: colonnes 'subject', 'point', 'value' + colonnes méta (classif_col, confounds).
        classif_col: covariable d'intérêt (groupe catégoriel ou variable continue).
        effect_type: 'group' (dummy-codée) ou 'continuous'.
        nbasis: nombre de fonctions de base B-spline pour beta(t).
        n_perm: nombre de permutations pour le contrôle du FWER (statistique du max).
        lambda_smooth: paramètre de lissage pour les profils sujets.
        min_obs: nombre minimal de points observés pour préserver un sujet.

    Returns:
        (df_results, info) où df_results a les colonnes:
          point, beta, se, t_stat, p_raw, q (FDR-BH), p_perm (FWER), sig_fdr, sig_fwer
        info contient des diagnostics (n_subjects, n_points, colonnes du design, lambda choisi, r2...).
    """
    info = {'n_subjects_initial': 0, 'n_subjects_used': 0, 'n_points_used': 0,
            'n_subjects_dropped_sparse': 0, 'n_subjects_dropped_cov': 0,
            'confounds_used': [], 'design_columns': [], 'lam': None, 'r2': None,
            'lambda_smooth': lambda_smooth}

    if classif_col not in long_df.columns:
        if verbose:
            print(f"  [SKIP] Colonne '{classif_col}' non trouvée")
        return pd.DataFrame(), info

    confounds = confounds or []
    available_confounds = [c for c in confounds if c in long_df.columns
                            and long_df.drop_duplicates('subject')[c].notna().mean() >= 0.5]
    info['confounds_used'] = available_confounds

    cols_needed = ['subject', 'point', 'value', classif_col] + available_confounds
    df = long_df[cols_needed].dropna(subset=[classif_col])

    pivot = df.pivot_table(index='subject', columns='point', values='value')
    meta = df.drop_duplicates('subject').set_index('subject')

    info['n_subjects_initial'] = pivot.shape[0]

    points = pivot.columns.to_numpy(float)
    Y_smoothed, kept_subjects, n_obs = _smooth_subject_profiles(
        pivot, points, nbasis=nbasis, degree=3,
        penalty_order=2, lambda_smooth=lambda_smooth, min_obs=min_obs)

    if Y_smoothed.size == 0:
        if verbose:
            print("  [SKIP] Aucun sujet lissable avec le nombre minimal de points observés")
        return pd.DataFrame(), info

    meta = meta.reindex(kept_subjects).dropna(subset=[classif_col] + available_confounds)
    valid_subjects = meta.index
    Y_smoothed = Y_smoothed[np.isin(kept_subjects, valid_subjects)]
    kept_subjects = kept_subjects[np.isin(kept_subjects, valid_subjects)]

    info['n_subjects_dropped_sparse'] = int((pivot.shape[0] - len(kept_subjects))
                                            - np.sum(meta.index.notna()))
    info['n_subjects_dropped_cov'] = int(pivot.shape[0] - len(kept_subjects) - info['n_subjects_dropped_sparse'])

    if effect_type == 'group' and meta[classif_col].nunique() < 2:
        if verbose:
            print(f"  [SKIP] Moins de 2 groupes pour '{classif_col}'")
        return pd.DataFrame(), info

    if Y_smoothed.shape[0] < 10 or Y_smoothed.shape[1] < 5:
        if verbose:
            print(f"  [SKIP] Pas assez de données: {Y_smoothed.shape[0]} sujets, {Y_smoothed.shape[1]} points")
        return pd.DataFrame(), info

    points = points
    Y = Y_smoothed
    X, design_columns = _build_design_matrix(meta.loc[kept_subjects], classif_col,
                                             available_confounds, effect_type)

    info['n_subjects_used'] = Y.shape[0]
    info['n_points_used'] = Y.shape[1]
    info['design_columns'] = design_columns
    info['miss_frac'] = float(np.mean(np.isnan(pivot.loc[kept_subjects].to_numpy(float))))
    info['med_obs'] = float(np.median(n_obs[np.isin(kept_subjects, kept_subjects)]))

    effect_idx = 1  # colonne juste après l'intercept (première dummy / variable continue)
    nb = min(nbasis, max(4, Y.shape[1] // 3))

    perm_res = fosr_permutation_test(Y, X, col_idx=effect_idx, t=points, nbasis=nb,
                                      n_perm=n_perm, random_state=random_state)
    fit_full = perm_res['fit_full']
    info['lam'] = fit_full['lam']
    info['r2'] = fit_full['r2']
    info['effect_name'] = design_columns[effect_idx]

    df_results = pd.DataFrame({
        'point': points,
        'beta': perm_res['beta'],
        'se': perm_res['se'],
        't_stat': perm_res['t_stat_obs'],
        'p_raw': perm_res['p_raw'],
        'p_perm': perm_res['p_perm'],
    })
    valid_mask = df_results['p_raw'].notna()
    df_results['q'] = np.nan
    if valid_mask.any():
        _, q_fdr, _, _ = multipletests(df_results.loc[valid_mask, 'p_raw'], method='fdr_bh')
        df_results.loc[valid_mask, 'q'] = q_fdr
    df_results['sig_fdr'] = df_results['q'] < ALPHA
    df_results['sig_fwer'] = df_results['p_perm'] < ALPHA
    info['p_global'] = perm_res['p_global']

    return df_results, info


# =============================================================================
# Visualisation
# =============================================================================

def plot_fosr_results(results_df: pd.DataFrame, metric_name: str, bundle_name: str,
                       effect_name: str, out_path: str, p_global: Optional[float] = None) -> None:
    """Trace beta(t) (IC 95%) et les p-values ponctuelles / corrigées (FDR et FWER)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    ci = 1.96 * results_df['se']
    ax1.fill_between(results_df['point'], results_df['beta'] - ci, results_df['beta'] + ci,
                      alpha=0.3, color='tab:blue', label='IC 95%')
    ax1.plot(results_df['point'], results_df['beta'], color='tab:blue', linewidth=2, label='β(t) (FoSR)')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)

    if results_df['sig_fwer'].any():
        sig = results_df[results_df['sig_fwer']]
        ax1.scatter(sig['point'], sig['beta'], color='red', s=50, zorder=5, label='p_FWER < 0.05')
        _highlight_clusters(ax1, results_df['point'].values, results_df['sig_fwer'].values,
                             color='red', alpha=0.15)

    title = f'{bundle_name} - {metric_name} - Effet: {effect_name} (FoSR)'
    if p_global is not None:
        title += f' — p_global={p_global:.3f}'
    ax1.set_title(title)
    ax1.set_ylabel('β(t)')
    ax1.legend(loc='upper right', fontsize=9)

    ax2 = axes[1]
    ax2.bar(results_df['point'], results_df['p_raw'], color='gray', alpha=0.5, width=1.0, label='p brute')
    ax2.plot(results_df['point'], results_df['q'], color='green', linewidth=1.5, label='q (FDR)')
    ax2.plot(results_df['point'], results_df['p_perm'], color='red', linewidth=1.5, label='p (FWER, permutation)')
    ax2.axhline(ALPHA, color='black', linestyle='--', linewidth=1.0)
    ax2.set_ylabel('p-value')
    ax2.set_xlabel('Segment le long du faisceau (t)')
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _highlight_clusters(ax, points, sig_mask, color='red', alpha=0.18):
    if len(points) == 0 or not np.any(sig_mask):
        return
    start, prev = None, None
    for p, s in zip(points, sig_mask):
        if s and start is None:
            start = p
        if (not s) and start is not None:
            ax.axvspan(start, prev, color=color, alpha=alpha, linewidth=0)
            start = None
        prev = p
    if start is not None:
        ax.axvspan(start, prev, color=color, alpha=alpha, linewidth=0)


# =============================================================================
# Script d'exemple (exécution directe)
# =============================================================================
if __name__ == '__main__':
    import os
    from TractoPL.data.loader import Dataset
    from TractoPL.set_config import get_HCP_bundle_names

    pipeline = 'hcp_association_new_50pts_mcm_tensors_staniz_longcentral'
    sub_infos = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx"
    OUTDIR = 'fosr_report'
    os.makedirs(os.path.join(OUTDIR, 'figures'), exist_ok=True)

    bundle_names = get_HCP_bundle_names().keys()#['CC2', 'CGleft', 'STFOright', 'TPREMleft']
    
    metrics = ['FA', 'IFW']

    ds = Dataset(restore='pouet')
    sub_info_df = pd.read_excel(sub_infos)

    all_results = []
    for bundle in bundle_names:
        bundle_paths = ds.get_global(pipeline=pipeline, bundle=bundle, suffix='mean', extension='csv')
        if not bundle_paths:
            continue
        for metric in metrics:
            rows = []
            for bp in bundle_paths:
                csv = pd.read_csv(bp.path)
                col = metric + '_mean'
                if col not in csv.columns:
                    break
                csv = csv[['point_id', col]].rename(columns={col: 'value', 'point_id': 'point'})
                csv['subject'] = bp.subject
                rows.append(csv)
            if not rows:
                continue
            long_df = pd.concat(rows, ignore_index=True)
            long_df['participant_id'] = 'sub-' + long_df['subject'].astype(str)
            long_df = long_df.merge(sub_info_df, on='participant_id', how='left')
            
            long_df = long_df[long_df['group']=='dep']

            res, info = fosr_group_test(long_df, classif_col='apathy', confounds=['age'],
                                         effect_type='group', n_perm=0, verbose=True)
            if res.empty:
                continue
            res['bundle'] = bundle
            res['metric'] = metric
            all_results.append(res)
            plot_fosr_results(res, metric, bundle, info['effect_name'],
                               os.path.join(OUTDIR, 'figures', f'{bundle}_{metric}_fosr.png'),
                               p_global=info['p_global'])
            print(f"{bundle}/{metric}: n={info['n_subjects_used']}, "
                  f"lambda={info['lam']:.3g}, r2={info['r2']:.3f}, p_global={info['p_global']:.3g}")

    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        all_df.to_csv(os.path.join(OUTDIR, 'results.csv'), index=False)
