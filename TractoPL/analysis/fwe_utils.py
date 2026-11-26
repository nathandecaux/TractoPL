"""
Family-wise error (FWE) methods shared by population_analysis.py and the result
readers (population_analysis_report.py, report_analysis.py).

The per-point CSVs written by population_analysis.py contain everything needed to
apply any method afterwards:

- p_raw: pointwise p-value;
- alphaFWE: min p-value threshold;
- clusterFWE: minimum length of a significant run of points with p < alpha;
- p_tfce: TFCE FWE-corrected p-value (and tfce, tfceFWE);
- sig_<method>: significance of each point for every method; sig_afq is the
  method chosen in the configuration (FWE_METHOD).

Older CSVs without the sig_<method> columns are handled by recomputing the
significance from p_raw and the thresholds (TFCE needs p_tfce).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

FWE_METHODS = ("clusterFWE", "alphaFWE", "mixed", "TFCE")


def true_runs(mask, starts: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """(start, end) half-open index ranges of the runs of True values; a run is also
    split where `starts` is True (first point of a new bundle in stacked profiles)."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    first = mask & ~padded[:-2]
    last = mask & ~padded[2:]
    if starts is not None:
        starts = np.asarray(starts, dtype=bool)
        first |= mask & starts
        last |= mask & np.r_[starts[1:], True]
    return list(zip(np.flatnonzero(first), np.flatnonzero(last) + 1))


def significance_mask(p: np.ndarray, alpha_fwe: float, cluster_fwe: float,
                      method: str, alpha: float) -> np.ndarray:
    """
    Significant points from the pointwise p-values and the FWE thresholds:

    - alphaFWE: p < alphaFWE;
    - clusterFWE: runs of p < alpha at least clusterFWE points long;
    - mixed: such runs that also contain a point with p < alphaFWE.

    (TFCE uses the corrected p-values p_tfce, see point_significance.)
    """
    p = np.asarray(p, dtype=float)
    sig = np.zeros(len(p), dtype=bool)
    if method == "alphaFWE":
        return p < alpha_fwe if np.isfinite(alpha_fwe) else sig
    if not np.isfinite(cluster_fwe) or (method == "mixed" and not np.isfinite(alpha_fwe)):
        return sig
    min_size = max(1, int(cluster_fwe))
    for start, end in true_runs(p < alpha):
        if end - start < min_size:
            continue
        if method == "mixed" and not np.any(p[start:end] < alpha_fwe):
            continue
        sig[start:end] = True
    return sig


def _as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0")).to_numpy()


def _first(d: pd.DataFrame, col: str) -> float:
    if col not in d:
        return np.nan
    v = pd.to_numeric(d[col], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else np.nan


def point_significance(d: pd.DataFrame, method: str, alpha: float = 0.05
                       ) -> Optional[np.ndarray]:
    """
    Significant points of a per-point result table for an FWE method: the saved
    sig_<method> column when present, else recomputed from p_raw and the thresholds
    (or from p_tfce for TFCE). None when the table lacks what the method needs.
    """
    if method not in FWE_METHODS:
        raise ValueError(f"Unknown FWE method '{method}', expected one of {FWE_METHODS}")
    col = f"sig_{method}"
    if col in d:
        return _as_bool(d[col])
    if method == "TFCE":
        if "p_tfce" not in d:
            return None
        return (pd.to_numeric(d["p_tfce"], errors="coerce") <= alpha).to_numpy()
    if "p_raw" not in d:
        return None
    alpha_fwe, cluster_fwe = _first(d, "alphaFWE"), _first(d, "clusterFWE")
    if (method in ("alphaFWE", "mixed") and "alphaFWE" not in d) or \
            (method in ("clusterFWE", "mixed") and "clusterFWE" not in d):
        return None
    p = pd.to_numeric(d["p_raw"], errors="coerce").to_numpy(dtype=float)
    return significance_mask(p, alpha_fwe, cluster_fwe, method, alpha)


def n_significant(sig: np.ndarray, method: str) -> int:
    """n_sig of the summary: number of clusters for clusterFWE, else of points."""
    sig = np.asarray(sig, dtype=bool)
    return len(true_runs(sig)) if method == "clusterFWE" else int(sig.sum())
