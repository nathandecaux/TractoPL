#!/usr/bin/env python3
"""
Summary report (PDF and/or HTML) of the results of population_analysis.py.

Reads <report_dir>/summary_results.csv and the per-point CSVs of
<report_dir>/figures, then produces:

0. for every variable and metric, a heatmap of the statistic (r / t / F) of all the
   bundles (rows) x nodes (columns), squares on the FWE-significant points;
1. a table of the significant bundles (rows: bundle / metric) x variables (columns):
   each cell gives the peak statistic of the significant clusters (partial r for the
   correlations, t of the group difference for the groups) with stars for its p-value;
2. the bundle profile (statistic along the bundle, significant clusters, p-values) of
   every significant bundle x variable;
3. at the end, for every variable and metric, the profiles of all the bundles on a
   single A4 page.

The FWE method used for the significance (clusterFWE, alphaFWE, mixed, TFCE) is
chosen with --fwe (default: FWE_METHOD of the analysis config): it is read from the
per-point CSVs (sig_<method> columns), or recomputed from p_raw and the thresholds
for results of older versions.

Usage:
    python population_analysis_report.py /path/to/report_dir [--format pdf|html|both]
                                         [--fwe clusterFWE|alphaFWE|mixed|TFCE]
                                         [-o output_basename]
"""

import argparse
import base64
import io
import json
import math
import os
from dataclasses import dataclass
from os.path import join as opj
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

try:
    from TractoPL.analysis.fwe_utils import FWE_METHODS, n_significant, point_significance
except ImportError:  # run as a standalone script
    from fwe_utils import FWE_METHODS, n_significant, point_significance  # noqa: E402

A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)
STAR_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
SIG_COLOR = "#d62728"
LINE_COLOR = "#1f77b4"
GROUP_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


# ============================================================================
# DATA
# ============================================================================

def stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    for level, s in STAR_LEVELS:
        if p < level:
            return s
    return ""


def _true_runs(mask) -> List[Tuple[int, int]]:
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


def load_summary(report_dir: str) -> pd.DataFrame:
    path = opj(report_dir, "summary_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"summary_results.csv not found in {report_dir}")
    df = pd.read_csv(path)
    kind_var = df["type"].str.split("_", n=1, expand=True)
    df["kind"], df["var"] = kind_var[0], kind_var[1]
    df["n_sig"] = pd.to_numeric(df["n_sig"], errors="coerce").fillna(0).astype(int)
    df["file_unit"] = [unit_label(b, c) for b, c in zip(df["bundle"], df["centroid_id"])]
    n_cent = df.groupby("bundle")["centroid_id"].nunique(dropna=False)
    df["unit"] = [fu if n_cent[b] > 1 else b for b, fu in zip(df["bundle"], df["file_unit"])]
    return df


def unit_label(bundle: str, cid) -> str:
    if cid is None or (isinstance(cid, float) and not np.isfinite(cid)):
        return bundle
    return f"{bundle}_cent{int(cid)}"


def point_csv(report_dir: str, row) -> str:
    suffix = "group_corrected" if row["kind"] == "group" else "corr_partial"
    return opj(report_dir, "figures", f"{row['file_unit']}_{row['metric']}_{row['var']}_{suffix}.csv")


@dataclass
class Profile:
    """Per-point results of one bundle x metric x variable analysis."""
    kind: str
    points: np.ndarray
    stat: np.ndarray          # r (corr) or t of group 1 - group 0 (group)
    p: np.ndarray
    sig: np.ndarray           # significant points for the selected FWE method
    group_means: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], int]]
    alpha_fwe: float
    cluster_fwe: float
    p_tfce: Optional[np.ndarray] = None

    @property
    def stat_label(self) -> str:
        if self.kind == "corr":
            return "r"
        if len(self.group_means) == 2:
            g0, g1 = list(self.group_means)
            return f"t ({g1} - {g0})"
        return "F"

    def peak(self) -> Tuple[float, float, int]:
        """(stat, p, n_clusters) at the most significant point of the significant clusters."""
        if not self.sig.any():
            return np.nan, np.nan, 0
        idx = np.where(self.sig)[0]
        best = idx[np.nanargmin(self.p[idx])]
        return float(self.stat[best]), float(self.p[best]), len(_true_runs(self.sig))


def read_profile(path: str, kind: str, method: str, alpha: float) -> Optional[Profile]:
    """Per-point results of one analysis, significance according to `method` (None
    when the CSV lacks what the method needs, e.g. TFCE on old results)."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path).sort_values("point").reset_index(drop=True)
    p = pd.to_numeric(d["p_raw"], errors="coerce").to_numpy(dtype=float)
    sig = point_significance(d, method, alpha)
    if sig is None:
        return None
    group_means = {}
    if kind == "corr":
        stat = d["r"].to_numpy(dtype=float)
    else:
        levels = [c[len("mean_"):] for c in d.columns
                  if c.startswith("mean_") and c not in ("mean_g0", "mean_g1")
                  and not c.startswith("mean_g")]
        for lv in levels:
            sd = d[f"std_{lv}"].to_numpy(dtype=float) if f"std_{lv}" in d else None
            n = int(d[f"n_{lv}"].iloc[0]) if f"n_{lv}" in d else 0
            group_means[_level_name(lv)] = (d[f"mean_{lv}"].to_numpy(dtype=float), sd, n)
        stat = d["stat"].to_numpy(dtype=float)
        if len(levels) == 2:
            stat = -stat  # 'stat' follows ttest_ind(g0, g1): flip to get g1 - g0
    return Profile(kind=kind, points=d["point"].to_numpy(), stat=stat, p=p, sig=sig,
                   group_means=group_means,
                   alpha_fwe=float(d["alphaFWE"].iloc[0]) if "alphaFWE" in d else np.nan,
                   cluster_fwe=float(d["clusterFWE"].iloc[0]) if "clusterFWE" in d else np.nan,
                   p_tfce=(pd.to_numeric(d["p_tfce"], errors="coerce").to_numpy(dtype=float)
                           if "p_tfce" in d else None))


def _level_name(level: str) -> str:
    try:
        f = float(level)
        return str(int(f)) if f.is_integer() else level
    except ValueError:
        return level


def load_profiles(report_dir: str, summary: pd.DataFrame, method: str, alpha: float
                  ) -> Dict[Tuple, Profile]:
    """Per-point results of every analysis; updates summary["n_sig"] for `method`."""
    profiles, n_sig = {}, []
    for _, row in summary.iterrows():
        prof = read_profile(point_csv(report_dir, row), row["kind"], method, alpha)
        n_sig.append(0 if prof is None else n_significant(prof.sig, method))
        if prof is not None:
            profiles[(row["unit"], row["metric"], row["var"])] = prof
    summary["n_sig"] = n_sig
    return profiles


# ============================================================================
# SIGNIFICANCE TABLE
# ============================================================================

def build_table(summary: pd.DataFrame, profiles: Dict[Tuple, Profile]
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(text, value, kind) tables: rows 'unit / metric', columns = variables with at least
    one significant result."""
    sig = summary[summary["n_sig"] > 0]
    var_order = [v for v in dict.fromkeys(summary["var"]) if v in set(sig["var"])]
    rows = sorted(dict.fromkeys(zip(sig["unit"], sig["metric"])))
    index = [f"{u} / {m}" for u, m in rows]
    text = pd.DataFrame("", index=index, columns=var_order)
    value = pd.DataFrame(np.nan, index=index, columns=var_order)
    for _, r in sig.iterrows():
        prof = profiles.get((r["unit"], r["metric"], r["var"]))
        if prof is None:
            continue
        stat, p, ncl = prof.peak()
        if not np.isfinite(stat):
            continue
        key = f"{r['unit']} / {r['metric']}"
        prefix = "r=" if prof.kind == "corr" else ("t=" if len(prof.group_means) == 2 else "F=")
        text.loc[key, r["var"]] = (f"{prefix}{stat:.2f}{stars(p)}"
                                   + (f"\n({ncl}cl)" if ncl > 1 else ""))
        # Color scale: r directly, t mapped to a pseudo-r so both share the colormap
        value.loc[key, r["var"]] = stat if prof.kind == "corr" else np.tanh(stat / 4)
    kinds = summary.drop_duplicates("var").set_index("var")["kind"]
    return text, value, kinds


def _short(name: str, width: int = 18) -> str:
    name = name.replace("walking_", "w_")
    return name if len(name) <= width else name[:width - 1] + "…"


def _wrap(name: str, width: int = 7, max_lines: int = 6) -> str:
    """Wrap a variable name on '_' boundaries."""
    lines, cur = [], ""
    for part in name.replace("walking_", "w_").split("_"):
        cand = f"{cur}_{part}" if cur else part
        if cur and len(cand) > width:
            lines.append(cur + "_")
            cur = part
        else:
            cur = cand
    lines.append(cur)
    return "\n".join(lines[:max_lines])


def table_pages(text: pd.DataFrame, value: pd.DataFrame, kinds: pd.Series,
                header_lines: List[str], max_cols: int = 18, max_rows: int = 38
                ) -> List[plt.Figure]:
    """Significance table split on A4 landscape pages (columns then rows)."""
    if text.empty:
        fig = plt.figure(figsize=A4_LANDSCAPE)
        fig.text(0.5, 0.5, "No significant result.", ha="center", fontsize=14)
        return [fig]
    norm, cmap = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), plt.get_cmap("RdBu_r")
    figs, first = [], True
    col_chunks = [list(text.columns[i:i + max_cols]) for i in range(0, text.shape[1], max_cols)]
    for cols in col_chunks:
        sub = text[cols]
        sub = sub[(sub != "").any(axis=1)]
        for start in range(0, len(sub), max_rows):
            block = sub.iloc[start:start + max_rows]
            fig = plt.figure(figsize=A4_LANDSCAPE)
            top = 0.96
            if first:
                fig.text(0.03, 0.97, "Significant bundles x variables", fontsize=13,
                         weight="bold", va="top")
                for i, line in enumerate(header_lines):
                    fig.text(0.03, 0.94 - i * 0.017, line, fontsize=6.5, va="top", color="#333")
                top = 0.94 - len(header_lines) * 0.017 - 0.01
                first = False
            ax = fig.add_axes([0.03, 0.03, 0.94, top - 0.03])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            label_w = 0.13
            col_w = 0.045  # fixed narrow width: variable names wrap on several lines
            head_h = 0.11
            row_h = min(0.035, (1 - head_h) / max(len(block), 1))
            y = 1 - head_h
            for j, c in enumerate(cols):
                x = label_w + j * col_w
                ax.add_patch(plt.Rectangle((x, y), col_w, head_h, fc="#eeeeee", ec="#999", lw=0.3))
                ax.text(x + col_w / 2, y + head_h / 2,
                        f"{_wrap(c)}\n({kinds.get(c, '')})", ha="center", va="center",
                        fontsize=5, weight="bold", linespacing=1.05)
            for i, idx in enumerate(block.index):
                yi = y - (i + 1) * row_h
                ax.add_patch(plt.Rectangle((0, yi), label_w, row_h,
                                           fc="#f6f6f6" if i % 2 else "white", ec="#999", lw=0.3))
                ax.text(label_w - 0.005, yi + row_h / 2, idx, ha="right", va="center", fontsize=6.5)
                for j, c in enumerate(cols):
                    v = value.loc[idx, c]
                    fc = "white"
                    if np.isfinite(v):
                        rgb = cmap(norm(v))[:3]
                        fc = tuple(0.5 + 0.5 * x for x in rgb)
                    ax.add_patch(plt.Rectangle((label_w + j * col_w, yi), col_w, row_h,
                                               fc=fc, ec="#999", lw=0.3))
                    if block.loc[idx, c]:
                        ax.text(label_w + (j + 0.5) * col_w, yi + row_h / 2, block.loc[idx, c],
                                ha="center", va="center", fontsize=5, linespacing=1.0)
            figs.append(fig)
    return figs


# ============================================================================
# PROFILE PLOTS
# ============================================================================

def _shade(ax, prof: Profile):
    for s, e in _true_runs(prof.sig):
        ax.axvspan(prof.points[s] - 0.5, prof.points[e] + 0.5, color=SIG_COLOR,
                   alpha=0.15, linewidth=0)


def plot_profile(ax, prof: Profile, title: str, alpha: float = 0.05, detailed: bool = True,
                 method: str = "clusterFWE"):
    """Statistic (r) or group means along the bundle + significant clusters; in
    detailed mode, -log10(p) bars on a secondary axis."""
    _shade(ax, prof)
    if prof.kind == "corr" or not prof.group_means:
        ax.plot(prof.points, prof.stat, color=LINE_COLOR, lw=1 if detailed else 0.7)
        ax.axhline(0, color="black", lw=0.4)
        if prof.sig.any():
            ax.scatter(prof.points[prof.sig], prof.stat[prof.sig], color=SIG_COLOR,
                       s=4 if detailed else 1.5, zorder=3, linewidths=0)
        if detailed:
            ax.set_ylabel("partial r" if prof.kind == "corr" else "stat")
    else:
        for i, (lv, (m, sd, n)) in enumerate(prof.group_means.items()):
            col = GROUP_COLORS[i % len(GROUP_COLORS)]
            ax.plot(prof.points, m, color=col, lw=1 if detailed else 0.6,
                    label=f"{lv} (n={n})")
            if sd is not None and n > 1:
                ci = 1.96 * sd / np.sqrt(n)
                ax.fill_between(prof.points, m - ci, m + ci, color=col, alpha=0.15, lw=0)
        if detailed:
            ax.set_ylabel("adjusted mean ±95% CI")
            ax.legend(loc="lower center", frameon=False, ncol=len(prof.group_means),
                      handlelength=1)
    ax.set_title(title, pad=2)
    ax.set_xlim(prof.points.min() - 0.5, prof.points.max() + 0.5)
    if detailed:
        ax2 = ax.twinx()
        mlogp = -np.log10(np.clip(prof.p, 1e-300, 1))
        ax2.bar(prof.points, mlogp, width=1, color="gray", alpha=0.25, lw=0)
        ax2.axhline(-np.log10(alpha), color="gray", ls=":", lw=0.7)
        ymax = np.nanmax(mlogp)
        if method in ("alphaFWE", "mixed") and np.isfinite(prof.alpha_fwe) and prof.alpha_fwe > 0:
            ax2.axhline(-np.log10(prof.alpha_fwe), color="green", ls=":", lw=0.7)
        if method == "TFCE" and prof.p_tfce is not None:
            mlogp_tfce = -np.log10(np.clip(prof.p_tfce, 1e-300, 1))
            ax2.plot(prof.points, mlogp_tfce, color="purple", lw=0.8, label="p TFCE")
            ymax = np.nanmax([ymax, np.nanmax(mlogp_tfce)])
        ax2.set_ylim(0, max(3 * -np.log10(alpha), ymax * 1.1))
        ax2.set_ylabel("-log10(p)", color="gray")
        ax2.tick_params(axis="y", colors="gray")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xlabel("point")
        stat, p, ncl = prof.peak()
        if np.isfinite(stat):
            ax.text(0.99, 0.97, f"peak {prof.stat_label}={stat:.2f}, p={p:.1e}{stars(p)}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    else:
        ax.tick_params(length=1.5, pad=1, labelsize=4)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_color(SIG_COLOR if prof.sig.any() else "#999999")
            spine.set_linewidth(1.0 if prof.sig.any() else 0.4)


def detail_pages(sig_rows: pd.DataFrame, profiles: Dict[Tuple, Profile], alpha: float,
                 method: str, ncols: int = 2, nrows: int = 4) -> List[plt.Figure]:
    """Profiles of the significant analyses, ncols x nrows per A4 portrait page."""
    figs = []
    items = [(r, profiles.get((r["unit"], r["metric"], r["var"]))) for _, r in sig_rows.iterrows()]
    items = [(r, p) for r, p in items if p is not None]
    per_page = ncols * nrows
    for start in range(0, len(items), per_page):
        fig, axes = plt.subplots(nrows, ncols, figsize=A4_PORTRAIT, squeeze=False)
        fig.suptitle(f"Profiles of the significant bundles ({start + 1}-"
                     f"{min(start + per_page, len(items))} / {len(items)})",
                     fontsize=10, weight="bold", y=0.985)
        for ax, (r, prof) in zip(axes.flat, items[start:start + per_page]):
            plot_profile(ax, prof, f"{r['unit']} · {r['metric']} · {r['var']}", alpha,
                         method=method)
        for ax in axes.flat[len(items[start:start + per_page]):]:
            ax.axis("off")
        fig.tight_layout(rect=(0.02, 0.01, 0.98, 0.97), h_pad=1.4, w_pad=2.5)
        figs.append(fig)
    return figs


def overview_page(var: str, metric: str, kind: str, units: List[str],
                  profiles: Dict[Tuple, Profile], summary_var: pd.DataFrame) -> plt.Figure:
    """All the bundles of one variable x metric on a single A4 portrait page."""
    n = len(units)
    ncols = max(1, math.ceil(math.sqrt(n * A4_PORTRAIT[0] / A4_PORTRAIT[1] * 1.3)))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=A4_PORTRAIT, squeeze=False)
    n_sig = int((summary_var["n_sig"] > 0).sum())
    what = "partial r" if kind == "corr" else "adjusted group means"
    fig.suptitle(f"{var} — {metric}  ({what}; {n_sig} significant bundle(s), red frame)",
                 fontsize=9, weight="bold", y=0.993)
    # Shared y-range for the correlations so the bundles are comparable
    ylim = None
    if kind == "corr":
        vals = np.concatenate([profiles[(u, metric, var)].stat for u in units
                               if (u, metric, var) in profiles] or [np.zeros(1)])
        lim = max(0.2, float(np.nanmax(np.abs(vals))) * 1.05) if np.isfinite(vals).any() else 1
        ylim = (-lim, lim)
    for ax, unit in zip(axes.flat, units):
        prof = profiles.get((unit, metric, var))
        if prof is None:
            ax.set_title(unit, pad=1, fontsize=5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes,
                    fontsize=5, color="gray")
            continue
        stat, p, _ = prof.peak()
        title = unit + (f"  {stat:.2f}{stars(p)}" if np.isfinite(stat) else "")
        plot_profile(ax, prof, title, detailed=False)
        ax.title.set_fontsize(5)
        if prof.sig.any():
            ax.title.set_color(SIG_COLOR)
        if ylim is not None:
            ax.set_ylim(*ylim)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.01, top=0.965, wspace=0.35, hspace=0.38)
    return fig


def heatmap_page(var: str, metric: str, units: List[str], profiles: Dict[Tuple, Profile],
                 method: str) -> Optional[plt.Figure]:
    """Statistic (r / t / F) of every bundle (rows) x node (columns) for one variable x
    metric; squares mark the FWE-significant points."""
    profs = [(u, profiles.get((u, metric, var))) for u in units]
    profs = [(u, p) for u, p in profs if p is not None and len(p.points)]
    if not profs:
        return None
    pmin = int(min(p.points.min() for _, p in profs))
    pmax = int(max(p.points.max() for _, p in profs))
    mat = np.full((len(profs), pmax - pmin + 1), np.nan)
    sig_y, sig_x = [], []
    for i, (_, p) in enumerate(profs):
        cols = p.points.astype(int) - pmin
        mat[i, cols] = p.stat
        sig_y += [i] * int(p.sig.sum())
        sig_x += list(cols[p.sig])

    ref = profs[0][1]
    label = ref.stat_label
    vals = mat[np.isfinite(mat)]
    if label == "F":
        cmap = plt.get_cmap("magma_r").copy()
        vmax = float(np.nanpercentile(vals, 99)) if vals.size else 1
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max(vmax, 1e-6))
    else:
        cmap = plt.get_cmap("RdBu_r").copy()
        lim = float(np.nanpercentile(np.abs(vals), 99)) if vals.size else 1
        lim = min(lim, 1.0) if ref.kind == "corr" else lim
        norm = TwoSlopeNorm(vmin=-max(lim, 1e-6), vcenter=0, vmax=max(lim, 1e-6))
    cmap.set_bad("white")

    fig = plt.figure(figsize=A4_PORTRAIT)
    ax = fig.add_axes((0.15, 0.05, 0.68, 0.91))
    cax = fig.add_axes((0.86, 0.35, 0.02, 0.3))
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", extent=(pmin - 0.5, pmax + 0.5, len(profs) - 0.5, -0.5))
    if sig_x:
        ax.scatter(np.asarray(sig_x) + pmin, sig_y, marker="s",
                   s=max(1.5, min(8.0, 400 / len(profs))), facecolor="white",
                   edgecolor="black", linewidths=0.4, zorder=3)
    ax.set_yticks(range(len(profs)))
    ax.set_yticklabels([u for u, _ in profs], fontsize=max(3.5, min(6.5, 450 / len(profs))))
    for tick, (_, p) in zip(ax.get_yticklabels(), profs):
        if p.sig.any():
            tick.set_color(SIG_COLOR)
            tick.set_fontweight("bold")
    ax.set_xlabel("node")
    ax.tick_params(length=1.5, pad=1)
    what = {"r": "partial r"}.get(label, label)
    ax.set_title(f"{var} — {metric}: {what} per node  (□ = significant point, {method}; "
                 f"red label = significant bundle)", fontsize=8, pad=4)
    extend = "neither" if label == "r" else ("max" if label == "F" else "both")
    cb = fig.colorbar(im, cax=cax, extend=extend)
    cb.set_label(what, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return fig


def heatmap_pages(summary: pd.DataFrame, profiles: Dict[Tuple, Profile], method: str):
    units = list(dict.fromkeys(summary["unit"]))
    for var in dict.fromkeys(summary["var"]):
        sv = summary[summary["var"] == var]
        for metric in dict.fromkeys(sv["metric"]):
            fig = heatmap_page(var, metric, units, profiles, method)
            if fig is not None:
                yield (var, metric), fig


def overview_pages(summary: pd.DataFrame, profiles: Dict[Tuple, Profile]):
    units = list(dict.fromkeys(summary["unit"]))
    for var in dict.fromkeys(summary["var"]):
        sv = summary[summary["var"] == var]
        kind = sv["kind"].iloc[0]
        for metric in dict.fromkeys(sv["metric"]):
            yield (var, metric), overview_page(var, metric, kind, units, profiles,
                                               sv[sv["metric"] == metric])


# ============================================================================
# REPORT
# ============================================================================

def header_lines(report_dir: str, summary: pd.DataFrame, method: str) -> List[str]:
    lines = [f"Report directory: {report_dir}  |  FWE method of this report: {method}"]
    cfg_path = opj(report_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        lines.append(
            f"Model: {cfg.get('STAT_MODEL')}  |  FWE (analysis): {cfg.get('FWE_METHOD')} "
            f"(alpha={cfg.get('AFQ_ALPHA')}, {cfg.get('AFQ_NPERM')} permutations)  |  "
            f"multi-tract: {cfg.get('CORRECT_MULTI_TRACT')}  |  "
            f"pipeline: {cfg.get('hcp_asso_pipeline')}")
    conf = ", ".join(sorted(set(summary["confounds_used"].dropna().astype(str))))
    n_used = summary["subjects_used"]
    lines.append(f"Confounds: {conf}  |  subjects used: {int(n_used.min())}-{int(n_used.max())}  |  "
                 f"{summary['unit'].nunique()} bundles, {summary['metric'].nunique()} metrics, "
                 f"{summary['var'].nunique()} variables  |  "
                 f"{int((summary['n_sig'] > 0).sum())} significant analyses / {len(summary)}")
    lines.append("Cell: statistic at the most significant point of the FWE-significant clusters "
                 "(r = partial correlation; t = group difference, last level - first level); "
                 "stars = pointwise p at that point: * <0.05, ** <0.01, *** <0.001; "
                 "(kcl) = k significant clusters.")
    return lines


def _fig_to_png64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_html(path: str, title: str, header: List[str], text: pd.DataFrame,
               value: pd.DataFrame, kinds: pd.Series, detail_pngs, overview_pngs,
               heatmap_pngs=()):
    cmap, norm = plt.get_cmap("RdBu_r"), TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    def bg(v):
        if not np.isfinite(v):
            return ""
        c = cmap(norm(v))
        rgb = [int(255 * (0.55 + 0.45 * x)) for x in c[:3]]
        return f' style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})"'

    head = "".join(f"<th>{c}<br><small>{kinds.get(c, '')}</small></th>" for c in text.columns)
    body = "".join(
        f"<tr><th>{idx}</th>" + "".join(
            f"<td{bg(value.loc[idx, c])}>{text.loc[idx, c]}</td>" for c in text.columns)
        + "</tr>" for idx in text.index)
    details = "".join(f'<img src="data:image/png;base64,{png}">' for png in detail_pngs)
    heatmaps = "".join(f'<h3>{v} — {m}</h3><img src="data:image/png;base64,{png}">'
                       for (v, m), png in heatmap_pngs)
    overviews = "".join(f'<h3>{v} — {m}</h3><img loading="lazy" src="data:image/png;base64,{png}">'
                        for (v, m), png in overview_pngs)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:sans-serif;margin:16px;background:#fff;color:#222}}
table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #ccc;padding:3px 6px;text-align:center;white-space:nowrap}}
tbody th{{text-align:right;background:#f6f6f6}} thead th{{background:#eee;position:sticky;top:0}}
.wrap{{overflow-x:auto}} img{{max-width:100%;display:block;margin:8px auto;border:1px solid #eee}}
p.h{{margin:2px 0;font-size:12px;color:#444}}
</style></head><body><h1>{title}</h1>
{''.join(f'<p class="h">{line}</p>' for line in header)}
{f'<h2>Statistic per bundle x node</h2>{heatmaps}' if heatmaps else ''}
<h2>Significant bundles x variables</h2><div class="wrap"><table>
<thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>
<h2>Profiles of the significant bundles</h2>{details}
<h2>All bundles, per variable and metric</h2>{overviews}
</body></html>"""
    with open(path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report_dir", help="Output directory of population_analysis.py")
    parser.add_argument("--format", choices=("pdf", "html", "both"), default="pdf")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path without extension (default: <report_dir>/summary_report, "
                             "summary_report_<method> with --fwe)")
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip the per-variable pages with all the bundles")
    parser.add_argument("--variables", nargs="+", default=None,
                        help="Restrict the report to these variables")
    parser.add_argument("--fwe", choices=FWE_METHODS, default=None,
                        help="FWE method of the significance (default: FWE_METHOD of the config)")
    args = parser.parse_args()

    report_dir = os.path.abspath(args.report_dir)
    if report_dir.endswith(".csv"):
        report_dir = os.path.dirname(report_dir)
    summary = load_summary(report_dir)
    if args.variables:
        summary = summary[summary["var"].isin(args.variables)]
    cfg = {}
    cfg_path = opj(report_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    alpha = float(cfg.get("AFQ_ALPHA") or 0.05)
    method = args.fwe or cfg.get("FWE_METHOD") or "clusterFWE"
    out = args.output or opj(report_dir, "summary_report" + (f"_{args.fwe}" if args.fwe else ""))

    print(f"Loading {len(summary)} analyses from {report_dir}/figures (FWE: {method}) ...")
    profiles = load_profiles(report_dir, summary, method, alpha)
    missing = len(summary) - len(profiles)
    if missing:
        print(f"  {missing} per-point CSV(s) not found or without the data needed by "
              f"{method} (skipped)")

    text, value, kinds = build_table(summary, profiles)
    header = header_lines(report_dir, summary, method)
    sig_rows = summary[summary["n_sig"] > 0].sort_values(["unit", "metric", "var"])
    title = f"Population analysis — {os.path.basename(report_dir)} ({method})"

    want_pdf = args.format in ("pdf", "both")
    want_html = args.format in ("html", "both")
    pdf = PdfPages(out + ".pdf") if want_pdf else None
    html_details, html_overviews, html_heatmaps = [], [], []

    def emit(fig, html_list=None, key=None):
        if pdf is not None:
            pdf.savefig(fig)
        if html_list is not None and want_html:
            png = _fig_to_png64(fig, dpi=110 if key is None else 90)
            html_list.append(png if key is None else (key, png))
        plt.close(fig)

    for key, fig in heatmap_pages(summary, profiles, method):
        emit(fig, html_heatmaps, key)
    for fig in table_pages(text, value, kinds, [title] + header):
        emit(fig)
    for fig in detail_pages(sig_rows, profiles, alpha, method):
        emit(fig, html_details)
    if not args.no_overview:
        for key, fig in overview_pages(summary, profiles):
            emit(fig, html_overviews, key)
    if pdf is not None:
        pdf.infodict()["Title"] = title
        pdf.close()
        print(f"PDF: {out}.pdf")
    if want_html:
        write_html(out + ".html", title, header, text, value, kinds, html_details, html_overviews,
                   html_heatmaps)
        print(f"HTML: {out}.html")


if __name__ == "__main__":
    main()
