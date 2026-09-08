import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import nibabel as nib
from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from shapely.geometry import LineString
import shapely

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from TractoPL.data.vtk_loader import load_vtk, save_vtk

# ====================== I/O ======================

def load_vtk_streamlines(vtk_file_path):
    streamlines, _ = load_vtk(vtk_file_path)
    return streamlines


def load_trk_streamlines(trk_file_path):
    """
    nibabel loads .trk streamlines in RAS+ world coordinates (mm), while the
    .vtk bundles in this pipeline are in LPS (ITK/Slicer convention). Flip
    x and y to bring .trk streamlines into the same frame as .vtk ones.
    """
    tfile = nib.streamlines.load(trk_file_path)
    ras_to_lps = np.array([-1.0, -1.0, 1.0])
    return [np.asarray(s) * ras_to_lps for s in tfile.streamlines]


def load_streamlines(bundle_path):
    ext = os.path.splitext(bundle_path)[1].lower()
    if ext == '.vtk':
        return load_vtk_streamlines(bundle_path)
    elif ext == '.trk':
        return load_trk_streamlines(bundle_path)
    else:
        raise ValueError(f"Unsupported bundle format: {ext} (expected: .vtk or .trk)")


# ====================== Orientation ======================

def _make_qb(threshold, nb_points):
    """QuickBundles configured to accept streamlines with nb_points (otherwise QB forces 12)."""
    feature = ResampleFeature(nb_points=nb_points)
    metric = AveragePointwiseEuclideanMetric(feature=feature)
    return QuickBundles(threshold=threshold, metric=metric)

def compute_mean_streamline_quick(sl, nb_points):
    qb = _make_qb(threshold=1000.0, nb_points=nb_points)  # very large: just to extract a reference "mean"
    clusters = qb.cluster(sl)
    return clusters.centroids[0]

def longest_fraction_centroid(streamlines, frac, nb_points):
    """QuickBundles centroid (threshold=1000mm, single cluster) of the `frac` longest fibers."""
    lengths = [LineString(np.asarray(s)).length for s in streamlines]
    order = np.argsort(lengths)[::-1]
    n_top = max(1, int(np.ceil(len(streamlines) * frac)))
    top = [streamlines[i] for i in order[:n_top]]
    top_rs = set_number_of_points(Streamlines(top), nb_points)
    return compute_mean_streamline_quick(top_rs, nb_points)

def reorient_to_reference_manhattan(streamlines, reference):
    if reference is None:
        return Streamlines(streamlines), [False]*len(streamlines)
    oriented, flips = [], []
    ref = np.asarray(reference)
    for s in streamlines:
        s = np.asarray(s)
        d_orig = np.sum(np.abs(s - ref))
        s_rev = s[::-1]
        d_flip = np.sum(np.abs(s_rev - ref))
        flip = d_flip < d_orig
        oriented.append(s_rev if flip else s)
        flips.append(flip)
    return Streamlines(oriented), flips

def reorient_from_indices(streamlines, flips):
    out = []
    for s, f in zip(streamlines, flips):
        out.append(s[::-1] if f else s)
    return Streamlines(out)


# ====================== Distances / Normalization ======================

def frechet_distance(s1, s2):
    return shapely.frechet_distance(LineString(np.asarray(s1)), LineString(np.asarray(s2)))

def compute_distance_matrix(streamlines):
    """NxN matrix of Fréchet distances (symmetric, diag=0) with tqdm."""
    n = len(streamlines)
    D = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n), desc="Frechet: lignes"):
        si = streamlines[i]
        for j in range(i+1, n):
            d = frechet_distance(si, streamlines[j])
            D[i, j] = D[j, i] = d
    return D

def normalize_distances(D, method="p95"):
    if method == "max":
        s = np.max(D)
    elif method == "p95":
        s = np.percentile(D[np.triu_indices_from(D, 1)], 95)
    else:
        s = 1.0
    s = max(s, 1e-12)
    return D / s


# ====================== Hierarchical (≤3) ======================

def best_hierarchical_clustering(D_norm, max_k=3, linkage_method="complete",sil_threshold=None, return_all=False):
    """
    Choose k in {1..max_k} maximizing the silhouette (metric='precomputed').
    Returns labels (0..k-1), chosen k, silhouette.
    If return_all=True, also returns a dict k -> {labels, sil, sil_samples}.
    """
    n = D_norm.shape[0]
    if n <= 1:
        empty = {} if return_all else None
        if return_all:
            return np.zeros(n, dtype=int), min(n, 1), np.nan, {}, empty
        return np.zeros(n, dtype=int), min(n, 1), np.nan

    Z = linkage(squareform(D_norm, checks=False), method=linkage_method)

    best_k, best_sil, best_labels = 1, -np.inf, np.zeros(n, dtype=int)
    sil_scores = {}  # k -> score
    all_candidates = {}  # k -> {labels, sil, sil_samples}
    for k in tqdm(range(1, max_k+1), desc="Trying k (silhouette)"):
        labels = fcluster(Z, t=k, criterion='maxclust') - 1  # 0..k-1
        sil_samples = None
        if k == 1 or len(set(labels)) == 1 or min(np.bincount(labels)) == 1:
            sil = -np.inf
        else:
            try:
                sil = silhouette_score(D_norm, labels, metric='precomputed')
                sil_samples = silhouette_samples(D_norm, labels, metric='precomputed')
            except Exception:
                sil = -np.inf
        sil_scores[k] = float(sil) if np.isfinite(sil) else None
        if return_all:
            all_candidates[k] = {
                'labels': labels.copy(),
                'sil': float(sil) if np.isfinite(sil) else None,
                'sil_samples': sil_samples,
            }
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels
    if sil_threshold is not None and best_sil < sil_threshold:
        if return_all:
            return np.zeros(n, dtype=int), 1, best_sil, sil_scores, all_candidates
        return np.zeros(n, dtype=int), 1, best_sil, sil_scores
    if return_all:
        return best_labels, best_k, best_sil, sil_scores, all_candidates
    return best_labels, best_k, best_sil, sil_scores


# ====================== VTK writers ======================

def direction_rgb(points):
    """
    Per-point RGB (uint8) encoding the local tangent direction (|dx|,|dy|,|dz|).
    Endpoints reuse the neighbor's tangent. Returns array of shape (N,3) uint8.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return np.zeros((n, 3), dtype=np.uint8)
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0]    = pts[1] - pts[0]
    tangents[-1]   = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    rgb = np.abs(tangents) / norms
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

def write_centroids_vtk(path, centroids_streamlines, point_arrays=None, add_direction_rgb=False):
    """
    point_arrays: optional dict name -> array of length len(centroids_streamlines),
    one scalar per centroid replicated on each of its points.
    """
    scalar_dict = {
        'centroid_index': np.concatenate([
            np.full(len(centroid), centroid_id, dtype=int)
            for centroid_id, centroid in enumerate(centroids_streamlines)
        ]) if centroids_streamlines else np.array([], dtype=int),
    }
    for cid, c in enumerate(centroids_streamlines):
        for name, values in (point_arrays or {}).items():
            scalar_dict.setdefault(name, []).append(
                np.full(len(c), values[cid], dtype=float)
            )
    if add_direction_rgb:
        scalar_dict['direction_rgb'] = [direction_rgb(centroid) for centroid in centroids_streamlines]
    save_vtk(centroids_streamlines, path, scalar_dict=scalar_dict)

def write_model_with_labels_vtk(path, streamlines, labels, per_streamline_arrays=None, add_direction_rgb=False):
    """
    per_streamline_arrays: optional dict name -> array of length len(streamlines),
    one scalar per streamline replicated on each of its points.
    """
    scalar_dict = {
        'centroid_index': [np.full(len(streamline), labels[index], dtype=int)
                           for index, streamline in enumerate(streamlines)],
        'point_index': [np.arange(len(streamline), dtype=int) for streamline in streamlines],
    }
    for sidx, s in enumerate(streamlines):
        for name, values in (per_streamline_arrays or {}).items():
            scalar_dict.setdefault(name, []).append(
                np.full(len(s), values[sidx], dtype=float)
            )
    if add_direction_rgb:
        scalar_dict['direction_rgb'] = [direction_rgb(streamline) for streamline in streamlines]
    save_vtk(streamlines, path, scalar_dict=scalar_dict)


# ====================== Bundle pipeline (2 steps) ======================

def process_single_bundle(bundle_path, output_dir, resample_points=24, qb_threshold_mm=10.0,
                           max_k_final=3, normalization=None, debug=False, longest_frac=0.15):
    bundle_name = os.path.splitext(os.path.basename(bundle_path))[0]
    print(f"\n==================== {bundle_name} ====================")

    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(output_dir, exist_ok=True)
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    # 0) Load & orient
    sl_src = load_streamlines(bundle_path)
    sl_rs  = set_number_of_points(Streamlines(sl_src), resample_points)
    ref    = compute_mean_streamline_quick(sl_rs, nb_points=resample_points)
    sl_rs_oriented, flips = reorient_to_reference_manhattan(sl_rs, ref)
    sl_src_oriented = reorient_from_indices(Streamlines(sl_src), flips)
    print(f"-> {len(sl_rs_oriented)} streamlines (resample={resample_points})")

    # 1) QuickBundles (fine pre-clustering)
    print(f"Step 1: QuickBundles threshold={qb_threshold_mm} mm (resample={resample_points} pts)")
    qb = _make_qb(threshold=qb_threshold_mm, nb_points=resample_points)
    qb_clusters = qb.cluster(sl_rs_oriented)
    C = len(qb_clusters)
    print(f"-> {C} QB clusters (fine) | total fibers={sum(len(c.indices) for c in qb_clusters)}")

    # QB centroids (representatives for step 2)
    qb_centroids = list(qb_clusters.centroids)  # list of polylines (Nx3)

    if len(qb_centroids) == 0:
        # nothing to cluster
        write_model_with_labels_vtk(
            os.path.join(output_dir, f"{bundle_name}_model_with_centroid_index.vtk"),
            sl_src_oriented,
            np.zeros(len(sl_src_oriented), dtype=int)
        )
        return bundle_name, {'n_clusters': 1, 'silhouette_scores': {}, 'qb_stats': {}}

    # Resample the centroid based on their arc-length to have a more accurate Frechet distance
    resampled_qb_centroids = []
    lengths= []
    for c in qb_centroids:
        # Use shapely to compute the arc-length and resample the centroid
        line = LineString(c)
        length = line.length
        lengths.append(length)

    # Resample each centroid to ~1 point per mm (at least 2 points)
    for c, l in zip(qb_centroids, lengths):
        n_points = max(2, int(round(l)) + 1)
        resampled = set_number_of_points(Streamlines([c]), n_points)[0]
        resampled_qb_centroids.append(resampled)

    qb_centroids = resampled_qb_centroids

    # QB stats
    centroid_lengths = [LineString(c).length for c in qb_centroids]
    qb_stats = {
        'n_qb_clusters': C,
        'centroid_length_min': float(np.min(centroid_lengths)),
        'centroid_length_mean': float(np.mean(centroid_lengths)),
        'centroid_length_max': float(np.max(centroid_lengths)),
    }

    # 2) Hierarchical clustering on QB centroids using Fréchet distance
    print("Step 2: Hierarchical (Fréchet) on QB centroids")
    D = compute_distance_matrix(qb_centroids)                 # Fréchet distances between QB centroids
    Dn = normalize_distances(D, method=normalization)         # robust normalization
    if debug:
        labels_reps, k_chosen, sil, sil_scores, all_candidates = best_hierarchical_clustering(
            Dn, max_k=max_k_final, linkage_method="complete", return_all=True)
    else:
        labels_reps, k_chosen, sil, sil_scores = best_hierarchical_clustering(
            Dn, max_k=max_k_final, linkage_method="complete")
        all_candidates = None
    print(f"-> chosen k = {k_chosen} (silhouette={sil if np.isfinite(sil) else 'n/a'}) | scores/k: {sil_scores}")

    # DEBUG: QB centroids alone + all candidate clusterings (per K)
    if debug:
        # QB centroids alone (before hierarchical clustering)
        write_centroids_vtk(
            os.path.join(debug_dir, f"{bundle_name}_qb_centroids.vtk"),
            qb_centroids,
            point_arrays={'qb_centroid_id': np.arange(C, dtype=float),
                          'qb_cluster_size': np.array([len(cl.indices) for cl in qb_clusters], dtype=float)},
            add_direction_rgb=True,
        )

        # For each candidate k: write QB centroids colored by label + silhouette per centroid,
        # and the full propagated model.
        for k, info in (all_candidates or {}).items():
            labels_k = info['labels']
            sil_samples = info['sil_samples']
            if sil_samples is None:
                sil_samples = np.full(C, np.nan, dtype=float)

            extras = {
                'cluster_label': labels_k.astype(float),
                'silhouette': np.asarray(sil_samples, dtype=float),
            }
            write_centroids_vtk(
                os.path.join(debug_dir, f"{bundle_name}_qb_centroids_k{k}.vtk"),
                qb_centroids,
                point_arrays=extras,
                add_direction_rgb=True,
            )

            # Propagate to all fibers for this k
            labels_all_k = np.full(len(sl_rs_oriented), -1, dtype=int)
            sil_per_streamline = np.full(len(sl_rs_oriented), np.nan, dtype=float)
            for cid, cl in enumerate(qb_clusters):
                for sidx in cl.indices:
                    labels_all_k[sidx] = labels_k[cid]
                    sil_per_streamline[sidx] = sil_samples[cid]
            write_model_with_labels_vtk(
                os.path.join(debug_dir, f"{bundle_name}_model_k{k}.vtk"),
                sl_src_oriented,
                labels_all_k,
                per_streamline_arrays={'silhouette': sil_per_streamline},
                add_direction_rgb=True,
            )

    # 3) Label propagation: each fiber inherits the label of its QB cluster
    #    (mapping QB_cluster_id -> hierarchical label)
    rep_label = {cid: labels_reps[cid] for cid in range(C)}
    labels_all = np.full(len(sl_rs_oriented), -1, dtype=int)
    for cid, cl in enumerate(qb_clusters):
        lab = rep_label[cid]
        for sidx in cl.indices:
            labels_all[sidx] = lab
    assert np.all(labels_all >= 0), "Missing label on some streamlines."

    # 4) Final centroids = medoids among QB centroids for each hierarchical cluster
    centroids_final = []
    for lab in sorted(set(labels_reps)):
        idx = np.where(labels_reps == lab)[0]
        if len(idx) == 1:
            centroids_final.append(qb_centroids[idx[0]])
        else:
            sub = D[np.ix_(idx, idx)]            # Fréchet distances between reps of the same cluster
            sums = sub.sum(axis=1)
            medoid_idx = idx[np.argmin(sums)]    # global index in qb_centroids
            centroids_final.append(qb_centroids[medoid_idx])

    # 4b) Alternative centroids = QuickBundles (threshold=1000mm) of the `longest_frac`
    #     longest fibers of each final cluster (on original, non-resampled fibers)
    centroids_longest = []
    for lab in sorted(set(labels_all)):
        idx = np.where(labels_all == lab)[0]
        cluster_streamlines = [sl_src_oriented[i] for i in idx]
        centroids_longest.append(longest_fraction_centroid(cluster_streamlines, longest_frac, resample_points))

    # 5) VTK outputs
    write_centroids_vtk(
        os.path.join(output_dir, f"{bundle_name}_centroids_qb_then_hier_medoid.vtk"),
        centroids_final
    )
    write_centroids_vtk(
        os.path.join(output_dir, f"{bundle_name}_centroids.vtk"),
        centroids_longest
    )
    write_model_with_labels_vtk(
        os.path.join(output_dir, f"{bundle_name}_model_with_centroid_index.vtk"),
        sl_src_oriented,
        labels_all
    )
    n_clusters = int(len(set(labels_all)))
    print(f"VTK export — final clusters: {n_clusters}")

    info = {
        'n_clusters': n_clusters,
        'silhouette_scores': sil_scores,
        'qb_stats': qb_stats,
    }
    return bundle_name, info


# ====================== CLI ======================

def parse_args():
    p = argparse.ArgumentParser(
        description="Hierarchical clustering (Fréchet distance) of a single streamline bundle.")
    p.add_argument("bundle", help="Path to the bundle file to process (.vtk or .trk)")
    p.add_argument("output_dir", help="Output directory")
    p.add_argument("--resample-points", type=int, default=24,
                    help="Number of resampling points per streamline (default: 24)")
    p.add_argument("--qb-threshold", type=float, default=10.0,
                    help="QuickBundles threshold in mm (default: 10.0)")
    p.add_argument("--max-k", type=int, default=3,
                    help="Max number of hierarchical clusters tested (default: 3)")
    p.add_argument("--normalization", choices=["max", "p95", "none"], default="none",
                    help="Fréchet distance normalization method (default: none)")
    p.add_argument("--debug", action="store_true",
                    help="Save candidate clusterings, silhouettes, and QB centroids into a debug/ subfolder")
    p.add_argument("--longest-frac", type=float, default=0.15,
                    help="Fraction of the longest fibers (per final cluster) used for the alternative centroid (default: 0.15)")
    return p.parse_args()

def main():
    args = parse_args()
    normalization = None if args.normalization == "none" else args.normalization

    bundle_name, info = process_single_bundle(
        args.bundle,
        args.output_dir,
        resample_points=args.resample_points,
        qb_threshold_mm=args.qb_threshold,
        max_k_final=args.max_k,
        normalization=normalization,
        debug=args.debug,
        longest_frac=args.longest_frac,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{bundle_name}_info.json")
    with open(out_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\nInfo saved: {out_path}")

if __name__ == "__main__":
    main()
