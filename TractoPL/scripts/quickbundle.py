import os
from subprocess import run
import argparse
from glob import glob
from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric, mean_manhattan_distance
from TractoPL.data.loader import load_streamlines, save_streamlines


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Quickbundle all bundles in a tractogram")
    parser.add_argument("input_tractograms", nargs="+",
                        help="Paths to the input tractogram files"
                        )
    parser.add_argument("-o", "--out-dir",
                        help="Output folder for clustered tractogram")
    parser.add_argument(
        "--output-format", "-of",
        choices=["vtk", "trk", "tck"],
        help="Global output format (replaces the extension of each input file)."
    )
    parser.add_argument(
        "-r", "--reference", help="Reference file for the input tractogram (TRK/TCK only)")
    parser.add_argument("-n", "--nb-points", type=int, default=24,
                        help="Number of points to resample each streamline to (default: 24)")
    parser.add_argument("-t", "--threshold", type=float,
                        default=1.0, help="Clustering threshold (default: 1.0)")
    parser.add_argument("--space", choices=['LPS', 'RAS'], default='LPS',
                        help="Space of the input tractogram (TRK/TCK only, default: LPS)")
    parser.add_argument("--min-cluster-size", type=int,
                        default=1, help="Minimum cluster size (default: 1)")
    return parser.parse_args()


def quickbundle_streamlines(streamlines, cluster_thr, nb_points=24):
    print(f"Resampling streamlines to {nb_points} points each.")
    streamlines = set_number_of_points(streamlines, nb_points)
    print(f"Clustering streamlines with threshold {cluster_thr}.")
    metric = AveragePointwiseEuclideanMetric()
    qb_subject = QuickBundles(threshold=cluster_thr, metric=metric)
    clusters = qb_subject.cluster(streamlines)
    return clusters


def main():

    args = parse_arguments()
    input_tractograms = args.input_tractograms
    out_dir = args.out_dir
    out_format = args.output_format
    threshold = args.threshold
    min_cluster_size = args.min_cluster_size
    print(f"Input tractograms: {input_tractograms}")
    print(f"Output folder: {out_dir}")
    print(f"Clustering threshold: {threshold}")
    print(f"Minimum cluster size: {min_cluster_size}")
    print(f"Output format: {out_format}")
    print(f"Number of points per streamline: {args.nb_points}")
    print(f"Reference: {args.reference}")
    print(f"Space: {args.space}")

    os.makedirs(out_dir, exist_ok=True)
    # Load the input tractogram
    for input_tractogram in input_tractograms:
        tractogram = load_streamlines(
            input_tractogram, reference=args.reference, space=args.space)
        print(f"Loaded {len(tractogram)} streamlines from {os.path.basename(input_tractogram)}")
        # Perform quickbundling
        clusters = quickbundle_streamlines(
            tractogram, cluster_thr=threshold, nb_points=args.nb_points)
        print(f"Formed {len(clusters)} initial clusters.")
        # Filter clusters by minimum cluster size

        filtered_clusters = [centroid for i, centroid in enumerate(
            clusters.centroids) if len(clusters[i]) >= min_cluster_size]
        print(
            f"Filtered down to {len(filtered_clusters)} clusters with minimum size {min_cluster_size}.")

        # Save the clustered tractogram
        output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(input_tractogram))[0] + f".{out_format}")
        save_streamlines(filtered_clusters, output_path,
                        reference=args.reference, space=args.space)
        print(f"Saved clustered tractogram to {output_path}.")


if __name__ == "__main__":
    main()
