"""Generate a QuickBundles centroid from the longest fibers of a tractogram."""

import argparse
from pathlib import Path

from dipy.tracking.streamline import Streamlines, set_number_of_points

from TractoPL.atlasing.frechet_clustering import (
	compute_mean_streamline_quick,
	load_streamlines,
	longest_fraction_centroid,
	reorient_from_indices,
	reorient_to_reference_manhattan,
)
from TractoPL.data.vtk_loader import save_vtk


def parse_args():
	parser = argparse.ArgumentParser(
		description="Generate a VTK centroid from the longest fibers of a tractogram."
	)
	parser.add_argument("tractogram", help="Input tractogram (.vtk or .trk)")
	parser.add_argument("output_vtk", help="Output centroid VTK path")
	parser.add_argument(
		"--longest-frac",
		type=float,
		default=0.15,
		help="Fraction of the longest fibers used for the centroid (default: 0.15)",
	)
	parser.add_argument(
		"--resample-points",
		type=int,
		default=24,
		help="Number of points used to compute the centroid (default: 24)",
	)
	return parser.parse_args()


def generate_centroid(tractogram_path, output_vtk_path, longest_frac=0.15, resample_points=24):
	"""Compute and save the centroid of the longest fraction of input streamlines."""
	if not 0 < longest_frac <= 1:
		raise ValueError("longest_frac must be greater than 0 and less than or equal to 1.")
	if resample_points < 2:
		raise ValueError("resample_points must be at least 2.")

	streamlines = load_streamlines(tractogram_path)
	if not streamlines:
		raise ValueError("The input tractogram does not contain any streamlines.")

	resampled = set_number_of_points(Streamlines(streamlines), resample_points)
	reference = compute_mean_streamline_quick(resampled, nb_points=resample_points)
	_, flips = reorient_to_reference_manhattan(resampled, reference)
	oriented = reorient_from_indices(Streamlines(streamlines), flips)
	centroid = longest_fraction_centroid(oriented, longest_frac, resample_points)

	output_path = Path(output_vtk_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	save_vtk([centroid], str(output_path))
	return centroid


def main():
	args = parse_args()
	centroid = generate_centroid(
		args.tractogram,
		args.output_vtk,
		longest_frac=args.longest_frac,
		resample_points=args.resample_points,
	)
	print(f"Centroid saved to {args.output_vtk} ({len(centroid)} points)")


if __name__ == "__main__":
	main()
