#!/usr/bin/env python3

import argparse
import os
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from TractoPL.data.vtk_loader import load_vtk, save_vtk
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert tractograms between formats in batch, with optional orientation flipping (LPS<->RAS)."
    )
    # Multiple input files
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to the input tractogram files"
    )
    # Either multiple -o values (one per input) or a global --output-format
    parser.add_argument(
        "--output-file", "-o",
        action="append", nargs="+", default=[],
        help="Output paths. Repeat the option or provide multiple values to match the input files."
    )
    parser.add_argument(
        "--output-format", "-of",
        choices=["vtk", "trk", "tck"],
        help="Global output format (replaces the extension of each input file)."
    )
    # References: one for all inputs or one per input
    parser.add_argument(
        "--reference", "-r",
        action="append", default=[],
        help="Reference file(s). One for all inputs or one per input."
    )
    parser.add_argument(
        "--flip", "-f",
        action="store_true",
        help="Enable LPS<->RAS orientation flipping"
    )
    return parser.parse_args()

def _flatten(list_of_lists):
    flat = []
    for item in list_of_lists:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat

def _strip_all_ext(path):
    # Remove .gz and the previous extension if present (e.g. .tck.gz -> base)
    base, ext = os.path.splitext(path)
    if ext == ".gz":
        base, _ = os.path.splitext(base)
    return base

def _is_ext(path, ext_no_dot):
    # True if path ends with .ext or .ext.gz
    return path.endswith(f".{ext_no_dot}") or path.endswith(f".{ext_no_dot}.gz")

def _derive_outputs(inputs, output_files_chunks, output_format):
    outputs = _flatten(output_files_chunks)
    if output_format and outputs:
        raise ValueError("Ne pas utiliser simultanément -o/--output-file et --output-format.")
    if output_format:
        ext = f".{output_format}"
        return [_strip_all_ext(p) + ext for p in inputs]
    if len(outputs) != len(inputs):
        raise ValueError(f"Le nombre de sorties (-o) doit égaler le nombre d'entrées ({len(inputs)}).")
    return outputs

def _derive_references(inputs, refs):
    if not refs:
        return [None] * len(inputs)
    if len(refs) == 1:
        return [refs[0]] * len(inputs)
    if len(refs) == len(inputs):
        return refs
    raise ValueError("Le nombre de références (-r) doit être 1 ou égal au nombre d'entrées.")

def flip_tractogram(tractogram):
    """
    Flip LPS<->RAS: invert x and y on each streamline.
    """
    # Flip x and y coordinates (first and second columns)
    flipped = []

    for sl in tractogram:
        sl2 = sl.copy()
        if sl2.shape[1] >= 2:
            sl2[:, 0] *= -1  # x
            sl2[:, 1] *= -1  # y
        flipped.append(sl2)
    return flipped

def convert_tractogram(in_path, out_path, ref, flip):
    # Use 'same' for .trk if no reference is provided
    if ref is None and _is_ext(in_path, "vtk") and _is_ext(out_path, "trk"):
        raise ValueError("Impossible de convertir un fichier VTK en TRK sans référence.")
        
    loader_ref = "same" if ref is None and _is_ext(in_path, "trk") else ref
    tractogram = None
    if _is_ext(in_path, "vtk"):
        streamlines,arrays = load_vtk(in_path)
        if flip:
            streamlines = flip_tractogram(streamlines)
    else:
        tractogram = load_tractogram(
            in_path,
            loader_ref,
            bbox_valid_check=False,
            trk_header_check=False,
            to_space=Space.LPSMM
        )
        arrays = None

        if flip:
            tractogram.streamlines = flip_tractogram(tractogram.streamlines)
        streamlines=tractogram.streamlines

    if _is_ext(out_path, "vtk"):
        save_vtk(streamlines, out_path, scalar_dict=arrays)
    else:
        if not tractogram:
            tractogram = StatefulTractogram(streamlines,reference=ref, space=Space.LPSMM)
        save_tractogram(
            tractogram,
            out_path,
            bbox_valid_check=False,
            to_space=Space.LPSMM
        )

def main():
    args = parse_arguments()

    inputs = args.input_files
    try:
        outputs = _derive_outputs(inputs, args.output_file, args.output_format)
        refs = _derive_references(inputs, args.reference)
    except ValueError as e:
        print(f"Erreur: {e}")
        raise SystemExit(2)

    # Specific validation: .tck requires a reference
    for i, in_path in enumerate(inputs):
        if _is_ext(in_path, "tck") and refs[i] is None:
            print(f"Erreur: {in_path} est un .tck et nécessite une référence (-r).")
            raise SystemExit(2)

    for in_path, out_path, ref in zip(inputs, outputs, refs):
        print(f"Chargement: {in_path}")
        # Utiliser 'same' pour .trk si pas de référence fournie
        # loader_ref = "same" if ref is None and _is_ext(in_path, "trk") else ref
        # tractogram = load_tractogram(
        #     in_path,
        #     loader_ref,
        #     bbox_valid_check=False,
        #     trk_header_check=False,
        #     to_space=Space.LPSMM
        # )
        # # Only flip if specified (not the default behavior anymore)
        # if args.flip:
        #     print(f"Flip tractogram (LPS<->RAS)...")
        #     tractogram = flip_tractogram(tractogram)
        # else:
        #     print("Format conversion only (no flip)...")

        # print(f"Sauvegarde: {out_path}")
        # save_tractogram(
        #     tractogram,
        #     out_path,
        #     bbox_valid_check=False,
        #     to_space=Space.LPSMM
        # )
        # print("OK.")
        print(f"Conversion de {in_path} vers {out_path} avec flip={args.flip}...")
        convert_tractogram(in_path, out_path, ref, args.flip)
        print(f"Conversion terminée vers {out_path}.")
    print("Terminé.")
if __name__ == "__main__":
    main()
