#!/usr/bin/env python3
"""
Script pour appliquer des transformations ANTs à des fichiers VTK PolyData ou
tractographies (.tck / .trk).

Les formats sont tous traités comme une liste de streamlines: les points sont
aplatis avant l'appel à ANTs, puis reconstruits selon leurs longueurs et le
format de sortie demandé. Les conversions VTK utilisent TractoPL.data.vtk_loader.

Pour lire une entrée .tck/.trk, fournir --moving-image. Pour produire une sortie
.tck/.trk, fournir --fixed-image. Les coordonnées sont supposées cohérentes avec
les transformations ANTs; aucun recalage voxel/world n'est effectué.
"""

import argparse
import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
from TractoPL.data.vtk_loader import load_vtk_streamlines, save_vtk
from TractoPL.scripts.convert_tractogram import flip_tractogram
try:
    from dipy.io.streamline import load_tractogram, save_tractogram, StatefulTractogram, Space
    DIPY_AVAILABLE = True
except ImportError:
    DIPY_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Coordonnees
# --------------------------------------------------------------------------------------

def ras_to_lps(pts):
    """Convertit RAS -> LPS (inverse sur X,Y)."""
    out = pts.copy()
    out[:, 0] *= -1
    out[:, 1] *= -1
    return out

def lps_to_ras(pts):
    """Convertit LPS -> RAS."""
    return ras_to_lps(pts)  # même opération

# --------------------------------------------------------------------------------------
# Tractographie
# --------------------------------------------------------------------------------------

def load_tractogram_as_points(input_file, moving_image):
    """Charge et aplatit un tractogramme en RASMM, avec les longueurs des streamlines."""
    if not DIPY_AVAILABLE:
        raise RuntimeError("dipy requis pour les fichiers tractographie (.tck/.trk)")
    sft = load_tractogram(input_file, moving_image, bbox_valid_check=False,
                          trk_header_check=False, to_space=Space.LPSMM)
    streamlines = list(sft.streamlines)
    if len(streamlines) == 0:
        raise ValueError("Tractogramme vide")
    lengths = [sl.shape[0] for sl in streamlines]
    points = np.vstack(streamlines)
    return points, lengths

def reconstruct_streamlines(points, lengths):
    """Reconstruit les streamlines depuis un tableau de points aplati."""
    if len(points) != sum(lengths):
        raise RuntimeError("Incohérence nombre de points transformés")
    streamlines = []
    start = 0
    for length in lengths:
        streamlines.append(points[start:start + length])
        start += length
    return streamlines


def reconstruct_and_save_tractogram(points_world_lps, lengths, fixed_image_path, output_file):
    """Reconstruit et sauvegarde un tractogramme en espace LPSMM de l'image fixed."""
    if not DIPY_AVAILABLE:
        raise RuntimeError("dipy requis pour sauvegarder le tractogramme")
    new_streamlines = reconstruct_streamlines(points_world_lps, lengths)
    sft_new = StatefulTractogram(new_streamlines, fixed_image_path, Space.LPSMM)
    save_tractogram(sft_new, output_file, bbox_valid_check=False)

# --------------------------------------------------------------------------------------
# CSV / ANTs
# --------------------------------------------------------------------------------------

def points_to_csv(points, csv_filename, dimensionality=3):
    """
    Convertit un array de points en fichier CSV pour antsApplyTransformsToPoints.
    
    Args:
        points (numpy.ndarray): Array de points (N x 3)
        csv_filename (str): Nom du fichier CSV de sortie
        dimensionality (int): Dimensionnalité (2 ou 3)
        
    Note:
        antsApplyTransformsToPoints attend toujours x,y,z,t même en 2D
    """
    if points.shape[1] < dimensionality:
        raise ValueError(f"Les points doivent avoir au moins {dimensionality} dimensions")
    data = {
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2] if points.shape[1] > 2 else np.zeros(len(points)),
        't': np.zeros(len(points))
    }
    if dimensionality == 2:
        data['z'] = np.zeros(len(points))
    pd.DataFrame(data).to_csv(csv_filename, index=False)
    print(f"Points sauvegardés en CSV: {csv_filename} ({len(points)} points)")


def csv_to_points(csv_filename):
    """
    Lit un fichier CSV et retourne les points comme array numpy.
    
    Args:
        csv_filename (str): Nom du fichier CSV
        
    Returns:
        numpy.ndarray: Array de points (N x 3)
    """
    df = pd.read_csv(csv_filename)
    for c in ['x', 'y', 'z']:
        if c not in df.columns:
            raise ValueError(f"Colonne '{c}' manquante dans {csv_filename}")
    return df[['x', 'y', 'z']].values


def apply_ants_transforms(input_csv, output_csv, transforms, dimensionality=3, invert_affine=False):
    """
    Applique les transformations ANTs aux points en utilisant antsApplyTransformsToPoints.
    
    Args:
        input_csv (str): Fichier CSV d'entrée avec les points
        output_csv (str): Fichier CSV de sortie avec les points transformés
        transforms (list): Liste des fichiers de transformation
        dimensionality (int): Dimensionnalité (2 ou 3)
        invert_affine (bool): Si True, inverse automatiquement les transformations affines
        
    Returns:
        bool: True si succès, False sinon
    """
    # Construire la commande antsApplyTransformsToPoints
    cmd = ['antsApplyTransformsToPoints', '-d', str(dimensionality), '-i', input_csv, '-o', output_csv]
    for transform in transforms:
        if invert_affine and ( transform.endswith('.mat') or transform.endswith('.txt') ) :
            # Inverser automatiquement les transformations affines
            cmd.extend(['-t', f'[{transform},1]'])
            print(f"Transformation affine inversée: {transform}")
        else:
            cmd.extend(['-t', transform])
    print(f"Exécution: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print("Erreur antsApplyTransformsToPoints", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Erreur: antsApplyTransformsToPoints introuvable", file=sys.stderr)
        return False

# --------------------------------------------------------------------------------------
def show_examples():
    examples = """
EXEMPLES:
1. VTK -> TRK:
    python apply_trans_to_vtk.py fibers.vtk -t transform.mat -o fibers_tx.trk \
         --fixed-image template_T1.nii.gz

2. TCK -> VTK:
    python apply_trans_to_vtk.py fibers.tck -t warp.nii.gz -t affine.mat --invert-affine \
         --moving-image subj_T1.nii.gz -o fibers_tx.vtk
"""
    print(examples)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Applique des transformations ANTs à un VTK ou tractogramme (.tck/.trk)")
    parser.add_argument("input_vtk", help="Fichier d'entrée (.vtk .tck .trk)")
    parser.add_argument("-o","--output", help="Fichier de sortie (.vtk .tck .trk ou .csv)")
    parser.add_argument("-t","--transform", action="append", help="Transformation(s) ANTs (pile)")
    parser.add_argument("--invert-affine", action="store_true", help="Inverse les .mat affines")
    parser.add_argument("--example", action="store_true", help="Exemples")
    parser.add_argument("-d","--dimensionality", type=int, default=3, choices=[2,3], help="Dimension (2/3)")
    parser.add_argument("--keep-csv", action="store_true", help="Conserver CSV")
    parser.add_argument("--csv-only", action="store_true", help="Sortie seulement CSV")
    parser.add_argument("--moving-image", help="Image moving (NIfTI) pour tractogramme")
    parser.add_argument("--fixed-image", help="Image fixed (NIfTI) pour sortie tractogramme")
    parser.add_argument("--ants-space", choices=["LPS","RAS"], default="LPS", help="Espace attendu par ANTs (défaut LPS)")
    parser.add_argument("--flip", action="store_true", help="Activer le basculement LPS<->RAS pour les tractogrammes")
    args = parser.parse_args()

    if args.example:
        show_examples(); sys.exit(0)
    if not os.path.exists(args.input_vtk):
        print("Erreur: fichier d'entrée inexistant", file=sys.stderr); sys.exit(1)
    if not args.transform:
        print("Erreur: au moins une transformation -t requise", file=sys.stderr); sys.exit(1)
    for tf in args.transform:
        if not os.path.exists(tf):
            print(f"Erreur: transformation introuvable {tf}", file=sys.stderr); sys.exit(1)

    input_ext = os.path.splitext(args.input_vtk)[1].lower()
    input_is_tracto = input_ext in ('.tck', '.trk')
    if input_ext not in ('.vtk', '.tck', '.trk'):
        print("Erreur: format d'entrée non pris en charge", file=sys.stderr); sys.exit(1)
    if input_is_tracto and args.dimensionality != 3:
        print("Info: dimension forcée à 3 pour tractographie"); args.dimensionality = 3
    if input_is_tracto:
        if not args.moving_image or not os.path.exists(args.moving_image):
            print("Erreur: --moving-image existante requise pour une entrée tractographie", file=sys.stderr); sys.exit(1)
        if not DIPY_AVAILABLE:
            print("Erreur: dipy requis (pip install dipy)", file=sys.stderr); sys.exit(1)

    if args.output:
        output_file = args.output
    else:
        base = os.path.splitext(args.input_vtk)[0]
        output_file = f"{base}_transformed{'.csv' if args.csv_only else input_ext}"

    output_ext = os.path.splitext(output_file)[1].lower()
    output_is_tracto = output_ext in ('.tck', '.trk')
    if not args.csv_only and output_ext not in ('.vtk', '.tck', '.trk'):
        print("Erreur: format de sortie non pris en charge", file=sys.stderr); sys.exit(1)
    if output_is_tracto:
        if not args.fixed_image or not os.path.exists(args.fixed_image):
            print("Erreur: --fixed-image existante requise pour une sortie tractographie", file=sys.stderr); sys.exit(1)
        if not DIPY_AVAILABLE:
            print("Erreur: dipy requis (pip install dipy)", file=sys.stderr); sys.exit(1)

    print(f"Entrée: {args.input_vtk}")
    print(f"Transformations: {', '.join(args.transform)}")
    print(f"Sortie: {output_file}")
    print(f"Type d'entrée: {'Tractographie' if input_is_tracto else 'VTK'}")

    try:
        # Préparation des points
        if input_is_tracto:
            points, tract_lengths = load_tractogram_as_points(args.input_vtk, args.moving_image)
            print(f"Streamlines: {len(tract_lengths)} | Points: {len(points)}")
        else:
            streamlines, _ = load_vtk_streamlines(args.input_vtk)
            if not streamlines:
                raise ValueError("VTK sans streamlines")
            tract_lengths = [len(streamline) for streamline in streamlines]
            points = np.vstack(streamlines)
            print(f"Streamlines: {len(tract_lengths)} | Points: {len(points)}")
        # points = ras_to_lps(points) if args.ants_space == 'LPS' else points

        # CSV input
        if args.keep_csv:
            base_in = os.path.splitext(args.input_vtk)[0]
            input_csv = f"{base_in}_input.csv"
        else:
            input_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        points_to_csv(points, input_csv, args.dimensionality)

        # CSV output
        if args.csv_only:
            output_csv = output_file
        elif args.keep_csv:
            base_out = os.path.splitext(output_file)[0]
            output_csv = f"{base_out}_output.csv"
        else:
            output_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name

        print("Application transformations ANTs...")
        if not apply_ants_transforms(input_csv, output_csv, args.transform, args.dimensionality, args.invert_affine):
            print("Échec transformations", file=sys.stderr); sys.exit(1)

        if not args.csv_only:
            transformed_points = csv_to_points(output_csv)
            # transformed_ras = lps_to_ras(transformed_points) if args.ants_space == 'LPS' else transformed_points
            # if output_is_tracto:
            #     if args.flip:
            #         transformed_ras = flip_tractogram(transformed_ras)
            #     reconstruct_and_save_tractogram(transformed_ras, tract_lengths, args.fixed_image, output_file)
            #     print(f"Tractogramme sauvegardé: {output_file}")
            # else:
            #     save_vtk(reconstruct_streamlines(transformed_ras, tract_lengths), output_file)
            #     print(f"VTK sauvegardé: {output_file}")
            reconstruct_and_save_tractogram(transformed_points, tract_lengths, args.fixed_image, output_file)
            print(f"Tractogramme sauvegardé: {output_file}")
        else:
            print(f"CSV sauvegardé: {output_file}")

        # Nettoyage
        if not args.keep_csv:
            if os.path.exists(input_csv): os.unlink(input_csv)
            if not args.csv_only and os.path.exists(output_csv): os.unlink(output_csv)
        print("Terminé avec succès")
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr); sys.exit(1)

if __name__ == '__main__':
    main()
