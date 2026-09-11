from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config, get_HCP_bundle_names
from TractoPL.data.loader import Subject, move2nii, parse_filename, BIDSFile, copy2nii
from TractoPL.data.mcmfile import MCMFile, read_mcm_file
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, CLIArg, run_anima_command, run_cli_command
from TractoPL.utils.converters import call_dipy_converter
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import xml.etree.ElementTree as ET
import zipfile
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata, interpn, interp1d
from scipy.spatial import cKDTree
import numpy as np
import nibabel as nib
import time
from vtk.numpy_interface import dataset_adapter as dsa
from pathlib import Path
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingFreeType
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonComputationalGeometry import vtkCardinalSpline
from vtkmodules.vtkCommonCore import (
    vtkMinimalStandardRandomSequence, vtkPoints
)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData)
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersGeneral import vtkSplineFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor
)
from TractoPL.data.loader import Dataset

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.utils import length
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa
from TractoPL.data.io import copy_from_dict
import multiprocessing as mp
from functools import partial
import traceback
import uuid
import socket
from TractoPL.scripts.convert_tractogram import convert_tractogram

name_mapping = {
        "Fractional anisotropy": "FA",
        "Mean diffusivity": "MD",
        "Parallel diffusivity": "AD",
        "Perpendicular diffusivity": "RD",
        "Isotropic restricted water fraction": "IRF",
        "Free water fraction": "IFW"
    }

def get_transform_center(transform):
    """
    Extrait le centre de rotation d'une transformation ANTs.
    
    Parameters
    ----------
    transform : SimpleITK Transform
        Transformation ANTs chargée avec SimpleITK
        
    Returns
    -------
    ndarray
        Centre de rotation (x, y, z)
    """
    try:
        # Pour les transformations affines, ANTs stocke parfois le centre
        if hasattr(transform, 'GetCenter'):
            center = transform.GetCenter()
            return np.array(center)
        elif hasattr(transform, 'GetFixedParameters'):
            # Le centre peut être dans les paramètres fixes
            fixed_params = transform.GetFixedParameters()
            if len(fixed_params) >= 3:
                return np.array(fixed_params[:3])
    except:
        pass
    
    # Retourner l'origine par défaut
    return np.array([0.0, 0.0, 0.0])
def apply_transformation_with_center(points, transformation_matrix, center):
    """
    Applique une transformation affine avec un centre de rotation spécifique.
    
    La transformation se fait en 3 étapes :
    1. Translater vers l'origine (soustraire le centre)
    2. Appliquer la transformation complète
    3. Translater de retour (ajouter le centre)
    
    Parameters
    ----------
    points : ndarray
        Points à transformer, shape (N, 3)
    transformation_matrix : ndarray  
        Matrice de transformation 4x4
    center : ndarray
        Centre de rotation (x, y, z)
        
    Returns
    -------
    ndarray
        Points transformés, shape (N, 3)
    """
    # Extraire la rotation et la translation
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    
    # 1. Translater vers l'origine (centre -> origine)
    points_centered = points - center
    
    # 2. Appliquer la rotation
    points_rotated = (R @ points_centered.T).T
    
    # 3. Ajouter la translation et remettre le centre
    # Formula: T * R * (p - c) + c + t
    # où T est la rotation, R est déjà appliquée, c est le centre, t est la translation
    points_final = points_rotated + center + t
    
    return points_final

def get_image_center_from_nifti(nifti_path):
    """
    Calcule le centre d'une image NIfTI en coordonnées physiques.
    
    Parameters
    ----------
    nifti_path : str
        Chemin vers le fichier NIfTI
        
    Returns
    -------
    ndarray
        Centre de l'image en coordonnées physiques (x, y, z)
    """
    try:
        img = nib.load(nifti_path)
        affine = img.affine
        shape = img.shape
        
        # Centre en coordonnées voxel (au milieu de l'image)
        center_voxel = np.array([(shape[i] - 1) / 2.0 for i in range(3)])
        
        # Conversion en coordonnées physiques
        center_voxel_homog = np.append(center_voxel, 1.0)
        center_physical = (affine @ center_voxel_homog)[:3]
        
        return center_physical
        
    except Exception as e:
        print(f"Error calculating image center: {e}")
        return np.array([0.0, 0.0, 0.0])
    
def mcm_estimator(dwi, bval, bvec, mask, compart_map=None, n_comparts=None, **kwargs):
    """
    Calls the animaMCMEstimator to estimate the MCM coefficients from the DWI data.

    Parameters
    ----------
    dwi : BIDSFile
        The DWI data to estimate the responses from
    bval : BIDSFile
        The bval file associated with the DWI data
    bvec : BIDSFile
        The bvec file associated with the DWI data
    mask : BIDSFile
        Brain mask to use for the estimation
    n_comparts : int
        Number of anisotropic compartments to estimate
    kwargs : dict  
        Additional arguments to pass to the script
    """
    # Préparer les entrées
    inputs = {
        "dwi": dwi,
        "bval": bval,
        "bvec": bvec,
        "mask": mask
    }

    # Construire les arguments de commande avec des références symboliques (copie les fichiers dans un dossier temporaire)
    command_args = [
        "-b", "$bval",
        "-g", "$bvec",
        "-i", "$dwi",
        "-m", "$mask",
        "-o", "mcm"
        # "-n", str(n_comparts)
    ]

    # Définir les sorties attendues (pattern de base, sera complété après exécution)
    output_pattern = {
        'mcm.mcm': {"model": "MCM", "extension": ".mcm"}
    }


    if compart_map is not None:
        inputs['compart_map'] = compart_map
        command_args += ["-I", '$compart_map']
    elif n_comparts is not None:
        command_args += ["-n", str(n_comparts)]
    
    elif 'M' in kwargs:
        command_args += ["-a", 'aic.nii.gz']
    else:
        raise ValueError("Either compart_map or n_comparts must be provided.")


    # Exécuter la commande
    result = run_anima_command(
        "animaMCMEstimator",
        inputs,
        output_pattern,
        dwi,
        command_args=command_args,
        **kwargs
    )

    # Chemin vers le fichier MCM généré
    mcm_file = next(path for path in result.keys() if path.endswith('.mcm'))

    if not os.path.exists(mcm_file):
        raise FileNotFoundError(f"MCM file not found: {mcm_file}")
    mcm_obj = MCMFile(mcm_file)
    mcm_obj.write(mcm_file.replace('.mcm', '.mcmx'))
    base_entities = dwi.get_entities() if isinstance(dwi, BIDSFile) else dwi.copy()
    result = {
        mcm_file.replace('.mcm', '.mcmx'): upt_dict(
            base_entities,
            model='MCM',
            extension='.mcmx'
        )
    }

    if 'M' in kwargs:
        result[opj(os.path.dirname(mcm_file), 'aic.nii.gz')] = upt_dict(
            base_entities,
            extension='.nii.gz',
            model='MCM',
            suffix='AIC'
        )

    # # Lire le modèle MCM pour identifier les compartiments
    # mcm_model = read_mcm_file(mcm_file)
    # tmp_folder = os.path.dirname(mcm_file)

    # # Ajouter les compartiments au dictionnaire de résultats
    # base_entities = dwi.get_entities() if isinstance(dwi, BIDSFile) else dwi.copy()
    # base_entities = upt_dict(base_entities, model='MCM', extension='nrrd')

    # # Ajouter les fichiers de compartiments
    # for comp in mcm_model['compartments']:
    #     comp_path = opj('mcm', comp['filename'])
    #     comp_num = comp['filename'].split('_')[-1].split('.')[0]  # Get number from filename
    #     result[opj(tmp_folder, comp_path)] = upt_dict(
    #         base_entities.copy(),
    #         compartment=comp_num,
    #         extension='.nrrd',
    #         type=comp['type'].lower()
    #     )

    # # Ajouter le fichier de poids
    # result[opj(tmp_folder, 'mcm/mcm_weights.nrrd')] = upt_dict(
    #     base_entities.copy(),
    #     extension='.nrrd',
    #     model='MCM',
    #     type='weights'
    # )

    return result


def update_mcm_info_file(copy_dict):
    """
    Update the MCM info file with the new output directory.

    Parameters
    ----------
    copy_dict : dict
        Dictionary containing the mapping of source files to destination files
    """
    print(copy_dict)
    mcm_file = [v for k, v in copy_dict.items() if k.endswith('.mcm')][0]

    copy_dict = {
        os.path.basename(k): v
        for k, v in copy_dict.items() if not k.endswith('.mcm')
    }
    if mcm_file is None:
        raise ValueError("No MCM file found in the copy dictionary.")

    # Update in the XML file using copy_dict (replace copy_dict.keys() strings in XML by copy_dict.values())
    tree = ET.parse(mcm_file)
    root = tree.getroot()
    for elem in root.iter():
        if (elem.tag == 'FileName'
                or elem.tag == 'Weights') and elem.text in copy_dict:
            elem.text = os.path.basename(copy_dict[elem.text])

    # Write back to the file
    tree.write(mcm_file)
    return True


def convert_vtk_to_csv(vtk_file_path, output_base_path):
    """
    Convert a VTK PolyData file to multiple CSV files:
    - One CSV for fiber connectivity (FiberID, PointOrder, PointID).
    - One CSV per point data array (X, Y, Z, ArrayComponent1, ArrayComponent2, ...).

    Parameters
    ----------
    vtk_file_path : str
        Path to the input VTK file.
    output_base_path : str
        Base path for the output CSV files (e.g., '/path/to/output/prefix').
        Array names or 'fibers' will be appended (e.g., prefix_ArrayName.csv).

    Returns
    -------
    dict
        A dictionary mapping descriptive keys (like 'fibers' or array names)
        to the paths of the generated CSV files.
        Returns an empty dictionary if conversion fails.
    """
    generated_csv_files = {}

    # Create a reader for VTK files (assuming legacy .vtk format)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    if reader.GetOutput() is None or reader.GetOutput().GetNumberOfPoints() == 0:
        print(
            f"Error: Could not read VTK file or file is empty: {vtk_file_path}")
        return generated_csv_files

    polydata = reader.GetOutput()

    # 1. Extract Point Data Arrays
    point_data = polydata.GetPointData()
    num_points = polydata.GetNumberOfPoints()

    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = array.GetName()
        if not array_name:
            array_name = f"UnnamedArray_{i}"

        num_components = array.GetNumberOfComponents()

        csv_file_path = f"{output_base_path}_{array_name}.csv"

        data_for_df = []
        header = ['X', 'Y', 'Z']
        if num_components == 1:
            header.append(array_name)
        else:
            for comp_idx in range(num_components):
                header.append(f"{array_name}_{comp_idx}")

        for pt_id in range(num_points):
            coords = polydata.GetPoint(pt_id)
            row = list(coords)
            value_tuple = array.GetTuple(pt_id)
            row.extend(list(value_tuple))
            data_for_df.append(row)

        df = pd.DataFrame(data_for_df, columns=header)
        df.to_csv(csv_file_path, index=False)
        generated_csv_files[array_name] = csv_file_path

    # 2. Extract Fiber Connectivity (Lines)
    lines = polydata.GetLines()
    if lines and lines.GetNumberOfCells() > 0:
        csv_fibers_path = f"{output_base_path}_fibers.csv"

        fiber_data_for_df = []
        header_fibers = ['FiberID', 'PointOrder', 'PointID']

        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        fiber_id_counter = 0
        while lines.GetNextCell(id_list):
            for point_order in range(id_list.GetNumberOfIds()):
                point_id = id_list.GetId(point_order)
                fiber_data_for_df.append(
                    [fiber_id_counter, point_order, point_id])
            fiber_id_counter += 1

        df_fibers = pd.DataFrame(fiber_data_for_df, columns=header_fibers)
        df_fibers.to_csv(csv_fibers_path, index=False)
        generated_csv_files['fibers'] = csv_fibers_path

    return generated_csv_files


def generate_scalar_map(vtk_file_path, reference_nifti_path, output_dir, output_file_prefix="scalar_map"):
    """
    Generates NIfTI scalar maps from each point data array in a VTK PolyData file,
    interpolating the values onto the grid of a reference NIfTI image.

    Parameters
    ----------
    vtk_file_path : str
        Path to the input VTK PolyData file (.vtk).
    reference_nifti_path : str
        Path to the reference NIfTI image file.
    output_dir : str
        Directory where the output NIfTI files will be saved.
    output_file_prefix : str
        Prefix for the output NIfTI file names. Array name will be appended.

    Returns
    -------
    list
        A list of paths to the generated NIfTI files.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_nifti_paths = []

    vtk_file=vtk_file_path if isinstance(vtk_file_path, BIDSFile) else BIDSFile(vtk_file_path)
    vtk_file_path = vtk_file.path
    name_mapping = {
        "Fractional anisotropy": "FA",
        "Mean diffusivity": "MD",
        "Parallel diffusivity": "AD",
        "Perpendicular diffusivity": "RD",
        "Isotropic restricted water fraction": "IRF",
        "Free water fraction": "IFW"
    }

    # Read the VTK PolyData
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    if not polydata or polydata.GetNumberOfPoints() == 0:
        print(f"Error: Could not read VTK file or it's empty: {vtk_file_path}")
        return output_nifti_paths

    # Read the reference NIfTI image
    try:
        ref_image = nib.load(reference_nifti_path)
        ref_data = ref_image.get_fdata()
        ref_affine = ref_image.affine
        ref_shape = ref_data.shape[:3]  # Ensure we get only spatial dimensions
    except Exception as e:
        print(f"Error reading reference NIfTI file: {reference_nifti_path}")
        print(e)
        return output_nifti_paths

    # Read all arrays from the VTK file
    point_data = polydata.GetPointData()
    num_points = polydata.GetNumberOfPoints()
    num_arrays = point_data.GetNumberOfArrays()

    if num_arrays == 0:
        print(
            f"Error: No point data arrays found in VTK file: {vtk_file_path}")
        return output_nifti_paths

    # Extract VTK points and convert to physical coordinates
    vtk_points = np.array([polydata.GetPoint(i) for i in range(num_points)])

    # Create grid points in voxel space
    x, y, z = np.meshgrid(
        np.arange(ref_shape[0]),
        np.arange(ref_shape[1]),
        np.arange(ref_shape[2]),
        indexing='ij'
    )

    grid_coords = np.vstack(
        [x.ravel(), y.ravel(), z.ravel(), np.ones_like(x.ravel())])
    world_coords = np.dot(ref_affine, grid_coords)[
        :3].T  # Transform and get (N, 3) shape

    for idx_array in range(num_arrays):
        array = point_data.GetArray(idx_array)
        array_name = array.GetName()
        if not array_name:
            array_name = f"UnnamedArray_{idx_array}"

        num_components = array.GetNumberOfComponents()

        # Skip arrays with multiple components for simplicity
        if num_components > 1:
            print(
                f"Skipping array '{array_name}' with {num_components} components")
            continue

        # Get scalar values from the VTK array
        vtk_values = vtk_to_numpy(array)
        #

        inv_ref_affine = np.linalg.inv(ref_affine)

        sum_fa_map = np.zeros(ref_shape, dtype=float)
        count_map = np.zeros(ref_shape, dtype=int)

        # Transformation de tous les points VTK en coordonnées voxel NIfTI
        # Ajout de la coordonnée homogène aux points VTK
        vtk_points_homogeneous = np.hstack(
            (vtk_points, np.ones((vtk_points.shape[0], 1))))
        # Transformation en coordonnées voxel
        vtk_points_voxel_coords_float = np.dot(
            inv_ref_affine, vtk_points_homogeneous.T)[:3, :].T

        # Itérer sur chaque point VTK et sa valeur de FA
        for i in range(vtk_points.shape[0]):
            point_coord_voxel_float = vtk_points_voxel_coords_float[i]
            fa_value = vtk_values[i]

            # Convertir les coordonnées voxel flottantes en indices entiers
            # et s'assurer qu'ils sont dans les limites de l'image
            vx = int(round(point_coord_voxel_float[0]))
            vy = int(round(point_coord_voxel_float[1]))
            vz = int(round(point_coord_voxel_float[2]))

            if (0 <= vx < ref_shape[0] and
                0 <= vy < ref_shape[1] and
                    0 <= vz < ref_shape[2]):

                sum_fa_map[vx, vy, vz] += fa_value
                count_map[vx, vy, vz] += 1

        # Calculer la carte de FA moyenne
        # Éviter la division par zéro pour les voxels sans points
        mean_fa_map = np.zeros(ref_shape, dtype=float)
        # Utiliser np.true_divide pour gérer la division par zéro et obtenir des NaN si souhaité,
        # ou vérifier count_map > 0 pour affecter 0 aux voxels vides.
        non_zero_counts = count_map > 0
        mean_fa_map[non_zero_counts] = sum_fa_map[non_zero_counts] / \
            count_map[non_zero_counts]

        #Flip the mean FA in z 
        ref_affine[0,:]= -ref_affine[0,:]
        ref_affine[1,:]= -ref_affine[1,:]
        # Sauvegarder la carte de FA moyenne
        output_mean_fa_path = os.path.join(
            output_dir, f"{output_file_prefix}_{name_mapping[array_name]}.nii.gz")
        mean_fa_img = nib.Nifti1Image(mean_fa_map, ref_affine)
        nib.save(mean_fa_img, output_mean_fa_path)
        print(f"Generated Mean FA in Voxel NIfTI file: {output_mean_fa_path}")
        output_nifti_paths.append(output_mean_fa_path)
    
    # Prepare result dictionary with proper entities
    base_entities = vtk_file.get_full_entities()
    
    res_dict = {}
    for nifti_path in output_nifti_paths:
        # Extract metric name from filename
        filename = os.path.basename(nifti_path)
        # Remove extension and prefix to get metric name
        metric_name = filename.replace(output_file_prefix + '_', '').replace('.nii.gz', '')
        
        res_dict[nifti_path] = upt_dict(
            base_entities.copy(),
            metric=metric_name,
            extension='.nii.gz'
        )
    return res_dict

def _interpolate_scalar_along_fiber(
    original_fiber_coords_np, 
    original_fiber_point_ids, 
    polydata_point_data, 
    scalar_array_name, 
    target_proportion
):
    """Interpolates a scalar value along an original fiber at a target proportion of its length."""
    scalar_array_vtk = polydata_point_data.GetArray(scalar_array_name)
    if not scalar_array_vtk:
        return None

    num_orig_points = original_fiber_coords_np.shape[0]
    if num_orig_points == 0:
        return None
    
    num_components = scalar_array_vtk.GetNumberOfComponents()
    
    # Use vtk_to_numpy for faster bulk extraction
    original_scalar_values = vtk_to_numpy(scalar_array_vtk)[original_fiber_point_ids]
    
    if num_orig_points == 1:
        return original_scalar_values[0]

    # Vectorized distance calculation
    distances = np.linalg.norm(np.diff(original_fiber_coords_np, axis=0), axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_distances[-1]

    if total_length == 0:
        return original_scalar_values[0]

    target_dist = np.clip(target_proportion * total_length, 0, total_length)
    
    # Use searchsorted for faster interpolation position finding
    if target_dist <= cumulative_distances[0]:
        return original_scalar_values[0]
    elif target_dist >= cumulative_distances[-1]:
        return original_scalar_values[-1]
    
    # Find interpolation segment
    idx = np.searchsorted(cumulative_distances, target_dist) - 1
    
    # Linear interpolation between two points
    t = (target_dist - cumulative_distances[idx]) / (cumulative_distances[idx + 1] - cumulative_distances[idx])
    interpolated_val = original_scalar_values[idx] * (1 - t) + original_scalar_values[idx + 1] * t
    
    return interpolated_val


def project_to_central_line_from_vtk(vtk_file_path, reference_nifti_path, output_path, num_points_central_line=100, transformation_matrix=None, transformation_center=None, rotate_z_180=False, center_vtk=True):
    """
    Clusters fibers into a single central line and projects all array values onto this line.

    Parameters
    ----------
    vtk_file_path : str
        Path to the input VTK PolyData file (.vtk).
    reference_nifti_path : str
        Path to the reference NIfTI image file.
    output_path : str
        Directory where the output VTK PolyData file will be saved.
        The output file will be named "central_line.vtk".
    num_points_central_line : int, optional
        Number of points to define the central line, by default 100.
    transformation_matrix : ndarray, optional
        Transformation matrix to apply to the fibers. Shape (4, 4).
    transformation_center : ndarray, optional
        Center of rotation for the transformation. Shape (3,). If None, uses (0, 0, 0).
    rotate_z_180 : bool, optional
        Whether to apply a 180 degree rotation around the z-axis, by default False.
    center_vtk : bool, optional
        Whether to center the VTK points before applying the transformation, by default True.

    Returns
    -------
    str or None
        Path to the generated VTK file containing the central line, or None if an error occurs
        or no fibers are found.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_vtk_filepath = os.path.join(output_path, "central_line.vtk")

    # Read the VTK PolyData
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    if not polydata or polydata.GetNumberOfLines() == 0:
        print(f"Error: No lines found in VTK file: {vtk_file_path}")
        return None
    
    # Load transformation matrix if provided and ensure it's in the right format
    if transformation_matrix is not None:
        # Ensure transformation_matrix is a numpy array and 4x4
        transformation_matrix = np.array(transformation_matrix)
        print(f"Transformation matrix final:\n{transformation_matrix}")


    # Extract point coordinates of each line
    lines = polydata.GetLines()
    lines_points = []
    
    # Initialize line cell traversal
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    
    # Collect all points first to calculate center if needed
    all_line_points = []
    while lines.GetNextCell(id_list):
        line_points = []
        for j in range(id_list.GetNumberOfIds()):
            point_id = id_list.GetId(j)
            point = polydata.GetPoint(point_id)
            line_points.append(point)
        all_line_points.append(np.array(line_points))

    
    # Process each line (transformation will be applied later to avoid double application)
    for line_points_array in all_line_points:
        lines_points.append(line_points_array)
    
    print(f"Number of lines found: {len(lines_points)}")
    if len(lines_points) > 0:
        print(f"First line has {len(lines_points[0])} points")
        
        # Additional debug: check if points are reasonable
        all_points = np.vstack(lines_points)
        overall_mean = np.mean(all_points, axis=0)
        overall_std = np.std(all_points, axis=0)
        print(f"Overall points stats - Mean: {overall_mean}")
        print(f"Overall points stats - Std: {overall_std}")
        
        # Check if points seem to be in a reasonable coordinate system
        if np.any(np.abs(overall_mean) > 1000) or np.any(overall_std > 1000):
            print("Warning: Points seem to have very large coordinates, transformation might be incorrect")
    
    #Get stat of the lengths of the lines
    line_lengths = np.array([len(line) for line in lines_points])
    
    print(f'Line lengths: min={line_lengths.min()}, max={line_lengths.max()}, mean={line_lengths.mean()}')

    #Interpolate lines to a common number of points
    interpolated_lines = []
    for line in lines_points:
        if len(line) < 2:
            # Skip lines with less than 2 points
            continue
        # Create a linear space of points along the line
        # Calculate cumulative distances along the line
        distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        total_length = cumulative_distances[-1]
        
        # Create target distances for interpolation
        target_distances = np.linspace(0, total_length, num_points_central_line)
        
        # Interpolate each coordinate (x, y, z) separately
        interpolated_line = np.zeros((num_points_central_line, 3))
        for coord_idx in range(3):  # x, y, z coordinates
            interp_func = interp1d(cumulative_distances, line[:, coord_idx], 
                     kind='linear', bounds_error=False, 
                     fill_value=(line[0, coord_idx], line[-1, coord_idx]))
            interpolated_line[:, coord_idx] = interp_func(target_distances)
        interpolated_lines.append(interpolated_line)
    
    # Create a new polydata with interpolated fibers (fixed number of points)
    polydata_interpolated = vtk.vtkPolyData()
    
    # Create points for the interpolated lines
    interpolated_points = vtk.vtkPoints()
    interpolated_lines_vtk = vtk.vtkCellArray()
    
    point_counter = 0
    for line in interpolated_lines:
        # Add points for this line
        line_point_ids = []
        for point in line:
            interpolated_points.InsertNextPoint(point[0], point[1], point[2])
            line_point_ids.append(point_counter)
            point_counter += 1
        
        # Create line connectivity
        interpolated_lines_vtk.InsertNextCell(len(line_point_ids))
        for point_id in line_point_ids:
            interpolated_lines_vtk.InsertCellPoint(point_id)
    
    # Set points and lines to the new polydata
    polydata_interpolated.SetPoints(interpolated_points)
    polydata_interpolated.SetLines(interpolated_lines_vtk)
    
    # Interpolate scalar values for each point of the interpolated lines
    polydata_point_data = polydata.GetPointData()
    scalar_arrays = [polydata_point_data.GetArray(i).GetName() for i in range(polydata_point_data.GetNumberOfArrays()) if polydata_point_data.GetArray(i).GetName() != 'colors']

    print(f"Scalar arrays found: {scalar_arrays}")
    
    # For each scalar array, create interpolated values
    for array_name in scalar_arrays:
        print(f"Interpolating scalar values for array: {array_name}")
        array_vtk = polydata_point_data.GetArray(array_name)
        num_components = array_vtk.GetNumberOfComponents()
        
        # Create array to hold all interpolated values
        all_interpolated_values = []
        
        # Process each original line
        lines = polydata.GetLines()
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        line_idx = 0
        
        while lines.GetNextCell(id_list):
            if line_idx >= len(interpolated_lines):
                break
                
            # Get original fiber coordinates and point IDs
            original_fiber_point_ids = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
            original_fiber_coords_np = np.array([polydata.GetPoint(pid) for pid in original_fiber_point_ids])
            
            # Interpolate scalar values for each point of the interpolated line
            for t in np.linspace(0, 1, num_points_central_line):
                interpolated_val = _interpolate_scalar_along_fiber(
                    original_fiber_coords_np, 
                    original_fiber_point_ids, 
                    polydata_point_data, 
                    array_name, 
                    target_proportion=t
                )
                if interpolated_val is not None:
                    all_interpolated_values.append(interpolated_val)
                else:
                    # Fill with zeros if interpolation fails
                    all_interpolated_values.append(np.zeros(num_components))
            
            line_idx += 1
        
        # Convert to numpy array and add to polydata
        if all_interpolated_values:
            interpolated_values_np = np.array(all_interpolated_values)
            vtk_array = numpy_to_vtk(interpolated_values_np, deep=True)
            vtk_array.SetName(array_name)
            polydata_interpolated.GetPointData().AddArray(vtk_array)

    qb = QuickBundles(
        threshold=100.0,  # Adjust threshold as needed
        metric=AveragePointwiseEuclideanMetric()
    )

    # Convert the interpolated lines to a Streamlines object
    streamlines = Streamlines(interpolated_lines)
    
    # Perform quickbundling to get central lines
    print("Performing quickbundling to get central lines...")
    qb_result = qb.cluster(streamlines)
    print(f"Quickbundling resulted in {len(qb_result)} clusters.")
    if len(qb_result) == 0:
        print("No central lines found after quickbundling.")
        return None
    
    # Create a new polydata for the central line
    central_line_polydata = vtk.vtkPolyData()
    central_line_points = vtk.vtkPoints()
    central_line_lines = vtk.vtkCellArray()
    
    point_counter = 0
    for i, cluster in enumerate(qb_result):
        # Get the centroid of the cluster (this is the representative streamline)
        centroid = cluster.centroid
        
        # Add points for the centroid line
        line_point_ids = []
        for point in centroid:
            # Ensure point coordinates are float type
            central_line_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            line_point_ids.append(point_counter)
            point_counter += 1
        
        # Create a line connectivity for the centroid
        central_line_lines.InsertNextCell(len(line_point_ids))
        for point_id in line_point_ids:
            central_line_lines.InsertCellPoint(point_id)
    # Set points and lines to the new polydata
    central_line_polydata.SetPoints(central_line_points)
    central_line_polydata.SetLines(central_line_lines)

    # Populate scalar arrays for central_line_polydata by averaging from member streamlines
    for array_name in scalar_arrays:
        print(f"Averaging scalar array '{array_name}' onto central line polydata.")
        
        # Scalars for interpolated streamlines (input to QuickBundles)
        interpolated_vtk_array = polydata_interpolated.GetPointData().GetArray(array_name)
        if not interpolated_vtk_array:
            print(f"  Skipping array {array_name}, not found in polydata_interpolated.")
            continue
        
        interpolated_np_array = vtk_to_numpy(interpolated_vtk_array)
        num_components = interpolated_vtk_array.GetNumberOfComponents()
        total_central_points = central_line_points.GetNumberOfPoints()
        
        sum_values = np.zeros((total_central_points, num_components), dtype=np.float64)
        count_values = np.zeros((total_central_points, 1), dtype=np.float64) # Use float for count to avoid type issues in division if sum is float

        current_centroid_point_offset = 0
        for cluster_idx, cluster in enumerate(qb_result):
            centroid_length = len(cluster.centroid)
            
            for i in range(centroid_length): # For each point on this centroid
                global_centroid_point_idx = current_centroid_point_offset + i
                
                for member_streamline_idx_in_interpolated_list in cluster.indices:
                    # member_streamline_idx_in_interpolated_list is an index into `interpolated_lines`
                    # Calculate the global point index for this member's point in `interpolated_np_array`
                    # Each interpolated streamline has `num_points_central_line` points.
                    point_idx_on_member_streamline = (member_streamline_idx_in_interpolated_list * num_points_central_line) + i
                    
                    if point_idx_on_member_streamline < interpolated_np_array.shape[0]:
                        value_from_member = interpolated_np_array[point_idx_on_member_streamline]
                        sum_values[global_centroid_point_idx] += value_from_member
                        count_values[global_centroid_point_idx] += 1
            
            current_centroid_point_offset += centroid_length

        central_line_final_data = np.zeros((total_central_points, num_components), dtype=np.float32)
        # Avoid division by zero: only divide where count is non-zero
        valid_counts_mask = count_values > 0 # Shape (total_central_points, 1)
        
        # Create a 1D boolean mask for selecting rows
        row_mask = valid_counts_mask[:, 0] # Shape (total_central_points,)

        # Perform division only for rows with valid counts
        if np.any(row_mask):
            selected_sum_values = sum_values[row_mask]     # Shape (N, num_components)
            selected_count_values = count_values[row_mask] # Shape (N, 1)
            
            division_result = selected_sum_values / selected_count_values # Shape (N, num_components)
            
            central_line_final_data[row_mask] = division_result.astype(np.float32)
                
        vtk_central_line_array = numpy_to_vtk(central_line_final_data, deep=True)
        vtk_central_line_array.SetName(array_name)
        central_line_polydata.GetPointData().AddArray(vtk_central_line_array)

    # --- DEBUG: Save original streamlines with centroid-mapped metrics using cKDTree ---
    debug_polydata = vtk.vtkPolyData()
    debug_points = vtk.vtkPoints()
    debug_lines = vtk.vtkCellArray()
    debug_point_counter = 0
    debug_metric_arrays = {array_name: [] for array_name in scalar_arrays}

    # Build a dict of centroid metrics for each cluster and array
    centroid_metrics_per_cluster = {}
    for array_name in scalar_arrays:
        centroid_metrics_per_cluster[array_name] = []
        arr = None
        for arr_idx in range(central_line_polydata.GetPointData().GetNumberOfArrays()):
            arr_candidate = central_line_polydata.GetPointData().GetArray(arr_idx)
            if arr_candidate.GetName() == array_name:
                arr = arr_candidate
                break
        if arr is not None: # arr is from central_line_polydata, which is now populated
            arr_np = vtk_to_numpy(arr)
            # Split arr_np into clusters (since all clusters are concatenated)
            offset = 0
            for cluster in qb_result:
                npts = len(cluster.centroid)
                centroid_metrics_per_cluster[array_name].append(arr_np[offset:offset+npts])
                offset += npts
        else: # Fallback if array was not found in central_line_polydata (should not happen)
            for cluster in qb_result:
                # Get num_components from the original input polydata as a last resort
                original_input_array = polydata.GetPointData().GetArray(array_name)
                num_components = original_input_array.GetNumberOfComponents() if original_input_array else 1
                centroid_metrics_per_cluster[array_name].append(
                    np.zeros((len(cluster.centroid), num_components), dtype=np.float32)
                )

    for cluster_idx, cluster in enumerate(qb_result):
        centroid = cluster.centroid  # (num_points_central_line, 3)
        centroid_tree = cKDTree(centroid)
        # Get centroid metrics for this cluster
        centroid_metrics = {array_name: centroid_metrics_per_cluster[array_name][cluster_idx] for array_name in scalar_arrays}
        for member_idx in cluster.indices:
            orig_line = lines_points[member_idx]
            orig_len = len(orig_line)
            if orig_len < 1:
                continue
            # Associe chaque point au point du centroïde le plus proche
            dists, idxs = centroid_tree.query(orig_line)
            # Add points and lines
            line_point_ids = []
            for pt_idx, pt in enumerate(orig_line):
                debug_points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
                line_point_ids.append(debug_point_counter)
                debug_point_counter += 1
                for array_name in scalar_arrays:
                    debug_metric_arrays[array_name].append(centroid_metrics[array_name][idxs[pt_idx]])
            debug_lines.InsertNextCell(len(line_point_ids))
            for pid in line_point_ids:
                debug_lines.InsertCellPoint(pid)
    debug_polydata.SetPoints(debug_points)
    debug_polydata.SetLines(debug_lines)
    for array_name in scalar_arrays:
        arr_np = np.array(debug_metric_arrays[array_name], dtype=np.float32)
        arr = numpy_to_vtk(arr_np, deep=True)
        arr.SetName(array_name)
        debug_polydata.GetPointData().AddArray(arr)

    if reference_nifti_path:
        # Load the reference NIfTI image to get the affine transformation
        try:
            ref_image = nib.load(reference_nifti_path)
            ref_affine = ref_image.affine
            print(f"Reference NIfTI affine:\n{ref_affine}")
        except Exception as e:
            print(f"Error loading reference NIfTI file: {reference_nifti_path}")
            print(e)
            ref_affine = np.eye(4)
    
    # Apply transformation matrix to the polydata if provided
    if transformation_matrix is not None:
        print("Applying transformation matrix to polydata...")
        
        # Utiliser le centre de transformation si fourni, sinon calculer le centre de l'image de référence
        if transformation_center is None:
            if reference_nifti_path:
                transformation_center = get_image_center_from_nifti(reference_nifti_path)
                print(f"Using reference image center: {transformation_center}")
            else:
                transformation_center = np.array([0.0, 0.0, 0.0])
                print("No reference image, using origin as center")
        else:
            print(f"Using provided transformation center: {transformation_center}")
        
        # Debug: afficher quelques points avant transformation
        if debug_polydata.GetNumberOfPoints() > 0:
            sample_points_before = []
            for i in range(min(5, debug_polydata.GetNumberOfPoints())):
                pt = debug_polydata.GetPoint(i)
                sample_points_before.append(pt)
            print(f"Sample points before transformation: {sample_points_before}")
        
        # Appliquer la transformation manuellement avec le centre correct
        # Extraire tous les points des polydata
        debug_points_array = []
        for i in range(debug_polydata.GetNumberOfPoints()):
            pt = debug_polydata.GetPoint(i)
            debug_points_array.append([pt[0], pt[1], pt[2]])
        debug_points_array = np.array(debug_points_array)
        
        # Appliquer la transformation avec le centre
        transformed_debug_points = apply_transformation_with_center(
            debug_points_array, transformation_matrix, transformation_center
        )
        
        # Mettre à jour les points dans debug_polydata
        debug_points_vtk = vtk.vtkPoints()
        for i, pt in enumerate(transformed_debug_points):
            debug_points_vtk.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
        debug_polydata.SetPoints(debug_points_vtk)
        
        # Faire de même pour central_line_polydata
        central_points_array = []
        for i in range(central_line_polydata.GetNumberOfPoints()):
            pt = central_line_polydata.GetPoint(i)
            central_points_array.append([pt[0], pt[1], pt[2]])
        central_points_array = np.array(central_points_array)
        
        transformed_central_points = apply_transformation_with_center(
            central_points_array, transformation_matrix, transformation_center
        )
        
        central_points_vtk = vtk.vtkPoints()
        for i, pt in enumerate(transformed_central_points):
            central_points_vtk.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
        central_line_polydata.SetPoints(central_points_vtk)
        
        # Debug: afficher quelques points après transformation
        if debug_polydata.GetNumberOfPoints() > 0:
            sample_points_after = []
            for i in range(min(5, debug_polydata.GetNumberOfPoints())):
                pt = debug_polydata.GetPoint(i)
                sample_points_after.append(pt)
            print(f"Sample points after transformation: {sample_points_after}")
    else:
        print("No transformation matrix provided, skipping transformation.")
    debug_vtk_path = output_path.replace('desc-centroid','desc-debug')
    writer_debug = vtk.vtkPolyDataWriter()
    writer_debug.SetFileName(debug_vtk_path)
    writer_debug.SetInputData(debug_polydata)
    writer_debug.Write()
    print(f"Debug: Original streamlines with centroid-mapped metrics saved to: {debug_vtk_path}")

    # Save the central line polydata to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(central_line_polydata)
    writer.Write()
    print(f"Central line saved to: {output_path}")
    
    return output_path
    
def apply_transformation_to_vtk(vtk_file_path, reference_nifti_path, transformation_matrix=None, transformation_center=None):
    """
    Applies a transformation matrix to a VTK PolyData object, optionally centered around a specified point.

    Parameters
    ----------
    vtk_file_path : str
        Path to the input VTK file.
    reference_nifti_path : str
        Path to the reference NIfTI image file to get the affine transformation.
    transformation_matrix : np.ndarray, optional
        A 4x4 transformation matrix to apply. If None, no transformation is applied.
    transformation_center : np.ndarray, optional
        A 3-element array specifying the center of rotation for the transformation. If None,
        the center of the reference NIfTI image is used.

    Returns
    -------
    str
        Path to the transformed VTK file.
    """

    # Read the VTK PolyData
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    if not polydata or polydata.GetNumberOfPoints() == 0:
        print(f"Error: Could not read VTK file or file is empty: {vtk_file_path}")
        return vtk_file_path

    if transformation_matrix is None:
        print("No transformation matrix provided, returning original polydata.")
        return vtk_file_path

    # Load the reference NIfTI image to get the affine transformation
    try:
        ref_image = nib.load(reference_nifti_path)
        ref_affine = ref_image.affine
        print(f"Reference NIfTI affine:\n{ref_affine}")
    except Exception as e:
        print(f"Error loading reference NIfTI file: {reference_nifti_path}")
        print(e)
        ref_affine = np.eye(4)

    # Ensure transformation_matrix is a numpy array and 4x4
    transformation_matrix = np.array(transformation_matrix)
    if transformation_matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be a 4x4 numpy array.")
    print(f"Transformation matrix final:\n{transformation_matrix}")
    
    # Utiliser le centre de transformation si fourni, sinon calculer le centre de l'image de référence
    if transformation_center is None:
        if reference_nifti_path:
            transformation_center = get_image_center_from_nifti(reference_nifti_path)
            print(f"Using reference image center: {transformation_center}")
        else:
            transformation_center = np.array([0.0, 0.0, 0.0])
            print("No reference image, using origin as center")
    else:
        print(f"Using provided transformation center: {transformation_center}")
    
    # Debug: afficher quelques points avant transformation
    if polydata.GetNumberOfPoints() > 0:
        sample_points_before = []
        for i in range(min(5, polydata.GetNumberOfPoints())):
            pt = polydata.GetPoint(i)
            sample_points_before.append(pt)
        print(f"Sample points before transformation: {sample_points_before}")

    # Create a new polydata to store the transformed data
    transformed_polydata = vtk.vtkPolyData()
    
    # Create new points for the transformed data
    transformed_points = vtk.vtkPoints()
    
    # Extraire tous les points du polydata
    points_array = []
    for i in range(polydata.GetNumberOfPoints()):
        pt = polydata.GetPoint(i)
        points_array.append([pt[0], pt[1], pt[2]])
    points_array = np.array(points_array)
    
    # Appliquer la transformation avec le centre
    transformed_points_array = apply_transformation_with_center(
        points_array, transformation_matrix, transformation_center
    )
    
    # Mettre à jour les points dans le nouveau polydata
    for pt in transformed_points_array:
        transformed_points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
    
    transformed_polydata.SetPoints(transformed_points)
    
    # Copier les cellules (lignes, polygones, etc.) du polydata original
    transformed_polydata.SetPolys(polydata.GetPolys())
    transformed_polydata.SetLines(polydata.GetLines())
    transformed_polydata.SetVerts(polydata.GetVerts())
    transformed_polydata.SetStrips(polydata.GetStrips())
    
    # Copier les attributs de points (point data) du polydata original
    original_point_data = polydata.GetPointData()
    for i in range(original_point_data.GetNumberOfArrays()):
        array = original_point_data.GetArray(i)
        transformed_polydata.GetPointData().AddArray(array)
    
    # Copier les attributs de cellules (cell data) du polydata original
    original_cell_data = polydata.GetCellData()
    for i in range(original_cell_data.GetNumberOfArrays()):
        array = original_cell_data.GetArray(i)
        transformed_polydata.GetCellData().AddArray(array)
    
    # Debug: afficher quelques points après transformation
    if transformed_polydata.GetNumberOfPoints() > 0:
        sample_points_after = []
        for i in range(min(5, transformed_polydata.GetNumberOfPoints())):
            pt = transformed_polydata.GetPoint(i)
            sample_points_after.append(pt)
        print(f"Sample points after transformation: {sample_points_after}")

    # Enregistrer le polydata transformé dans un nouveau fichier VTK
    transformed_vtk_path = os.path.join(
        tempfile.gettempdir(), f"transformed_polydata{uuid.uuid4()}.vtk"
    )
    
    # Utiliser vtkPolyDataWriter pour l'écriture ASCII standard
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(transformed_vtk_path)
    writer.SetInputData(transformed_polydata)
    writer.SetFileTypeToASCII()  # Utiliser le format ASCII pour une meilleure compatibilité
    writer.Write()

    
    print(f"Transformed polydata saved to: {transformed_vtk_path}")
    return transformed_vtk_path



def add_mcm_to_tracts(tracts, mcm_file, reference=None, full=False,**kwargs):
    """
    Add MCM information to tracts.

    Parameters
    ----------
    tracts : BIDSFile
        The tracts to which MCM information will be added
    mcm_file : BIDSFile
        The MCM file containing the compartment information (.mcmx)
    reference : BIDSFile, optional
        Reference image for tractogram conversion if needed.
    kwargs : dict
        Additional arguments for run_cli_command.
    """
    print("Adding MCM to tracts...")
    inputs = {"tracts": tracts, "mcm_file": mcm_file}
    if reference:
        # Ensure reference is available if needed by commands run by run_cli_command
        inputs["reference"] = reference

    tracts_extension = os.path.basename(tracts.path).split('.')[-1].lower()

    temp_dir_for_conversion = None  # To store path to temp directory if created
    converted_input_vtk_path = tracts.path

    mcm_obj = MCMFile(mcm_file.path)


    if tracts_extension != 'vtk':
        if reference is None:
            raise ValueError(
                "Reference image is required for tractogram conversion.")

        temp_dir_for_conversion = tempfile.mkdtemp(suffix='tracts_conversion')

        print("Temporary directory for conversion:", temp_dir_for_conversion, os.path.exists(temp_dir_for_conversion))
        os.chdir(temp_dir_for_conversion)
        
        base_name = os.path.basename(tracts.path)
        vtk_name = os.path.splitext(base_name)[0] + '.vtk'
        converted_input_vtk_path = opj(temp_dir_for_conversion, vtk_name)

        mcm_weights = mcm_obj.get_weights()
        print("MCM Weights file:", mcm_weights)

        convert_tractogram(tracts.path, converted_input_vtk_path, reference.path, flip=False)
        print('coucou')
        # call(cmd)


    # Output filename for animaTracksMCMPropertiesExtraction, relative to its execution directory
    output_vtk_filename_in_tmp = "tracts_mcm_out.vtk"

    command_args = [
        # Path to VTK (original or temp converted)
        "-i", converted_input_vtk_path,
        "-m", mcm_obj.mcmfile,          # Path to .mcm file
        "-o", output_vtk_filename_in_tmp
    ]

    base_entities = tracts.get_full_entities()

    # Define the expected VTK output pattern for run_cli_command
    output_pattern = {output_vtk_filename_in_tmp:
                      upt_dict(base_entities, {
                          "model": "MCM", 'extension': "vtk", 'desc':base_entities.get('desc',None) if not full else 'full'})}
    

    # Run the anima command
    cli_results = run_cli_command(
        "animaTracksMCMPropertiesExtraction" if not full else "animaAddMCMToTracts",
        inputs,
        output_pattern,
        tracts,  # Base BIDSFile for deriving output metadata
        command_args=command_args,
        **kwargs
    )

    # # Cleanup the temporary directory used for VTK conversion
    # if temp_dir_for_conversion:
    #     shutil.rmtree(temp_dir_for_conversion)

    if not cli_results:
        print("Error: animaTracksMCMPropertiesExtraction failed or produced no output.")
        return {}

    # Get the absolute path of the generated VTK file from cli_results
    # Assuming cli_results has one entry for the VTK file
    generated_vtk_path = next(iter(cli_results.keys()))
    # # Prepare the final result dictionary, starting with the VTK file result
    final_results = cli_results.copy()

    # # Convert the output VTK to CSVs
    # # Base path for CSV names is the VTK path without extension
    # csv_output_base_path = os.path.splitext(generated_vtk_path)[0]

    # csv_files_dict = convert_vtk_to_csv(
    #     generated_vtk_path, csv_output_base_path)

    # # Add CSV files to the result dictionary
    # # Use original tracts for base entities
    # base_entities_for_csv = tracts.get_entities()

    # for content_key, csv_path_val in csv_files_dict.items():
    #     final_results[csv_path_val] = upt_dict(
    #         base_entities_for_csv.copy(),
    #         model="MCM",
    #         extension="csv",
    #         desc=content_key  # Store array name or 'fibers' in 'desc' field
    #     )

    # Generate scalar maps
    if reference and not full:
        output_directory_for_nifti = os.path.dirname(generated_vtk_path)
        # Prefix for NIfTI files, e.g., "tracts_mcm_out" from "tracts_mcm_out.vtk"
        nifti_file_prefix = os.path.splitext(
            os.path.basename(generated_vtk_path))[0]

        generated_nifti_paths = generate_scalar_map(  # Calling the new multi-output function
            vtk_file_path=generated_vtk_path,
            reference_nifti_path=reference.path,
            output_dir=output_directory_for_nifti,
            output_file_prefix=nifti_file_prefix
        )

        # Remove the pipeline
        print("Base entities for NIfTI:", base_entities)
        for nifti_path in generated_nifti_paths:
            # e.g., "tracts_mcm_out_FA.nii.gz"
            filename_base = os.path.basename(nifti_path)

            # Extract array name from filename for the 'desc' field
            temp_name = filename_base
            if temp_name.endswith(".nii.gz"):
                temp_name = temp_name[:-len(".nii.gz")]
            # temp_name is now, e.g., "tracts_mcm_out_FA"

            array_desc_name = "unknown_array"
            if temp_name.startswith(nifti_file_prefix + "_"):
                array_desc_name = temp_name[len(
                    nifti_file_prefix) + 1:]

            final_results[nifti_path] = upt_dict(
                base_entities.copy(),
                model="MCM",
                extension="nii.gz",
                metric=f"{array_desc_name}"
            )

    pprint(final_results)
    return final_results
    
def convert_transform_matrix_to_nibabel(transform_matrix):
    """
    Convert a transformation matrix from SimpleITK to nibabel format.
    SimpleITK uses LPS (Left-Posterior-Superior) coordinate system,
    while nibabel uses RAS (Right-Anterior-Superior).
    """
    # Create the coordinate system conversion matrices
    lps_to_ras = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Convert the matrix
    # First convert from LPS to RAS, then apply inverse for tract coordinates
    ras_matrix = lps_to_ras @ transform_matrix @ np.linalg.inv(lps_to_ras)
    
    return np.linalg.inv(ras_matrix)

def invert_affine_transformation(transformation_matrix):
    """
    Inverse une matrice de transformation affine 4x4 de manière correcte.
    
    Pour une matrice de la forme:
    [R t]
    [0 1]
    
    L'inverse est:
    [R^-1  -R^-1*t]
    [0     1      ]
    
    Parameters
    ----------
    transformation_matrix : ndarray
        Matrice de transformation affine 4x4
        
    Returns
    -------
    ndarray
        Matrice inverse 4x4
    """
    # Extraire la rotation et la translation
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    
    # Calculer l'inverse de la rotation
    R_inv = np.linalg.inv(R)
    
    # Calculer l'inverse de la translation
    t_inv = -R_inv @ t
    
    # Construire la matrice inverse
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = t_inv
    
    return inverse_matrix

def process_subject(subject_id, dataset_path, output_base_path, atlas_dir):
    """
    Process a single subject - function for multiprocessing
    
    Parameters
    ----------
    subject_id : str
        Subject ID to process
    dataset_path : str
        Path to the ActiDep dataset
    output_base_path : str
        Base output path for results
    atlas_dir : str
        Path to the atlas directory
    """
    try:
        dataset = Dataset(dataset_path)
        subject = dataset.get_subject(subject_id)
        print(f"\nProcessing subject: {subject_id}")
        
        # Get all MCM VTK files for this subject to identify available bundles
        all_mcm_vtk_files = subject.get(
            desc='cleaned',
            model='MCM',
            suffix='tracto',
            extension='vtk'
        )
        
        if not all_mcm_vtk_files:
            print(f"  No MCM VTK files found for subject {subject_id}")
            return f"No MCM VTK files found for subject {subject_id}"
        
        # Extract unique bundle names from the files
        bundles = set()
        for file in all_mcm_vtk_files:
            if hasattr(file, 'entities') and 'bundle' in file.entities:
                bundles.add(file.entities['bundle'])
        
        if not bundles:
            print(f"  No bundle information found for subject {subject_id}")
            return f"No bundle information found for subject {subject_id}"
        
        bundles = [b for b in bundles if any(x in b for x in ['CSTright'])]
        print(f"  Found bundles: {list(bundles)}")


        
        # Get the FA reference file for this subject (same for all bundles)
        fa_files = subject.get(
            metric='FA',
            model='DTI',
            suffix='dwi',
            extension='nii.gz'
        )
        
        if not fa_files:
            print(f"  No FA file found for subject {subject_id}")
            return f"No FA file found for subject {subject_id}"
            
        fa_nii = fa_files[0].path
        print(f"  FA reference: {fa_nii}")

        transformation_matrix = subject.get(**{'pipeline':'tractometry', 
                                                   'from':'subject',
                                                   'to':'HCP',
                                                   'suffix':'xfm',
                                                   'extension':'mat'})
        
        #Load the transformation matrix
        if len(transformation_matrix)==1:
            transformation_matrix_path = transformation_matrix[0].path
            print(f"  Transformation matrix: {transformation_matrix_path}")
            try:
                # Chargement avec SimpleITK et reconstruction correcte de la matrice
                transform = sitk.ReadTransform(transformation_matrix_path)
                
                # Récupérer le centre de rotation de la transformation
                transform_center = get_transform_center(transform)
                print(f"  Transform center: {transform_center}")
                
                transformation_matrix = np.eye(4)
                
                # ANTs stocke les paramètres dans un ordre spécifique
                # Pour une transformation affine, les paramètres sont :
                # [Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz, Tx, Ty, Tz]
                params = transform.GetParameters()
                print(f"  Transform parameters: {params}")
                print(f"  Number of parameters: {len(params)}")
                
                if len(params) >= 12:  # Transformation affine
                    # Reconstruction de la matrice de rotation (row-major order)
                    rotation = np.array(params[:9]).reshape(3, 3)
                    translation = np.array(params[9:12])
                    
                    print(f"  Rotation matrix:\n{rotation}")
                    print(f"  Translation vector: {translation}")
                    
                    # Construction de la matrice de transformation
                    transformation_matrix[:3, :3] = rotation
                    transformation_matrix[:3, 3] = translation
                    
                    print(f"  Transformation matrix (ANTs convention, HCP->subject):\n{transformation_matrix}")
                    
                    # Inversion pour obtenir subject->HCP
                    transformation_matrix = invert_affine_transformation(transformation_matrix)
                    print(f"  Inverted transformation matrix (subject->HCP):\n{transformation_matrix}")
                    
                    # Stocker le centre pour utilisation ultérieure
                    # Utiliser le centre de l'image de référence si le centre de la transformation n'est pas disponible
                    if np.allclose(transform_center, [0, 0, 0]):
                        print("  Transform center is at origin, will use reference image center instead")
                        transformation_center = None  # Sera calculé plus tard avec l'image de référence
                    else:
                        transformation_center = transform_center
                    
                else:
                    print(f"  Unexpected number of parameters: {len(params)}")
                    transformation_matrix = np.eye(4)
                

                
            except ImportError:
                print("  SimpleITK is not available, using numpy load instead")
                transformation_matrix = np.loadtxt(transformation_matrix_path)


            except Exception as e:
                print(f"  Erreur lors du chargement de la matrice de transformation: {e}")
                print("  Utilisation d'une matrice identité à la place")
                transformation_matrix = np.eye(4)
                sys.exit(1)

        else:
            print("  Aucune matrice de transformation trouvée, utilisation d'une matrice identité")
            transformation_matrix = np.eye(4)
            transformation_center = np.array([0.0, 0.0, 0.0])
            transformation_matrix = np.eye(4)
        
        # Process each bundle
        # bundles = [b for b in bundles if b == 'CSTright']  # Filter for ATRright only
        bundle_results = []
        for bundle in bundles:
            print(f"    Processing bundle: {bundle}")
            
            # Get the MCM VTK file for this specific bundle
            mcm_vtk_files = subject.get(
                bundle=bundle,
                desc='cleaned',
                model='MCM',
                suffix='tracto',
                extension='vtk'
            )
            
            if not mcm_vtk_files:
                print(f"      No MCM VTK file found for bundle {bundle}")
                continue
            
            mcm_vtk = mcm_vtk_files[0].path
            print(f"      MCM VTK: {mcm_vtk}")
            
            # Create output directory for this subject and bundle
            # output_path = f'{output_base_path}/sub-{subject_id}/{bundle}'
            output_path=f'/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/tractometry/sub-{subject_id}/tracto/sub-{subject_id}_bundle-{bundle}_desc-centroid_model-MCM_tracto.vtk'            
            # Generate clustered points version
            # cluster_output_path = output_path.replace('_desc-centroid_', '_desc-clustered_')
            # cluster_vtk_path = cluster_points_by_distance_from_vtk(
            #     vtk_file_path=mcm_vtk,
            #     output_path=cluster_output_path,
            #     distance_threshold=5.0,
            #     min_cluster_size=3,
            #     transformation_matrix=transformation_matrix,
            #     transformation_center=transformation_center
            # )
            
            # if cluster_vtk_path:
            #     print(f"      ✓ Clustered points VTK saved to: {cluster_vtk_path}")
            #     bundle_results.append(f"✓ {bundle} (clustered)")
            
            # Generate central line
            output_vtk_path = project_to_central_line_from_vtk(
                vtk_file_path=mcm_vtk,
                reference_nifti_path=fa_nii,
                output_path=output_path,
                num_points_central_line=10,
                transformation_matrix=transformation_matrix,
                transformation_center=transformation_center
            )

            if output_vtk_path:
                print(f"      ✓ Central line VTK saved to: {output_vtk_path}")
                bundle_results.append(f"✓ {bundle}")
            else:
                print(f"      ✗ Failed to generate central line VTK for bundle {bundle}")
                bundle_results.append(f"✗ {bundle}")

            
        
        return f"Subject {subject_id}: {', '.join(bundle_results)}"
        
    except Exception as e:
        error_msg = f"Error processing subject {subject_id}: {str(e)}"
        full_traceback = traceback.format_exc()
        print(f"  ✗ {error_msg}")
        print(f"  Full traceback:\n{full_traceback}")
        return error_msg

def process_subject_apply_transform(subject_id, dataset_path, output_base_path, atlas_dir,overwrite=True):
    """
    Process a single subject and apply the transformation matrix to the MCM VTK files.
    
    Parameters
    ----------
    subject_id : str
        Subject ID to process
    dataset_path : str
        Path to the ActiDep dataset
    output_base_path : str
        Base output path for results
    atlas_dir : str
        Path to the atlas directory
    """
    try:
        dataset = Dataset(dataset_path)
        subject = dataset.get_subject(subject_id)
        print(f"\nProcessing subject: {subject_id}")
        
        # Get all MCM VTK files for this subject to identify available bundles
        all_mcm_vtk_files = subject.get(
            pipeline='mcm_tensors_staniz',
            model='MCM',
            suffix='tracto',
            extension='vtk'
        )


    
        if not all_mcm_vtk_files:
            print(f"  No MCM VTK files found for subject {subject_id}")
            return f"No MCM VTK files found for subject {subject_id}"
        
        # Extract unique bundle names from the files
        bundles = set()
        for file in all_mcm_vtk_files:
            if hasattr(file, 'entities') and 'bundle' in file.entities:
                bundles.add(file.entities['bundle'])
        
        if not bundles:
            print(f"  No bundle information found for subject {subject_id}")
            return f"No bundle information found for subject {subject_id}"
        
        print(f"  Found bundles: {list(bundles)}")

        # Get the FA reference file for this subject (same for all bundles)
        fa_files = subject.get(
            metric='FA',
            model='DTI',
            suffix='dwi',
            extension='nii.gz'
        )
        
        if not fa_files:
            print(f"  No FA file found for subject {subject_id}")
            return f"No FA file found for subject {subject_id}"
            
        fa_nii = fa_files[0].path
        print(f"  FA reference: {fa_nii}")

        transformation_matrix = subject.get(**{'pipeline':'tractometry', 
                                                   'from':'subject',
                                                   'to':'HCP',
                                                   'suffix':'xfm',
                                                   'extension':'mat'})
        
        #Load the transformation matrix
        if len(transformation_matrix)==1:
            transformation_matrix_path = transformation_matrix[0].path
            print(f"  Transformation matrix: {transformation_matrix_path}")
            try:
                # Chargement avec SimpleITK et reconstruction correcte de la matrice
                transform = sitk.ReadTransform(transformation_matrix_path)
                
                # Récupérer le centre de rotation de la transformation
                transform_center = get_transform_center(transform)
                print(f"  Transform center: {transform_center}")
                
                transformation_matrix = np.eye(4)
                
                # ANTs stocke les paramètres dans un ordre spécifique
                # Pour une transformation affine, les paramètres sont :
                # [Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz, Tx, Ty, Tz]
                params = transform.GetParameters()
                print(f"  Transform parameters: {params}")
                print(f"  Number of parameters: {len(params)}")
                
                if len(params) >= 12:  # Transformation affine
                    # Reconstruction de la matrice de rotation (row-major order)
                    rotation = np.array(params[:9]).reshape(3, 3)
                    translation = np.array(params[9:12])
                    
                    print(f"  Rotation matrix:\n{rotation}")
                    print(f"  Translation vector: {translation}")
                    
                    # Construction de la matrice de transformation
                    transformation_matrix[:3, :3] = rotation
                    transformation_matrix[:3, 3] = translation
                    
                    print(f"  Transformation matrix (ANTs convention, HCP->subject):\n{transformation_matrix}")
                    # Inversion pour obtenir subject->HCP
                    transformation_matrix = invert_affine_transformation(transformation_matrix)
                    print(f"  Inverted transformation matrix (subject->HCP): {transformation_matrix}")
                    # Stocker le centre pour utilisation ultérieure
                    # Utiliser le centre de l'image de référence si le centre de la transformation n'est
                    # pas disponible
                    if np.allclose(transform_center, [0, 0, 0]):
                        print("  Transform center is at origin, will use reference image center instead")
                        transformation_center = None
                    else:
                        transformation_center = transform_center
                else:
                    print(f"  Unexpected number of parameters: {len(params)}")
                    transformation_matrix = np.eye(4)
            except ImportError:
                print("  SimpleITK is not available, using numpy load instead")
                transformation_matrix = np.loadtxt(transformation_matrix_path)
            except Exception as e:
                print(f"  Erreur lors du chargement de la matrice de transformation: {e}")
                print("  Utilisation d'une matrice identité à la place")
                transformation_matrix = np.eye(4)
                sys.exit(1)
        else:
            print("  Aucune matrice de transformation trouvée, utilisation d'une matrice identité")
            transformation_matrix = np.eye(4)
            transformation_center = np.array([0.0, 0.0, 0.0])
        # Process each bundle
        already_done = subject.get(
            model='MCM',
            suffix='tracto',
            space='HCP',
            extension='vtk'
        )
        already_done = [f.get_full_entities()['bundle'] for f in already_done]
        if overwrite:
            print("  Overwrite is True, reprocessing all bundles")
            already_done = []
        print(f"  Already processed bundles: {already_done}")
        bundles = [b for b in bundles if b not in already_done]  # Filter out already processed bundles

        print(f"  Processing bundles: {list(bundles)}")
        bundle_results = []
        for bundle in bundles:
            print(f"    Processing bundle: {bundle}")
            
            # Get the MCM VTK file for this specific bundle
            mcm_vtk_files = subject.get(
                bundle=bundle,
                pipeline='mcm_tensors_staniz',
                model='MCM',
                suffix='tracto',
                extension='vtk'
            )
            
            if not mcm_vtk_files:
                print(f"      No MCM VTK file found for bundle {bundle}")
                continue
            
            mcm_vtk = mcm_vtk_files[0].path
            print(f"      MCM VTK: {mcm_vtk}")
            
            # Apply transformation to the MCM VTK file
            transformed_vtk_path = apply_transformation_to_vtk(
                vtk_file_path=mcm_vtk,
                reference_nifti_path=fa_nii,
                transformation_matrix=transformation_matrix,
                transformation_center=transformation_center
            )
            entitites= mcm_vtk_files[0].get_full_entities()
            print(transformed_vtk_path)
            
            
            final_save_path=copy_from_dict(subject,{transformed_vtk_path: entitites},pipeline='mcm_to_hcp_space',space='HCP',remove_after_copy=False)

            #Remove the transformed_vtk_path file
            shutil.rmtree(transformed_vtk_path, ignore_errors=True)
            print(f"      ✓ Transformed VTK saved to: {final_save_path}")
            bundle_results.append(f"✓ {bundle}")
        print(f"Subject {subject_id} processed successfully: {', '.join(bundle_results)}")
        return f"Subject {subject_id}: {', '.join(bundle_results)}"
    except Exception as e:
        error_msg = f"Error processing subject {subject_id}: {str(e)}"
        full_traceback = traceback.format_exc()
        print(f"  ✗ {error_msg}")
        print(f"  Full traceback:\n{full_traceback}")
        return error_msg
        

                


def extract_vtk_to_csv(vtk_file_paths, output_base_path, bundle_names=None, reference_vtk_file_paths=None):
    """
    Extracts scalar values from multiple VTK files and saves them as CSV files with bundle names as columns.

    Parameters
    ----------
    vtk_file_paths : list
        List of paths to the input VTK files.
    output_base_path : str
        Base path for saving the output CSV files.
    bundle_names : list, optional
        List of bundle names corresponding to each VTK file. If None, uses file basenames.
    reference_vtk_file_paths : list, optional
        List of paths to reference VTK files to match orientation of streamlines.
        Should have the same length as vtk_file_paths if provided.
    
    Returns
    -------
    dict
        Dictionary mapping array names to their corresponding CSV file paths.
    """
    bundle_name_mapping = get_HCP_bundle_names()
    if bundle_names is None:
        bundle_names = bundle_name_mapping.keys()
        #Order by appearance in vtk_file_paths
        bundle_names = [x.split('bundle-')[-1].split('_')[0] for x in vtk_file_paths]
    
    # if len(vtk_file_paths) != len(bundle_names):
    #     raise ValueError("Number of VTK files must match number of bundle names")
    
    # # Valider les fichiers de référence si fournis
    # if reference_vtk_file_paths is not None:
    #     if len(reference_vtk_file_paths) != len(vtk_file_paths):
    #         raise ValueError("Number of reference VTK files must match number of VTK files")
    
    name_mapping = {
        "Fractional anisotropy": "FA",
        "Mean diffusivity": "MD",
        "Parallel diffusivity": "AD",
        "Perpendicular diffusivity": "RD",
        "Isotropic restricted water fraction": "IRF",
        "Free water fraction": "IFW"
    }
    
    # Dictionary to store data for each metric
    metric_data = {}
    csv_files = {}
    
    # Process each VTK file
    for idx, (vtk_path, bundle_name) in enumerate(zip(vtk_file_paths, bundle_names)):
        if not os.path.exists(vtk_path):
            print(f"Warning: VTK file does not exist: {vtk_path}")
            continue
        
        # Déterminer si l'orientation doit être inversée
        should_flip = False
        if reference_vtk_file_paths is not None:
            reference_path = [f for f in reference_vtk_file_paths if bundle_name in f][0]
            print(f"Reference path for bundle {bundle_name}: {reference_path}")
            print(f"Reference VTK file paths: {vtk_file_path}")
            if os.path.exists(reference_path):
                should_flip = determine_streamline_orientation(vtk_path, reference_path)
                if should_flip:
                    print(f"Flipping orientation for bundle {bundle_name}")
            else:
                print(f"Warning: Reference VTK file does not exist: {reference_path}")
            
        # Read the VTK file
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_path)
        reader.Update()
        
        polydata = reader.GetOutput()
        point_data = polydata.GetPointData()
        
        # Extract arrays from this VTK file
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            
            if array_name not in name_mapping:
                continue
                
            metric_name = name_mapping[array_name]
            
            # Convert VTK array to numpy array
            np_array = vtk_to_numpy(array)
            
            # Appliquer le flip si nécessaire
            if should_flip:
                np_array = flip_array_data(np_array)
            
            # Initialize metric data dictionary if needed
            if metric_name not in metric_data:
                metric_data[metric_name] = {}
            
            # Store data with bundle name as key
            metric_data[metric_name][bundle_name] = np_array
    
    # Create CSV files for each metric
    os.makedirs(output_base_path, exist_ok=True)
    
    for metric_name, bundle_data in metric_data.items():
        # Create DataFrame with bundle names as columns
        df = pd.DataFrame(bundle_data)
        
        csv_file_path = os.path.join(output_base_path, f"{metric_name}.csv")
        df.to_csv(csv_file_path, index=False,sep=';')
        
        csv_files[metric_name] = csv_file_path
        print(f"Saved {metric_name} data to: {csv_file_path}")
    
    return csv_files

def determine_streamline_orientation(vtk_file_path, reference_vtk_file_path=None):
    """
    Détermine l'orientation d'une streamline par rapport à une référence.
    
    Parameters
    ----------
    vtk_file_path : str
        Chemin vers le fichier VTK à analyser
    reference_vtk_file_path : str, optional
        Chemin vers le fichier VTK de référence
        
    Returns
    -------
    bool
        True si l'orientation doit être inversée, False sinon
    """
    if reference_vtk_file_path is None:
        return False
    
    # Si c'est le même fichier, pas besoin de retourner
    if os.path.abspath(vtk_file_path) == os.path.abspath(reference_vtk_file_path):
        return False
    
    try:
        # Charger le fichier VTK principal
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_file_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        # Charger le fichier VTK de référence
        ref_reader = vtk.vtkPolyDataReader()
        ref_reader.SetFileName(reference_vtk_file_path)
        ref_reader.Update()
        ref_polydata = ref_reader.GetOutput()
        
        if polydata.GetNumberOfPoints() == 0 or ref_polydata.GetNumberOfPoints() == 0:
            return False
        
        # Extraire les points de la première ligne de chaque fichier
        lines = polydata.GetLines()
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        
        if not lines.GetNextCell(id_list) or id_list.GetNumberOfIds() < 2:
            return False
            
        # Premier et dernier point du fichier principal
        first_point = np.array(polydata.GetPoint(id_list.GetId(0)))
        last_point = np.array(polydata.GetPoint(id_list.GetId(id_list.GetNumberOfIds() - 1)))
        
        # Extraire les points de la première ligne du fichier de référence
        ref_lines = ref_polydata.GetLines()
        ref_lines.InitTraversal()
        ref_id_list = vtk.vtkIdList();
        
        if not ref_lines.GetNextCell(ref_id_list) or ref_id_list.GetNumberOfIds() < 2:
            return False
            
        # Premier et dernier point du fichier de référence
        ref_first_point = np.array(ref_polydata.GetPoint(ref_id_list.GetId(0)))
        ref_last_point = np.array(ref_polydata.GetPoint(ref_id_list.GetId(ref_id_list.GetNumberOfIds() - 1)))
        
        # Calculer les distances entre les extrémités
        dist_first_to_first = np.linalg.norm(first_point - ref_first_point)
        dist_first_to_last = np.linalg.norm(first_point - ref_last_point)
        dist_last_to_first = np.linalg.norm(last_point - ref_first_point)
        dist_last_to_last = np.linalg.norm(last_point - ref_last_point)
        
        # Déterminer l'orientation
        # Si first->first + last->last < first->last + last->first, alors même orientation
        same_orientation_score = dist_first_to_first + dist_last_to_last
        opposite_orientation_score = dist_first_to_last + dist_last_to_first
        
        # Retourner True si l'orientation doit être inversée
        return opposite_orientation_score < same_orientation_score
        
    except Exception as e:
        print(f"Error determining orientation for {vtk_file_path}: {e}")
        return False

def flip_array_data(data_array):
    """
    Inverse l'ordre des données dans un array.
    
    Parameters
    ----------
    data_array : ndarray
        Array de données à inverser
        
    Returns
    -------
    ndarray
        Array avec l'ordre inversé
    """
    return np.flip(data_array, axis=0)

def cluster_points_by_distance_from_vtk(vtk_file_path, output_path, distance_threshold=5.0, min_cluster_size=3, transformation_matrix=None, transformation_center=None):
    """
    Clusters points from different streamlines based on spatial proximity and creates
    a PolyData with cluster indices.

    Parameters
    ----------
    vtk_file_path : str
        Path to the input VTK PolyData file (.vtk).
    output_path : str
        Path where the output VTK PolyData file will be saved.
    distance_threshold : float, optional
        Maximum distance for points to be considered neighbors, by default 5.0.
    min_cluster_size : int, optional
        Minimum number of points required to form a cluster, by default 3.
    transformation_matrix : ndarray, optional
        Transformation matrix to apply to the points. Shape (4, 4).
    transformation_center : ndarray, optional
        Center of rotation for the transformation. Shape (3,). If None, uses (0, 0, 0).

    Returns
    -------
    str or None
        Path to the generated VTK file containing clustered points, or None if an error occurs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read the VTK PolyData
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    if not polydata or polydata.GetNumberOfLines() == 0:
        print(f"Error: No lines found in VTK file: {vtk_file_path}")
        return None

    # Extract all points and their streamline membership
    all_points = []
    point_to_streamline = []
    point_to_local_index = []
    
    lines = polydata.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    streamline_idx = 0
    
    while lines.GetNextCell(id_list):
        for local_idx in range(id_list.GetNumberOfIds()):
            point_id = id_list.GetId(local_idx)
            point = polydata.GetPoint(point_id)
            all_points.append(point)
            point_to_streamline.append(streamline_idx)
            point_to_local_index.append(local_idx)
        streamline_idx += 1

    all_points = np.array(all_points)
    point_to_streamline = np.array(point_to_streamline)
    point_to_local_index = np.array(point_to_local_index)
    
    print(f"Total points extracted: {len(all_points)}")
    print(f"Number of streamlines: {streamline_idx}")

    # Apply transformation if provided
    if transformation_matrix is not None:
        if transformation_center is None:
            transformation_center = np.array([0.0, 0.0, 0.0])
        
        print("Applying transformation to points...")
        all_points = apply_transformation_with_center(
            all_points, transformation_matrix, transformation_center
        )

    # Build cKDTree for efficient neighbor search
    print("Building spatial index...")
    tree = cKDTree(all_points)
    
    # Find clusters using distance-based grouping
    print(f"Finding clusters with distance threshold: {distance_threshold}")
    visited = np.zeros(len(all_points), dtype=bool)
    cluster_labels = np.full(len(all_points), -1, dtype=int)
    cluster_id = 0
    
    for point_idx in range(len(all_points)):
        if visited[point_idx]:
            continue
            
        # Find all neighbors within distance threshold
        neighbor_indices = tree.query_ball_point(all_points[point_idx], r=distance_threshold)
        
        if len(neighbor_indices) >= min_cluster_size:
            # Create new cluster
            for neighbor_idx in neighbor_indices:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    cluster_labels[neighbor_idx] = cluster_id
            cluster_id += 1
        else:
            # Mark as noise/outlier
            visited[point_idx] = True
            cluster_labels[point_idx] = -1

    print(f"Found {cluster_id} clusters")
    
    # Count points per cluster
    unique_clusters, cluster_counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique_clusters, cluster_counts))}")

    # Create new PolyData with clustered points
    clustered_polydata = vtk.vtkPolyData()
    clustered_points = vtk.vtkPoints()
    
    # Add all points to the new polydata
    for point in all_points:
        clustered_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    
    clustered_polydata.SetPoints(clustered_points)

    # Create vertices for visualization (each point as a vertex)
    vertices = vtk.vtkCellArray()
    for i in range(len(all_points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    clustered_polydata.SetVerts(vertices)

    # Add cluster indices as point data
    cluster_array = numpy_to_vtk(cluster_labels.astype(np.int32), deep=True)
    cluster_array.SetName("cluster_indices")
    clustered_polydata.GetPointData().AddArray(cluster_array)
    clustered_polydata.GetPointData().SetActiveScalars("cluster_indices")

    # Add streamline membership information
    streamline_array = numpy_to_vtk(point_to_streamline.astype(np.int32), deep=True)
    streamline_array.SetName("streamline_id")
    clustered_polydata.GetPointData().AddArray(streamline_array)

    # Add local index within streamline
    local_index_array = numpy_to_vtk(point_to_local_index.astype(np.int32), deep=True)
    local_index_array.SetName("local_index")
    clustered_polydata.GetPointData().AddArray(local_index_array)

    # Copy scalar arrays from original polydata if they exist
    original_point_data = polydata.GetPointData()
    for i in range(original_point_data.GetNumberOfArrays()):
        original_array = original_point_data.GetArray(i)
        array_name = original_array.GetName()
        
        if array_name and array_name != 'colors':
            print(f"Copying scalar array: {array_name}")
            
            # Create new array for clustered points
            new_array_data = []
            
            # For each point in all_points, find corresponding original point
            lines = polydata.GetLines()
            lines.InitTraversal()
            id_list = vtk.vtkIdList()
            current_streamline = 0
            
            while lines.GetNextCell(id_list):
                for local_idx in range(id_list.GetNumberOfIds()):
                    original_point_id = id_list.GetId(local_idx)
                    
                    # Get the scalar value for this original point
                    if original_array.GetNumberOfComponents() == 1:
                        value = original_array.GetValue(original_point_id)
                        new_array_data.append(value)
                    else:
                        value_tuple = original_array.GetTuple(original_point_id)
                        new_array_data.append(list(value_tuple))
                
                current_streamline += 1
            
            # Convert to numpy and add to polydata
            new_array_np = np.array(new_array_data, dtype=np.float32)
            new_vtk_array = numpy_to_vtk(new_array_np, deep=True)
            new_vtk_array.SetName(array_name)
            clustered_polydata.GetPointData().AddArray(new_vtk_array)

    # Save the clustered polydata to VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(clustered_polydata)
    writer.Write()
    
    print(f"Clustered points saved to: {output_path}")
    
    return output_path

def project_to_reference_space(streamlines, reference: BIDSFile) -> list:
    """
    Projects streamlines to the space of a reference image.

    Parameters
    ----------
    streamlines : list
        List of streamlines (each streamline is an array of shape (n_points, 3)).
    reference : BIDSFile
        An BIDSFile instance containing the reference image.

    Returns
    -------
    projected_streamlines : list
        List of streamlines projected to the reference image space.
    """
    ref_img = nib.load(reference.path)
    inv_affine = np.linalg.inv(ref_img.affine)

    projected_streamlines = []
    for sl in streamlines:
        projected_sl = []
        for point in sl:
            point_homogeneous = np.append(point, 1)
            voxel_coords = inv_affine @ point_homogeneous
            projected_sl.append(voxel_coords[:3])
        projected_streamlines.append(np.array(projected_sl))
    
    return projected_streamlines

if __name__ == "__main__":
    config, tools = set_config()
    

    #### Central line extraction example 
    ## Get all subjects in the dataset
    dataset_path = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids"
    # dataset_path = '/home/ndecaux/Data/dysdiago/'
    output_base_path = '/home/ndecaux/Code/Data/actidep_atlas/moved_mcm_projections'
    atlas_dir = '/home/ndecaux/Data/actidep_atlas'
    
    dataset = Dataset(dataset_path,restore=False)
    subjects = dataset.get_subjects()
    subjects = [s for s in subjects]# if '01001' in s]  # Filter for specific subject IDs, e.g., '01002'
    print(f"Processing {len(subjects)} subjects...")
    
    # Set up multiprocessing
    num_processes = 1#min(mp.cpu_count() - 1, len(subjects))  # Use all CPUs minus 1, or number of subjects if less
    print(f"Using {num_processes} processes for parallel processing")
    
    #If hostname == 'calcarine'
    if os.uname()[1] == 'calcarine':
        num_processes = 16

    # Create partial function with fixed parameters
    # process_func = partial(
    #     process_subject_apply_transform,
    #     dataset_path=dataset_path,
    #     output_base_path=output_base_path,
    #     atlas_dir=atlas_dir
    # )
    
    # Process subjects in parallel
    # with mp.Pool(processes=num_processes) as pool:
    #     results = pool.map(process_func, subjects)
    
    # Print summary
    # print("\n" + "="*50)
    # print("PROCESSING SUMMARY")
    # print("="*50)
    # for result in results:
    #     print(result)
    
    # print("\nProcessing complete!")

    vtk_in_hcp=dataset.get_global(bundle='CSTleft',
                             pipeline='mcm_to_hcp_space',
                             extension='vtk')
    
    for f in vtk_in_hcp:
        if f.subject != '03025':
            continue
        print(f.path)
        res_dict=generate_scalar_map(
            vtk_file_path=f,
            reference_nifti_path="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_fa.nii.gz",
            output_dir=f'/tmp/{f.subject}',
            )
    
        copy_from_dict(subject=dataset.get_subject(f.subject),
                          file_dict=res_dict, 
                          pipeline='mcm_to_hcp_space',
                          datatype='metric'
        )



    # generate_scalar_map(
    #     vtk_file_path=mcm_vtk,
    #     reference_nifti_path=fa_nii,
    #     output_dir='test_scalar_map',
    #     output_file_prefix="tracts_mcm_out"
    # )


    # ds = Dataset('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
    # first_sub = ds.get_subjects()[0]
    # print(f"First subject: {first_sub}")

    # first_sub_centroids = [f.path for f in ds.get(
    #     first_sub,
    #     model='MCM',
    #     desc='centroid',
    #     suffix='tracto',
    #     extension='vtk',
    #     pipeline='tractometry'
    # )]

    # print(first_sub_centroids)

    # for sub in ds.get_subjects():
    #     print(f"Processing subject: {sub}")
    #     sub_centroids = ds.get(sub,
    #         model='MCM',
    #         desc='centroid',
    #         suffix='tracto',
    #         extension='vtk',
    #         pipeline='tractometry'
    #     )
    #     if not sub_centroids:
    #         print(f"No centroid VTK files found for subject {sub}")
    #         continue
        
    #     csv_files=extract_vtk_to_csv(vtk_file_paths=[s.path for s in sub_centroids], 
    #                        output_base_path=os.path.join('/tmp/',sub),
    #                        reference_vtk_file_paths=first_sub_centroids)
        
    #     res_dict = {v: {'metric': k} for k, v in csv_files.items()}

    #     copy_from_dict(
    #         subject=ds.get_subject(sub),
    #         file_dict=res_dict,
    #         pipeline='tractometry',
    #         model='staniz',
    #         suffix='mean',
    #         datatype='metric'
    #     )


