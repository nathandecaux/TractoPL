from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa



#### Custom imports ####
from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config,get_HCP_bundle_names
from TractoPL.data.loader import Subject, parse_filename, BIDSFile, Dataset
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import dipy
from dipy.io.stateful_tractogram import Space, StatefulTractogram,Origin
from dipy.io.streamline import save_tractogram, load_tractogram
from time import process_time
import vtk
from dipy.tracking.streamline import transform_streamlines
from scipy.io import loadmat
import nibabel as nib
import pandas as pd
from dipy.io.streamline import load_tractogram, load_trk
import datetime
from dipy.tracking.streamline import orient_by_streamline

def _get_length_best_orig_peak(predicted_img, orig_img, x, y, z):
    predicted = predicted_img[x, y, z, :]       # 1 peak
    orig = [orig_img[x, y, z, 0:3], orig_img[x, y, z, 3:6], orig_img[x, y, z, 6:9]]     # 3 peaks

    angle1 = abs(np.dot(predicted, orig[0]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[0]) + 1e-7))
    angle2 = abs(np.dot(predicted, orig[1]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[1]) + 1e-7))
    angle3 = abs(np.dot(predicted, orig[2]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[2]) + 1e-7))

    argmax = np.argmax([angle1, angle2, angle3])
    best_peak_len = np.linalg.norm(orig[argmax])
    return best_peak_len

def reorient_streamlines(streamlines):
    """
    Reorient streamlines to have the same order of points, using dipy's orient_by_streamline.
    Parameters
    ----------
    streamlines : list of Streamlines
        List of streamlines to reorient.
    Returns
    -------
    list of Streamlines
        List of reoriented streamlines.
    """
    if len(streamlines) == 0:
        return streamlines
    # Use the longest streamline as the reference
    longest_streamline = max(streamlines, key=len)
    # Reorient all streamlines to match the longest streamline
    reoriented_streamlines = orient_by_streamline(streamlines, longest_streamline)
    return reoriented_streamlines


def orient_streamlines(streamlines):
    oriented = []
    for sl in streamlines:
        if len(sl) == 0:
            oriented.append(sl)
            continue
        if sl[0, 0] > sl[-1, 0]:
            oriented.append(sl[::-1])
        else:
            oriented.append(sl)
    return oriented

def evaluate_along_streamlines(scalar_img,
                               streamlines,
                               nr_points,
                               beginnings=None,
                               dilate=0,
                               predicted_peaks=None,
                               affine=None):
    # Runtime:
    # - default:                2.7s (test),    56s (all),      10s (test 4 bundles, 100 points)
    # - map_coordinate order 1: 1.9s (test),    26s (all),       6s (test 4 bundles, 100 points)
    # - map_coordinate order 3: 2.2s (test),    33s (all),
    # - values_from_volume:     2.5s (test),    43s (all),
    # - AFQ:                      ?s (test),     ?s (all),      85s  (test 4 bundles, 100 points)
    # => AFQ a lot slower than others

    streamlines = list(
        transform_streamlines(streamlines, np.linalg.inv(affine)))

    if beginnings is not None:
        streamlines = orient_streamlines(streamlines)
        # for i in range(dilate):
        #     beginnings = binary_dilation(beginnings)
        # beginnings = beginnings.astype(np.uint8)
        # streamlines = fiber_utils.orient_to_same_start_region(
        #     streamlines, beginnings)

    if predicted_peaks is not None:
        # scalar img can also be orig peaks
        best_orig_peaks = fiber_utils.get_best_original_peaks(
            predicted_peaks, scalar_img, peak_len_thr=0.00001)
        scalar_img = np.linalg.norm(best_orig_peaks, axis=-1)

    algorithm = "distance_map"  # equal_dist | distance_map | cutting_plane | afq

    if algorithm == "equal_dist":
        ### Sampling ###
        streamlines = fiber_utils.resample_fibers(streamlines,
                                                  nb_points=nr_points)
        values = map_coordinates(scalar_img, np.array(streamlines).T, order=1)
        ### Aggregation ###
        values_mean = np.array(values).mean(axis=1)
        values_std = np.array(values).std(axis=1)
        return values_mean, values_std

    if algorithm == "distance_map":  # cKDTree

        ### Sampling ###
        streamlines = fiber_utils.resample_fibers(streamlines,
                                                  nb_points=nr_points)
        values = map_coordinates(scalar_img, np.array(streamlines).T, order=1)

        ### Aggregating by cKDTree approach ###
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines)
        centroids = Streamlines(clusters.centroids)
        if len(centroids) > 1:
            print("WARNING: number clusters > 1 ({})".format(len(centroids)))
        _, segment_idxs = cKDTree(centroids.get_data(), 1,
                                  copy_data=True).query(streamlines,
                                                        k=1)  # (2000, 100)

        values_t = np.array(values).T  # (2000, 100)

        # If we want to take weighted mean like in AFQ:
        # weights = dsa.gaussian_weights(Streamlines(streamlines))
        # values_t = weights * values_t
        # return np.sum(values_t, 0), None

        results_dict = defaultdict(list)
        for idx, sl in enumerate(values_t):
            for jdx, seg in enumerate(sl):
                results_dict[segment_idxs[idx, jdx]].append(seg)

        if len(results_dict.keys()) < nr_points:
            print(
                "WARNING: found less than required points. Filling up with centroid values."
            )
            centroid_values = map_coordinates(scalar_img,
                                              np.array([centroids[0]]).T,
                                              order=1)
            for i in range(nr_points):
                if len(results_dict[i]) == 0:
                    results_dict[i].append(np.array(centroid_values).T[0, i])

        results_mean = []
        results_std = []
        for key in sorted(results_dict.keys()):
            value = results_dict[key]
            if len(value) > 0:
                results_mean.append(np.array(value).mean())
                results_std.append(np.array(value).std())
            else:
                print("WARNING: empty segment")
                results_mean.append(0)
                results_std.append(0)
        return results_mean, results_std

    elif algorithm == "cutting_plane":
        # This will resample all streamline to have equally distant points (resulting in a different number of points
        # in each streamline). Then the "middle" of the tract will be estimated taking the middle element of the
        # centroid (estimated with QuickBundles). Then each streamline the point closest to the "middle" will be
        # calculated and points will be indexed for each streamline starting from the middle. Then averaging across
        # all streamlines will be done by taking the mean for points with same indices.

        ### Sampling ###
        streamlines = fiber_utils.resample_to_same_distance(
            streamlines, max_nr_points=nr_points)
        # map_coordinates does not allow streamlines with different lengths -> use values_from_volume
        values = np.array(
            values_from_volume(scalar_img, streamlines, affine=np.eye(4))).T

        ### Aggregating by Cutting Plane approach ###
        # Resample to all fibers having same number of points -> needed for QuickBundles
        streamlines_resamp = fiber_utils.resample_fibers(streamlines,
                                                         nb_points=nr_points)
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines_resamp)
        centroids = Streamlines(clusters.centroids)

        # index of the middle cluster
        middle_idx = int(nr_points / 2)
        middle_point = centroids[0][middle_idx]
        # For each streamline get idx for the point which is closest to the middle
        segment_idxs = fiber_utils.get_idxs_of_closest_points(
            streamlines, middle_point)

        # Align along the middle and assign indices
        segment_idxs_eqlen = []
        base_idx = 1000  # use higher index to avoid negative numbers for area below middle
        for idx, sl in enumerate(streamlines):
            sl_middle_pos = segment_idxs[idx]
            before_elems = sl_middle_pos
            after_elems = len(sl) - sl_middle_pos
            # indices for one streamline e.g. [998, 999, 1000, 1001, 1002, 1003]; 1000 is middle
            r = range((base_idx - before_elems), (base_idx + after_elems))
            segment_idxs_eqlen.append(r)
        segment_idxs = segment_idxs_eqlen

        # Calcuate maximum number of indices to not result in more indices than nr_points.
        # (this could be case if one streamline is very off-center and therefore has a lot of points only on one
        # side. In this case the values too far out of this streamline will be cut off).
        max_idx = base_idx + int(nr_points / 2)
        min_idx = base_idx - int(nr_points / 2)

        # Group by segment indices
        results_dict = defaultdict(list)
        for idx, sl in enumerate(values):
            for jdx, seg in enumerate(sl):
                current_idx = segment_idxs[idx][jdx]
                if current_idx >= min_idx and current_idx < max_idx:
                    results_dict[current_idx].append(seg)

        # If values missing fill up with centroid values
        if len(results_dict.keys()) < nr_points:
            print(
                "WARNING: found less than required points. Filling up with centroid values."
            )
            centroid_sl = [centroids[0]]
            centroid_sl = np.array(centroid_sl).T
            centroid_values = map_coordinates(scalar_img, centroid_sl, order=1)
            for idx, seg_idx in enumerate(range(min_idx, max_idx)):
                if len(results_dict[seg_idx]) == 0:
                    results_dict[seg_idx].append(
                        np.array(centroid_values).T[0, idx])

        # Aggregate by mean
        results_mean = []
        results_std = []
        for key in sorted(results_dict.keys()):
            value = results_dict[key]
            if len(value) > 0:
                results_mean.append(np.array(value).mean())
                results_std.append(np.array(value).std())
            else:
                print("WARNING: empty segment")
                results_mean.append(0)
                results_std.append(0)
        return results_mean, results_std

    elif algorithm == "afq":
        ### sampling + aggregation ###
        streamlines = fiber_utils.resample_fibers(streamlines,
                                                  nb_points=nr_points)
        streamlines = Streamlines(streamlines)
        weights = dsa.gaussian_weights(streamlines)
        results_mean = dsa.afq_profile(scalar_img,
                                       streamlines,
                                       affine=np.eye(4),
                                       weights=weights)
        results_std = np.zeros(nr_points)
        return results_mean, results_std




def process_projection(tracto_dict, metric_dict, beginnings_dict={}, **kwargs):
    """
    Process the projection of the streamlines in the bundlesegmentation tractography.

    Parameters
    ----------
    tracto_dict : list of BIDSFile
        List of tractography files to process
    ref_img : BIDSFile
        Reference image for the projection
    endings : BIDSFile, optional
        Endings segmentation file. If None, no endings will be used.
    kwargs : dict
        Additional keyword arguments for processing
    """


    bundle_name_dict = get_HCP_bundle_names()


    if not "nr_points" in list(kwargs.keys()):
        nr_points = 100
    else:
        nr_points = kwargs['nr_points']


    res_list = []
    # Process each tractography file
    for bundle_name,tracto in tracto_dict.items():
        # Load the reference image
        ref_img = nib.load(metric_dict[bundle_name].path)
        affine = ref_img.affine
        beginnings = beginnings_dict.pop(bundle_name, None)
        scalar_img = ref_img.get_fdata()
        tracto_data = nib.streamlines.load(tracto.path).streamlines
        beginnings = nib.load(beginnings.path).get_fdata() if beginnings is not None else None
        
        # If the tractography is in RAS space, flip the affine to LPS
        # affine[0,:] = -affine[0, :]
        # affine[1,:] = -affine[1, :]
        mean, std = evaluate_along_streamlines(scalar_img,
                                               tracto_data,
                                               beginnings=beginnings,
                                               nr_points=nr_points,
                                               affine=affine,
                                               **kwargs)

        # Remove first and last segment as those tend to be more noisy
        mean = mean[1:-1]
        #Convert nan to 0
        mean = np.nan_to_num(mean, nan=0.0)
        std = std[1:-1]

        tractseg_bundle_name = bundle_name_dict[bundle_name]
        # Save the results
        entities= tracto.get_full_entities()
        res_dict = upt_dict(entities, tractseg_name=tractseg_bundle_name, mean=mean, std=std)
        res_list.append(res_dict)
    # Save the results to a CSV file
    res_df = pd.DataFrame(res_list)
    #Create temporary directory
    temp_dir = tempfile.mkdtemp()

    res_df.to_csv(os.path.join(temp_dir, 'projection.csv'), index=False)

    ref_entities = metric_dict[list(tracto_dict.keys())[0]].get_full_entities()
    del ref_entities['bundle']

    csv_path = os.path.join(temp_dir, 'mean.csv')

    # Organiser les données pour avoir une colonne par tractseg_name
    mean_data = {}
    for tract_data in res_list:
        tractseg_name = tract_data['tractseg_name']
        mean_values = tract_data['mean']

        # S'assurer que toutes les moyennes ont la même longueur
        if not mean_data:  # Premier faisceau, initialiser la structure
            # Créer un tableau de lignes (correspond aux points le long du faisceau)
            rows = np.zeros((len(mean_values), 0))
            mean_data = {
                'data': rows,
                'columns': []
            }

        # Ajouter les données pour ce faisceau comme nouvelle colonne
        mean_data['data'] = np.column_stack((mean_data['data'], mean_values))
        mean_data['columns'].append(tractseg_name)

    # Enregistrer le CSV avec une colonne par faisceau
    header = ";".join(mean_data['columns'])
    np.savetxt(csv_path, mean_data['data'], delimiter=";", header=header, comments="")

    out_dict = {
        os.path.join(temp_dir, 'projection.csv'):
        upt_dict(
            ref_entities, {
                'suffix': 'projection',
                'extension': 'csv',
                'datatype': kwargs.get('datatype', 'projection')
            }),
        os.path.join(temp_dir, 'mean.csv'):
        upt_dict(
            ref_entities, {
                'suffix': 'mean',
                'extension': 'csv',
                'datatype': kwargs.get('datatype', 'projection')
            })
    }

    return out_dict


def process_tractseg_analysis(
        subjects_txt,
        dataset_path="/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids",
        with_3dplot=False,
        metric='FA',
        pipeline='tractometry'):
    """
    Process the tractseg analysis for the given subjects.

    Parameters
    ----------
    subjects_txt : str
        Path to the text file containing the list of subjects.
    """

    bundle_name_dict = get_HCP_bundle_names()

    available_bundles = [
        'AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CC_1', 'CC_2', 'CC_3',
        'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left',
        'CST_right', 'FPT_left', 'FPT_right', 'ICP_left', 'ICP_right',
        'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'OR_left',
        'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
        'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right',
        'SLF_III_left', 'SLF_III_right', 'STR_left', 'STR_right', 'UF_left',
        'UF_right', 'T_PREM_left', 'T_PREM_right', 'T_PAR_left', 'T_PAR_right',
        'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right',
        'ST_PREM_left', 'ST_PREM_right'
    ]

    # Load the subjects
    with open(subjects_txt, 'r') as f:
        subjects = [line.strip() for line in f.readlines() if line.strip()]
    tracto_path = subjects.pop(0)

    if 'bundle' in subjects[0]:
        bundle_list = subjects.pop(0)
        bundle_header, no_header_bundle_list = bundle_list.split('=')
        curated_list = [
            b.strip() for b in no_header_bundle_list.split()
            if b.strip() in available_bundles
        ]

        bundle_list = f"{bundle_header}={' '.join(curated_list)}"

        print(
            f"Excluded bundles not in available bundles: {set(no_header_bundle_list.split()) - set(available_bundles)}"
        )
    else:
        bundle_list = ''

    if 'plot_3D' in subjects[0]:
        plot_3D = subjects.pop(0)
        plot_3D_sub = plot_3D.split('/')[1]
        print(f"plot_3D_sub: {plot_3D_sub}")
    else:
        plot_3D = ''
        plot_3D_sub = ''
        with_3dplot = False
    # Parse subject information (format: "subject_id group")
    subjects_data = [line.split(' ', 1) for line in subjects]
    subjects_df = pd.DataFrame(subjects_data, columns=['subject', 'group'])

    #Change group by target if present in the original file
    if 'target' in subjects_df.columns:
        subjects_df['group'] = subjects_df['target']

    

    # Skip the header row if present
    if len(subjects_df) > 0:
        subjects_df = subjects_df.iloc[1:]

    ds = Dataset(dataset_path)
    #Créer un dossier temporaire pour stocker les résultats
    temp_dir = tempfile.mkdtemp()
    tracto_path = tracto_path.replace('TEMPDIR', temp_dir)
    print(f"Temporary directory created at: {temp_dir}")

    if with_3dplot:
        plot_3D = plot_3D.replace('TEMPDIR', temp_dir)

        print(f"3D plot directory: {plot_3D}")

    for sub in subjects_df['subject']:
        print(f"Processing subject: {sub}")
        sub_id = sub.split('-')[-1]
        files = ds.get(sub_id,
                       pipeline=pipeline,
                       suffix='tractsegmean',
                       metric=metric,
                       extension='csv')
        if len(files) == 0:
            print(f"No files found for subject {sub_id}. Skipping.")
            continue
        else:
            tracto_metrics = files[0]
        #Copy to <temp_dir>/<sub>/Tractometry.csv
        sub_temp_dir = os.path.join(temp_dir, sub)
        os.makedirs(sub_temp_dir, exist_ok=True)
        shutil.copy(tracto_metrics.path,
                    os.path.join(sub_temp_dir, 'Tractometry.csv'))

        if with_3dplot and plot_3D_sub == sub:
            bundles = ds.get(sub_id,
                             pipeline='bundle_seg',
                             suffix='tracto',
                             extension='tck')
            brain_mask = ds.get(sub_id,
                                pipeline='preprocessing',
                                suffix='mask',
                                label='brain')[0]

            if len(bundles) == 0:
                print(
                    f"No bundles found for subject {sub_id}. Skipping 3D plot."
                )
                with_3dplot = False
                continue

            bundles = [(b.path,
                        bundle_name_dict[parse_filename(b.path).get('bundle')])
                       for b in bundles]
            for bundle, bundle_name in bundles:
                bundle_symlink = pathlib.Path(plot_3D.split(
                    '=')[-1]) / 'TOM_trackings' / f"{bundle_name}.tck"
                bundle_symlink.parent.mkdir(parents=True, exist_ok=True)

                fake_ending = pathlib.Path(
                    plot_3D.split('=')[-1]) / 'endings_segmentations' / f"{bundle_name}_b.nii.gz"
                fake_ending.parent.mkdir(parents=True, exist_ok=True)

                brain_mask_path = pathlib.Path(
                    sub_temp_dir) / "nodif_brain_mask.nii.gz"
                brain_mask_path.parent.mkdir(parents=True, exist_ok=True)


                if not bundle_symlink.exists():
                    bundle_symlink.symlink_to(bundle)
                    # Create a fake endings segmentation file
                    if not fake_ending.exists():
                        fake_ending.symlink_to(brain_mask.path)
                    # Copy the brain mask to the temp directory
                    if brain_mask_path.exists():
                        brain_mask_path.unlink()  # Supprimer le lien existant avant d'en créer un nouveau
                    brain_mask_path.symlink_to(brain_mask.path)
                    print(
                        f"Created symlink for bundle {bundle_name} at {bundle_symlink}"
                    )
                else:
                    print(
                        f"Symlink for bundle {bundle_name} already exists at {bundle_symlink}"
                    )

    # Read the original file and only update the tractometry_path line
    with open(subjects_txt, 'r') as f:
        original_lines = f.readlines()
    
    # Update only the first line (tractometry_path) with the new temp directory path
    with open(os.path.join(temp_dir, 'subjects.txt'), 'w') as f:
        for i, line in enumerate(original_lines):
            if i == 0:
                # Replace TEMPDIR in the first line only
                f.write(line.replace('TEMPDIR', temp_dir))
            else:
                f.write(line)

    cmd = [
        'plot_tractometry_results', '-i',
        os.path.join(temp_dir, 'subjects.txt'), '-o',
        os.path.join(temp_dir, 'tractometry_results.png'),
        "--save_csv",
    ]
    if with_3dplot:
        cmd += ['--plot3D', 'pval', '--tracking_format', 'tck']
    print(f"Running command: {' '.join(cmd)}")
    call(cmd)
    print(
        f"Tractometry results saved to {os.path.join(temp_dir, 'tractometry_results.png')}"
    )
    
    # Create a timestamped folder in ~/Data/Tractometry/
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.expanduser(f"~/Data/Tractometry/tractometry_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the results to the output directory
    shutil.copy(os.path.join(temp_dir, 'tractometry_results.png'), output_dir)
    shutil.copy(os.path.join(temp_dir, 'subjects.txt'), output_dir)
    
    # Create a JSON file with the parameters
    params = {
        "timestamp": timestamp,
        "metric": metric,
        "with_3dplot": with_3dplot,
        "dataset_path": dataset_path,
        "temp_dir": temp_dir,
        "subjects_file": subjects_txt,
        "command": " ".join(cmd)
    }
    
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Results saved to {output_dir}")

def process_tractseg_analysis_new(subjects_txt,
        dataset_path="/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids",
        with_3dplot=False,
        metric='FA',
        pipeline='tractometry'):
    """
    Process the tractseg analysis for the given subjects.
    This function only updates the tractometry_path in the subjects.txt file.

    Parameters
    ----------
    subjects_txt : str
        Path to the text file containing the list of subjects.
    dataset_path : str
        Path to the BIDS dataset.
    with_3dplot : bool
        Whether to generate 3D plots.
    metric : str
        The metric to use for tractometry (e.g., 'FA', 'MD').
    pipeline : str
        The pipeline to use for fetching tractometry data.
    """
    
    # Get bundle names mapping
    bundle_name_dict = get_HCP_bundle_names()

    # Read the original file
    with open(subjects_txt, 'r') as f:
        original_lines = f.readlines()
    
    # Parse subjects from the file (skip header lines starting with #)
    subjects_list = []
    target_bundles = []
    for line in original_lines:
        stripped = line.strip()
        if stripped.startswith('# bundles='):
            target_bundles = stripped.split('=')[1].split()

        if stripped and not stripped.startswith('#'):
            # First non-comment line is the column header, skip it
            if not subjects_list:
                subjects_list.append(None)  # Placeholder for header
            else:
                subject = stripped.split()[0]  # First column is subject_id
                subjects_list.append(subject)
    
    # Remove the header placeholder
    subjects_list = [s for s in subjects_list[1:] if s is not None]
    
    ds = Dataset(dataset_path)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created at: {temp_dir}")
    
    # Copy tractometry CSV files for each subject, en remplaçant les NaN par 0 dans la copie
    for sub in subjects_list:
        print(f"Processing subject: {sub}")
        sub_id = sub.split('-')[-1]
        files = ds.get(sub_id,
                       pipeline=pipeline,
                       suffix='tractsegmean',
                       metric=metric,
                       extension='csv')
        if len(files) == 0:
            print(f"No files found for subject {sub_id}. Skipping.")
            continue
        else:
            tracto_metrics = files[0]

        # Charger le CSV, remplacer les NaN par 0, sauvegarder dans le dossier temporaire
        import pandas as pd
        import numpy as np
        sub_temp_dir = os.path.join(temp_dir, sub)
        os.makedirs(sub_temp_dir, exist_ok=True)
        src_csv = tracto_metrics.path
        dst_csv = os.path.join(sub_temp_dir, 'Tractometry.csv')
        try:
            df = pd.read_csv(src_csv, sep=None, engine='python')
            df = df.rename(columns=bundle_name_dict)
            if target_bundles:
                df = df[[c for c in df.columns if c in target_bundles]]
            df = df.fillna(0)
            # Add point index
            df.insert(0, 'index', range(len(df)))
            df.to_csv(dst_csv, index=False, sep=';')
        except Exception as e:
            print(f"Erreur lors du traitement du CSV pour {sub}: {e}. Copie brute.")
            shutil.copy(src_csv, dst_csv)
    
    # Write the subjects.txt file, only replacing TEMPDIR in all lines
    with open(os.path.join(temp_dir, 'subjects.txt'), 'w') as f:
        for line in original_lines:
            f.write(line.replace('TEMPDIR', temp_dir))
    
    # Run plot_tractometry_results
    cmd = [
        'plot_tractometry_results', '-i',
        os.path.join(temp_dir, 'subjects.txt'), '-o',
        os.path.join(temp_dir, 'tractometry_results.png'),
        '--mc',
        "--save_csv",
    ]
    if with_3dplot:
        cmd += ['--plot3D', 'pval', '--tracking_format', 'tck']
    print(f"Running command: {' '.join(cmd)}")
    call(cmd)
    print(f"Tractometry results saved to {os.path.join(temp_dir, 'tractometry_results.png')}")
    
    # Create a timestamped folder in ~/Data/Tractometry/
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.expanduser(f"~/Data/Tractometry/tractometry_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the results to the output directory
    shutil.copy(os.path.join(temp_dir, 'tractometry_results.png'), output_dir)
    shutil.copy(os.path.join(temp_dir, 'subjects.txt'), output_dir)
    
    # Create a JSON file with the parameters
    params = {
        "timestamp": timestamp,
        "metric": metric,
        "with_3dplot": with_3dplot,
        "dataset_path": dataset_path,
        "temp_dir": temp_dir,
        "subjects_file": subjects_txt,
        "command": " ".join(cmd)
    }
    
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Results saved to {output_dir}")
    return output_dir


def prepare_subject_txt(pipeline,target_col,confond_cols,bundle_list='ALL'):
    """
    Prepare the subjects.txt file for tractometry analysis.

    Parameters
    ----------
    bundle_list : list of str
        List of bundles to include in the analysis.
    target_col : str
        Name of the target column in the subjects dataframe.
    confond_cols : list of str
        List of confounding columns in the subjects dataframe.
    """
    ds=Dataset()
    subjects= set(['sub-'+x.subject for x in ds.get_global(pipeline=pipeline)])
    infos = pd.read_excel('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx')
    subjects_df = infos[infos['participant_id'].isin(subjects)]

    #Set col_type based on target_col. If target_col is binary, col_type is 'group', else 'target'
    if subjects_df[target_col].nunique() == 2:
        col_type = 'group'
    else:
        col_type = 'target'
    
    print(f"Column {target_col} has {subjects_df[target_col].nunique()} unique values. Setting col_type to '{col_type}'.")

    #Get subjects where all confond_cols and target_col are not null
    subjects_df = subjects_df.dropna(subset=[target_col] + confond_cols)
    #for non numeric confond_cols or target_col, provides a mapping to numeric
    for col in [target_col] + confond_cols:
        if not np.issubdtype(subjects_df[col].dtype, np.number):
            print(f"Column {col} is not numeric, providing a mapping to numeric.")
            unique_values = subjects_df[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            subjects_df[col] = subjects_df[col].map(mapping)
            print(f"Mapping for column {col}: {mapping}")

    # Si col_type == 'group', forcer le mapping à 0/1
    if col_type == 'group':
        if target_col in subjects_df.columns:
            unique_values = sorted(subjects_df[target_col].unique())
            if set(unique_values) != {0, 1}:
                # Remap group to 0/1
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                subjects_df[target_col] = subjects_df[target_col].map(mapping)
                print(f"Mapping forcé pour 'group': {mapping}")
    
    list_valid_bundles = "AF_left AF_right ATR_left ATR_right CC_1 CC_2 CC_3 CC_7 CG_left CG_right CST_left CST_right FPT_left FPT_right ICP_left ICP_right IFO_left IFO_right ILF_left ILF_right MCP OR_left OR_right POPT_left POPT_right SCP_left SCP_right SLF_III_left SLF_III_right SLF_II_left SLF_II_right SLF_I_left SLF_I_right STR_left STR_right ST_FO_left ST_FO_right ST_PREM_left ST_PREM_right T_OCC_left T_OCC_right T_PAR_left T_PAR_right T_PREM_left T_PREM_right UF_left UF_right"

    if bundle_list == 'ALL':
        bundle_list = list_valid_bundles.split(' ')
    else:
        bundle_list = [b for b in get_HCP_bundle_names().values() if b.replace('_','') in [x.replace('_','') for x in bundle_list]]

    temp_dir = tempfile.mkdtemp()
    subjects_txt_path = os.path.join(temp_dir, 'subjects.txt')
    with open(subjects_txt_path, 'w') as f:
        f.write(f"# tractometry_path=TEMPDIR/SUBJECT_ID/Tractometry.csv\n")
        # Ensure bundle_list is a list of strings
        bundle_list = list(bundle_list)
        f.write(f"# bundles={' '.join(bundle_list)}\n")
        f.write(f"# plot_3D=TEMPDIR/{list(subjects)[0]}/tracto/\n")
        # Write header: subject_id followed by target and confounding columns
        header_cols = ['subject_id', col_type] + confond_cols
        f.write(' '.join(header_cols) + "\n")
        for idx, row in subjects_df.iterrows():
            subject_line = f"{row['participant_id']} "
            subject_line += f"{row[target_col]} "
            for confond in confond_cols:
                subject_line += f"{row[confond]} "
            f.write(subject_line.strip() + "\n")
    print(f"Subjects.txt file created at: {subjects_txt_path}")
    return subjects_txt_path

if __name__ == "__main__":
    new_txt=prepare_subject_txt(pipeline='hcp_association_50pts', target_col='aes', confond_cols=['age','sex','city','duration_dep'],bundle_list='ALL')
    process_tractseg_analysis_new(new_txt,metric='FA',with_3dplot=False,pipeline='hcp_association_50pts')

#    process_tractseg_analysis("/home/ndecaux/Code/actiDep/subjects.txt",metric='FA',with_3dplot=False,pipeline='hcp_association_tractseg')
    # sub = Subject('03005',db_root='/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
    # fa = sub.get_unique(pipeline='mcm_tensors_staniz',desc='cleaned',metric='FA',extension='nii.gz',bundle='UFright')
    # tracto = sub.get_unique(pipeline='bundle_seg',bundle='UFright',suffix='tracto',extension='trk')
    # print(fa.path, tracto.path)
    # fa_nii = nib.load(fa.path)
    # fa_img= fa_nii.get_fdata()
    # affine = fa_nii.affine

    # # Use dipy's load_tractogram to load the streamlines
    # tractogram = load_trk(tracto.path, 'same', bbox_valid_check=False)
    # tracto_data = tractogram.streamlines

    # #Print the affine in the tractogram header
    # print("Affine in tractogram header:")
    # print(tractogram.affine)
    # # Print the affine in the fa_nii header
    # print("Affine in fa_nii header:")
    # print(fa_nii.affine)
    

    # fa = sub.get_unique(pipeline='mcm_tensors_staniz',desc='cleaned',metric='FA',extension='nii.gz',bundle='ATRleft')
    # tracto = sub.get_unique(pipeline='bundle_seg',bundle='ATRleft',suffix='tracto',extension='trk')
    # print(fa.path, tracto.path)
    # fa_nii = nib.load(fa.path)
    # fa_img= fa_nii.get_fdata()
    # affine = fa_nii.affine

    # # Use dipy's load_tractogram to load the streamlines
    # tractogram = load_trk(tracto.path, 'same', bbox_valid_check=False)
    # tracto_data = tractogram.streamlines

    # #Print the affine in the tractogram header
    # print("Affine in tractogram header:")
    # print(tractogram.affine)
    # # Print the affine in the fa_nii header
    # print("Affine in fa_nii header:")
    # print(fa_nii.affine)
    
    # #Flip affine in z
    # affine[0,:] = -affine[0, :]
    # affine[1,:] = -affine[1, :]

  

    # print(evaluate_along_streamlines(fa_img, tracto_data, nr_points=100, affine=affine))