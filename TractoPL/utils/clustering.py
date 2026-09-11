import os
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, Dataset
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template
from TractoPL.utils.tractography import get_tractogram_endings
from TractoPL.utils.mcm import name_mapping as mcm_name_mapping
from TractoPL.analysis.tractometry import process_projection
import multiprocessing
import dipy
import tempfile
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.utils import length
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.io.streamline import load_tractogram
from dipy.segment.metric import mdf, mean_manhattan_distance
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric, mdf
from scipy.spatial import cKDTree
from dipy.tracking.streamline import set_number_of_points
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import Streamlines
from scipy.interpolate import interp1d
import os
from dipy.align.streamlinear import StreamlineLinearRegistration
import shapely
from shapely import LineString, MultiLineString, Polygon
import csv
from TractoPL.data.vtk_loader import load_vtk_streamlines, save_vtk
import nibabel as ni
from TractoPL.utils.registration import apply_transforms
import ants
from subprocess import run,call
from scipy.spatial import cKDTree
import numpy as np
def load_vtk_streamlines(vtk_file_path):
    """
    Charge les streamlines depuis un fichier VTK.
    
    Returns
    -------
    streamlines : list
        Liste des streamlines (chaque streamline est un array numpy (N, 3))
    scalar_arrays : dict
        Dictionnaire des arrays scalaires {nom: valeurs}
    """
    # Lire le fichier VTK
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Extraire les streamlines
    lines = polydata.GetLines()
    streamlines = []

    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        line_points = []
        for j in range(id_list.GetNumberOfIds()):
            point_id = id_list.GetId(j)
            point = polydata.GetPoint(point_id)
            line_points.append(point)
        streamlines.append(np.array(line_points))

    # Extraire les arrays scalaires
    point_data = polydata.GetPointData()
    scalar_arrays = {}

    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = array.GetName()
        if array_name and array_name != 'colors':
            scalar_arrays[array_name] = vtk_to_numpy(array)


    print(f"Chargé : {len(streamlines)} streamlines")
    print(f"Arrays scalaires disponibles : {list(scalar_arrays.keys())}")
    return streamlines, scalar_arrays

def cluster_streamlines(tractogram, threshold=100.):
    tracto = load_tractogram(tractogram, "same", bbox_valid_check=False)

    #Resample to 100 points
    from dipy.tracking.streamline import set_number_of_points
    streamlines = set_number_of_points(tracto.streamlines, 100)

    #Save
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=100., metric=metric)
    clusters = qb.cluster(streamlines)
    print('N clusters',len(clusters))

    centroids = Streamlines(clusters.centroids)



def get_longest_streamlines(streamlines, number=15):
    """
    Trouve les N streamlines les plus longues.
    Parameters
    ----------
    streamlines : list
        Liste des streamlines (chaque streamline est un array numpy (N, 3))
    number : int
        Nombre de streamlines à retourner (par défaut 15)
    Returns
    -------
    longest_streamlines : list
        Liste des N streamlines les plus longues
    """
    # Calculer la longueur de chaque streamline
    lengths = [np.linalg.norm(s[-1] - s[0]) for s in streamlines]

    # Obtenir les indices des N plus longues
    longest_indices = np.argsort(lengths)[-number:]

    # Retourner les streamlines correspondantes
    longest_streamlines = [streamlines[i] for i in longest_indices]

    print(f"Streamlines les plus longues : {len(longest_streamlines)}")
    return longest_streamlines

def create_central_line_mix(streamlines):
    """
    Crée une ligne centrale en clusterisant les streamlines.
    
    Parameters
    ----------
    streamlines : list
        Streamlines interpolées
    threshold : float
        Seuil de distance pour le clustering
        
    Returns
    -------
    central_line : numpy.ndarray
        Ligne centrale (N, 3)
    cluster_indices : list
        Indices des streamlines dans chaque cluster
    """

    # Convertir en format DIPY
    streamlines = Streamlines(streamlines)
    streamlines_n_points= set_number_of_points(streamlines, 100)  # Interpoler à 100 points par streamline
    # Clustering avec QuickBundles
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=100., metric=metric)
    clusters = qb.cluster(streamlines_n_points)
    centroids = Streamlines(clusters.centroids)
    print(f"Clustering : {len(clusters)} clusters trouvés")

    len_centroids = np.max([s.shape[0] for s in streamlines])
    print(len_centroids)
    fake_streamlines = []
    for i, s in enumerate(streamlines):
        if s.shape[0] < len_centroids:
            fake_streamlines.append(np.concatenate((s, [s[-1]] * (len_centroids - s.shape[0]))))
        else:
            fake_streamlines.append(s)


    

    # print(f"Streamlines shape: {fake_streamlines.shape}")  # (2000, 100, 3)
    print(centroids.get_data().shape, fake_streamlines[0].shape)
    _, segment_idxs = cKDTree(centroids.get_data(), 1, copy_data=True).query(fake_streamlines, k=1)  # (15, len_centroids)

    print("n_seg",len(segment_idxs))
    filtered_segment_idxs = []
    #Remove points that were artificially added to match the length of the longest streamline
    for i in range(len(segment_idxs)):
        s = fake_streamlines[i]
        true_s= streamlines[i]
        if s.shape[0] > true_s.shape[0]:
            filtered_segment_idxs.append(segment_idxs[i][:true_s.shape[0]])
        else:
            filtered_segment_idxs.append(segment_idxs[i])

    segment_idxs = filtered_segment_idxs
    for i in range(len(segment_idxs)):
        segment_idxs[i] = np.round(segment_idxs[i])

    print([s.shape for s in segment_idxs])
    return segment_idxs, centroids, streamlines

def cluster_vtk(vtk_file,reference_file):
    """
    Cluster les streamlines d'un fichier VTK et crée une ligne centrale.
    
    Parameters
    ----------
    vtk_file : BIDSFile
        Fichier VTK contenant les streamlines à clusteriser.
    reference_file : BIDSFile
        Fichier de référence pour la transformation des streamlines.
    """

    streamlines, scalar_arrays = load_vtk_streamlines(vtk_file.path)
    longest_streamlines = get_longest_streamlines(streamlines, number=100)
    segment_idxs, centroids, streamlines_100_points = create_central_line_mix(longest_streamlines)
    #Create VTK PolyData from segment_idxs and streamlines_100_points
    output_vtk_path = 'clusters_mix.vtk'
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_path)
    polydata_filtered = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polydata_filtered.SetPoints(points)
    lines = vtk.vtkCellArray()
    for streamline in streamlines_100_points:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(streamline))
        for i, point in enumerate(streamline):
            point_id = points.InsertNextPoint(point)
            line.GetPointIds().SetId(i, point_id)
        lines.InsertNextCell(line)

    polydata_filtered.SetLines(lines)
    #Add scalar data for segment_idxs - flatten the indices to match point data
    flattened_segment_idxs = np.concatenate(segment_idxs)
    scalar_data = numpy_to_vtk(flattened_segment_idxs, deep=True)

    scalar_data.SetName('segment_idxs')
    polydata_filtered.GetPointData().AddArray(scalar_data)

    writer.SetInputData(polydata_filtered)
    writer.Write()
    print(f"Streamlines les plus longues sauvegardées dans {output_vtk_path}")

    output_vtk_path = 'centroids_mix.vtk'
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_path)
    polydata_filtered = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polydata_filtered.SetPoints(points)
    lines = vtk.vtkCellArray()
    print(centroids[0].shape)
    line.GetPointIds().SetNumberOfIds(len(centroids[0]))
    for i, point in enumerate(centroids[0]):
        point_id = points.InsertNextPoint(point)
        line.GetPointIds().SetId(i, point_id)
    lines.InsertNextCell(line)
    polydata_filtered.SetLines(lines)
    #Save centroids
    writer.SetInputData(polydata_filtered)
    writer.Write()
    print(f"Centroids sauvegardés dans {output_vtk_path}")


def associate_subject_to_centroids(subject_bundle,
                                   model_centroids_path,
                                   model_full_bundle_path,
                                   reference_nifti,
                                   n_pts=50,
                                   temp_dir=None,
                                   slr=True,
                                   model_clustering=5.0
                                   ):
    """
    Associe les streamlines d'un bundle de sujet aux centroids d'un modèle.
    
    Parameters
    ----------
    subject_bundle : BIDSFile
        Fichier VTK contenant les streamlines du sujet à associer
    model_centroids_path : str
        Chemin vers le fichier VTK contenant les centroids du modèle
    model_full_bundle_path : str
        Chemin vers le fichier VTK contenant le bundle complet du modèle
    reference_nifti : str
        Chemin vers l'image de référence NIfTI
    temp_dir : str, optional
        Répertoire temporaire (par défaut utilise tempfile.mkdtemp())
        
    Returns
    -------
    res_dict : dict
        Dictionnaire contenant les résultats et chemins des fichiers générés
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    # Charger les streamlines du sujet
    streamlines, scalar_arrays = load_vtk_streamlines(subject_bundle.path)
    centroids, _ = load_vtk_streamlines(model_centroids_path)

    slr_n_points= 12
    streamline_association_n_points = 12  # Nombre de points par streamline pour l'association
    points_association_n_points= n_pts

    if isinstance(points_association_n_points, str) and points_association_n_points.endswith('mm'):
        # Convertir la longueur d'arc des centroids en nombre de points pour l'association
        mm_value = float(points_association_n_points.split('mm')[0])
        new_centroids = []
        for centroid in centroids:
            arc_length=int(LineString(centroid).length)
            # arc_length = float(np.sum(np.linalg.norm(np.diff(centroid, axis=0), axis=1)))
            # n_points = max(2, int(np.ceil(arc_length / mm_value)))  # Au moins 2 points
            n_points = max(2, int(np.ceil(arc_length / mm_value)))  # Au moins 2 points
            centroid = set_number_of_points(centroid, n_points)
            new_centroids.append(centroid)
        centroids = new_centroids

    else:
        centroids = set_number_of_points(centroids, points_association_n_points)  # Interpoler à 100 points par centroid

    # Charger les modèles

    # model_tractogram = load_tractogram(model_centroids_path,
    #                                    reference_nifti,
    #                                    bbox_valid_check=False)
    # model_full_bundle = load_tractogram(model_full_bundle_path,
    #                                     reference_nifti,
    #                                     bbox_valid_check=False)



    print(f"Nombre de streamlines du sujet : {len(streamlines)}")
    print(f"Nombre de centroids du modèle : {len(centroids)}")


    if slr:

        if model_full_bundle_path.endswith('.trk'):
            model_full_bundle_tracto = load_tractogram(model_full_bundle_path,
                                            reference_nifti,
                                            bbox_valid_check=False)
            model_full_bundle = model_full_bundle_tracto.streamlines
        else:
            model_full_bundle,_ = load_vtk_streamlines(model_full_bundle_path)

        print(f"Nombre de streamlines du bundle complet : {len(model_full_bundle)}")

        # Clustering des streamlines du sujet pour réduire la charge SLR
        subject_streamlines_resampled = [set_number_of_points(s, slr_n_points) for s in streamlines]  # Interpoler à 100 points par streamline
        subject_streamlines_obj = Streamlines(subject_streamlines_resampled)
        
        metric = AveragePointwiseEuclideanMetric()
        qb_subject = QuickBundles(threshold=15., metric=metric)
        subject_clusters = qb_subject.cluster(set_number_of_points(subject_streamlines_obj, slr_n_points))
        subject_centroids = Streamlines(subject_clusters.centroids)
        print(f"Clustering du sujet : {len(subject_clusters)} clusters créés")

        # Clustering du modèle complet pour réduire la charge SLR
        model_streamlines_resampled = [set_number_of_points(s, slr_n_points) for s in model_full_bundle]  # Interpoler à 100 points par streamline
        model_streamlines_obj = Streamlines(model_streamlines_resampled)
        
        qb_model = QuickBundles(threshold=15., metric=metric)
        model_clusters = qb_model.cluster(set_number_of_points(model_streamlines_obj, slr_n_points))
        model_centroids = Streamlines(model_clusters.centroids)
        print(f"Clustering du modèle : {len(model_clusters)} clusters créés")
        print("Réalisation du SLR...")
        time_start = os.times()
        try:
            slr = StreamlineLinearRegistration()
            # Inverser l'ordre : recaler le modèle sur le sujet
            slm = slr.optimize(subject_centroids, model_centroids)
            # Appliquer la transformation aux centroids du modèle
            transformed_centroids = slm.transform(Streamlines(centroids))
            print("SLR avec clustering terminé avec succès - modèle recalé sur sujet")
            time_end = os.times()
            print(f"Temps écoulé pour le SLR : {time_end[0] - time_start[0]} secondes")
        except Exception as e:
            print(f"Erreur lors du SLR: {e}")
            transformed_centroids = centroids
    else:
        transformed_centroids = centroids
        print("SLR non effectué, utilisation des centroids d'origine")

    # Association des streamlines du sujet avec les centroids transformés
    print(
        "Calcul des associations entre les streamlines du sujet et les centroids transformés..."
    )

    associations = compute_distance_from_centroids(
        streamlines, transformed_centroids, n_pts=streamline_association_n_points)
            
    # transformed_centroids = set_number_of_points(transformed_centroids, n_pts)  # Interpoler à n_pts points par centroid
    # centroids = transformed_centroids#set_number_of_points(centroids, 100)  # Interpoler à 100 points par centroid

    print(f"Nombre d'associations : {len(associations)}")

    # Créer les arrays d'association
    streamline_cluster_ids = np.full(len(streamlines), -1, dtype=int)
    streamline_distances = np.full(len(streamlines), 999.0)

    for streamline_idx, centroid_idx, distance, flip in associations:
        streamline_cluster_ids[streamline_idx] = centroid_idx
        streamline_distances[streamline_idx] = distance

    point_centroid_distances = []
    point_indices_correspondance = []
    point_distances = []
    point_cluster_ids = []

    # Arrays pour stocker les indices
    centroid_indices = []
    global_point_indices = []
    global_point_counter = 0

    for i, streamline in enumerate(streamlines):
        cluster_id = streamline_cluster_ids[i]
        # Récupérer le flip optimal
        flip = False
        flip = associations[i][3]
        assoc_distance= associations[i][2]
        
        # streamline = set_number_of_points(streamline, points_association_n_points)
        if cluster_id >= 0:
            centroid = transformed_centroids[cluster_id].copy()
            n_centroid_pts = len(centroid)
            n_streamline_pts = len(streamline)

            # Indices normalisés des points du centroid (avec flip si nécessaire)
            raw_indices = np.arange(n_centroid_pts)
            if flip:
                norm_idx_centroid = (n_centroid_pts - 1 - raw_indices) / n_centroid_pts
            else:
                norm_idx_centroid = raw_indices / n_centroid_pts

            norm_idx_streamline = np.arange(n_streamline_pts) / n_streamline_pts

            # alpha = longueur d'arc de la streamline :
            # ainsi alpha * Δidx_voisins ≈ distance inter-points moyenne,
            # ce qui équilibre automatiquement pénalité d'ordre et distance spatiale
            # quelle que soit la résolution ou la taille de la fibre.
            arc_length = float(np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1)))
            alpha = arc_length
            centroid_augmented = np.column_stack([centroid, alpha * norm_idx_centroid])
            streamline_augmented = np.column_stack([streamline, alpha * norm_idx_streamline])

            # KDTree sur le centroid augmenté → une seule requête pour tous les points
            tree = cKDTree(centroid_augmented)
            _, indices_order = tree.query(streamline_augmented, k=1)

            # Distance 3D pure (métrique de sortie, sans la composante d'indice)
            dists_3d = np.linalg.norm(streamline - centroid[indices_order], axis=1)

            point_centroid_distances.extend(dists_3d.tolist())
            point_indices_correspondance.extend(indices_order.tolist())
            point_distances.extend([assoc_distance] * n_streamline_pts)
            point_cluster_ids.extend([cluster_id] * n_streamline_pts)

        else:
            point_centroid_distances.extend([999.0] * len(streamline))
            point_indices_correspondance.extend([-1] * len(streamline))
            point_distances.extend([999.0] * len(streamline))
            point_cluster_ids.extend([-1] * len(streamline))

    
    print(len(point_indices_correspondance))
    # Sauvegarder les résultats dans le répertoire temporaire
    output_vtk_path = os.path.join(temp_dir,
                                   'streamlines_with_associations.vtk')
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_path)
    polydata_associations = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polydata_associations.SetPoints(points)
    lines = vtk.vtkCellArray()

    for streamline in streamlines:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(streamline))
        for i, point in enumerate(streamline):
            point_id = points.InsertNextPoint(point)
            line.GetPointIds().SetId(i, point_id)
        lines.InsertNextCell(line)

    polydata_associations.SetLines(lines)

    print('Nombre centroids 1 : ',np.unique(np.array(point_cluster_ids)))

    # Ajouter les arrays scalaires
    cluster_array = numpy_to_vtk(np.array(point_cluster_ids), deep=True)
    cluster_array.SetName('centroid_index')
    polydata_associations.GetPointData().AddArray(cluster_array)

    distance_array = numpy_to_vtk(np.array(point_distances), deep=True)
    distance_array.SetName('association_distance')
    polydata_associations.GetPointData().AddArray(distance_array)

    centroid_distance_array = numpy_to_vtk(np.array(point_centroid_distances),
                                           deep=True)
    centroid_distance_array.SetName('centroid_point_distance')
    polydata_associations.GetPointData().AddArray(centroid_distance_array)

    point_correspondence_array = numpy_to_vtk(
        np.array(point_indices_correspondance), deep=True)
    point_correspondence_array.SetName('point_index')
    polydata_associations.GetPointData().AddArray(point_correspondence_array)

    writer.SetInputData(polydata_associations)
    writer.Write()

    # Sauvegarder les centroids transformés avec les indices de points
    centroids_vtk_path = os.path.join(temp_dir, 'transformed_centroids.vtk')
    centroid_writer = vtk.vtkPolyDataWriter()
    centroid_writer.SetFileName(centroids_vtk_path)
    centroid_polydata = vtk.vtkPolyData()
    centroid_points = vtk.vtkPoints()
    centroid_polydata.SetPoints(centroid_points)
    centroid_lines = vtk.vtkCellArray()

    print('Nbr centroids 2 : ',len(centroids))
    for centroid_idx, centroid in enumerate(centroids):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(centroid))
        for i, point in enumerate(centroid):
            point_id = centroid_points.InsertNextPoint(point)
            line.GetPointIds().SetId(i, point_id)
            centroid_indices.append(centroid_idx)
            global_point_indices.append(global_point_counter)
            global_point_counter += 1
        centroid_lines.InsertNextCell(line)

    centroid_polydata.SetLines(centroid_lines)

    # Ajouter les arrays d'indices
    centroid_index_array = numpy_to_vtk(np.array(centroid_indices), deep=True)
    centroid_index_array.SetName('centroid_index')
    centroid_polydata.GetPointData().AddArray(centroid_index_array)

    point_index_array = numpy_to_vtk(np.array(global_point_indices), deep=True)
    point_index_array.SetName('point_index')
    centroid_polydata.GetPointData().AddArray(point_index_array)

    centroid_writer.SetInputData(centroid_polydata)
    centroid_writer.Write()

    entities = subject_bundle.get_full_entities()

    # Créer le dictionnaire de résultats
    res_dict = {
        output_vtk_path: upt_dict(entities, desc='associations'),
        centroids_vtk_path: upt_dict(entities, desc='centroids', space='subject'),
    }
    csv_path = os.path.join(temp_dir, 'centroid_scalar_means.csv')

    # write_csv_from_scalars(scalar_arrays,
    #                        streamlines,
    #                         point_indices_correspondance,
    #                         transformed_centroids,
    #                         streamline_cluster_ids,
    #                         temp_dir)

    csv_path = write_csv_from_scalars(scalar_arrays,
                                      point_indices_correspondance,
                                      point_cluster_ids,
                                      temp_dir)
      # Ajouter le CSV au dictionnaire des résultats

    res_dict[csv_path] = upt_dict(entities, suffix='mean',datatype='metric',extension='csv')
    # --- Fin de la nouvelle section ---

    print(f"Résultats sauvegardés dans {temp_dir}")
    return res_dict


def compute_distance_from_centroids(streamlines,
                                   centroids,n_pts=12):
    """
    Calcule la distance de chaque point de chaque streamline aux centroids associés.
    Parameters
    ----------
    streamlines : list
        Liste des streamlines (chaque streamline est un array numpy (N, 3))
    centroids : list
        Liste des centroids (chaque centroid est un array numpy (M, 3))
    Returns
    -------
    distances : list
        Liste des distances pour chaque point de chaque streamline
    """ 

    associations = []

    for index_streamline, streamline in enumerate(streamlines):
        streamline_resampled = set_number_of_points(streamline, n_pts)
        min_distance = 999.0
        best_centroid = -1
        best_flip = False

        for index_centroid, centroid in enumerate(centroids):
            centroid_resampled = set_number_of_points(centroid, n_pts)
            # Calculer la somme des distances euclidiennes point à point pour les deux ordres
            # Use DIPY's AveragePointwiseEuclideanMetric instead of MDF
            dist_direct = mean_manhattan_distance(streamline_resampled, centroid_resampled)
            dist_flipped = mean_manhattan_distance(streamline_resampled, centroid_resampled[::-1])
            # dist_direct = shapely.frechet_distance(LineString(streamline_resampled), LineString(centroid_resampled))
            # dist_flipped = shapely.frechet_distance(LineString(streamline_resampled), LineString(centroid_resampled[::-1]))
            if dist_direct < dist_flipped:
                distance = dist_direct
                flip = False
            else:
                distance = dist_flipped
                flip = True

            if distance < min_distance:
                min_distance = distance
                best_centroid = index_centroid
                best_flip = flip

        if best_centroid != -1:
            associations.append(
                (index_streamline, best_centroid, min_distance, best_flip))

    return associations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import os

def write_csv_from_scalars(scalar_arrays, 
                           point_indices_correspondance, 
                           point_cluster_ids,
                           out_dir):
    """
    Calcule statistiques (mean, std, median, GMM) par point de centroid.
    """
    # 1. Préparation des données dans un DataFrame pour faciliter les groupements
    # On crée une structure plate : chaque point de chaque streamline du sujet est une ligne
    data_list = []
    
    # Récupérer les noms des scalaires
    scalar_names = list(scalar_arrays.keys())
    
    # On convertit point_indices_correspondance et point_cluster_ids en arrays pour le slicing
    corr_arr = np.array(point_indices_correspondance)
    clust_arr = np.array(point_cluster_ids)
    
    # Filtrer les points non associés (id < 0)
    mask = (clust_arr >= 0) & (corr_arr >= 0)
    
    df_dict = {
        'centroid_id': clust_arr[mask],
        'point_id': corr_arr[mask]
    }
    
    for name in scalar_names:
        # On suppose ici que scalar_arrays[name] est déjà un array plat "par point" 
        # de même longueur que point_indices_correspondance
        df_dict[name] = np.array(scalar_arrays[name])[mask]
        
    df = pd.DataFrame(df_dict)
    
    # 2. Calcul des statistiques classiques via GroupBy
    stats_df = df.groupby(['centroid_id', 'point_id']).agg(['mean', 'std', 'median'])
    
    # Aplatir les colonnes (ex: 'FA_mean', 'FA_std', etc.)
    stats_df.columns = [f"{mcm_name_mapping.get(col[0], col[0])}_{col[1]}" for col in stats_df.columns]
    stats_df = stats_df.reset_index()

    # # 3. Tentative de Statistique par Mixture de Gaussienne (GMM)
    # # On va calculer la moyenne de la composante principale de la GMM pour chaque point
    # gmm_results = []
    
    # for (c_id, p_id), group in df.groupby(['centroid_id', 'point_id']):
    #     res = {'centroid_id': c_id, 'point_id': p_id}
    #     for name in scalar_names:
    #         values = group[name].values.reshape(-1, 1)
    #         mapped_name = mcm_name_mapping.get(name, name)
            
    #         if len(values) > 1:
    #             try:
    #                 # On fit une GMM simple (1 composante pour avoir le mode robuste, 
    #                 # ou 2 si on veut détecter des sous-populations)
    #                 gmm = GaussianMixture(n_components=1, random_state=42)
    #                 gmm.fit(values)
    #                 res[f"{mapped_name}_gmm_mean"] = gmm.means_[0][0]
    #                 res[f"{mapped_name}_gmm_prec"] = gmm.precisions_[0][0][0]
    #             except:
    #                 res[f"{mapped_name}_gmm_mean"] = np.nan
    #         else:
    #             res[f"{mapped_name}_gmm_mean"] = values[0][0] if len(values) == 1 else np.nan
    #     gmm_results.append(res)
    
    # gmm_df = pd.DataFrame(gmm_results)
    
    # Fusionner les stats classiques et GMM
    final_df = stats_df#pd.merge(stats_df, gmm_df, on=['centroid_id', 'point_id'], how='left')

    # Sauvegarde
    csv_path = os.path.join(out_dir, 'centroid_scalar_stats.csv')
    final_df.to_csv(csv_path, index=False)
    print(f"Statistiques sauvegardées dans {csv_path}")
    return csv_path

    
def streamlines_to_parcellation(streamlines, parcellation_data, parcellation_affine):
    # Détection de l'orientation du volume (RAS/LPS)
    import nibabel as nib
    orientation = nib.aff2axcodes(parcellation_affine)
    print(f"Orientation de la parcellation : {orientation}")
    flip_x = orientation[0] == 'L'  # flip X si L (Left)
    flip_y = orientation[1] == 'P'  # flip Y si P (Posterior)
    print(f"Flip X: {flip_x}, Flip Y: {flip_y}")

    """
    Associe les streamlines à une parcellation NIfTI.
    Si un point d'une streamline tombe dans une région de la parcellation,
    cette streamline est associée à cette région. Si elle ne traverse aucune région (valeur 0), on associe à la parcelle la plus proche.

    Parameters
    ----------
    streamlines : list
        Liste des streamlines (chaque streamline est un array numpy (N, 3))
    parcellation_nifti : nibabel.Nifti1Image
        Image NIfTI de la parcellation
    affine : numpy.ndarray
        Matrice affine de l'image NIfTI
    Returns
    -------
    parcellation_labels : list
        Liste des labels de parcellation associés à chaque streamline
    """
    # Debug : carte binaire des voxels visités
    binary_map = np.zeros(parcellation_data.shape, dtype=np.uint8)

    # Précalculer les positions des voxels non-zéro pour optimisation
    non_zero_indices = np.where(parcellation_data != 0)
    non_zero_world = []
    labels_list = []
    #Flip z
    for x, y, z in zip(*non_zero_indices):
        world_pos = (parcellation_affine @ np.array([x, y, z, 1]))[:3]
        non_zero_world.append(world_pos)
        labels_list.append(parcellation_data[x, y, z])
    non_zero_world = np.array(non_zero_world)
    labels_list = np.array(labels_list)
    
    if len(non_zero_world) > 0:
        tree = cKDTree(non_zero_world)
    else:
        tree = None

    parcellation_labels = []
    inv_affine = np.linalg.inv(parcellation_affine)

    for streamline in streamlines:
        point_labels = []
        for point in streamline:
            # Correction LPS/RAS si nécessaire
            point_corr = point.copy()
            if flip_x:
                point_corr[0] = -point_corr[0]
            if flip_y:
                point_corr[1] = -point_corr[1]
            # Projeter le point corrigé dans l'espace voxel
            voxel_coord = inv_affine @ np.append(point_corr, 1)
            x, y, z = np.round(voxel_coord[:3]).astype(int)
            # Debug : marquer le voxel visité
            if (0 <= x < parcellation_data.shape[0] and
                0 <= y < parcellation_data.shape[1] and
                0 <= z < parcellation_data.shape[2]):
                binary_map[x, y, z] = 1
            label = None
            if (0 <= x < parcellation_data.shape[0] and
                0 <= y < parcellation_data.shape[1] and
                0 <= z < parcellation_data.shape[2]):
                label = parcellation_data[x, y, z]
            if label is not None and label != 0:
                point_labels.append(label)
            else:
                # Associer systématiquement à la parcelle la plus proche (KDTree)
                if tree is not None:
                    dist, idx_tree = tree.query(point_corr)
                    point_labels.append(labels_list[idx_tree])
                else:
                    point_labels.append(None)
        parcellation_labels.append(point_labels)

    # Sauvegarde la carte binaire pour debug
    import nibabel as nib
    binary_img = nib.Nifti1Image(binary_map, parcellation_affine)
    nib.save(binary_img, 'debug_voxel_projection.nii.gz')

    return parcellation_labels

def associate_subject_to_parcellation(subject_bundle,
                                   parcellation,
                                   subject_reference_nifti=None,
                                   transform_list=None
                                   ):
    """
    Associe les streamlines à une parcellation et calcule les stats par parcelle
    en réutilisant write_csv_from_scalars.
    """
    temp_dir = tempfile.mkdtemp()
    streamlines, scalar_arrays = load_vtk_streamlines(subject_bundle.path)

    # 1. Gestion du chemin et de la transformation de la parcellation
    parcellation_path = parcellation if isinstance(parcellation, str) else parcellation.path
    
    if transform_list and subject_reference_nifti:
        transformed_parcellation_path = os.path.join(temp_dir, 'parcellation_transformed.nii.gz')
        apply_trans_cmd = [
            'antsApplyTransforms', '-d', '3',
            '-i', parcellation_path,
            '-r', subject_reference_nifti,
            '-o', transformed_parcellation_path,
            '-n', 'NearestNeighbor'
        ]
        for transform in transform_list:
            apply_trans_cmd.extend(['-t', transform])
        
        print("Application des transformations à la parcellation...")
        run(apply_trans_cmd, check=True)
        parcellation_path = transformed_parcellation_path

    # 2. Association des points aux labels de parcelles
    parc_seg = ni.load(parcellation_path)
    # parc_streamlines est une liste de listes: [[p1_label, p2_label, ...], [streamline2], ...]
    parc_streamlines = streamlines_to_parcellation(streamlines, parc_seg.get_fdata(), parc_seg.affine)

    # 3. Préparation des entrées pour write_csv_from_scalars
    # Ta fonction originale attend 'point_indices_correspondance' (index dans le centroid).
    # Pour une parcellation, on va utiliser l'ID de la parcelle comme "index de point" 
    # et considérer qu'il n'y a qu'un seul cluster global (ID 0) ou mapper 1 cluster = 1 parcelle.
    
    # Version simple : Chaque point du sujet est associé à sa parcelle via point_indices_correspondance
    # On crée un "faux" centroid qui contient tous les IDs de parcelles uniques.
    flat_labels = []
    for s_labels in parc_streamlines:
        flat_labels.extend(s_labels)
    
    unique_parcels = np.unique(flat_labels)
    # On crée un dictionnaire pour mapper l'ID de parcelle à un index 0..N
    parcel_to_idx = {p: i for i, p in enumerate(unique_parcels)}
    
    # point_indices_correspondance : l'index dans la liste des parcelles uniques
    point_indices_correspondance = [parcel_to_idx[l] for l in flat_labels]
    
    # streamline_cluster_ids : on peut mettre tout le monde à 0 (un seul groupe géant)
    # ou donner l'ID de la parcelle majoritaire. Ici, mettre 0 suffit car 
    # c'est point_indices_correspondance qui dicte le regroupement final dans ton CSV.
    streamline_cluster_ids = np.zeros(len(streamlines), dtype=int)
    
    # Fake centroids pour la dimension (longueur = nombre de parcelles uniques)
    fake_centroids = [np.zeros((1, 3)) for _ in unique_parcels]

    # 4. Sauvegarde VTK
    output_vtk_path = os.path.join(temp_dir, 'streamlines_parcellation.vtk')
    save_vtk(streamlines=streamlines, 
             output_vtk_path=output_vtk_path,
             scalar_dict={'parcellation': parc_streamlines})

    # 5. Calcul des stats (Appel de ta fonction originale)
    # Note : Le CSV s'appellera 'centroid_scalar_means.csv' selon ton code
    write_csv_from_scalars(
        scalar_arrays,
        point_indices_correspondance,
        point_cluster_ids=point_indices_correspondance,  # On utilise les mêmes IDs pour le clustering
        out_dir=temp_dir
    )

    # 6. Post-processing du CSV pour renommer les IDs (optionnel)
    # Par défaut, le CSV aura 0, 1, 2 dans 'point_id'. 
    # On pourrait mapper ces IDs vers les vrais labels unique_parcels.

    entities = subject_bundle.get_full_entities()
    res_dict = {
        output_vtk_path: upt_dict(entities, desc='parcellation', space='subject'),
        os.path.join(temp_dir, 'centroid_scalar_stats.csv'): upt_dict(entities, suffix='stats', datatype='metric', extension='csv'),
    }

    return res_dict
    
    

    
    
def get_stat_from_association_file(association_file, metrics_file):
    """
    Extrait des statistiques à partir d'un fichier d'association de streamlines.
    
    Parameters
    ----------
    association_file : BIDSFile
        Fichier VTK contenant les associations de streamlines.
    metrics_file : BIDSFile
        Fichier VTK contenant les métriques des streamlines.
        
    Returns
    -------
    result_dict : dict
        Dictionnaire contenant les chemins des fichiers CSV générés avec leurs entités.
    """
    streamlines, scalar_arrays = load_vtk_streamlines(association_file.path)
    streamlines_metrics, scalar_arrays_metrics = load_vtk_streamlines(metrics_file.path)
    if 'centroid_index' not in scalar_arrays:
        if 'cluster_association' in scalar_arrays:
            scalar_arrays['centroid_index'] = scalar_arrays['cluster_association']
        else:
            raise ValueError("Le fichier d'association doit contenir les arrays 'centroid_index' ou 'cluster_association'.")

    centroid_indices = scalar_arrays['centroid_index']
    association_distances = scalar_arrays['association_distance']
    centroid_point_distances = scalar_arrays['centroid_point_distance']
    point_indices = scalar_arrays['point_index']

    unique_centroids = np.unique(centroid_indices[centroid_indices >= 0])
    n_centroids = len(unique_centroids)
    n_streamlines = len(streamlines)

    temp_dir = tempfile.mkdtemp()
    entities = association_file.get_full_entities()
    result_dict = {}

    # 1. CSV avec les statistiques globales
    global_stats = {
        'n_centroids': n_centroids,
        'n_streamlines': n_streamlines,
        'mean_association_distance': np.mean(association_distances[centroid_indices >= 0]),
        'std_association_distance': np.std(association_distances[centroid_indices >= 0]),
        'min_association_distance': np.min(association_distances[centroid_indices >= 0]),
        'max_association_distance': np.max(association_distances[centroid_indices >= 0]),
        'mean_centroid_point_distance': np.mean(centroid_point_distances[centroid_indices >= 0]),
        'std_centroid_point_distance': np.std(centroid_point_distances[centroid_indices >= 0]),
        'min_centroid_point_distance': np.min(centroid_point_distances[centroid_indices >= 0]),
        'max_centroid_point_distance': np.max(centroid_point_distances[centroid_indices >= 0]),
    }

    for name, values in scalar_arrays_metrics.items():
        if len(values) == len(streamlines_metrics):
            mean_val = np.mean([values[i] for i in range(len(streamlines_metrics)) if centroid_indices[i] >= 0])
            std_val = np.std([values[i] for i in range(len(streamlines_metrics)) if centroid_indices[i] >= 0])
            global_stats[f'mean_{mcm_name_mapping.get(name, name)}'] = mean_val
            global_stats[f'std_{mcm_name_mapping.get(name, name)}'] = std_val

    csv_global_path = os.path.join(temp_dir, 'stats_global.csv')
    with open(csv_global_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in global_stats.items():
            writer.writerow([key, value])
    
    result_dict[csv_global_path] = upt_dict(entities, desc='global', suffix='stats', datatype='metric', extension='csv')
    print(f"Statistiques globales sauvegardées dans {csv_global_path}")

    # 2. CSV avec les statistiques par centroïde
    centroid_stats_rows = []
    for centroid in unique_centroids:
        mask = centroid_indices == centroid
        row = {
            'centroid_id': int(centroid),
            'n_streamlines': int(np.sum(mask)),
            'mean_association_distance': float(np.mean(association_distances[mask])),
            'std_association_distance': float(np.std(association_distances[mask])),
            'min_association_distance': float(np.min(association_distances[mask])),
            'max_association_distance': float(np.max(association_distances[mask])),
            'mean_centroid_point_distance': float(np.mean(centroid_point_distances[mask])),
            'std_centroid_point_distance': float(np.std(centroid_point_distances[mask])),
            'min_centroid_point_distance': float(np.min(centroid_point_distances[mask])),
            'max_centroid_point_distance': float(np.max(centroid_point_distances[mask])),
        }
        
        for name, values in scalar_arrays_metrics.items():
            if len(values) == len(streamlines_metrics):
                mean_val = np.mean([values[i] for i in range(len(streamlines_metrics)) if mask[i]])
                std_val = np.std([values[i] for i in range(len(streamlines_metrics)) if mask[i]])
                row[f'mean_{mcm_name_mapping.get(name, name)}'] = float(mean_val)
                row[f'std_{mcm_name_mapping.get(name, name)}'] = float(std_val)
        
        centroid_stats_rows.append(row)

    csv_centroid_path = os.path.join(temp_dir, 'stats_bycentroid.csv')
    if centroid_stats_rows:
        with open(csv_centroid_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=centroid_stats_rows[0].keys())
            writer.writeheader()
            writer.writerows(centroid_stats_rows)
        
        result_dict[csv_centroid_path] = upt_dict(entities, desc='bycentroid', suffix='stats', datatype='metric', extension='csv')
        print(f"Statistiques par centroïde sauvegardées dans {csv_centroid_path}")

    # 3. CSV avec les statistiques par point
    point_centroid_indices = scalar_arrays['centroid_index']
    unique_point_indices = np.unique(point_indices[point_centroid_indices >= 0])
    point_values = {name: [] for name in scalar_arrays_metrics.keys() if len(scalar_arrays_metrics[name]) == len(point_indices)}
    
    for i, idx in enumerate(point_indices):
        if point_centroid_indices[i] >= 0:
            for name in point_values.keys():
                point_values[name].append((idx, scalar_arrays_metrics[name][i]))

    point_stats_rows = []
    for name, vals in point_values.items():
        vals_by_point = {}
        for idx, val in vals:
            if idx not in vals_by_point:
                vals_by_point[idx] = []
            vals_by_point[idx].append(val)
        
        for idx, vlist in sorted(vals_by_point.items()):
            mean_val = np.mean(vlist)
            std_val = np.std(vlist)
            point_stats_rows.append({
                'point_index': int(idx),
                'metric': mcm_name_mapping.get(name, name),
                'mean': float(mean_val),
                'std': float(std_val),
                'n_samples': len(vlist)
            })

    csv_point_path = os.path.join(temp_dir, 'stats_bypoint.csv')
    if point_stats_rows:
        with open(csv_point_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['point_index', 'metric', 'mean', 'std', 'n_samples'])
            writer.writeheader()
            writer.writerows(point_stats_rows)
        
        result_dict[csv_point_path] = upt_dict(entities, desc='bypoint', suffix='stats', datatype='metric', extension='csv')
        print(f"Statistiques par point sauvegardées dans {csv_point_path}")
    print(result_dict)
    return result_dict
    


if __name__ == "__main__":
    tracto = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/bundle_seg/sub-01002/tracto/sub-01002_bundle-CSTleft_desc-cleaned_tracto.trk"

    cluster_streamlines(tracto, threshold=100.)
