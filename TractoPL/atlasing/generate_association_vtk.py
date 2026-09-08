import os
import tempfile
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric, mean_manhattan_distance
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.io.streamline import load_tractogram
from TractoPL.data.vtk_loader import load_vtk, save_vtk

def compute_distance_from_centroids(streamlines, centroids, n_pts=12):
    associations = []
    for index_streamline, streamline in enumerate(streamlines):
        streamline_resampled = set_number_of_points(streamline, n_pts)
        min_distance = 999.0
        best_centroid = -1
        best_flip = False

        for index_centroid, centroid in enumerate(centroids):
            centroid_resampled = set_number_of_points(centroid, n_pts)
            dist_direct = mean_manhattan_distance(streamline_resampled, centroid_resampled)
            dist_flipped = mean_manhattan_distance(streamline_resampled, centroid_resampled[::-1])
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


def associate_vtk_to_centroids(subject_vtk_path,
                               model_centroids_path,
                               model_full_bundle_path,
                               reference_nifti,
                               n_pts=50,
                               temp_dir=None,
                               slr=False,
                               model_clustering=5.0):
    """
    Associe les streamlines d'un VTK sujet aux centroids d'un modèle.

    Parameters
    ----------
    subject_vtk_path : str
        Chemin vers le fichier VTK contenant les streamlines du sujet
    model_centroids_path : str
        Chemin vers le fichier VTK contenant les centroids du modèle
    model_full_bundle_path : str
        Chemin vers le fichier VTK contenant le bundle complet du modèle
    reference_nifti : str
        Chemin vers l'image de référence NIfTI
    n_pts : int, optional
        Nombre de points pour l'association point-à-point (défaut: 50)
    temp_dir : str, optional
        Répertoire temporaire (par défaut utilise tempfile.mkdtemp())
    slr : bool, optional
        Utiliser le Streamline Linear Registration (défaut: True)
    model_clustering : float, optional
        Seuil de clustering du modèle (défaut: 5.0)

    Returns
    -------
    res_dict : dict
        Dictionnaire avec les clés :
        - 'associations_vtk' : chemin vers le VTK des streamlines avec associations
        - 'centroids_vtk' : chemin vers le VTK des centroids transformés
        - 'point_index_per_cluster' : dict {cluster_id: array d'indices de points}
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    # Charger les streamlines du sujet
    streamlines, scalar_arrays = load_vtk(subject_vtk_path)
    centroids, _ = load_vtk(model_centroids_path)

    slr_n_points = 12
    streamline_association_n_points = 12
    points_association_n_points = n_pts

    centroids = set_number_of_points(centroids, points_association_n_points)

    print(f"Nombre de streamlines du sujet : {len(streamlines)}")
    print(f"Nombre de centroids du modèle : {len(centroids)}")

    if slr:
        if model_full_bundle_path.endswith('.trk'):
            model_full_bundle_tracto = load_tractogram(
                model_full_bundle_path, reference_nifti, bbox_valid_check=False)
            model_full_bundle = model_full_bundle_tracto.streamlines
        else:
            model_full_bundle, _ = load_vtk_streamlines(model_full_bundle_path)

        print(f"Nombre de streamlines du bundle complet : {len(model_full_bundle)}")

        # Clustering du sujet
        subject_streamlines_resampled = [
            set_number_of_points(s, slr_n_points) for s in streamlines
        ]
        subject_streamlines_obj = Streamlines(subject_streamlines_resampled)

        metric = AveragePointwiseEuclideanMetric()
        qb_subject = QuickBundles(threshold=15., metric=metric)
        subject_clusters = qb_subject.cluster(
            set_number_of_points(subject_streamlines_obj, slr_n_points))
        subject_centroids = Streamlines(subject_clusters.centroids)
        print(f"Clustering du sujet : {len(subject_clusters)} clusters créés")

        # Clustering du modèle
        model_streamlines_resampled = [
            set_number_of_points(s, slr_n_points) for s in model_full_bundle
        ]
        model_streamlines_obj = Streamlines(model_streamlines_resampled)

        qb_model = QuickBundles(threshold=15., metric=metric)
        model_clusters = qb_model.cluster(
            set_number_of_points(model_streamlines_obj, slr_n_points))
        model_centroids = Streamlines(model_clusters.centroids)
        print(f"Clustering du modèle : {len(model_clusters)} clusters créés")

        print("Réalisation du SLR...")
        time_start = os.times()
        try:
            slr_reg = StreamlineLinearRegistration()
            slm = slr_reg.optimize(subject_centroids, model_centroids)
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

    # Association des streamlines avec les centroids transformés
    print("Calcul des associations entre les streamlines du sujet et les centroids transformés...")

    associations = compute_distance_from_centroids(
        streamlines, transformed_centroids, n_pts=streamline_association_n_points)

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

    # Arrays pour les centroids
    centroid_indices = []
    global_point_indices = []
    global_point_counter = 0

    for i, streamline in enumerate(streamlines):
        cluster_id = streamline_cluster_ids[i]
        flip = associations[i][3]
        assoc_distance = associations[i][2]

        if cluster_id >= 0:
            centroid = transformed_centroids[cluster_id].copy()
            n_centroid_pts = len(centroid)
            n_streamline_pts = len(streamline)

            raw_indices = np.arange(n_centroid_pts)
            if flip:
                norm_idx_centroid = (n_centroid_pts - 1 - raw_indices) / n_centroid_pts
            else:
                norm_idx_centroid = raw_indices / n_centroid_pts

            norm_idx_streamline = np.arange(n_streamline_pts) / n_streamline_pts

            arc_length = float(
                np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1)))
            alpha = arc_length
            centroid_augmented = np.column_stack(
                [centroid, alpha * norm_idx_centroid])
            streamline_augmented = np.column_stack(
                [streamline, alpha * norm_idx_streamline])

            tree = cKDTree(centroid_augmented)
            _, indices_order = tree.query(streamline_augmented, k=1)

            dists_3d = np.linalg.norm(
                streamline - centroid[indices_order], axis=1)

            point_centroid_distances.extend(dists_3d.tolist())
            point_indices_correspondance.extend(indices_order.tolist())
            point_distances.extend([assoc_distance] * n_streamline_pts)
            point_cluster_ids.extend([cluster_id] * n_streamline_pts)
        else:
            point_centroid_distances.extend([999.0] * len(streamline))
            point_indices_correspondance.extend([-1] * len(streamline))
            point_distances.extend([999.0] * len(streamline))
            point_cluster_ids.extend([-1] * len(streamline))

    # Sauvegarder le VTK des streamlines avec associations
    output_vtk_path = os.path.join(temp_dir, 'streamlines_with_associations.vtk')
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_path)
    polydata_associations = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polydata_associations.SetPoints(points)
    lines = vtk.vtkCellArray()

    for streamline in streamlines:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(streamline))
        for j, point in enumerate(streamline):
            point_id = points.InsertNextPoint(point)
            line.GetPointIds().SetId(j, point_id)
        lines.InsertNextCell(line)

    polydata_associations.SetLines(lines)

    print('Nombre centroids uniques : ', np.unique(np.array(point_cluster_ids)))

    cluster_array = numpy_to_vtk(np.array(point_cluster_ids), deep=True)
    cluster_array.SetName('centroid_index')
    polydata_associations.GetPointData().AddArray(cluster_array)

    distance_array = numpy_to_vtk(np.array(point_distances), deep=True)
    distance_array.SetName('association_distance')
    polydata_associations.GetPointData().AddArray(distance_array)

    centroid_distance_array = numpy_to_vtk(
        np.array(point_centroid_distances), deep=True)
    centroid_distance_array.SetName('centroid_point_distance')
    polydata_associations.GetPointData().AddArray(centroid_distance_array)

    point_correspondence_array = numpy_to_vtk(
        np.array(point_indices_correspondance), deep=True)
    point_correspondence_array.SetName('point_index')
    polydata_associations.GetPointData().AddArray(point_correspondence_array)

    # Ajouter un array point_index par cluster (avec -1 pour les points hors cluster)
    point_cluster_ids_arr = np.array(point_cluster_ids)
    point_indices_arr = np.array(point_indices_correspondance)
    unique_clusters = np.unique(point_cluster_ids_arr)
    for cid in unique_clusters:
        if cid >= 0:
            arr = np.full(len(point_cluster_ids_arr), -1, dtype=int)
            mask = point_cluster_ids_arr == cid
            arr[mask] = point_indices_arr[mask]
            vtk_arr = numpy_to_vtk(arr, deep=True)
            vtk_arr.SetName(f'point_index_cluster_{int(cid)}')
            polydata_associations.GetPointData().AddArray(vtk_arr)

    writer.SetInputData(polydata_associations)
    writer.Write()

    # Sauvegarder les centroids transformés
    centroids_vtk_path = os.path.join(temp_dir, 'transformed_centroids.vtk')
    centroid_writer = vtk.vtkPolyDataWriter()
    centroid_writer.SetFileName(centroids_vtk_path)
    centroid_polydata = vtk.vtkPolyData()
    centroid_points = vtk.vtkPoints()
    centroid_polydata.SetPoints(centroid_points)
    centroid_lines = vtk.vtkCellArray()

    print('Nbr centroids : ', len(centroids))
    for centroid_idx, centroid in enumerate(centroids):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(centroid))
        for j, point in enumerate(centroid):
            point_id = centroid_points.InsertNextPoint(point)
            line.GetPointIds().SetId(j, point_id)
            centroid_indices.append(centroid_idx)
            global_point_indices.append(global_point_counter)
            global_point_counter += 1
        centroid_lines.InsertNextCell(line)

    centroid_polydata.SetLines(centroid_lines)

    centroid_index_array = numpy_to_vtk(np.array(centroid_indices), deep=True)
    centroid_index_array.SetName('centroid_index')
    centroid_polydata.GetPointData().AddArray(centroid_index_array)

    point_index_array = numpy_to_vtk(np.array(global_point_indices), deep=True)
    point_index_array.SetName('point_index')
    centroid_polydata.GetPointData().AddArray(point_index_array)

    centroid_writer.SetInputData(centroid_polydata)
    centroid_writer.Write()

    res_dict = {
        'associations_vtk': output_vtk_path,
        'centroids_vtk': centroids_vtk_path,
    }

    return res_dict

from glob import glob
import shutil
from multiprocessing import Pool


def process_bundle(args):
    bundle, clustering_method, reference_nifti = args
    bundle_name = bundle.split('summed_')[-1].split('.vtk')[0]

    output_dir = os.path.join(clustering_method, 'associations')
    dest_path = os.path.join(output_dir, f'{bundle_name}_associations.vtk')

    if os.path.exists(dest_path):
        print(f"SKIP {dest_path} (déjà traité)")
        return

    print(f"Traitement du bundle {bundle_name} avec la méthode de clusterisation {clustering_method}...")

    if not 'umap' in clustering_method:
        centroid = glob(f'{clustering_method}/*_{bundle_name}_centroids.vtk')
    else:
        centroid = glob(f'{clustering_method}/{bundle_name}_centroids.vtk')
    if len(centroid) != 1:
        print(f"Erreur : trouvé {len(centroid)} fichiers de centroids pour le bundle {bundle_name} et la méthode {clustering_method}.")
        raise ValueError("Il doit y avoir exactement un fichier de centroids correspondant.")
    centroid_path = centroid[0]
    res = associate_vtk_to_centroids(subject_vtk_path=bundle,
                                     model_centroids_path=centroid_path,
                                     model_full_bundle_path=bundle,
                                     reference_nifti=reference_nifti,
                                     n_pts=50)

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(res['associations_vtk'], dest_path)
    print(f"Copié vers {dest_path}")
    #Remove temp files
    os.remove(res['associations_vtk'])
    os.remove(res['centroids_vtk'])


if __name__ == '__main__':
    pwd = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    os.chdir(pwd)
    CLUSTERING_METHODS = ['flipped/centroids_frechetlong2', 'flipped/centroids_longcentral', 'UMAP_endpoints/centroids_umapendpoints']

    bundles = glob('vtk/summed_*.vtk')
    reference_nifti = 'average_fa.nii.gz'

    tasks = [(bundle, cm, reference_nifti)
             for cm in CLUSTERING_METHODS
             for bundle in bundles]

    with Pool(8) as pool:
        pool.map(process_bundle, tasks)

