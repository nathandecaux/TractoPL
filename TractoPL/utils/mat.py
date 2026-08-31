from TractoPL.data.vtk_loader import load_vtk_streamlines, save_vtk
from TractoPL.data.mcmfile import MCMFile, read_mcm_file
from TractoPL.data.loader import BIDSFile
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import os

def tensor_components_to_matrix(components: np.ndarray) -> np.ndarray:
    """Convertit les 6 composantes DTI en matrices 3x3 symétriques."""
    spatial_shape = components.shape[:-1]
    tensors = np.zeros((*spatial_shape, 3, 3), dtype=np.float32)
    
    # Ordre Anima : Dxx, Dxy, Dyy, Dxz, Dyz, Dzz
    tensors[..., 0, 0] = components[..., 0]  # Dxx
    tensors[..., 0, 1] = components[..., 1]  # Dxy
    tensors[..., 1, 0] = components[..., 1]  # Dxy
    tensors[..., 1, 1] = components[..., 2]  # Dyy
    tensors[..., 0, 2] = components[..., 3]  # Dxz
    tensors[..., 2, 0] = components[..., 3]  # Dxz
    tensors[..., 1, 2] = components[..., 4]  # Dyz
    tensors[..., 2, 1] = components[..., 4]  # Dyz
    tensors[..., 2, 2] = components[..., 5]  # Dzz
    
    return tensors


def extract_principal_eigenvector(tensors: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Extrait le vecteur propre principal de chaque tenseur."""
    spatial_shape = tensors.shape[:-2]
    peaks = np.zeros((*spatial_shape, 3), dtype=np.float32)
    
    flat_tensors = tensors.reshape(-1, 3, 3)
    flat_peaks = peaks.reshape(-1, 3)
    
    for i, tensor in enumerate(flat_tensors):
        try:
            eigenvals, eigenvecs = np.linalg.eigh(tensor)
            # Trier par valeur propre décroissante
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Prendre le vecteur propre principal si la valeur propre est suffisante
            if True:#eigenvals[0] > threshold:
                principal_vec = eigenvecs[:, 0]
                # Amplitude = racine de la valeur propre principale
                amplitude = np.sqrt(max(0, eigenvals[0]))
                flat_peaks[i] = principal_vec * amplitude
        except np.linalg.LinAlgError:
            # En cas d'erreur, laisser le pic à zéro
            pass
    
    return peaks


def lps_vectors_to_ras(lps_peaks: np.ndarray) -> np.ndarray:
    """Convertit les vecteurs LPS vers RAS."""
    ras_peaks = lps_peaks.copy()
    # ras_peaks[..., 2] *= -1.0  # Inverser l'axe Z (Sup -> Inf)
    ras_peaks[..., 0] *= -1.0  # L -> R (Left -> Right)
    ras_peaks[..., 1] *= -1.0  # P -> A (Posterior -> Anterior)
    return ras_peaks


def build_ras_affine(lps_affine: np.ndarray) -> np.ndarray:
    """Convertit l'affine LPS vers RAS."""
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    # flip = np.diag([1.0, 1.0, 1.0, 1.0])
    return flip @ lps_affine

def save_as_peaks(peaks, streamlines, reference, output_path):
    """
    Sauvegarde les pics extraits des streamlines au format MRtrix peaks.
    Assigne le peak le plus proche pour chaque voxel.
    
    Parameters
    ----------
    peaks : list
        Liste de pics pour chaque streamline (un array de shape (n_points, 3) par streamline)
    streamlines : list
        Liste des streamlines (coordonnées spatiales des points)
    reference : BIDSFile
        Image de référence pour définir l'espace et l'affine
    output_path : str or Path
        Chemin de sortie pour l'image des pics
    """
    # Charger l'image de référence
    ref_img = nib.load(reference.path)
    spatial_shape = ref_img.shape[:3]
    
    # Initialiser l'image de pics (3 composantes par voxel pour un seul pic)
    peaks_image = np.zeros((*spatial_shape, 3), dtype=np.float32)
    peak_count = np.zeros(spatial_shape, dtype=np.int32)  # Compteur pour moyenner
    
    # Pour chaque streamline
    for sl_idx, (streamline, sl_peaks) in enumerate(zip(streamlines, peaks)):
        # Convertir les coordonnées physiques en coordonnées voxel
        inv_affine = np.linalg.inv(ref_img.affine)
        
        for pt_idx, (point, peak) in enumerate(zip(streamline, sl_peaks)):
            # Convertir coordonnées physiques -> voxel
            point_homogeneous = np.append(point, 1)
            voxel_coords = inv_affine @ point_homogeneous
            voxel_coords = voxel_coords[:3].astype(int)
            
            # Vérifier que le voxel est dans les limites
            if (0 <= voxel_coords[0] < spatial_shape[0] and
                0 <= voxel_coords[1] < spatial_shape[1] and
                0 <= voxel_coords[2] < spatial_shape[2]):
                
                i, j, k = voxel_coords
                
                # Si c'est le premier pic pour ce voxel, l'assigner directement
                if peak_count[i, j, k] == 0:
                    peaks_image[i, j, k] = peak.squeeze()
                else:
                    # Sinon, garder le pic avec la plus grande amplitude
                    current_peak = peaks_image[i, j, k]
                    current_amplitude = np.linalg.norm(current_peak)
                    new_amplitude = np.linalg.norm(peak)
                    
                    if new_amplitude > current_amplitude:
                        peaks_image[i, j, k] = peak.squeeze()
                
                peak_count[i, j, k] += 1
    
    # Convertir LPS vers RAS si nécessaire
    peaks_image = lps_vectors_to_ras(peaks_image)
    
    # Construire l'affine RAS
    ras_affine = build_ras_affine(ref_img.affine)
    
    # Créer l'en-tête de sortie
    output_header = ref_img.header.copy()
    output_header.set_data_dtype(np.float32)
    output_header.set_data_shape(peaks_image.shape)
    output_header.set_qform(ras_affine, code=1)
    output_header.set_sform(ras_affine, code=1)
    
    # Sauvegarder
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_img = nib.Nifti1Image(peaks_image, ras_affine, output_header)
    nib.save(output_img, str(output_path))
    
    print(f"Pics sauvegardés : {output_path}")
    print(f"Forme de sortie : {peaks_image.shape}")
    print(f"Voxels avec pics : {np.sum(peak_count > 0)}")


def compute_mcm_metrics(mcm_vtk,mcm,reference,**kwargs):
    """
    Compute MCM metrics for given streamlines and associated scalar data.

    Parameters
    ----------
    mcm_vtk : BIDSFile
        An BIDSFile instance containing the VTK streamlines and scalar data.
    mcm : BIDSFile
        An BIDSFile instance containing MCM data.
    **kwargs : dict
        Additional keyword arguments to pass to the metric computation function.

    Returns
    -------
    metrics_vtk : BIDSFile
        An BIDSFile instance containing the computed MCM metrics as scalar data
        associated with the streamlines.
    """

    streamlines, scalar_dict = load_vtk_streamlines(mcm_vtk.path, reshape_scalars=True)
    mcm_data = MCMFile(mcm.path)
    tensors = {k:v for k,v in mcm_data.compartments.items() if 'Tensor' in v['type']}
    #Add idx in tensors dict, as the order of the compartment key
    comp_idx=np.sort([int(k) for k in tensors.keys()])
    for idx,k in enumerate(comp_idx):
        tensors[str(k)]['idx']=idx+1

    mostColinear=scalar_dict['MostColinearIndex']

    #Compute angle between tensor main directions and the streamline direction, and FA for all tensors
    angles_mostcolinear=[]
    fas_mostcolinear=[]
    peaks=[]
    angles_all_tensors={str(k): [] for k in tensors.keys()}
    fas_all_tensors={str(k): [] for k in tensors.keys()}
    corrected_mostcolinear=[]
    modified_mostcolinear=[]
    
    for sl_idx,sl in enumerate(streamlines):
        sl_angles_mc=[]
        sl_fas_mc=[]
        sl_peaks=[]
        sl_angles_all={str(k): [] for k in tensors.keys()}
        sl_fas_all={str(k): [] for k in tensors.keys()}
        sl_corrected_mc=[]
        sl_modified_mc=[]
        
        for pt_idx,pt in enumerate(sl):
            # Calculer la tangente de la streamline
            if pt_idx==0:
                tangent=sl[pt_idx+1]-sl[pt_idx]
            elif pt_idx==len(sl)-1:
                tangent=sl[pt_idx]-sl[pt_idx-1]
            else:
                tangent=sl[pt_idx+1]-sl[pt_idx-1]
            tangent=tangent/np.linalg.norm(tangent)
            
            mc_idx=int(mostColinear[sl_idx][pt_idx])
            
            # Calculer les métriques pour tous les tenseurs
            tensor_angles={}
            tensor_fas={}
            
            for tensor_key in tensors.keys():
                tensor_idx=tensors[tensor_key]['idx']
                tensor_params=[scalar_dict[f'Tensor{tensor_idx}Parameter{p+1}'][sl_idx][pt_idx] for p in range(6)]
                tensor_matrix=tensor_components_to_matrix(np.array(tensor_params))
                
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(tensor_matrix)
                    # Trier par valeur propre décroissante
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    principal_vec = eigenvecs[:, 0]
                    
                    # Angle (entre 0 et 90 degrés)
                    cos_angle=np.clip(np.abs(np.dot(principal_vec,tangent)),0.0,1.0)
                    angle=np.arccos(cos_angle)*(180.0/np.pi)
                    tensor_angles[tensor_key]=angle
                    
                    # FA
                    l1,l2,l3=eigenvals
                    fa=np.sqrt(0.5*((l1 - l2)**2 + (l2 - l3)**2 + (l3 - l1)**2)) / np.sqrt(l1**2 + l2**2 + l3**2) if (l1**2 + l2**2 + l3**2)>0 else 0.0
                    tensor_fas[tensor_key]=fa
                except np.linalg.LinAlgError:
                    tensor_angles[tensor_key]=90.0  # Angle maximum en cas d'erreur
                    tensor_fas[tensor_key]=0.0
            
            # Vérifier que le MostColinear correspond bien au plus petit angle
            if mc_idx==0 or not tensor_angles:
                # Pas de tenseur
                sl_angles_mc.append(90.0)
                sl_fas_mc.append(0.0)
                sl_peaks.append(np.array([0.0,0.0,0.0]))
                sl_corrected_mc.append(0)
                sl_modified_mc.append(0)
                for tensor_key in tensors.keys():
                    sl_angles_all[tensor_key].append(90.0)
                    sl_fas_all[tensor_key].append(0.0)
            else:
                # Trouver le tenseur avec le plus petit angle
                min_angle_key = min(tensor_angles, key=tensor_angles.get)
                min_angle_idx = int(min_angle_key)
                
                # Stocker les métriques du tenseur avec le plus petit angle
                sl_corrected_mc.append(min_angle_idx)
                sl_modified_mc.append(int(min_angle_idx != mc_idx))
                sl_angles_mc.append(tensor_angles[min_angle_key])
                sl_fas_mc.append(tensor_fas[min_angle_key])
                
                # Extraire le peak du tenseur le plus colinéaire
                tensor_idx=tensors[min_angle_key]['idx']
                tensor_params=[scalar_dict[f'Tensor{tensor_idx}Parameter{p+1}'][sl_idx][pt_idx] for p in range(6)]
                tensor_matrix=tensor_components_to_matrix(np.array(tensor_params))
                peak=extract_principal_eigenvector(tensor_matrix[np.newaxis,...])
                sl_peaks.append(peak)
                
                # Stocker les métriques de tous les tenseurs
                for tensor_key in tensors.keys():
                    sl_angles_all[tensor_key].append(tensor_angles[tensor_key])
                    sl_fas_all[tensor_key].append(tensor_fas[tensor_key])
        
        angles_mostcolinear.append(sl_angles_mc)
        fas_mostcolinear.append(sl_fas_mc)
        peaks.append(sl_peaks)
        corrected_mostcolinear.append(sl_corrected_mc)
        modified_mostcolinear.append(sl_modified_mc)
        for tensor_key in tensors.keys():
            angles_all_tensors[tensor_key].append(sl_angles_all[tensor_key])
            fas_all_tensors[tensor_key].append(sl_fas_all[tensor_key])
    
    # Créer un répertoire temporaire
    temp_dir = tempfile.mkdtemp()
    
    # Sauvegarder les peaks
    peaks_path = os.path.join(temp_dir, 'peaks.nii.gz')
    save_as_peaks(peaks, streamlines, reference, output_path=peaks_path)

    # Calculer le pourcentage de tensors corrigés
    total_points = sum(len(sl_corrected) for sl_corrected in corrected_mostcolinear)
    corrected_count = 0
    for sl_idx, sl_corrected in enumerate(corrected_mostcolinear):
        for pt_idx, corrected_idx in enumerate(sl_corrected):
            original_idx = int(mostColinear[sl_idx][pt_idx])
            if corrected_idx != original_idx and corrected_idx != 0:
                corrected_count += 1
    
    correction_percentage = (corrected_count / total_points * 100) if total_points > 0 else 0
    print(f"Tensors corrigés : {corrected_count}/{total_points} ({correction_percentage:.2f}%)")

    new_scalar_dict={}
    # Ajouter les métriques au dictionnaire de scalaires
    new_scalar_dict['Tensor_Angle_MostColinear'] = angles_mostcolinear
    new_scalar_dict['Tensor_FA_MostColinear'] = fas_mostcolinear
    new_scalar_dict['CorrectedMostColinearIndex'] = corrected_mostcolinear
    new_scalar_dict['ModifiedMostColinear'] = modified_mostcolinear
    
    # Ajouter les angles et FA pour tous les tenseurs
    for tensor_key in tensors.keys():
        tensor_idx = tensors[tensor_key]['idx']
        new_scalar_dict[f'Tensor{tensor_idx}_Angle'] = angles_all_tensors[tensor_key]
        new_scalar_dict[f'Tensor{tensor_idx}_FA'] = fas_all_tensors[tensor_key]
    
    # Sauvegarder le VTK avec les nouvelles métriques
    vtk_path = os.path.join(temp_dir, 'metrics.vtk')
    save_vtk(streamlines, vtk_path, new_scalar_dict)
    
    # Créer le dictionnaire de résultats avec entities
    entities = mcm_vtk.get_entities()
    entities.update(kwargs)
    
    res_dict = {
        vtk_path: dict(entities, suffix='tracto', extension='vtk', desc='tensors',datatype='mat'),
        peaks_path: dict(entities, suffix='peaks', extension='nii.gz', datatype='dwi',desc='mostcolinear')
    }
    
    return res_dict

if __name__ == "__main__":
    mcm = BIDSFile("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors_staniz/sub-01001/dwi/sub-01001_desc-preproc_model-MCM_dwi.mcmx")
    mcm_vtk  = BIDSFile("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors_staniz/sub-01001/tracto/sub-01001_bundle-CSTleft_desc-full_model-MCM_tracto.vtk")
    reference= BIDSFile("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/preprocessing/sub-01001/dwi/sub-01001_metric-FA_model-DTI_dwi.nii.gz")
    print(mcm.path)
    res = compute_mcm_metrics(mcm_vtk,mcm,reference)
    from pprint import pprint
    pprint(res)