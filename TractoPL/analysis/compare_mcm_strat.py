from TractoPL.data.loader import Dataset, Subject, BIDSFile
from TractoPL.data.mcmfile import MCMFile
import SimpleITK as sitk
import numpy as np
from collections import defaultdict
import pandas as pd
import os


ds = Dataset()

mcms = ds.get_global(model='MCM', datatype='dwi', extension='.mcmx', pipeline=['mcm_tensors_staniz_reproducible','mcm_tensors_no_isorestricted','mcm_tensors_Z','mcm_tensors_staniz_with_AIC',"mcm_tensors_S",'mcm_tensors_Z_with_AIC'])
mcms += ds.get_global(model='MCM', datatype='dwi', extension='.mcmx', pipeline='mcm_tensors_isorestricted',subject=[f.subject for f in mcms])
original_mcms = ds.get_global(model='MCM', datatype='dwi', extension='.mcmx', pipeline='mcm_tensors_staniz', subject=[f.subject for f in mcms])
original_tensor_segmentations=ds.get_global(pipeline='msmt_csd',suffix='density',desc='fixels2peaks',extension='.nii.gz',subject=[f.subject for f in mcms])
print(f'Found {len(mcms)} MCM files in the dataset.')
print(f'Found {len(original_mcms)} original MCM files in the dataset.')

# Organiser les fichiers par sujet et pipeline
mcms_by_subject = defaultdict(dict)
for mcm_file in mcms:
    mcms_by_subject[mcm_file.subject][mcm_file.pipeline] = mcm_file

original_by_subject = {mcm_file.subject: mcm_file for mcm_file in original_mcms}

# Liste pour stocker les résultats
results = []

# Pour chaque sujet, comparer les poids du compartiment FreeWater
for subject in sorted(mcms_by_subject.keys()):
    print(f"\n{'='*80}")
    print(f"Sujet: {subject}")
    print(f"{'='*80}")
    
    if subject not in original_by_subject:
        print(f"  ⚠️  Pas de MCM original trouvé pour {subject}")
        continue
    
    # Charger le MCM original
    original_mcm = MCMFile(original_by_subject[subject].path)
    
    # Trouver le compartiment FreeWater dans l'original
    original_fw_comp = None
    for comp_num, comp_info in original_mcm.compartments.items():
        if 'free' in comp_info['type'].lower() or 'water' in comp_info['type'].lower():
            original_fw_comp = comp_num
            break
    
    if original_fw_comp is None:
        print(f"  ⚠️  Pas de compartiment FreeWater trouvé dans le MCM original")
        continue
    
    print(f"  Compartiment FreeWater original: {original_fw_comp} ({original_mcm.compartments[original_fw_comp]['type']})")
    
    # Charger les données des poids du compartiment FreeWater original
    if not hasattr(original_mcm, 'weights') or not os.path.exists(original_mcm.weights):
        print(f"  ⚠️  Fichier de poids introuvable pour l'original")
        continue
    
    original_fw_index = original_mcm.compartments[original_fw_comp]['index']
    original_weights_img = sitk.ReadImage(original_mcm.weights)
    original_weights_data = sitk.GetArrayFromImage(original_weights_img)
    original_fw_weights = original_weights_data[..., original_fw_index]
    
    # Pour chaque pipeline, comparer avec l'original
    for pipeline in sorted(mcms_by_subject[subject].keys()):
        print(f"\n  Pipeline: {pipeline}")
        print(f"  {'-'*76}")
        
        mcm_file = mcms_by_subject[subject][pipeline]
        mcm = MCMFile(mcm_file.path)
        
        # Trouver le compartiment FreeWater
        fw_comp = None
        for comp_num, comp_info in mcm.compartments.items():
            if 'free' in comp_info['type'].lower() or 'water' in comp_info['type'].lower():
                fw_comp = comp_num
                break
        
        if fw_comp is None:
            print(f"    ⚠️  Pas de compartiment FreeWater trouvé")
            continue
        
        print(f"    Compartiment: {fw_comp} ({mcm.compartments[fw_comp]['type']})")
        
        # Charger les données des poids
        if not hasattr(mcm, 'weights') or not os.path.exists(mcm.weights):
            print(f"    ⚠️  Fichier de poids introuvable")
            continue
        
        fw_index = mcm.compartments[fw_comp]['index']
        weights_img = sitk.ReadImage(mcm.weights)
        weights_data = sitk.GetArrayFromImage(weights_img)
        fw_weights = weights_data[..., fw_index]
        
        # Comparer les données
        if fw_weights.shape != original_fw_weights.shape:
            print(f"    ⚠️  Dimensions différentes: {fw_weights.shape} vs {original_fw_weights.shape}")
            results.append({
                'subject': subject,
                'pipeline': pipeline,
                'fw_compartment': fw_comp,
                'status': 'different_dimensions',
                'original_shape': str(original_fw_weights.shape),
                'pipeline_shape': str(fw_weights.shape)
            })
            continue
        
        # Calculer les statistiques de différence
        diff = fw_weights - original_fw_weights
        mask = ~np.isnan(original_fw_weights) & ~np.isnan(fw_weights)
        
        if np.sum(mask) == 0:
            print(f"    ⚠️  Aucune donnée valide pour la comparaison")
            results.append({
                'subject': subject,
                'pipeline': pipeline,
                'fw_compartment': fw_comp,
                'status': 'no_valid_data'
            })
            continue
        
        # Calculer les statistiques
        original_min = np.nanmin(original_fw_weights)
        original_max = np.nanmax(original_fw_weights)
        original_mean = np.nanmean(original_fw_weights)
        
        pipeline_min = np.nanmin(fw_weights)
        pipeline_max = np.nanmax(fw_weights)
        pipeline_mean = np.nanmean(fw_weights)
        
        diff_min = np.min(diff[mask])
        diff_max = np.max(diff[mask])
        diff_mean = np.mean(diff[mask])
        abs_diff_mean = np.mean(np.abs(diff[mask]))
        correlation = np.corrcoef(original_fw_weights[mask].flatten(), fw_weights[mask].flatten())[0,1]
        
        print(f"    Statistiques de comparaison:")
        print(f"      Original  - min: {original_min:.6f}, max: {original_max:.6f}, mean: {original_mean:.6f}")
        print(f"      Pipeline  - min: {pipeline_min:.6f}, max: {pipeline_max:.6f}, mean: {pipeline_mean:.6f}")
        print(f"      Différence - min: {diff_min:.6f}, max: {diff_max:.6f}, mean: {diff_mean:.6f}")
        print(f"      Différence absolue moyenne: {abs_diff_mean:.6f}")
        print(f"      Corrélation: {correlation:.6f}")
        
        # Vérifier si les données sont identiques
        is_identical = np.allclose(fw_weights[mask], original_fw_weights[mask], rtol=1e-5, atol=1e-8)
        max_diff_abs = np.max(np.abs(diff[mask]))
        
        if is_identical:
            print(f"    ✓ Les données sont identiques (tolerance: rtol=1e-5, atol=1e-8)")
            status = 'identical'
        else:
            print(f"    ✗ Les données diffèrent (différence max absolue: {max_diff_abs:.6f})")
            status = 'different'
        
        # Ajouter les résultats
        results.append({
            'subject': subject,
            'pipeline': pipeline,
            'fw_compartment': fw_comp,
            'status': status,
            'original_min': original_min,
            'original_max': original_max,
            'original_mean': original_mean,
            'pipeline_min': pipeline_min,
            'pipeline_max': pipeline_max,
            'pipeline_mean': pipeline_mean,
            'diff_min': diff_min,
            'diff_max': diff_max,
            'diff_mean': diff_mean,
            'abs_diff_mean': abs_diff_mean,
            'max_diff_abs': max_diff_abs,
            'correlation': correlation
        })

# Pour les pipelines avec _with_AIC, créer des segmentations des compartiments Tensor
print(f"\n{'='*80}")
print("Création des segmentations pour les pipelines _with_AIC")
print(f"{'='*80}")

for subject in sorted(mcms_by_subject.keys()):
    if subject not in original_by_subject:
        continue
    
    for pipeline in sorted(mcms_by_subject[subject].keys()):
        if '_with_AIC' not in pipeline:
            continue
        
        print(f"\nSujet: {subject}, Pipeline: {pipeline}")
        
        mcm_file = mcms_by_subject[subject][pipeline]
        mcm = MCMFile(mcm_file.path)
        
        if not hasattr(mcm, 'weights') or not os.path.exists(mcm.weights):
            print(f"  ⚠️  Fichier de poids introuvable")
            continue
        
        # Charger les données des poids
        weights_img = sitk.ReadImage(mcm.weights)
        weights_data = sitk.GetArrayFromImage(weights_img)
        
        # Trouver les compartiments Tensor
        tensor_compartments = []
        for comp_num, comp_info in mcm.compartments.items():
            if 'tensor' in comp_info['type'].lower():
                tensor_compartments.append(comp_info['index'])
        
        if not tensor_compartments:
            print(f"  ⚠️  Aucun compartiment Tensor trouvé")
            continue
        
        print(f"  Compartiments Tensor: {tensor_compartments}")
        
        # Créer la segmentation
        segmentation = np.zeros(weights_data.shape[:-1], dtype=np.uint8)  # Même shape que les données spatiales
        
        for idx in tensor_compartments:
            tensor_weights = weights_data[..., idx]
            binary_mask = (tensor_weights > 0).astype(np.uint8)
            segmentation += binary_mask
        
        # Sauvegarder comme nifti
        segmentation_img = sitk.GetImageFromArray(segmentation)
        segmentation_img.CopyInformation(weights_img)  # Copier les métadonnées
        
        output_filename = f"analysis/mcm_segmentations/{subject}_{pipeline}_tensor_segmentation.nii.gz"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        sitk.WriteImage(segmentation_img, output_filename)
        
        print(f"  ✓ Segmentation sauvegardée: {output_filename}")

# Créer des figures de comparaison avec les segmentations originales
print(f"\n{'='*80}")
print("Création des figures de comparaison avec les segmentations originales")
print(f"{'='*80}")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# Organiser les segmentations originales par sujet
original_seg_by_subject = {seg.subject: seg for seg in original_tensor_segmentations}

# Sélectionner quelques sujets pour les figures (par exemple les 2 premiers)
subjects_to_plot = sorted(list(original_seg_by_subject.keys()))[:2]

# Noms des axes pour les figures
axis_names = ['axial', 'coronal', 'sagittal']

for subject in subjects_to_plot:
    if subject not in original_seg_by_subject:
        continue
    
    print(f"Traitement du sujet: {subject}")
    
    # Charger la segmentation originale
    original_seg_path = original_seg_by_subject[subject].path
    original_seg_img = sitk.ReadImage(original_seg_path)
    original_seg_data = sitk.GetArrayFromImage(original_seg_img)
    
    # Trouver les pipelines _with_AIC pour ce sujet
    aic_pipelines = [p for p in mcms_by_subject[subject].keys() if '_with_AIC' in p]
    
    if not aic_pipelines:
        continue

    #Rotate by -90 degrees
    original_seg_data = np.rot90(original_seg_data, k=1, axes=(1,2))
    
    # Créer des figures pour chaque axe
    for axis_to_plot in range(3):
        print(f"  Axe {axis_names[axis_to_plot]} (axe {axis_to_plot})")

        # Coupes à différents niveaux selon l'axe
        slice_indices = [original_seg_data.shape[axis_to_plot] // 4, 
                        original_seg_data.shape[axis_to_plot] // 2, 
                        3 * original_seg_data.shape[axis_to_plot] // 4]
        
        # Créer une colormap discrète pour les labels (0 = transparent, 1+ = couleurs)
        max_label = np.max(original_seg_data)
        # Créer une colormap avec transparent pour 0
        colors = np.zeros((max_label + 1, 4))  # RGBA
        colors[0] = [0, 0, 0, 0]  # Transparent pour label 0
        jet_colors = plt.cm.jet(np.linspace(0, 1, max_label))
        colors[1:, :3] = jet_colors[:max_label, :3]  # RGB seulement
        colors[1:, 3] = 1  # Alpha = 1 pour les labels > 0
        cmap = ListedColormap(colors)

        # Créer une figure avec sous-plots : chaque pipeline + original, chaque coupe en ligne
        n_pipelines = len(aic_pipelines)
        n_slices = len(slice_indices)
        n_cols = n_pipelines + 1  # pipelines + original
        fig, axes = plt.subplots(n_slices, n_cols, figsize=(4*n_cols, 4*n_slices))
        if n_slices == 1:
            axes = np.expand_dims(axes, 0)
        if n_cols == 1:
            axes = np.expand_dims(axes, 1)
        # Afficher la segmentation originale dans la première colonne
        for j, slice_idx in enumerate(slice_indices):
            ax = axes[j, 0]
            if axis_to_plot == 0:  # Axial: slice, :, :
                original_slice = original_seg_data[slice_idx, :, :]
            elif axis_to_plot == 1:  # Coronal: :, slice, :
                original_slice = original_seg_data[:, slice_idx, :]
            else:  # Sagittal: :, :, slice
                original_slice = original_seg_data[:, :, slice_idx]
            # Appliquer une rotation de -90 degrés
            rot_axes= (0,1) if axis_to_plot==0 else (0,2) if axis_to_plot==1 else (1,2)
            if axis_to_plot==0:
                original_slice = np.rot90(original_slice,k=1,axes=rot_axes)
            im = ax.imshow(original_slice, cmap=cmap, origin='lower', vmin=0, vmax=max_label)
            ax.set_title(f'Original\nCoupe {slice_idx}')
            ax.axis('off')

        # Afficher chaque pipeline dans les colonnes suivantes
        for i, pipeline in enumerate(aic_pipelines):
            mcm_seg_path = f"analysis/mcm_segmentations/{subject}_{pipeline}_tensor_segmentation.nii.gz"
            if not os.path.exists(mcm_seg_path):
                continue
            mcm_seg_img = sitk.ReadImage(mcm_seg_path)
            mcm_seg_data = sitk.GetArrayFromImage(mcm_seg_img)
            
            mcm_seg_data = np.rot90(mcm_seg_data, k=1, axes=(1,2))
            # Vérifier que les dimensions correspondent et ajuster les indices de slice si nécessaire
            if mcm_seg_data.shape != original_seg_data.shape:
                if mcm_seg_data.shape[axis_to_plot] != original_seg_data.shape[axis_to_plot]:
                    mcm_slice_indices = [int(idx * mcm_seg_data.shape[axis_to_plot] / original_seg_data.shape[axis_to_plot]) for idx in slice_indices]
                else:
                    mcm_slice_indices = slice_indices
            else:
                mcm_slice_indices = slice_indices
                
            for j, slice_idx in enumerate(slice_indices):
                ax = axes[j, i+1]
                mcm_slice_idx = mcm_slice_indices[j]
                if axis_to_plot == 0:  # Axial
                    mcm_slice = mcm_seg_data[mcm_slice_idx, :, :]
                elif axis_to_plot == 1:  # Coronal
                    mcm_slice = mcm_seg_data[:, mcm_slice_idx, :]
                else:  # Sagittal
                    mcm_slice = mcm_seg_data[:, :, mcm_slice_idx]
                # Appliquer une rotation de -90 degrés
                if axis_to_plot==0:
                    mcm_slice =  np.rot90(mcm_slice,k=1,axes=(0,1))
                ax.imshow(mcm_slice, cmap=cmap, origin='lower', vmin=0, vmax=max_label)
                ax.set_title(f'{pipeline}\nCoupe {slice_idx}')
                ax.axis('off')

        # Créer une légende commune pour les labels > 0
        unique_labels = np.unique(original_seg_data)
        unique_labels = unique_labels[unique_labels > 0]  # Exclure 0
        handles = [plt.Rectangle((0,0),1,1, facecolor=colors[int(label)], edgecolor='black', linewidth=0.5) 
                   for label in unique_labels]
        labels_legend = [f'{int(label)} compartiment(s)' for label in unique_labels]
        fig.legend(handles=handles, labels=labels_legend, loc='upper right', title='Nombre de compartiments\nanisotropes', fontsize=8)

        plt.tight_layout()
        output_fig_path = f"analysis/mcm_segmentations/{subject}_segmentation_comparison_{axis_names[axis_to_plot]}.png"
        plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Figure sauvegardée: {output_fig_path}")

# Sauvegarder les résultats dans un CSV
if results:
    df = pd.DataFrame(results)
    output_path = 'analysis/mcm_comparison_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"✓ Résultats sauvegardés dans: {output_path}")
    print(f"  Total: {len(results)} comparaisons")
    print(f"{'='*80}")
else:
    print(f"\n{'='*80}")
    print(f"⚠️  Aucun résultat à sauvegarder")
    print(f"{'='*80}")
