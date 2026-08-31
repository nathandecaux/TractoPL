import os
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
from TractoPL.data.loader import Subject, Dataset, BIDSFile
from TractoPL.data.mcmfile import MCMFile,read_mcm_file
import tempfile
from subprocess import call
from pprint import pprint
import SimpleITK as sitk
import nrrd
import itk


def get_dti_principal_direction_sitk(dti_files, weights=None, output_file=None):
    """
    Extract the principal direction from one or multiple DTI files using SimpleITK.
    The length of the peaks is modulated by the weights if provided.
    Compatible with MRTrix peak format.
    """
    if isinstance(dti_files, str):
        dti_files = [dti_files]
    
    if output_file is None:
        output_file = '/tmp/principal_direction_sitk.nii.gz'
    
    all_principal_directions = []
    reference_image = None
    
    for i, dti_file in enumerate(dti_files):
        print(f"Processing DTI file {i+1}/{len(dti_files)}: {os.path.basename(dti_file)}")
        
        # Read DTI image using SimpleITK only
        dti_image = sitk.ReadImage(dti_file)
        dti_array = sitk.GetArrayFromImage(dti_image)
        
        print(f"DTI array shape: {dti_array.shape}")
        
        # Initialize output arrays
        principal_directions = np.zeros(dti_array.shape[:-1] + (3,), dtype=np.float32)
        principal_eigenvalues = np.zeros(dti_array.shape[:-1], dtype=np.float32)
        
        print(dti_array[0,0,0])
        

        #Extract a vector that is aligned with the principal direction of the diffusion tensor
        for x in range(dti_array.shape[0]):
            for y in range(dti_array.shape[1]):
                for z in range(dti_array.shape[2]):
                    tensor = dti_array[x, y, z, :]

                    #If tensor is full of zeros, skip
                    if np.all(tensor == 0):
                        principal_directions[x, y, z, :] = [0.0, 0.0, 0.0]
                        principal_eigenvalues[x, y, z] = 0.0
                        continue
                    
                    # Convert tensor to 3x3 matrix
                    tensor_matrix = np.array([[tensor[0], tensor[1], tensor[3]],
                                                [tensor[1], tensor[2], tensor[4]],
                                                [tensor[3], tensor[4], tensor[5]]], dtype=np.float64)
                    
                    # Ensure matrix is symmetric (numerical stability)
                    tensor_matrix = (tensor_matrix + tensor_matrix.T) / 2.0
                    
                    # Check if matrix is valid (finite values)
                    if not np.all(np.isfinite(tensor_matrix)):
                        principal_directions[x, y, z, :] = [0.0, 0.0, 0.0]
                        principal_eigenvalues[x, y, z] = 0.0
                        continue
                    
                    # Check if matrix has reasonable magnitude
                    matrix_norm = np.linalg.norm(tensor_matrix)
                    if matrix_norm < 1e-12:
                        principal_directions[x, y, z, :] = [0.0, 0.0, 0.0]
                        principal_eigenvalues[x, y, z] = 0.0
                        continue
                    
                    try:
                        # Compute eigenvalues and eigenvectors
                        eigenvalues, eigenvectors = np.linalg.eigh(tensor_matrix)  # Use eigh for symmetric matrices
                        
                        # Take real parts to handle numerical errors that might introduce tiny imaginary components
                        eigenvalues = np.real(eigenvalues)
                        eigenvectors = np.real(eigenvectors)
                        
                        # Sort eigenvalues and corresponding eigenvectors in descending order
                        sorted_indices = np.argsort(eigenvalues)[::-1]
                        principal_eigenvalue = eigenvalues[sorted_indices[0]]
                        principal_direction = eigenvectors[:, sorted_indices[0]]
                        
                        # Ensure eigenvalue is positive (for diffusion tensors)
                        if principal_eigenvalue <= 0:
                            principal_directions[x, y, z, :] = [0.0, 0.0, 0.0]
                            principal_eigenvalues[x, y, z] = 0.0
                            continue
                        
                        # Store results
                        principal_directions[x, y, z, :] = principal_direction
                        principal_eigenvalues[x, y, z] = principal_eigenvalue
                        
                    except np.linalg.LinAlgError:
                        # Handle singular matrices
                        principal_directions[x, y, z, :] = [0.0, 0.0, 0.0]
                        principal_eigenvalues[x, y, z] = 0.0
                        continue

        print(f"Principal directions shape: {principal_directions.shape}")
        print(f"Principal eigenvalues shape: {principal_eigenvalues.shape}")
        
        # Normalize the principal directions
        norm = np.linalg.norm(principal_directions, axis=-1, keepdims=True)
        valid_mask = (norm > 1e-12).squeeze()  # More robust threshold
        principal_directions = np.where(valid_mask[..., np.newaxis], 
                                      principal_directions / norm, 
                                      0.0)
        
        print(f"Normalized principal directions shape: {principal_directions.shape}")
        # Find a non-zero voxel for verification
        non_zero_indices = np.where(valid_mask)
        if len(non_zero_indices[0]) > 0:
            test_idx = (non_zero_indices[0][0], non_zero_indices[1][0], non_zero_indices[2][0])
            print(f"Principal directions at {test_idx}: {principal_directions[test_idx]} (should be a unit vector)")
            print(f"Norm: {np.linalg.norm(principal_directions[test_idx])}")
        else:
            print("No valid principal directions found")
        # Ensure the principal directions are in the correct format (3, x, y, z)
        print(f"Principal directions after normalization: {principal_directions[0,0,0
        ]} (should be a unit vector)")
        
       
        
        # Apply weights if provided
        # if weights is not None and i < weights.shape[-1]:
        #     weight_volume = weights[..., i]
        #     print(f"Applying weights for compartment {i}, weight shape: {weight_volume.shape}")
            
        #     # Modulate vector length by weight
        #     valid_mask = principal_eigenvalues > 0
            
        #     for dim in range(3):
        #         principal_directions[..., dim] = np.where(
        #             valid_mask,
        #             principal_directions[..., dim] * weight_volume,
        #             0.0
        #         )
        # else:
        #     # Without weights, use eigenvalue magnitude
        #     valid_mask = principal_eigenvalues > 0
        #     eigenvalue_sqrt = np.sqrt(np.maximum(principal_eigenvalues, 0))
            
        #     for dim in range(3):
        #         principal_directions[..., dim] = np.where(
        #             valid_mask,
        #             principal_directions[..., dim] * eigenvalue_sqrt,
        #             0.0
        #         )
        
        if reference_image is None:
            reference_image = dti_image
        
        # Transpose for concatenation (3, x, y, z)
        principal_direction_transposed = np.transpose(principal_directions, (3, 0, 1, 2))
        all_principal_directions.append(principal_direction_transposed)
    
    # Combine all directions
    combined_directions = np.concatenate(all_principal_directions, axis=0)
    print(f"Combined directions shape: {combined_directions.shape}")
    
    # Apply flip in z-axis
    combined_directions[2::3] *= -1  # Flip the z-component every 3 channels
    print(f"Applied flip in z-axis")
    
    # Create output image using SimpleITK
    output_image = sitk.GetImageFromArray(combined_directions, isVector=False)
    
    # Set spatial information from reference image
    original_spacing = list(reference_image.GetSpacing()[:3]) + [1.0]
    original_origin = list(reference_image.GetOrigin()[:3]) + [0.0]
    
    # Handle direction matrix
    ref_direction = reference_image.GetDirection()
    if len(ref_direction) == 9:  # 3D image
        direction_3d = np.array(ref_direction).reshape(3, 3)
        direction_4d = np.eye(4)
        direction_4d[:3, :3] = direction_3d
        output_image.SetDirection(direction_4d.flatten())
    else:
        output_image.SetDirection(ref_direction)
    
    output_image.SetSpacing(original_spacing)
    output_image.SetOrigin(original_origin)
    
    # Write using SimpleITK
    sitk.WriteImage(output_image, output_file)
    print(f"Saved principal directions to: {output_file}")
    
    return output_file

def extract_peaks_from_mcm(mcmfile):
    """
    Extract peaks from MCM file and save them in a temporary directory.
    """
    mcm = MCMFile(mcmfile.path)
    print(f"Extracting peaks from {mcmfile}...")
    print(mcm)
    #Get compartments that are type = 'tensor'
    tensors = [(comp_number, comp) for comp_number,comp in mcm.compartments.items() if comp['type'] == 'Tensor']
    print(tensors)
    print(f"Found {len(tensors)} tensors in {mcmfile}.")
    
    # Get weights file path and load the data
    weights_file_path = mcm.get_weights()
    print(f"Weights file path: {weights_file_path}")
    
    # Load weights using nrrd or SimpleITK
    if weights_file_path and os.path.exists(weights_file_path):
        if weights_file_path.endswith('.nrrd'):
            weights_data, _ = nrrd.read(weights_file_path)
            print(f"Weights shape: {weights_data.shape}")
            weights = weights_data
        else:
            weights_img = sitk.ReadImage(weights_file_path)
            weights = sitk.GetArrayFromImage(weights_img)
            print(f"Weights shape: {weights.shape}")
    else:
        print("Warning: No weights file found or file doesn't exist")
        weights = None
    
    # Extract paths of all tensor files
    tensor_files = [comp[1]['path'] for comp in tensors]
    print(f"Tensor files: {[os.path.basename(f) for f in tensor_files]}")
    
    # Extract weights for tensor compartments only
    if weights is not None:
        tensor_indices = [int(comp[0]) for comp in tensors]
        print(f"Tensor indices: {tensor_indices}")
        print(f"Original weights shape: {weights.shape}")
        
        tensor_weights_temp = weights[tensor_indices, ...]
        print(f"Tensor weights temp shape: {tensor_weights_temp.shape}")
        
        tensor_weights = np.transpose(tensor_weights_temp, (3, 2, 1, 0))
        print(f"Tensor weights final shape: {tensor_weights.shape}")
    else:
        tensor_weights = None
        print("No weights available, proceeding without weight modulation")
    
    print(f"\n=== Processing {len(tensor_files)} tensor files ===")
    output_file_sitk = get_dti_principal_direction_sitk(
        tensor_files, 
        weights=tensor_weights,
        output_file='/tmp/combined_principal_directions.nii.gz'
    )

    return output_file_sitk

sub = Subject('01001', '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
mcmfile = sub.get_unique(pipeline='mcm_tensors_staniz',extension='mcmx')
peaks = sub.get_unique(pipeline='msmt_csd', suffix='peaks', extension='nii.gz',desc='fixels2peaks')
print(nib.load(peaks.path).shape)
extract_peaks_from_mcm(mcmfile)


# get_dti_principal_direction_sitk(['/home/ndecaux/Code/Demo/fake_dti/fake_dti_anima.nii.gz'],output_file='/home/ndecaux/Code/Demo/fake_dti/principal_direction.nii.gz')

