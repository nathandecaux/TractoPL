import os 
from pprint import pprint
from os.path import join as opj
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import xml.etree.ElementTree as ET
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import read_mapping
import multiprocessing
from functools import partial
import traceback
import ants 
import vtk
from pprint import pprint
from vtk.util import numpy_support
from TractoPL.utils.registration import ants_registration
from TractoPL.data.io import copy_from_dict
from TractoPL.data.loader import Subject,Dataset, BIDSFile

def get_template_surface(template):
    """
    Get the template surface from the template file.

    Parameters
    ----------
    template : str
        Path to the template file (nifti)

    Returns
    -------
    template_surface : str
        Path to the template surface file (vtk)
    """

    template_surface = template.replace('.nii.gz', '_surface.vtk')
    
    # Load NIfTI image to get affine transformation
    nifti_img = nib.load(template)
    affine_matrix = np.linalg.inv(nifti_img.affine.copy())

    # Apply flip in axes 0 and 1
    affine_matrix[0,:] = -affine_matrix[0,:]  # Flip x-axis
    affine_matrix[1,:] = -affine_matrix[1,:]  # Flip y-axis
    
    #Load the NIfTI data directly with nibabel instead of VTK reader
    nifti_data = nifti_img.get_fdata()
    
    # Create VTK image data from numpy array
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(nifti_data.shape)
    vtk_image.SetOrigin(0, 0, 0)  # Set origin to 0,0,0 to work in voxel space
    vtk_image.SetSpacing(1, 1, 1)  # Set spacing to 1,1,1 to work in voxel space
    
    # Convert numpy array to VTK and set as scalars
    vtk_array = numpy_support.numpy_to_vtk(nifti_data.ravel(order='F'), deep=True)
    vtk_image.GetPointData().SetScalars(vtk_array)
    
    surface_extractor = vtk.vtkMarchingCubes()
    surface_extractor.SetInputData(vtk_image)
    surface_extractor.SetValue(0, 0.5)
    surface_extractor.Update()
    
    # Get the surface polydata
    surface_polydata = surface_extractor.GetOutput()
    
    # Transform points from voxel coordinates to world coordinates using affine
    points = surface_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    # Convert VTK points to numpy array
    points_array = numpy_support.vtk_to_numpy(points.GetData())
    print(f"Original VTK points range: X=[{points_array[:, 0].min():.2f}, {points_array[:, 0].max():.2f}], "
          f"Y=[{points_array[:, 1].min():.2f}, {points_array[:, 1].max():.2f}], "
          f"Z=[{points_array[:, 2].min():.2f}, {points_array[:, 2].max():.2f}]")
    print(f"NIfTI image shape: {nifti_img.shape}")
    print(f"NIfTI affine matrix:\n{nifti_img.affine}")
    
    # Add homogeneous coordinate (1s) for affine transformation
    homogeneous_points = np.hstack([points_array, np.ones((num_points, 1))])
    
    # Apply affine transformation to convert to world coordinates
    world_points = np.dot(homogeneous_points, affine_matrix.T)[:, :3]
    
    print(f"Transformed world points range: X=[{world_points[:, 0].min():.2f}, {world_points[:, 0].max():.2f}], "
          f"Y=[{world_points[:, 1].min():.2f}, {world_points[:, 1].max():.2f}], "
          f"Z=[{world_points[:, 2].min():.2f}, {world_points[:, 2].max():.2f}]")
    
    # Convert back to VTK points
    world_points_vtk = vtk.vtkPoints()
    world_points_vtk.SetData(numpy_support.numpy_to_vtk(world_points, deep=True))
    
    # Update the polydata with transformed points
    surface_polydata.SetPoints(world_points_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(template_surface)
    writer.SetInputData(surface_polydata)

    writer.Write()
    return template_surface

def get_transf_list(atlas_dir):
    """
    Get the list of transformation files in the atlas directory.

    Parameters
    ----------
    atlas_dir : str
        Path to the atlas directory

    Returns
    -------
    transf_dict : list
        Dict of transformation files
    """
    
    #List all .mat files in the atlas directory
    transf_dict = {}
    for root, dirs, files in os.walk(atlas_dir):
        for file in files:
            if file.endswith('.mat') and 'sub-' in file:
                subject_id = file.split('sub-')[1].split('_')[0]
                transf_path = os.path.join(root, file)
                transf_dict[subject_id] = transf_path
    # Sort the dictionary by subject ID
    transf_dict = dict(sorted(transf_dict.items()))

    return transf_dict

def register_subject_to_HCP(subject, template_path):
    """
    Register a subject to the HCP template space.

    Parameters
    ----------
    subject : Dataset Subject object
        The subject to register.
    template_path : str
        Path to the HCP template file.
    """ 

    res_dict = ants_registration(
        moving=subject.get_unique(pipeline='preprocessing',metric='FA',extension='.nii.gz'),
        fixed=template_path,
        fixed_space='HCP',
        moving_space='subject'
    )

    copy_from_dict(subject=subject,
                   file_dict=res_dict,
                   pipeline='tractometry'
                   )
    
    return res_dict

def create_actidep_template(dataset):
    """
    Create an ActiDep template from the dataset.

    Parameters
    ----------
    dataset : Dataset object
        The Dataset dataset to create the template from.
    
    Returns
    -------
    template_path : str
        Path to the created template file.
    """
    
    list_of_registered_fa = dataset.get_global(pipeline='tractometry',
                                              extension='.nii.gz',
                                              space='HCP')
    
    print(f"Number of registered subjects: {len(list_of_registered_fa)}")

    # Initialize the sum and count for iterative averaging
    fa_sum = None
    count = 0
    reference_affine = None
    
    print(f"Processing {len(list_of_registered_fa)} FA images iteratively...")
    
    for i, fa_file in enumerate(list_of_registered_fa):
        print(f"Processing FA image {i+1}/{len(list_of_registered_fa)}: {fa_file.path}")
        
        # Load the current FA image
        fa_img = nib.load(fa_file.path)
        fa_data = fa_img.get_fdata()
        
        # Store the reference affine from the first image
        if reference_affine is None:
            reference_affine = fa_img.affine
            fa_sum = np.zeros_like(fa_data)
        
        # Add to the running sum
        fa_sum += fa_data
        count += 1
    
    # Calculate the mean
    mean_fa = fa_sum / count
    print(f"Mean FA shape: {mean_fa.shape}")
    
    # Create and save the template
    mean_fa_img = nib.Nifti1Image(mean_fa, affine=reference_affine)
    template_path = '/home/ndecaux/Data/actidep_atlas/metric-FA_space-HCP_template.nii.gz'
    nib.save(mean_fa_img, template_path)

    # Create a surface from the mean FA image
    template_surface = get_template_surface(template_path)
    print(f"Template surface created at: {template_surface}")

    return template_path, template_surface



    
    


if __name__ == "__main__":
    # template = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/preprocessing/sub-01002/dwi/sub-01002_space-B0_label-brain_mask.nii.gz'
    # template = '/home/ndecaux/Code/Data/actidep_atlas/fiber_density_template0.nii.gz'
    # template_surface = get_template_surface(template)
    # print(f"Template surface file: {template_surface}")

    # pprint(get_transf_list('/home/ndecaux/Code/Data/actidep_atlas'))
    db_root= '/home/ndecaux/Data/dysdiago/'
    # subjects = Dataset(db_root = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/')

    subjects = Dataset(db_root)
    for subject_id in subjects.get_subjects():
        subject = subjects.get_subject(subject_id)
        print(f"Processing subject: {subject_id}")
        
        # Get the template path
        template_path = '/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz'
        
        already_exists = len(subject.get(pipeline='tractometry',extension='.json')) > 0
        if already_exists:
            print(f"Subject {subject_id} already registered to HCP template space. Skipping registration.")
            continue

        # Register the subject to HCP template space
        res_dict = register_subject_to_HCP(subject, template_path)
        
        # Print the result dictionary
        pprint(res_dict)


    ## Create an ActiDep template from the dataset
    # create_actidep_template(subjects)