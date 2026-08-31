import os
import numpy as np
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, parse_filename, BIDSFile
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import tempfile
from dipy.io.stateful_tractogram import Space, StatefulTractogram,Origin
from dipy.io.streamline import save_tractogram, load_tractogram
from time import process_time
import vtk
from dipy.tracking.streamline import transform_streamlines
from scipy.io import loadmat
import nibabel as nib


def rotation(in_file, out_file, angle=180, x=0, y=0, z=1):
    '''Rotate a vtp or vtk file by ${angle}° along x, y or z axis.'''
    if in_file[-3:] == 'vtp':
        reader = vtk.vtkXMLPolyDataReader()
        writer = vtk.vtkXMLPolyDataWriter()
    elif in_file[-3:] == 'vtk':
        reader = vtk.vtkPolyDataReader()
        writer = vtk.vtkPolyDataWriter()
    else:
        print("Unrecognized input file. Must be vtp or vtk.")
        return
    reader.SetFileName(in_file)
    reader.Update()
    translation = vtk.vtkTransform()
    translation.RotateWXYZ(angle, x, y, z)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.SetTransform(translation)
    transformFilter.Update()
    writer.SetFileName(out_file)
    writer.SetInputConnection(transformFilter.GetOutputPort())
    # the constant 42 is defined in IO/Legacy/vtkDataWriter.h (c++ code), it corresponds to VTK_LEGACY_READER_VERSION_4_2
    writer.SetFileVersion(42)
    writer.Update()
    writer.Write()


# command = [
#     "tckgen",
#     "-algorithm", "iFOD2",
#     "-seed_image", seeds,
#     odf_mrtrix,
#     output,
#     "-force"
# ]

def generate_ifod2_tracto(odf, seeds, **kwargs):

    inputs = {
        "odf": odf,
        "seeds": seeds
    }

    command_args = [
        "$odf",
        "-algorithm", "iFOD2",
        "-seed_image", "$seeds",
        "-force",
        "-debug",
        "tracto.tck"
    ]

    output_patterns = {
        "tracto.tck": {
            "suffix": "tracto",
            "datatype": "dwi",
            "extension": "tck"
        }
    }
    return run_mrtrix_command('tckgen', inputs, output_patterns, entities_template=odf.get_entities(), command_args=command_args, **kwargs)

def load_matrix_in_any_format(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    elif ext == '.mat':
        # .mat are actually dictionnary. This function support .mat from
        # antsRegistration that encode a 4x4 transformation matrix.
        transfo_dict = loadmat(filepath)
        print(transfo_dict)
        lps2ras = np.diag([-1, -1, 1])

        rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
        trans = transfo_dict['AffineTransform_float_3_3'][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))

    return data

def apply_affine(tracto, affine_mat, reference, target):
    """
    Apply the given affine matrix to the tractography file.

    Parameters
    ----------
    tracto : BIDSFile
        BIDSFile object containing the tractography file to transform.
    affine_mat : str
        Path to the affine matrix file.
    reference : str
        Path to the reference image file.
    target : str
        Path to the target image file.
    Returns
    -------
    str
        Path to the transformed tractography file.
    """

    #Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Load the affine matrix
    affine_matrix = load_matrix_in_any_format(affine_mat)
    affine_matrix = np.linalg.inv(affine_matrix)
    # Load the tractogram file
    tractogram = load_tractogram(tracto.path, reference)
    tractogram.to_space(Space.RASMM)
    # Apply the affine transformation

    tractogram.streamlines = transform_streamlines(tractogram.streamlines, affine_matrix)

    tractogram= StatefulTractogram(tractogram.streamlines, target, Space.RASMM)

    file_extension = tracto.path.split('.')[-1]
    # Save the transformed tractogram
    output_file = f'{temp_dir}/transformed_tractogram.{file_extension}'
    save_tractogram(tractogram, output_file, bbox_valid_check=False)
    return output_file

def generate_trekker_tracto(odf, seeds, n_seeds=1000, **kwargs):

    inputs = {
        "odf": odf,
        "seeds": seeds
    }

    # odf_to_lps = run_cli_command('convert_fod', {'odf': odf}, {'odf_lps.nii.gz': odf.get_entities()}, command_args=[
    #                              '-i', odf.path, '-o', 'odf_lps.nii.gz', '-c', 'MRTRIX2ANIMA'])
    # #Get first key of odf_to_lps
    # odf_to_lps = list(odf_to_lps.keys())[0]

    command_args = [
        "track",
        odf.path,
        "--seed", seeds if isinstance(seeds, str) else seeds.path,
        "--seed_count", str(n_seeds),
        "-o", "tracto.vtk",
        "--force"
    ]

    output_patterns = {
        "tracto.vtk": {
            "suffix": "tracto",
            "datatype": "tracto",
            "extension": "vtk"
        }
    }
    result_dict = run_cli_command('trekker_linux', inputs, output_patterns, entities_template=odf.get_entities(
    ), command_args=command_args, **kwargs, use_sym_link=True)
    tract_path, tract_entities = list(result_dict.items())[0]

    # Rotate the VTK file
    rotated_path = tract_path.replace('.vtk', '_rotated.vtk')
    rotation(tract_path, rotated_path, angle=180, x=0, y=0, z=1)

    # # Convert VTK to TCK
    # call(["trekker_linux", "convert", tract_path, tract_path.replace('.vtk','.tck'), '--force'])

    result_dict[tract_path.replace('.vtk', '_rotated.vtk')] = upt_dict(tract_entities, orient='LPS')

    return result_dict


def generate_trekker_tracto_tck(odf, seeds, n_seeds=1000, **kwargs):

    inputs = {"odf": odf, "seeds": seeds}

    # odf_to_lps = run_cli_command('convert_fod', {'odf': odf}, {'odf_lps.nii.gz': odf.get_entities()}, command_args=[
    #                              '-i', odf.path, '-o', 'odf_lps.nii.gz', '-c', 'MRTRIX2ANIMA'])
    # #Get first key of odf_to_lps
    # odf_to_lps = list(odf_to_lps.keys())[0]

    command_args = [
        "track", odf.path, "--seed",
        seeds if isinstance(seeds, str) else seeds.path, "--seed_count",
        str(n_seeds), "-o", "tracto.tck", "--force"
    ]

    output_patterns = {
        "tracto.tck": {
            "suffix": "tracto",
            "datatype": "tracto",
            'algo': 'trekker',
            "extension": "tck"
        }
    }
    result_dict = run_cli_command('trekker_linux',
                                  inputs,
                                  output_patterns,
                                  entities_template=odf.get_entities(),
                                  command_args=command_args,
                                  **kwargs,
                                  use_sym_link=True)

    return result_dict


def _find_tractogram_endings(tractogram, reference):
    """
    Get the endings segmentation of the streamlines in the tractogram.
    Ensures all tracts are oriented in the same direction.

    Parameters
    ----------
    tractogram : StatefulTractogram
        The tractogram to process.
        
    reference : str
        The reference image

    Returns
    -------
    endings : dict
        Dictionary containing the start and end binary masks of the streamlines in the tractogram.
    """
    # Load reference image
    import nibabel as nib
    from dipy.tracking.streamline import orient_by_streamline
    
    ref_img = nib.load(reference)
    shape = ref_img.shape
    affine = ref_img.affine
    
    # Ensure tractogram is in the correct space
    tractogram.to_space(Space.RASMM)
    
    # Create empty binary volumes for start and end points
    start_volume = np.zeros(shape)
    end_volume = np.zeros(shape)
    
    # Get streamlines from tractogram
    streamlines = tractogram.streamlines
    
    # Check if we have valid streamlines
    if len(streamlines) == 0:
        start_img = nib.Nifti1Image(start_volume.astype('uint8'), affine)
        end_img = nib.Nifti1Image(end_volume.astype('uint8'), affine)
        return {'start': start_img, 'end': end_img}
    
    # Find the longest streamline to use as standard for orientation
    lengths = [len(s) for s in streamlines]
    standard_idx = np.argmax(lengths)
    standard = streamlines[standard_idx]
    
    # Orient all streamlines to match the standard
    oriented_streamlines = orient_by_streamline(streamlines, standard, n_points=12, in_place=False)
    
    # Get the inverse affine to convert from mm to voxel coordinates
    inv_affine = np.linalg.inv(affine)
    # Process each oriented streamline
    for streamline in oriented_streamlines:
        if len(streamline) < 2:
            continue
            
        # Get the start and end points of the oriented streamline
        start_point = streamline[0]
        end_point = streamline[-1]
        
        # Transform points from RAS mm to voxel space
        start_voxel = np.round(nib.affines.apply_affine(inv_affine, start_point)).astype(int)
        end_voxel = np.round(nib.affines.apply_affine(inv_affine, end_point)).astype(int)
        
        # Check if points are within volume bounds and set binary mask
        if (0 <= start_voxel[0] < shape[0] and 
            0 <= start_voxel[1] < shape[1] and 
            0 <= start_voxel[2] < shape[2]):
            start_volume[start_voxel[0], start_voxel[1], start_voxel[2]] = 1
            
        if (0 <= end_voxel[0] < shape[0] and 
            0 <= end_voxel[1] < shape[1] and 
            0 <= end_voxel[2] < shape[2]):
            end_volume[end_voxel[0], end_voxel[1], end_voxel[2]] = 1
    
    # Create NIfTI images with binary masks
    start_img = nib.Nifti1Image((start_volume>0).astype('uint8'), affine)
    end_img = nib.Nifti1Image((end_volume>0).astype('uint8'), affine)
    
    return {'start': start_img, 'end': end_img}


def get_tractogram_endings(tractogram_file, reference):
    """
    Get the endings segmentation of the streamlines in the tractogram.

    Parameters
    ----------
    tractogram : BIDSFile or str
        The tractogram to process.
        
    reference : BIDSFile or str
        The reference image

    Returns
    -------
    result_dict : dict
        A dictionary containing the path to the generated endings segmentation files (beginning and end of each streamline).
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    tracto_path = tractogram_file if isinstance(tractogram_file, str) else tractogram_file.path
    ref_path = reference if isinstance(reference, str) else reference.path
    
    # Load the tractogram
    tractogram = load_tractogram(tracto_path, ref_path)
    
    # Get the tractogram entities if available
    if hasattr(tractogram_file, 'get_entities'):
        entities = tractogram_file.get_entities()
    else:
        # Create basic entities if not available
        filename = os.path.basename(tracto_path)
        entities = parse_filename(filename)
    
    # Get the endings segmentation
    endings = _find_tractogram_endings(tractogram, ref_path)
    
    # Save the endings segmentation
    start_file = f'{temp_dir}/streamlines_start.nii.gz'
    end_file = f'{temp_dir}/streamlines_end.nii.gz'
    
    nib.save(endings['start'], start_file)
    nib.save(endings['end'], end_file)
    
    # Create a dictionary to store the results
    result_dict = {
        start_file: upt_dict(entities, suffix='mask', label='start', desc='endings', datatype='tracto', extension='nii.gz'),
        end_file: upt_dict(entities, suffix='mask', label='end', desc='endings', datatype='tracto', extension='nii.gz')
    }
    
    return result_dict

def surface_projection(tractogram, surface, output_file, **kwargs):
    """
    Project the streamlines onto a surface.

    Parameters
    ----------
    """
    return True

def filter_tracto_by_endings(tracto, start_mask, end_mask,**kwargs):
    """
    Filter the streamlines in the tractogram based on their endpoints.

    Parameters
    ----------
    tracto : BIDSFile or str
        The tractogram to filter.
    start_mask : BIDSFile or str
        The binary mask defining the start region.
    end_mask : BIDSFile or str
        The binary mask defining the end region.
    Returns
    -------
    dict
        A dictionary containing the path to the filtered tractogram file.
    """
    inputs = {
        "tracto": tracto if isinstance(tracto, str) else tracto.path,
        "start_mask": start_mask if isinstance(start_mask, str) else start_mask.path,
        "end_mask": end_mask if isinstance(end_mask, str) else end_mask.path
    }

    command_args = [
        "$tracto",
        "-ends_only",
        "-include", "$start_mask",
        "-include", "$end_mask",
        "-force",
        "filtered_tracto.tck"
    ]
    entities = tracto.get_full_entities()
    output_patterns = {
        "filtered_tracto.tck": upt_dict(entities,desc='filtered',filter='endings')
    }
    return run_mrtrix_command('tckedit', inputs, output_patterns, entities_template=parse_filename(os.path.basename(tracto.path)) if hasattr(tracto, 'path') else {}, command_args=command_args, **kwargs)

def filter_tracto_by_endings_dipy(tracto, reference, start_mask, end_mask, output_file=None):
    """
    Filter the streamlines in the tractogram based on their endpoints using DIPY.

    Parameters
    ----------
    tracto : BIDSFile or str
        The tractogram to filter.
    reference : BIDSFile or str
        The reference image for the tractogram.
    start_mask : BIDSFile or str
        The binary mask defining the start region.
    end_mask : BIDSFile or str
        The binary mask defining the end region.
    output_file : str, optional
        Path to save the filtered tractogram. If None, a temporary file will be created.

    Returns
    -------
    str
        Path to the filtered tractogram file.
    """
    import nibabel as nib
    from dipy.tracking.streamline import set_number_of_points

    tracto_path = tracto if isinstance(tracto, str) else tracto.path
    ref_path = reference if isinstance(reference, str) else reference.path
    start_mask_path = start_mask if isinstance(start_mask, str) else start_mask.path
    end_mask_path = end_mask if isinstance(end_mask, str) else end_mask.path

    # Load the tractogram and masks
    tractogram = load_tractogram(tracto_path, ref_path)
    start_img = nib.load(start_mask_path)
    end_img = nib.load(end_mask_path)

    start_data = start_img.get_fdata().astype(bool)
    end_data = end_img.get_fdata().astype(bool)

    # Get the affine of the reference image
    affine = nib.load(ref_path).affine
    inv_affine = np.linalg.inv(affine)

    filtered_streamlines = []
    
    for sl in tractogram.streamlines:
        if len(sl) < 2:
            continue
        
        # Get start and end points in voxel space
        start_voxel = np.round(nib.affines.apply_affine(inv_affine, sl[0])).astype(int)
        end_voxel = np.round(nib.affines.apply_affine(inv_affine, sl[-1])).astype(int)
        
        # Check if points are within bounds and in the masks
        if (0 <= start_voxel[0] < start_data.shape[0] and 
            0 <= start_voxel[1] < start_data.shape[1] and
            0 <= start_voxel[2] < start_data.shape[2] and
            0 <= end_voxel[0] < end_data.shape[0] and
            0 <= end_voxel[1] < end_data.shape[1] and
            0 <= end_voxel[2] < end_data.shape[2]):
            if start_data[start_voxel[0], start_voxel[1], start_voxel[2]] and end_data[end_voxel[0], end_voxel[1], end_voxel[2]]:
                filtered_streamlines.append(sl)

    # Create a new tractogram with the filtered streamlines
    filtered_tractogram = StatefulTractogram(filtered_streamlines, ref_path, Space.RASMM)
    if output_file is None:
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, 'filtered_tracto.tck')
    save_tractogram(filtered_tractogram, output_file, bbox_valid_check=False)
    
    entities = tracto.get_full_entities()
    return {output_file: upt_dict(entities, desc='filtered', filter='endings')}

def get_fiber_density(tracto, reference, output_file=None):
    """
    Compute the fiber density map from the tractogram.

    Parameters
    ----------
    tracto : BIDSFile or str
        The tractogram to process.
    reference : BIDSFile or str
        The reference image for the tractogram.
    output_file : str, optional
        Path to save the fiber density map. If None, a temporary file will be created.

    Returns
    -------
    str
        Path to the fiber density map file.
    """
    inputs = {
        "tracto": tracto if isinstance(tracto, str) else tracto.path,
        "reference": reference if isinstance(reference, str) else reference.path
    }
    tracto_path = tracto if isinstance(tracto, str) else tracto.path
    ref_path = reference if isinstance(reference, str) else reference.path

    # Load tractogram and reference
    tractogram = load_tractogram(tracto_path, ref_path)
    ref_img = nib.load(ref_path)
    shape = ref_img.shape[:3]
    affine = ref_img.affine
    inv_affine = np.linalg.inv(affine)

    # Ensure streamlines are in RASMM world coordinates
    try:
        tractogram.to_space(Space.RASMM)
    except Exception:
        pass

    # Initialize density image (count of streamlines passing through each voxel)
    density = np.zeros(shape, dtype=np.int32)

    # If there are no streamlines, return zero image
    streamlines = list(tractogram.streamlines)
    if len(streamlines) == 0:
        if output_file is None:
            temp_dir = tempfile.mkdtemp()
            output_file = os.path.join(temp_dir, 'fiber_density.nii.gz')
        nib.save(nib.Nifti1Image(density, affine), output_file)
        entities = tracto.get_full_entities() if hasattr(tracto, 'get_full_entities') else parse_filename(os.path.basename(tracto_path))
        return {output_file: upt_dict(entities, suffix='density', desc='fiber_density', datatype='dwi', extension='nii.gz')}

    # For each streamline, find unique voxels it visits and increment count once per streamline per voxel
    for sl in streamlines:
        if len(sl) < 2:
            continue
        vox = np.round(nib.affines.apply_affine(inv_affine, sl)).astype(int)
        if vox.size == 0:
            continue
        unique_vox = np.unique(vox, axis=0)
        for x, y, z in unique_vox:
            if (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                density[x, y, z] += 1

    # Normalize by total number of streamlines
    n_streamlines = len(streamlines)
    density_normalized = density.astype(np.float32) / n_streamlines

    # Prepare output file
    if output_file is None:
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, 'fiber_density.nii.gz')

    nib.save(nib.Nifti1Image(density_normalized, affine), output_file)

    entities = tracto.get_full_entities() if hasattr(tracto, 'get_full_entities') else parse_filename(os.path.basename(tracto_path))

    #Create a binary map as well
    binary_density = (density > 0).astype(np.uint8)
    binary_output_file = output_file.replace('fiber_density.nii.gz', 'fiber_density_binary.nii.gz')
    nib.save(nib.Nifti1Image(binary_density, affine), binary_output_file)
    binary_entities = entities.copy()
    binary_entities['suffix'] = 'density_binary'

    return {
        output_file: upt_dict(entities, suffix='density', label='fibers', datatype='map', extension='nii.gz'),
        binary_output_file: upt_dict(binary_entities, suffix='mask', label='fibers', datatype='map', extension='nii.gz')
    }


def compare_fiber_density(density1, density2, output_dir=None, win=9):
    """
    Compute sliding window NCC between two fiber density maps.

    Parameters
    ----------
    density1 : str
        Path to the first fiber density NIfTI map.
    density2 : str
        Path to the second fiber density NIfTI map.
    output_dir : str, optional
        Directory to save outputs. If None, a temporary directory is used.
    win : int, optional
        Window size for sliding NCC (default: 9).

    Returns
    -------
    dict
        {ncc_map_path: entities, 'mean_ncc': float}
    """
    from scipy.ndimage import uniform_filter

    img1 = nib.load(density1)
    img2 = nib.load(density2)
    I = img1.get_fdata(dtype=np.float64)
    J = img2.get_fdata(dtype=np.float64)
    affine = img1.affine

    win_size = win ** 3

    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum  = uniform_filter(I,  size=win) * win_size
    J_sum  = uniform_filter(J,  size=win) * win_size
    I2_sum = uniform_filter(I2, size=win) * win_size
    J2_sum = uniform_filter(J2, size=win) * win_size
    IJ_sum = uniform_filter(IJ, size=win) * win_size

    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    # Mask: voxels where at least one density map is non-zero
    mask = (I != 0) | (J != 0)

    mean_ncc = float(cc[mask].mean()) if mask.any() else float('nan')

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    os.makedirs(output_dir, exist_ok=True)

    ncc_map_path = os.path.join(output_dir, 'fiber_density_ncc.nii.gz')
    nib.save(nib.Nifti1Image(cc.astype(np.float32), affine), ncc_map_path)

    mean_ncc_path = os.path.join(output_dir, 'fiber_density_mean_ncc.json')
    with open(mean_ncc_path, 'w') as f:
        import json as _json
        _json.dump({'mean_ncc': mean_ncc, 'n_voxels': int(mask.sum())}, f, indent=2)

    entities1 = parse_filename(os.path.basename(density1))
    ncc_entities = upt_dict(entities1, suffix='ncc', label='fibers', datatype='map', extension='nii.gz')

    return {
        ncc_map_path: ncc_entities,
        'mean_ncc': mean_ncc,
        'mean_ncc_path': mean_ncc_path,
    }


if __name__ == "__main__":
    # subject = Subject("03011")
    # odf = sub.get_unique(suffix='fod',  desc='preproc', label='WM')
    # seeds = sub.get_unique(suffix='mask', label='WM', space='B0')

    # output_dict = generate_ifod2_tracto(odf, seeds)
    # pprint(output_dict)
    # copy_from_dict(sub, output_dict,pipeline='msmt_csd')

    # odf = sub.get_unique(suffix='fod',  desc='preproc', label='WM')
    tracto = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/bundle_seg/sub-03011/tracto/sub-03011_bundle-CSTleft_desc-cleaned_tracto.trk'

    ref = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/preprocessing/sub-03011/dwi/sub-03011_metric-FA_model-DTI_dwi.nii.gz'

    print(get_tractogram_endings(tracto, ref))