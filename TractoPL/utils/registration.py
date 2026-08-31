from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, BIDSFile
from TractoPL.utils.tools import del_key, upt_dict, run_cli_command
from TractoPL.data.io import copy_from_dict
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants


def ants_registration(moving, fixed, outprefix='registered',transform_type='affine', **kwargs):
    """
    Calls the ANTs registration script on the given subject.

    Parameters
    ----------
    moving : BIDSFile or str
        Path to the moving image
    fixed : BIDSFile or str
        Path to the fixed image
    outprefix : str
        Prefix for the output files
    kwargs : dict
        Additional arguments to pass to the registration script

    Returns
    -------
    dict
        Dictionary with registration output files
    """
    inputs = {
        "moving": moving,
        "fixed": fixed
    }

    moving_space = kwargs.pop('moving_space', None)
    if moving_space is None:
        moving_space = moving.get_full_entities().get('space', 'datatype')

    fixed_space = kwargs.pop('fixed_space', None)
    if fixed_space is None:
        fixed_space = fixed.get_full_entities().get('space', 'datatype')

    # Prepare command arguments
    command_args = [
        "-d", "3",
        "-f", "$fixed",
        "-m", "$moving",
        "-o", outprefix
    ]


    # Get base entities
    base_entities = {}
    if hasattr(moving, "get_entities"):
        base_entities = moving.get_entities()

    # Define expected output files
    output_pattern = {
        f"{outprefix}0GenericAffine.mat": upt_dict(base_entities, {"suffix": "xfm", "extension": "mat", "from": moving_space, "to": fixed_space}),
        f"{outprefix}Warped.nii.gz": upt_dict(base_entities, {'space': fixed_space, "extension": "nii.gz"}),
        f"{outprefix}1Warp.nii.gz": upt_dict(base_entities, {"suffix": "xfm", "extension": "nii.gz", "from": moving_space, "to": fixed_space}),
        f"{outprefix}1InverseWarp.nii.gz": upt_dict(base_entities, {"suffix": "xfm", "extension": "nii.gz", "from": fixed_space, "to": moving_space}),
    }

    
    if transform_type == 'affine':
        command_args += ["-t", "a"]
        output_pattern.pop(f"{outprefix}1Warp.nii.gz", None)
        output_pattern.pop(f"{outprefix}1InverseWarp.nii.gz", None)
    elif transform_type == 'syn' or transform_type == 'synquick':
        command_args += ["-t", "s"]
    #Add the suffix of the keys in the output_pattern as a desc entity
    output_pattern = {k: upt_dict(v, {"desc": k.split(outprefix)[1].split('.')[0],"trans":transform_type}) for k, v in output_pattern.items()}

    main_cmd= "antsRegistrationSyN.sh" if 'quick' in transform_type.lower() else "antsRegistrationSyN.sh"
    # Run the command
    res_dict =  run_cli_command(
        main_cmd,
        inputs,
        output_pattern,
        base_entities,
        command_args=command_args
    )

    tmp_dir = os.path.dirname(list(res_dict.keys())[0])
    trans_list = []
    for trans in ['0GenericAffine','1Warp']:
        if any([f for f in output_pattern if trans in f]):
            trans_list.append(opj(tmp_dir,trans))

    trans_file = writeTransformSerie(moving, fixed,transform_list=trans_list)
    
    res_dict[trans_file] = upt_dict(base_entities, {"suffix": "xfm", "extension": "json", "from": moving_space, "to": fixed_space})
    return res_dict


def writeTransformSerie(moving_bidsfile, ref_bidsfile, transform_list, inverse_list=None):
    """
    Write a series of transformations to a single file.
    """

    # Using animaTransformSerieXmlGenerator
    output = transform_list[0].split('.')[0] + '.xml'
    print(output)
    # Get filename instead of full path
    moving_image = moving_bidsfile.filename if isinstance(moving_bidsfile, BIDSFile) else os.path.basename(moving_bidsfile)
    ref_image = ref_bidsfile.filename if isinstance(ref_bidsfile, BIDSFile) else os.path.basename(ref_bidsfile)
    transform_list = [os.path.basename(transform)
                      for transform in transform_list]
    if inverse_list is not None:
        inverse_list = [os.path.basename(inverse) for inverse in inverse_list]

    os.chdir(os.path.dirname(output))

    command = ['animaTransformSerieXmlGenerator', '-D']
    for transform in transform_list:
        command += ['-i', transform]
    # if inverse_list is not None:
    #     for inverse in inverse_list:
    #         command += ['-I',inverse]

    command += ['-o', output]

    print(' '.join(command))
    call(command)

    # Using a json file (BIDS compliant)
    output = output.replace('.xml', '.json')
    transform_dict = {}
    transform_dict['TransformList'] = transform_list
    transform_dict['InverseTransformList'] = inverse_list
    transform_dict['Reference'] = {'name': ref_image}
    transform_dict['Reference']['entities'] = ref_bidsfile.get_full_entities() if isinstance(ref_bidsfile, BIDSFile) else {}
    transform_dict['Moving'] = {'name': moving_image}
    transform_dict['Moving']['entities'] = moving_bidsfile.get_full_entities() if isinstance(moving_bidsfile, BIDSFile) else {}
    with open(output, 'w') as f:
        json.dump(transform_dict, f, indent=4)
    
    return output


def get_transform_list(transform_file):
    """
    Get the list of transformations from a file.

    Parameters
    ----------
    transform_file : BIDSFile
        JSON file describing transformation series

    Returns
    -------
    list
        List of BIDSFile objects for each transformation
    """
    with open(transform_file.path) as f:
        transform_dict = json.load(f)

    transform_list = []
    associated_files = {
        f.filename: f for f in transform_file.get_associations()}

    for transform in transform_dict['TransformList']:
        transform_list.append([v for k,v in associated_files.items() if transform in k][0])

    return transform_list


def apply_transforms(moving, trans_list, ref, **kwargs):
    """
    Apply a list of transformations to the moving image using ANTs.

    Parameters
    ----------
    moving : BIDSFile
        The moving image to transform
    trans_list : list of BIDSFile
        List of transformation files to apply
    ref : BIDSFile
        The reference image to use for the transformation
    kwargs : dict
        Additional arguments to pass to the transformation

    Returns
    -------
    ANTsImage
        The transformed image
    """
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    print("Applying transformations")

    seg_suffixes = ['mask', 'dseg', 'probseg']

    # Determine moving image path and pixel type
    if isinstance(moving, str):
        moving_path = moving
        basename = os.path.basename(moving)
        pixeltype = 'unsigned int' if any(seg in basename for seg in seg_suffixes) else 'double'
    else:
        moving_path = moving.path
        pixeltype = 'unsigned int' if moving.suffix in seg_suffixes else 'double'

    # Determine reference image path and pixel type
    if isinstance(ref, str):
        ref_path = ref
        basename = os.path.basename(ref)
        pixeltype_ref = 'unsigned int' if any(seg in basename for seg in seg_suffixes) else 'double'
    else:
        ref_path = ref.path
        pixeltype_ref = 'unsigned int' if ref.suffix in seg_suffixes else 'double'

    # Read moving and fixed images
    moving_image = ants.image_read(
        moving_path,
        dimension=3,
        pixeltype=pixeltype
    )
    ref_image = ants.image_read(
        ref_path,
        dimension=3,
        pixeltype=pixeltype_ref
    )


    # Process transform paths to handle .txt transforms
    new_trans_list = []
    for transform in trans_list:
        if isinstance(transform, str):
            transform_path = transform
        else:
            transform_path = transform.path
            
        if transform_path.endswith('.txt'):
            # Convert .txt affine transform to ANTs format
            trans_matrix = np.loadtxt(transform_path)
            
            # Reshape the matrix if needed (ensure it's a proper 4x4 or 3x3 affine)
            if trans_matrix.size == 12:  # 3x4 matrix
                # Convert to 4x4 homogeneous matrix
                affine_matrix = np.eye(4)
                affine_matrix[:3, :4] = trans_matrix.reshape(3, 4)
                trans_matrix = affine_matrix
            elif trans_matrix.size == 9:  # 3x3 matrix
                # Convert to 4x4 homogeneous matrix
                affine_matrix = np.eye(4)
                affine_matrix[:3, :3] = trans_matrix.reshape(3, 3)
                trans_matrix = affine_matrix
            
            # Get image dimensions for center calculation
            dims = ref_image.shape
            center = [float(d) / 2.0 for d in dims]
            
            # For ANTs affine transform, we need the fixed parameters (center of rotation)
            try:
                with open(transform_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'FixedParameters:' in line:
                            fixed_params = [float(x) for x in line.split(':')[1].strip().split()]
                            break
                    else:
                        # If not found in file, use the image center
                        fixed_params = center
            except:
                # Fallback to image center
                fixed_params = center
            
            # Extract the rotation/scale/shear part (3x3) and translation part (3x1)
            rotation_scale = trans_matrix[:3, :3].flatten().tolist()
            translation = trans_matrix[:3, 3].tolist()
            
            # Combine parameters in the correct order for ANTs (rotation/scale followed by translation)
            parameters = rotation_scale + translation
            
            ants_transform = ants.create_ants_transform(
                transform_type='AffineTransform',
                dimension=3,
                parameters=parameters,
                fixed_parameters=fixed_params
            )
        
            # Save the transform to a file
            transform_path = os.path.join(temp_dir, os.path.basename(transform_path).replace('.txt', '.mat'))
            ants.write_transform(ants_transform, transform_path)
        
            new_trans_list.append(transform_path)

        else:
            new_trans_list.append(transform_path)

    trans_list = new_trans_list

    # Apply transforms in sequence (reverse order for ANTs)
    transformed = ants.apply_transforms(
        fixed=ref_image,
        moving=moving_image,
        transformlist=trans_list[::-1],
        interpolator=kwargs.pop(
            'interpolator', 'genericLabel' if moving.suffix in seg_suffixes else 'linear'),
        verbose=True,
        dimension=3,
        **kwargs
    )

    return transformed

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

def apply_transform_trk(moving_path, trans_path,ref_path=None, output_dir=None):
    """
    Apply a transformation (only one, affine or dense) to a tractography file using dipy. both moving and ref must be in the same space.
    """
    


if __name__ == '__main__':
    subject = Subject('03011')

    # Test transform application
    t1 = subject.get(scope='raw', suffix='T1w', extension='nii.gz')[0]
    b0 = subject.get(pipeline='preprocessing', suffix='dwi',
                     extension='nii.gz', desc='b0ref')[0]
    # res_dict = ants_registration(moving=t1, fixed=b0,moving_space='T1w',fixed_space='B0')
    # pprint(res_dict)

    # copy_from_dict(subject, res_dict, pipeline='registration_test')

    trans_desc = subject.get_unique(**{'from':'T1w','to':'B0'}, extension='json', pipeline='registration_test')
    ref = subject.get_unique(pipeline='preprocessing', desc='b0ref', extension='nii.gz')
    trans_list = get_transform_list(transform_file=trans_desc)
    moved_t1 = apply_transforms(moving=t1, trans_list=trans_list, ref=ref)

    # Save transformed image to BIDS structure
    entities = {'datatype': 'anat', 'pipeline': 'registration_test', 'suffix': 'T1w', 'space': 'B0'}
    target = subject.write_object(moved_t1, **entities)
    print("Target path:", target)
