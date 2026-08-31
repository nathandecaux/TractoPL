from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, move2nii, parse_filename, BIDSFile, copy2nii
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command, run_multiple_commands
import SimpleITK as sitk

import shutil
import ants

import importlib.util
import nibabel as nib

# def register_fa_on_MNI(dwi_fa,**kwargs):
#     # flirt -ref tractseg/tractseg/resources/MNI_FA_template.nii.gz -in FA.nii.gz \
# # -out FA_MNI.nii.gz -omat FA_2_MNI.mat -dof 6 -cost mutualinfo -searchcost mutualinfo
#     """
#     Register the given FA image to MNI space using dipy.

#     Parameters
#     ----------
#     dwi_fa : BIDSFile
#         BIDSFile object containing the FA image to register
#     kwargs : dict
#         Additional arguments to pass to the registration algorithm

#     Returns
#     -------
#     dict
#         Dictionary containing the output files and their paths

#     """

#     inputs = {
#         "dwi_fa": dwi_fa
#     }

#     #Get the tractseg MNI FA template
#     #Get the tractseg site-package
#     tractseg_spec = importlib.util.find_spec("tractseg")
#     tractseg_dir = os.path.dirname(tractseg_spec.origin)
#     mni_fa = os.path.join(tractseg_dir, "resources", "MNI_FA_template.nii.gz")

#     command_args = ['-ref', mni_fa,
#                     '-in', '$dwi_fa',
#                     '-out', 'FA_MNI.nii.gz',
#                     '-omat', 'FA_2_MNI.mat',
#                     '-dof', '6',
#                     '-cost', 'mutualinfo',
#                     '-searchcost', 'mutualinfo']
    
#     # Define output patterns for registered files
#     output_patterns = {
#         "FA_2_MNI.mat": {
#             "suffix": "xfm",
#             "from" : "subject",
#             "to": "MNI",
#             "extension": "mat"
#         }
#     }

#     # Run the flirt command
#     res_dict = run_cli_command('flirt', inputs, output_patterns, command_args=command_args, entities_template=dwi_fa.get_entities(), **kwargs)
     
#     return res_dict

def register_fa_on_MNI(dwi_fa, **kwargs):
    """
    Register the given FA image to MNI space using ANTsPy

    Parameters
    ----------
    dwi_fa : BIDSFile
        BIDSFile object containing the FA image to register
    kwargs : dict
        Additional arguments to pass to the registration algorithm

    Returns
    -------
    dict
        Dictionary containing the output files and their paths
    """
    inputs = {
        "dwi_fa": dwi_fa
    }

    # Get the tractseg MNI FA template
    tractseg_spec = importlib.util.find_spec("tractseg")
    tractseg_dir = os.path.dirname(tractseg_spec.origin)
    mni_fa_path = os.path.join(tractseg_dir, "resources", "MNI_FA_template.nii.gz")

    # Load the fixed (MNI) and moving (subject FA) images with ANTs
    fixed_img = ants.image_read(mni_fa_path)
    moving_img = ants.image_read(dwi_fa.path)

    # Perform registration
    registration = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Affine',
        verbose=True
    )

    # Save the transformation matrix to a file
    transform_file = "/tmp/FA_2_MNI.mat"
    # ANTs transforms are saved during registration, we just need to use the fwdtransforms
    shutil.copy(registration['fwdtransforms'][0], transform_file)
    # Apply the transformation to the moving image
    moved_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_img,
        transformlist=registration['fwdtransforms'],
        interpolator='linear'
    )
    # Save the moved image
    moved_img_path = "/tmp/FA_MNI.nii.gz"
    ants.image_write(moved_img, moved_img_path)

    res_dict = {
        transform_file: upt_dict(dwi_fa.get_entities(), {
            "suffix": "xfm",
            "from": "subject",
            "to": "MNI",
            "extension": "mat"
        }),
        moved_img_path: upt_dict(dwi_fa.get_entities(), {
            "space": "MNI"
        })
    }

    return res_dict
    



def move_peaks_to_mni(peaks, affine, **kwargs):
    """
    Move the given peaks file to MNI space using SimpleITK.

    Parameters
    ----------
    peaks : BIDSFile
        BIDSFile object containing the peaks to move
    affine : str, BIDSFile or numpy.ndarray
        Affine transformation matrix to apply
    kwargs : dict
        Additional arguments to pass to the moving algorithm

    Returns
    -------
    dict
        Dictionary containing the output files and their paths
    """
    inputs = {
        "peaks": peaks
    }
    
    peaks_img = sitk.ReadImage(peaks.path)
    sitk_affine = sitk.ReadTransform(affine.path)
    
    # Créer un tableau pour stocker les sous-volumes transformés
    moved_subvolumes = []
    
    tractseg_spec = importlib.util.find_spec("tractseg")
    tractseg_dir = os.path.dirname(tractseg_spec.origin)
    mni_fa = os.path.join(tractseg_dir, "resources", "MNI_FA_template.nii.gz")
    ref = sitk.ReadImage(mni_fa)

    print(peaks_img.GetSize())
    # Appliquer la transformation à chaque sous-volume
    for i in range(peaks_img.GetSize()[-1]):
        subvolume = peaks_img[...,i]
        print(subvolume.GetSize())
        moved_subvolume = sitk.Resample(subvolume, ref, sitk_affine, sitk.sitkLinear, 0.0, subvolume.GetPixelID())
        moved_subvolumes.append(moved_subvolume)
    
    # Utiliser la méthode Join Series pour recombiner les volumes
    moved_peaks = sitk.JoinSeries(moved_subvolumes)
    print(moved_peaks.GetSize())
    return moved_peaks
    

def register_to_mni(dwi, bval, bvec, **kwargs):

    """
    Register the given DWI image to MNI space using FSL.

    Parameters
    ----------
    dwi : BIDSFile
        BIDSFile object containing the DWI image to register
    bval : BIDSFile
        BIDSFile object containing the b-values
    bvec : BIDSFile
        BIDSFile object containing the b-vectors
    kwargs : dict
        Additional arguments to pass to the registration algorithm

    Returns
    -------
    dict
        Dictionary containing the output files and their paths

    """

    inputs = {
        "dwi": dwi,
        "bval": bval,
        "bvec": bvec,
    }

    # Define output patterns for registered files
    output_patterns = {
        "dwi_mni.nii.gz": {
            "suffix": "dwi_mni",
            "datatype": "dwi",
            "extension": "nii.gz"
        },
        "bval_mni.bvals": {
            "suffix": "bval_mni",
            "datatype": "bval",
            "extension": "bvals"
        },
        "bvec_mni.bvecs": {
            "suffix": "bvec_mni",
            "datatype": "bvec",
            "extension": "bvecs"
        }
    }

    res_dict = run_cli_command('register_to_mni', inputs, output_patterns, entities_template=dwi.get_entities(), **kwargs)
    return res_dict

def tractseg(peaks,dwi=None,bval=None,bvec=None,**kwargs):
    """
    Call the TractSeg algorithm on the given peaks file.
    Parameters
    ----------
    peaks : BIDSFile
        BIDSFile object containing the peaks to segment (MRTrix format)
    kwargs : dict
        Additional arguments to pass to the TractSeg algorithm
    Returns
    -------
    dict
        Dictionary containing the output files and their paths

    """

    inputs = {
        "peaks": peaks,
    }

# calc_FA -i Diffusion.nii.gz -o FA.nii.gz --bvals Diffusion.bvals --bvecs Diffusion.bvecs \
# --brain_mask nodif_brain_mask.nii.gz

# flirt -ref tractseg/tractseg/resources/MNI_FA_template.nii.gz -in FA.nii.gz \
# -out FA_MNI.nii.gz -omat FA_2_MNI.mat -dof 6 -cost mutualinfo -searchcost mutualinfo

# flirt -ref tractseg/tractseg/resources/MNI_FA_template.nii.gz -in Diffusion.nii.gz \
# -out Diffusion_MNI.nii.gz -applyxfm -init FA_2_MNI.mat -dof 6
# cp Diffusion.bvals Diffusion_MNI.bvals
# rotate_bvecs -i Diffusion.bvecs -t FA_2_MNI.mat -o Diffusion_MNI.bvecs
    # TractSeg -i peaks.nii.gz --output_type tract_segmentation
    # TractSeg -i peaks.nii.gz --output_type endings_segmentation
    # TractSeg -i peaks.nii.gz --output_type TOM 
    # Tracking -i peaks.nii.gz

    register = False
    if all([dwi, bval, bvec]):
        inputs["dwi"] = dwi
        inputs["bval"] = bval
        inputs["bvec"] = bvec
        register = True

        calc_fa = [
            'calc_FA',
            '-i', '$dwi',
            '-o', 'FA.nii.gz',
            '--bvals', '$bval',
            '--bvecs', '$bvec'
        ]

    

    

    tract_segmentation = [
        "TractSeg",
        "-i", peaks.path,
        "--output_type", "tract_segmentation"
    ]

    endings_segmentation = [
        "TractSeg",
        "-i", peaks.path,
        "--output_type", "endings_segmentation"
    ]
    tom_segmentation = [
        "TractSeg",
        "-i", peaks.path,
        "--output_type", "TOM"
    ]
    tracking = [
        "Tracking",
        "-i", peaks.path,
    ]

    commands = [tract_segmentation, endings_segmentation, tom_segmentation, tracking]

    res_dict= run_multiple_commands(commands, inputs, {}, entities_template=peaks.get_entities(), **kwargs)

    print(res_dict)






if __name__ == "__main__":
    config, tools = set_config()
    sub = Subject("03011")
