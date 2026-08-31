from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, parse_filename, BIDSFile
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
from TractoPL.utils.registration import ants_registration
from TractoPL.utils.tractography import apply_affine
from TractoPL.set_config import get_HCP_bundle_names
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import dipy
from dipy.io.streamline import load_trk, save_trk
from dipy.segment.bundles import RecoBundles
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.utils import create_nifti_header
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from time import process_time
from multiprocessing import Pool, cpu_count
from os.path import join as pjoin
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking.streamline import transform_streamlines
import os
import sys
import argparse
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import transform_streamlines
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import dipy
from dipy.align import affine_registration
from subprocess import call

def create_whole_brain_tract(tract_list, ref_image=None, **kwargs):
    """
    Create a whole brain tractogram from a list of tracts.
    
    Parameters
    ----------
    tract_list : list
        List of tracts to concatenate
    ref_image : str
        Path to the reference image
    """

    whole_brain_tract = []
    for tract in tract_list:
        tract = load_trk(tract, reference=ref_image,trk_header_check=False)
        whole_brain_tract.extend(tract.streamlines)

    whole_brain_tract = StatefulTractogram(whole_brain_tract,reference=ref_image, space
                                           =Space.RASMM)
    #Save in first tract directory
    return whole_brain_tract


def process_subject(subject_data, HCP_anat_root, HCP_tract_root_dest):
    """Helper function to process a single subject for parallelization"""
    sub_id, sub_tracts = subject_data
    
    # Check if subject has already been processed
    if os.path.exists(opj(HCP_tract_root_dest, sub_id, "tracts", "whole_brain.trk")):
        return f'Skipping: {sub_id}'
    
    subject_tracts = [f for f in glob.glob(opj(sub_tracts, "tracts", "*.trk")) 
                     if os.path.isfile(f) and os.path.basename(f) != "whole_brain.trk"]
    ref_image = opj(HCP_anat_root, sub_id, "Images", "T1w_acpc_dc_restore_brain.nii.gz")
    whole_brain = create_whole_brain_tract(subject_tracts, ref_image)
    save_trk(whole_brain, opj(HCP_tract_root_dest, sub_id, "tracts", "whole_brain.trk"))
    return f'Processed: {sub_id}'


def create_HCP_whole_brain_tract(HCP_tract_root, HCP_anat_root, HCP_tract_root_dest=None, resume=True, n_jobs=-1, **kwargs):
    """
    Create a whole brain tractogram from the HCP dataset.
    
    Parameters
    ----------
    HCP_tract_root : str
        Path to the HCP tractogram root directory
    HCP_anat_root : str
        Path to the HCP anatomical root directory
    HCP_tract_root_dest : str, optional
        Path to save the whole brain tractograms
    resume : bool, default=True
        Whether to skip already processed subjects
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all available processors
    """

    if HCP_tract_root_dest is None:
        HCP_tract_root_dest = HCP_tract_root

    subject_list = [
        (os.path.basename(d), d) for d in glob.glob(opj(HCP_tract_root, "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ]
    
    if not resume:
        # Process all subjects without checking if they're already done
        subject_list_to_process = subject_list
    else:
        # Filter out already processed subjects
        subject_list_to_process = [
            (sub_id, sub_tracts) for sub_id, sub_tracts in subject_list
            if not os.path.exists(opj(HCP_tract_root_dest, sub_id, "tracts", "whole_brain.trk"))
        ]
    
    # If no subjects to process, exit early
    if not subject_list_to_process:
        print("No subjects to process.")
        return
    
    # Determine number of processes to use
    if n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = min(n_jobs, cpu_count())
    
    print(f"Processing {len(subject_list_to_process)} subjects using {n_jobs} processes")
    
    # Create a pool of workers and map the processing function to subjects
    with Pool(processes=n_jobs) as pool:
        args = [(subject_data, HCP_anat_root, HCP_tract_root_dest) 
                for subject_data in subject_list_to_process]
        
        # Use starmap to pass multiple arguments to the processing function
        results = pool.starmap(process_subject, args)
    
    for result in results:
        print(result)


def register_template_to_subject(streamlines_file, reference_file, atlas_name='ref', **kwargs):
    """
    Register the reference streamlines to the subject streamlines using the StreamlineLinearRegistration algorithm.

    Parameters
    ----------
    streamlines_file : BIDSFile
        BIDSFile object containing the streamlines to register
    reference_file : str
        Path to the reference file
    kwargs : dict
        Additional arguments to pass to the registration script

    Returns
    -------
    dict
        Dictionary with registration output files
    """

    reference_file = reference_file if isinstance(
        reference_file, list) else [reference_file]
    inputs = {
        "streamlines": streamlines_file
    }

    for i, ref in enumerate(reference_file):
        inputs[os.path.basename(ref)] = ref

    # Prepare command arguments
    command_args =  ['$streamlines']+reference_file+ ['--out_moved', "ref_registered.trk"]

    output_patterns = {
        "ref_registered.trk": {
            "suffix": "tracto",
            "datatype": "dwi",
            "extension": "trk",
            "atlas": atlas_name,
            "space": "subject"
        },
        "affine.txt": {
            "suffix": "xfm",
            "datatype": "tracto",
            "extension": "txt",
            "from": atlas_name,
            "to": "subject"
        }
    }

    res_dict =  run_cli_command('dipy_slr', inputs, output_patterns, entities_template=streamlines_file.get_entities(), command_args=command_args, **kwargs)
    return res_dict

from os.path import join as pjoin
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking.streamline import transform_streamlines
import os
import sys
import argparse
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import transform_streamlines
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import dipy

# def register_anat(moving_path, static_path, tract_path,output_dir='.', transform_ref=False):

#     moving, moving_affine = load_nifti(moving_path)
#     static, static_affine = load_nifti(static_path)
#     sft = load_tractogram(tract_path, moving_path, Space.RASMM,bbox_valid_check=False)

#     # Very simple registration parameters
#     pipeline = ["center_of_mass", "rigid", "affine"]


#     warped_b0, warped_b0_affine = affine_registration(
#             moving, static, moving_affine=moving_affine,
#             static_affine=static_affine, pipeline=pipeline)


#     print("Registration completed successfully.")


#     moved_streamlines = transform_streamlines(
#             sft.streamlines, np.linalg.inv(warped_b0_affine))

#     moved_tractogram = StatefulTractogram(
#         moved_streamlines, static_path, Space.RASMM)

#     # Save the transformed tractogram
#     save_tractogram(moved_tractogram, pjoin(output_dir, 'moved_tract.trk'), bbox_valid_check=False)
#     print(f"Transformed tractogram saved to {pjoin(output_dir, 'moved_tract.trk')}")

#     # Save the transformed reference image
#     transformed_ref_path = pjoin(output_dir, 'moved_ref.nii.gz')
#     if transform_ref:
#         dipy.io.image.save_nifti(transformed_ref_path, warped_b0, static_affine)
#         print(f"Transformed reference image saved to {transformed_ref_path}")

#     #Save the affine matrix
#     affine_matrix_path = pjoin(output_dir, 'affine_matrix.txt')
#     np.savetxt(affine_matrix_path, warped_b0_affine)
#     print(f"Affine matrix saved to {affine_matrix_path}")
#     return output_dir

def register_anat_to_template(subject, template_path, tractogram, atlas_name='ref', **kwargs):
    """
    Register the anatomical subject to the template using the specified atlas.

    Parameters
    ----------
    subject : str
        Path to the anatomical subject file
    template_path : str
        Path to the template file
    tractogram : BIDSFile
        BIDSFile object containing the streamlines to register
    atlas_name : str
        Name of the atlas to use for registration
    kwargs : dict
        Additional arguments to pass to the registration script

    Returns
    -------
    dict
        Dictionary with registration output files
    """

    inputs = {
        "tractogram": tractogram,
        "subject": subject,
        "template": template_path
    }

    ants_registration(moving=subject, static=template_path)
    

def register_anat_subject_to_template(subject, template_path, tractogram, atlas_name='ref', **kwargs):
    """
    Register the anatomical subject to the template using the specified atlas.

    Parameters
    ----------
    subject : str
        Path to the anatomical subject file
    template_path : str
        Path to the template file
    tractogram : BIDSFile
        BIDSFile object containing the streamlines to register
    atlas_name : str
        Name of the atlas to use for registration
    kwargs : dict
        Additional arguments to pass to the registration script

    Returns
    -------
    dict
        Dictionary with registration output files
    """

    inputs = {
        "tractogram": tractogram,
        "subject": subject,
        "template": template_path
    }

    #Move to a temporary directory
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    moving_img = ants.image_read(subject if isinstance(subject, str) else subject.path)
    print(template_path)
    fixed_img = ants.image_read(template_path if isinstance(template_path, str) else template_path.path)

    # Perform registration
    registration = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Affine',
        verbose=True
    )

    # Save the transformation matrix to a file
    transform_file = "/tmp/affine.mat"
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
    moved_img_path = "/tmp/moved_ref.nii.gz"
    ants.image_write(moved_img, moved_img_path)

    res_dict = {
        transform_file: upt_dict(subject.get_entities(), {
            "suffix": "xfm",
            "from": "subject",
            "to": atlas_name,
            "extension": "mat"
        }),
        moved_img_path: upt_dict(subject.get_entities(), {
            "space": atlas_name
        })
    }

    #Apply the affine transformation to the tractogram
    moved_tractogram = apply_affine(tractogram, transform_file, reference=subject.path,target=moved_img_path)

    res_dict[moved_tractogram] = upt_dict(tractogram.get_entities(), space = atlas_name,extension='trk')

    return res_dict

    

def call_recobundle(streamlines_file, model_file, **kwargs):
    """
    Call the RecoBundles algorithm on the given streamlines file.

    Parameters
    ----------
    streamlines_file : BIDSFile
        BIDSFile object containing the streamlines to segment
    model_file : str
        Path to the model file
    model_config : str
        Path to the model configuration file
    kwargs : dict
        Additional arguments to pass"
    """

    inputs = {
        "streamlines": streamlines_file,
        "model": model_file,
    }

    bundle_name = kwargs.pop("bundle_name", os.path.basename(model_file).split(".")[0])
    #Convert bundle_name to alphanumeric only
    bundle_name = ''.join(e for e in bundle_name if e.isalnum())

    atlas_name = kwargs.pop("atlas_name", "ref")

    output_patterns = {
        "recognized.trk": {
            "suffix": "tracto",
            "datatype": "tracto",
            "extension": "trk",
            "bundle": bundle_name
        },
        "labels.npy": {
            "suffix": "labels",
            "datatype": "tracto",
            "extension": "npy",
            "bundle": bundle_name
        }
    }

    res_dict = run_cli_command('dipy_recobundles', inputs, output_patterns, entities_template=streamlines_file.get_entities(), **kwargs)
    return res_dict

def prepare_atlas_for_recobundle(model_list, model_config, model_T1,temp_dir=None, **kwargs):
    """
    Prepare the atlas for RecoBundles by registering the model T1 to each model in the model list.

    Parameters
    ----------
    model_list : list
        List of model files (list of trk paths)
    model_config : float, list or dict
        If float, it is the threshold for all models
        If list, it is the list of thresholds for each model (paired with model_list)
        If dict, it is a dictionary with model names as keys and thresholds as values
    model_T1 : str
        Path to the model T1 file
    temp_dir : str, optional
        Path to a temporary directory to store symlink or create files
    kwargs : dict
        Additional arguments to pass to the registration script
    """

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    # Create a symbolic link to model_T1 and all models in model_list in temp_dir
    temp_model_T1 = os.path.abspath(os.path.join(temp_dir, 'atlas_anat.nii.gz'))
    if not os.path.exists(temp_model_T1):
        os.symlink(os.path.abspath(model_T1), temp_model_T1)

    temp_model_list = []
    #Create folders atlas/models/
    os.makedirs(opj(temp_dir, 'atlas', 'models'), exist_ok=True)
    for i, model in enumerate(model_list):
        temp_model = os.path.basename(model)
        if not os.path.exists(temp_model):
            os.symlink(os.path.abspath(model), opj(temp_dir, 'atlas', 'models',temp_model))
        temp_model_list.append(temp_model)
    
    #If model_config is dict, save it as json in temp_dir
    temp_model_config = os.path.abspath(os.path.join(temp_dir, 'config.json'))

    if isinstance(model_config, dict):
        with open(temp_model_config, 'w') as f:
            json.dump(model_config, f)
    elif isinstance(model_config, list):
        if len(model_config) != len(model_list):
            raise ValueError("If model_config is a list, it must have the same length as model_list")
        model_config_dict = {model: thresh for model, thresh in zip(temp_model_list, model_config)}
        with open(temp_model_config, 'w') as f:
            json.dump(model_config_dict, f)
    else :#isinstance(model_config, float) or isinstance(model_config, int) or isinstance(model_config, str):
        model_config_dict = {model:model_config for model in temp_model_list}
        with open(temp_model_config, 'w') as f:
            json.dump(model_config_dict, f)
    
    return temp_dir




def process_bundleseg(streamlines_file, fa_file, atlas_dir='/home/ndecaux/Data/SCIL_Atlas/',config='config.json',rbx_dir='/home/ndecaux/Git/rbx_flow', **kwargs):
    """
    Process the bundle segmentation using the BundleSeg pipeline.
    """
    #Create a temporary directory with the name of the subject
    sub_name = streamlines_file.get_entities()['subject']
    temp_dir = tempfile.mkdtemp(prefix=sub_name + "_")
    
    # temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    #Create symbolic links to the input files in temp_dir/S1/
    streamlines_path = streamlines_file if isinstance(streamlines_file, str) else streamlines_file.path

    fa_path = fa_file if isinstance(fa_file, str) else fa_file.path

    if not streamlines_path.endswith('.trk'):
        #Call flip_tractogram to convert to trk
        #eg : flip_tractogram $streamlines_path $streamlines_path.trk --reference $fa_path 
        old_streamlines_path = streamlines_path
        streamlines_path = old_streamlines_path.split('.')[0] + '.trk'

        if not os.path.exists(streamlines_path):
            call(['flip_tractogram', old_streamlines_path, streamlines_path, '--reference', fa_path])

    os.makedirs("S1", exist_ok=True)
    temp_streamlines = os.path.abspath(os.path.join("S1", 'tracking.trk'))
    temp_fa = os.path.abspath(os.path.join("S1", 'fa.nii.gz'))
    
    shutil.copy(streamlines_path, temp_streamlines)
    shutil.copy(fa_path, temp_fa)
    
    inputs = {
        "streamlines": temp_streamlines,
        "fa": temp_fa,
    }
    #Read main_HCP.nf, and replace config.json by 'config' value
    nf_file=os.path.join(rbx_dir, 'main_HCP.nf')
    with open(nf_file, 'r') as f:
        nf_content = f.read()
        nf_content = nf_content.replace('config.json', config)
    temp_nf_file = os.path.join(temp_dir, 'main_HCP.nf')
    with open(temp_nf_file, 'w') as f:
        f.write(nf_content)
    
    #Symlink the temp main_HCP.nf to rbx_dir (with name f{temp_dir}_main_HCP.nf)
    temp_id=os.path.basename(temp_dir)
    if not os.path.exists(os.path.join(rbx_dir, f'{temp_id}_main_HCP.nf')):
        os.symlink(temp_nf_file, os.path.join(rbx_dir, f'{temp_id}_main_HCP.nf'))



    cmd = f'run {rbx_dir}/{temp_id}_main_HCP.nf --input {temp_dir} --atlas_directory {atlas_dir} -with-singularity {rbx_dir}/scilus_latest.sif -w {temp_dir}'
    cmd = cmd.split(' ')
    #Set the following environment variable : NXF_VER=21.04.0
    os.environ['NXF_VER'] = '21.10.0'

    run_cli_command('nextflow', inputs, {}, entities_template=streamlines_file.get_entities(), command_args=cmd, **kwargs)

    #Use glob to find all recognized bundles in temp_dir/*/*/_cleaned.trk
    recognized_bundles = glob.glob(os.path.join(temp_dir, "*", "*", "*_cleaned.trk"))

    res_dict = {}
    bundle_map = get_HCP_bundle_names()
    entities=streamlines_file.get_entities()
    for bundle in recognized_bundles:
        #Extract bundle name from path 
        bundle_name = [i for i,b in bundle_map.items() if b in str(bundle) or i in str(bundle)]
        if len(bundle_name) != 0:
            bundle_name = bundle_name[0]
        else:
            print(f"Bundle name not found for {bundle}")
            raise
        res_dict[bundle] = upt_dict(entities, {
            "bundle": bundle_name,"extension": "trk"
        })
    
    return res_dict


def process_tractosearch(streamlines_file, models_dict, radius=None, **kwargs):
    """
    Call the TractoSearch algorithm on the given streamlines file.

    Parameters
    ----------
    streamlines_file : BIDSFile
        BIDSFile object containing the streamlines to segment
    models_dict : dict
        Dictonnary of paths to the model file or BIDSFile objects. Keys are bundle names.
    radius : float or dict
        Radius for the nearest neighbor search. If dict, keys are bundle names and values are radius for each bundle.

    kwargs : dict
        Additional arguments to pass"
    """

    temp_dir = tempfile.mkdtemp()
                
    
    if radius is None:
        cmd = ['tractosearch_nearest.py']
    else:
        cmd = ['tractosearch_nearest_in_radius.py']
        cmd.extend(['--mean_distance',str(radius)])


    cmd.extend(['--out_folder', temp_dir])

           
    cmd.extend([streamlines_file.path if isinstance(streamlines_file, str) else streamlines_file.path])
    for bundle_name, model_file in models_dict.items():
        model_path = model_file if isinstance(model_file, str) else model_file.path
        cmd.append(model_path)
    
    if 'in_nii' in kwargs:
        cmd.extend(['--in_nii', kwargs.pop('in_nii') if isinstance(kwargs['in_nii'], str) else kwargs['in_nii'].path])

    if 'ref_nii' in kwargs:
        cmd.extend(['--ref_nii', kwargs.pop('ref_nii') if isinstance(kwargs['ref_nii'], str) else kwargs['ref_nii'].path])



    print("Running command:", ' '.join(cmd))
    call(cmd)

    entities=streamlines_file.get_entities()
    entities.update(kwargs)

    trks = glob.glob(os.path.join(temp_dir, "*.trk"))
    print(trks)
    #trk files contains the original model filename 
    res_dict={}
    for bundle, model_path in models_dict.items():
        model_path_str = model_path if isinstance(model_path, str) else model_path.path
        model_filename = os.path.basename(model_path_str)
        matched_trk = [trk for trk in trks if bundle in os.path.basename(trk)]
        if len(matched_trk) == 0:
            print(f"No matching tract found for model {model_filename} in output folder {temp_dir}")
            continue
        elif len(matched_trk) > 1:
            print(f"Multiple matching tracts found for model {model_filename} in output folder {temp_dir}, taking the first one")
        matched_trk = matched_trk[0]
        res_dict[matched_trk] = upt_dict(entities, {
            "bundle": bundle,
            "extension": "trk"
        })

    return res_dict

if __name__ == "__main__":
    config, tools = set_config()
    sub = Subject("02",'/home/ndecaux/NAS_EMPENN/share/projects/amynet/bids')

    # Exemple d'utilisation :
    # models_path = "/path/to/models"
    # model_T1 = "/path/to/template_T1.nii.gz"
    # model_config = "/path/to/bundles_config.json"
    # process_recobundles(sub, models_path, model_T1, model_config)
    # template_path = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/695768/tracts/whole_brain.trk"
    # template_file_list = glob.glob(opj(template_path, "*.trk"))
    # tracto = sub.get_unique(suffix='tracto', pipeline='msmt_csd', label='WM')
    # output_dict = register_template_to_subject(tracto,template_path)
    # pprint(output_dict)
    # # print(output_dict)
    # #exemple ref_image /data/HCP_Data/Structural_Data_Preprocessed/100408/Images/T1w_acpc_dc_restore_brain.nii.gz
    # ref_image = "/data/HCP_Data/Structural_Data_Preprocessed/992774/Images/T1w_acpc_dc_restore.nii.gz"
    # whole_brain_tract = create_whole_brain_tract(template_file_list, ref_image)
    # save_trk(whole_brain_tract, "whole_brain_tract.trk")

    # HCP_tract_root = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat"
    # HCP_tract_root = "/data/HCP_Data/HCP105_Zenodo"
    # HCP_anat_root = "/data/HCP_Data/Structural_Data_Preprocessed"
    # create_HCP_whole_brain_tract(HCP_tract_root, HCP_anat_root,HCP_tract_root_dest="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat",n_jobs=8)

    #TractoSearch example
    model_files = sub.get(suffix='tracto', atlas='HCP',bundle='CSTleft')
    model_dict={m.bundle:m for m in model_files}
    tracto = sub.get_unique(suffix='tracto', pipeline='msmt_csd', extension='tck')
    in_nii = sub.get_unique(suffix='dwi', datatype='dwi', metric='FA',pipeline='preprocessing')
    output_dict = process_tractosearch(tracto, model_dict, radius=8.0,in_nii=in_nii, atlas='HCP')
    pprint(output_dict)