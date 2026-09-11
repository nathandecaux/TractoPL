import os

import os
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, Dataset
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template, prepare_atlas_for_recobundle, process_tractosearch
from TractoPL.utils.tractography import get_tractogram_endings, filter_tracto_by_endings, get_fiber_density
from TractoPL.utils.registration import ants_registration
from TractoPL.set_config import get_HCP_bundle_names
from TractoPL.analysis.tractometry import process_projection
import nibabel as ni
import multiprocessing
import tempfile
from glob import glob
import traceback
from subprocess import call
import json
from tqdm import tqdm
from time import sleep

def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True

def get_DTI_metrics(subject, bundle_seg_pipeline='bundle_seg', dti_pipeline='preprocessing', **kwargs):
    """
    Docstring pour get_DTI_metrics
    
    :param subject: Description
    :param bundle_seg_pipeline: Description
    :param dti_pipeline: Description
    :param kwargs: Description
    """
    
    dti_metrics = subject.get(pipeline=dti_pipeline, metric='*')
    bundle_masks=subject.get(pipeline=bundle_seg_pipeline, label='fibers',suffix='mask',datatype='map')

    temp_dir=tempfile.mkdtemp(prefix=f"tractometry_{subject.sub_id}_")
    print(f"Temporary directory created at {temp_dir}")

    res_dict={}
    for dti_metric in dti_metrics:
        print(f"Processing DTI metric: {dti_metric.metric}")
        metric_img = ni.load(dti_metric.path)
        metric_data = metric_img.get_fdata()

        for bundle_mask in bundle_masks:
            print(f"Processing bundle mask: {bundle_mask.bundle}")
            mask_img = ni.load(bundle_mask.path)
            mask_data = mask_img.get_fdata().astype(bool)
            masked_data = metric_data * mask_data

            print(mask_img.shape, metric_img.shape, masked_data.shape)

            #Save in temporary directory
            out_path = os.path.join(temp_dir, f"{subject.sub_id}_{bundle_mask.bundle}_{dti_metric.metric}_masked.nii.gz")
            masked_img = ni.Nifti1Image(masked_data.astype(metric_data.dtype), metric_img.affine, metric_img.header)
            ni.save(masked_img, out_path)
            print(f"Saved masked data to {out_path}")
            res_dict[out_path]=upt_dict(bundle_mask.get_full_entities(),metric=dti_metric.metric,label=None,model='DTI')
    #Copy results to subject database
    copy_from_dict(subject,res_dict, pipeline='DTI_analysis')


def get_DTI_metrics_atlas(subject, bundle_seg_pipeline='bundle_seg', dti_pipeline='preprocessing', atlas_name='HCP', **kwargs):

    dti_metrics = subject.get(pipeline=dti_pipeline, metric='*')
    bundle_masks=subject.get(pipeline=bundle_seg_pipeline, label='fibers',suffix='mask',datatype='atlasmap')

    temp_dir=tempfile.mkdtemp(prefix=f"tractometry_{subject.sub_id}_")
    print(f"Temporary directory created at {temp_dir}")

    res_dict={}
    for dti_metric in dti_metrics:
        print(f"Processing DTI metric: {dti_metric.metric}")
        metric_img = ni.load(dti_metric.path)
        metric_data = metric_img.get_fdata()

        for bundle_mask in bundle_masks:
            print(f"Processing bundle mask: {bundle_mask.bundle}")
            mask_img = ni.load(bundle_mask.path)
            mask_data = mask_img.get_fdata().astype(bool)
            masked_data = metric_data * mask_data

            print(mask_img.shape, metric_img.shape, masked_data.shape)

            #Save in temporary directory
            out_path = os.path.join(temp_dir, f"{subject.sub_id}_{bundle_mask.bundle}_{dti_metric.metric}_masked.nii.gz")
            masked_img = ni.Nifti1Image(masked_data.astype(metric_data.dtype), metric_img.affine, metric_img.header)
            ni.save(masked_img, out_path)
            print(f"Saved masked data to {out_path}")
            res_dict[out_path]=upt_dict(bundle_mask.get_full_entities(),metric=dti_metric.metric,label=None,model='DTI')
    #Copy results to subject database
    copy_from_dict(subject,res_dict, pipeline='DTI_analysis')



def segment_subject_bundleseg(subject,pipeline='recobundle_segmentation', **kwargs):
    """
    Process the MSMT-CSD pipeline on the given subject.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """

    if isinstance(subject, str):
        subject = Subject(subject)
    # Define processing steps
    pipeline_list = [
        'init',
        'get_DTI_metrics',
        # 'get_MCM_metrics',
        'get_DTI_metrics_atlas',
        # 'get_MCM_metrics_atlas'
    ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'get_DTI_metrics': lambda: get_DTI_metrics(subject, **kwargs),
        'get_DTI_metrics_atlas': lambda: get_DTI_metrics_atlas(subject, **kwargs),

    }

    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()
            sleep(5)
            subject=Subject(subject.sub_id, db_root=subject.db_root)  # Reload subject to update database
    
    return True


def process_single_subject(arg):
    """Process a single subject with the given arguments"""

    try :
        sub, dataset_path, pipeline = arg
        print(f"Processing subject {sub} with pipeline {pipeline}")
        subject = Subject(sub, db_root=dataset_path)
        return segment_subject_bundleseg(subject, pipeline=pipeline)
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

from pprint import pprint

if __name__ == "__main__":
    pipeline = 'bundle_seg'
    num_processes = 1

    if os.uname()[1] == 'calcarine':
        num_processes = 32
        print("calcarine")
        # tempfile.tempdir = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bundle_seg'
        tempfile.tempdir = '/local/ndecaux/bundle_seg'
        #also set the TMPDIR env variable
        os.environ['TMPDIR'] = tempfile.tempdir
    else:
        num_processes = 12
        print(f"Not calcarine, using {num_processes} processes")
        tempfile.tempdir = '/home/ndecaux/bundle_seg'
        os.environ['TMPDIR'] = tempfile.tempdir
    # else:
    #     #Tempdir on home
    #     tempfile.tempdir = os.path.join(os.path.expanduser('~'), 'bundle_seg')
    #     os.environ['TMPDIR'] = tempfile.tempdir


    config, tools = set_config()
    # subject = Subject('100206',db_root='/home/ndecaux/Data/HCP/')

    # dataset_path = '/home/ndecaux/Code/Data/comascore'
    # dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/amynet/bids'

    # dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'

    dataset_path = '/home/ndecaux/NAS_EMPENN/share/users/ndecaux/dysdiago/bids'
    # dataset_path='/home/ndecaux/NAS_EMPENN/share/projects/actidep/IRM_Cerveau_MOI/bids'
    ds = Dataset(dataset_path)

    subject_ids = ds.subject_ids
    args = [(sub, pipeline) for sub in subject_ids]
    args_filtered = []
    flag=False
    for arg in args:
        sub, pipeline = arg
        if flag == False and sub == '03026':
            continue
        else:
            flag=True
        
        # sub = Subject(sub, db_root=dataset_path)
        args_filtered.append((sub, dataset_path, pipeline))

    args = args_filtered
    print(f"Found {len(args)} subjects to process")

    # Exécution parallèle avec multiprocessing
    print(f"Démarrage du traitement parallèle avec {num_processes} processus")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_subject, args)
