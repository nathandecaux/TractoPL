from pprint import pprint
import os
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config, get_HCP_bundle_names
from TractoPL.data.loader import Subject, Dataset
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.mcm import mcm_estimator, update_mcm_info_file, add_mcm_to_tracts
from TractoPL.data.mcmfile import MCMFile
from TractoPL.utils.mat import compute_mcm_metrics

import tempfile
import glob
import shutil
import zipfile
import multiprocessing  # Ajout de multiprocessing
import traceback
import click


def _bundle_names(subject, bundle_name, pipeline, datatype='tracto'):
    if bundle_name != "ALL":
        return bundle_name if isinstance(bundle_name, list) else [bundle_name]
    bundles = subject.get(pipeline=pipeline, datatype=datatype, suffix='tracto')
    return sorted(
        {bundle.get_entities().get('bundle') for bundle in bundles if bundle.get_entities().get('bundle')}
    )

def get_dwi_data(subject):
    """Helper function to get DWI data for a subject"""
    dwi = subject.get_unique(suffix='dwi', desc='preproc',
                             extension='nii.gz', pipeline='preprocessing',model=None)
    bval = subject.get(extension='bval',scope='raw')[0]
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', space='B0')
    return dwi, bval, bvec, mask


# def init_pipeline(subject, pipeline, **kwargs):
#     """Initialize the MCM pipeline"""
#     create_pipeline_description(
#         pipeline,
#         layout=subject.layout,
#         **kwargs
#     )
#     return True


def process_mcm_estimation(subject, pipeline, **kwargs):
    mcmx=subject.get(extension='mcmx', pipeline=pipeline)
    if len(mcmx)>0:
        print(f"MCM estimation already done for subject {subject}. Skipping.")
        return True
    dwi, bval, bvec, mask = get_dwi_data(subject)
    if 'M' not in kwargs.keys():
        compart_map = subject.get_unique(
            suffix='density', extension='nii.gz', desc='fixels2peaks')
        res_dict = mcm_estimator(dwi, bval, bvec, mask,
                                compart_map=compart_map, **kwargs)
    else:
        print("Estimation with MCM model selection")
        res_dict = mcm_estimator(dwi, bval, bvec, mask, **kwargs)
    mapping = copy_from_dict(subject, res_dict, pipeline=pipeline,remove_after_copy=False)
    # update_mcm_info_file(mapping)
    return True

# def mcm_to_trekker_tracts(subject, pipeline, **kwargs):
#     """
#     Project MCM metrics to IFOD2 tracts.

#     Parameters
#     ----------
#     subject : Subject
#         The subject object containing the data
#     pipeline : str
#         The pipeline to use for processing
#     kwargs : dict
#         Additional keyword arguments for processing
#     """
#     mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
#     tracts= subject.get_unique(suffix='tracto', algo='trekker',orient='LPS',desc='normalized', extension='vtk',pipeline='msmt_csd')
#     reference = subject.get(space='B0', datatype='anat',extension='nii.gz')[0]

#     res_dict = add_mcm_to_tracts(mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
#     copy_from_dict(subject, res_dict, pipeline=pipeline)
#     return True


def mcm_to_bundleseg_tracts(subject, pipeline, bundle_name,bundle_pipeline='bundle_seg', **kwargs):
    """
    Project MCM metrics to bundlesegmentation tracts.

    Parameters
    ----------
    subject : Subject
        The subject object containing the data
    pipeline : str
        The pipeline to use for processing
    kwargs : dict
        Additional keyword arguments for processing
    """

    if 'overwrite' in kwargs:
        overwrite = kwargs.pop('overwrite')
    else:
        overwrite = False
    mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
    reference = subject.get(
        metric='FA', pipeline='preprocessing', datatype='dwi', extension='nii.gz')[0]

    bundle_list = _bundle_names(subject, bundle_name, bundle_pipeline)

    already_done = subject.get(suffix='tracto',
                               pipeline=pipeline, extension='vtk')
    if len(already_done) > 0 and not overwrite:
        already_done = [output.get_full_entities()['bundle'] for output in already_done]
        print(f"Already done: {already_done}")
        bundle_list = list(set(bundle_list) - set(already_done))

        # if len(already_done) > 40:
        #     print(
        #         f"Already done: {len(already_done)} bundles. Skipping.")
        #     return True

    for bundle_name in bundle_list:
        # #Empty the temp directories
        # for f in glob.glob(os.path.join(tempfile.gettempdir(), '*')):
        #     try:
        #         os.remove(f)
        #     except Exception as e:
        #         print(f"Error removing {f}: {e}")

        tracts = subject.get(datatype='tracto', bundle=bundle_name,
                             extension='trk', pipeline=bundle_pipeline)
        if len(tracts) == 0:
            print(
                f"Bundle {bundle_name} not found for subject {subject}. Skipping.")
            continue

        try:
            tracts = tracts[0]
            res_dict = add_mcm_to_tracts(
                mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
            copy_from_dict(subject, res_dict, pipeline=pipeline)
        except Exception as e:
            print(f"Error processing bundle {bundle_name} for subject {subject}: {e}")
            continue


        #Remove all files in res_dict.keys() basedir
        try:
            for k in res_dict.keys():
                tempfile_dir = os.path.dirname(k)
                if os.path.exists(tempfile_dir):
                    for f in glob.glob(os.path.join(tempfile_dir, '*')):
                            os.remove(f)
        except Exception as e:
            print(f"Error removing temporary files: {e}")
                    
        #Delete all the objects apart from the one used in the loop
        del res_dict
        del tracts

    
    return True

def mcm_to_bundleseg_tracts_full(subject, pipeline, bundle_name, **kwargs):
    """
    Project MCM metrics to bundlesegmentation tracts.

    Parameters
    ----------
    subject : Subject
        The subject object containing the data
    pipeline : str
        The pipeline to use for processing
    kwargs : dict
        Additional keyword arguments for processing
    """

    if 'overwrite' in kwargs:
        overwrite = kwargs.pop('overwrite')
    else:
        overwrite = False
    mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
    reference = subject.get(
        metric='FA', pipeline='preprocessing', datatype='dwi', extension='nii.gz')[0]

    bundle_list = _bundle_names(subject, bundle_name, 'bundle_seg')

    already_done = subject.get(suffix='tracto',
                               pipeline=pipeline, extension='vtk',desc='full')
    if len(already_done) > 0 and not overwrite:
        already_done = [output.get_full_entities()['bundle'] for output in already_done]
        print(f"Already done: {already_done}")
        bundle_list = list(set(bundle_list) - set(already_done))

        # if len(already_done) > 40:
        #     print(
        #         f"Already done: {len(already_done)} bundles. Skipping.")
        #     return True

    for bundle_name in bundle_list:
        # #Empty the temp directories
        # for f in glob.glob(os.path.join(tempfile.gettempdir(), '*')):
        #     try:
        #         os.remove(f)
        #     except Exception as e:
        #         print(f"Error removing {f}: {e}")

        tracts = subject.get(datatype='tracto', bundle=bundle_name,
                             extension='trk', pipeline='bundle_seg')
        if len(tracts) == 0:
            print(
                f"Bundle {bundle_name} not found for subject {subject}. Skipping.")
            continue

        try:
            tracts = tracts[0]
            res_dict = add_mcm_to_tracts(
                mcm_file=mcm_file, tracts=tracts, reference=reference,full=True, **kwargs)
            copy_from_dict(subject, res_dict, pipeline=pipeline,datatype='mat')
        except Exception as e:
            print(f"Error processing bundle {bundle_name} for subject {subject}: {e}")
            continue


        #Remove all files in res_dict.keys() basedir
        try:
            for k in res_dict.keys():
                tempfile_dir = os.path.dirname(k)
                if os.path.exists(tempfile_dir):
                    for f in glob.glob(os.path.join(tempfile_dir, '*')):
                            os.remove(f)
        except Exception as e:
            print(f"Error removing temporary files: {e}")
                    
        #Delete all the objects apart from the one used in the loop
        del res_dict
        del tracts

    
    return True

def get_mcm_metrics(subject, pipeline,bundle_name='ALL', **kwargs):
    """
    Compute a specific MCM metric for the subject.

    Parameters
    ----------
    subject : Subject
        The subject object containing the data
    pipeline : str
        The pipeline to use for processing
    metric_name : str
        The name of the metric to compute
    kwargs : dict
        Additional keyword arguments for processing
    """

    if 'overwrite' in kwargs:
        overwrite = kwargs.pop('overwrite')
    else:
        overwrite = False
    
    bundle_list = _bundle_names(subject, bundle_name, pipeline, datatype='mat')

    already_done = subject.get(suffix='tracto',
                               pipeline=pipeline, extension='vtk',desc='full')
    if len(already_done) > 0 and not overwrite:
        already_done = [output.get_full_entities()['bundle'] for output in already_done]
        print(f"Already done: {already_done}")
        bundle_list = list(set(bundle_list) - set(already_done))
    
    mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)

    reference = subject.get(metric='FA', pipeline='preprocessing', datatype='dwi', extension='nii.gz')[0]

    for bundle_name in bundle_list:
        # #Empty the temp directories
        # for f in glob.glob(os.path.join(tempfile.gettempdir(), '*')):
        #     try:
        #         os.remove(f)
        #     except Exception as e:
        #         print(f"Error removing {f}: {e}")
        
        tracts = subject.get(datatype='mat', pipeline=pipeline,extension='vtk',desc='full',bundle=bundle_name)

        if len(tracts) == 0:
            print(
                f"Bundle {bundle_name} not found for subject {subject}. Skipping.")
            continue

        try:
            tracts = tracts[0]
            res_dict = compute_mcm_metrics(mcm_vtk=tracts,mcm=mcm_file, reference=reference, **kwargs)
            copy_from_dict(subject, res_dict, pipeline=pipeline)
        except Exception as e:
            print(f"Error processing bundle {bundle_name} for subject {subject}: {e}")
            continue


        #Remove all files in res_dict.keys() basedir
        try:
            for k in res_dict.keys():
                tempfile_dir = os.path.dirname(k)
                if os.path.exists(tempfile_dir):
                    for f in glob.glob(os.path.join(tempfile_dir, '*')):
                            os.remove(f)
        except Exception as e:
            print(f"Error removing temporary files: {e}")
                    
        #Delete all the objects apart from the one used in the loop
        del res_dict
        del tracts
    return True

def mcm_to_hcp_bundles(subject, pipeline, bundle_name='ALL', **kwargs):
    """
    Project MCM metrics to bundlesegmentation tracts.

    Parameters
    ----------
    subject : Subject
        The subject object containing the data
    pipeline : str
        The pipeline to use for processing
    kwargs : dict
        Additional keyword arguments for processing
    """

    if 'overwrite' in kwargs:
        overwrite = kwargs.pop('overwrite')
    else:
        overwrite = False
    mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
    reference = subject.get(
        metric='FA', pipeline='preprocessing', datatype='dwi', extension='nii.gz')[0]

    bundle_list = _bundle_names(subject, bundle_name, 'bundle_seg')

    already_done = subject.get(suffix='tracto',
                               desc='cleaned',pipeline=pipeline, extension='vtk')
    if len(already_done) > 0 and not overwrite:
        already_done = [subject.get_full_entities()['bundle'] for subject in already_done]
        print(f"Already done: {already_done}")
        bundle_list = list(set(bundle_list) - set(already_done))

        # if len(already_done) > 40:
        #     print(
        #         f"Already done: {len(already_done)} bundles. Skipping.")
        #     return True

    for bundle_name in bundle_list:
        # #Empty the temp directories
        # for f in glob.glob(os.path.join(tempfile.gettempdir(), '*')):
        #     try:
        #         os.remove(f)
        #     except Exception as e:
        #         print(f"Error removing {f}: {e}")

        tracts = subject.get(suffix='tracto', bundle=bundle_name,
                             extension='trk', pipeline='bundle_seg',atlas='HCP')
        if len(tracts) == 0:
            print(
                f"Bundle {bundle_name} not found for subject {subject}. Skipping.")
            continue

        try:
            tracts = tracts[0]
            res_dict = add_mcm_to_tracts(
                mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
            copy_from_dict(subject, res_dict, pipeline='mcm_on_hcp_bundles')
        except Exception as e:
            print(f"Error processing bundle {bundle_name} for subject {subject}: {e}")
            continue


        #Remove all files in res_dict.keys() basedir
        try:
            for k in res_dict.keys():
                tempfile_dir = os.path.dirname(k)
                if os.path.exists(tempfile_dir):
                    for f in glob.glob(os.path.join(tempfile_dir, '*')):
                            os.remove(f)
        except Exception as e:
            print(f"Error removing temporary files: {e}")
                    
        #Delete all the objects apart from the one used in the loop
        del res_dict
        del tracts
    return True

def create_fake_mcm_from_dti(subject, pipeline='fake_mcm_dti'):
    """
    Create a fake MCM file from FA map for testing purposes.

    Parameters
    ----------
    subject : Subject
        The subject object containing the data
    pipeline : str
        The pipeline to use for processing
    """

    dti_file = subject.get_unique(model='DTI', pipeline='preprocessing', extension='nii.gz',metric=None)
    mask_file = subject.get_unique(suffix='mask', label='brain', space='B0', pipeline='preprocessing', extension='nii.gz')



    dti_path = dti_file.path
    mask_path = mask_file.path

    dti_basename = os.path.basename(dti_path)
    mask_basename = os.path.basename(mask_path)

    # Répertoire de travail temporaire
    temp_dir = tempfile.mkdtemp()
    mcm_name = 'mcm'

    # Copier les fichiers dans le sous-dossier mcm/
    subdir_path = os.path.join(temp_dir, mcm_name)
    os.makedirs(subdir_path, exist_ok=True)
    shutil.copy2(dti_path, os.path.join(subdir_path, dti_basename))
    shutil.copy2(mask_path, os.path.join(subdir_path, mask_basename))

    # Écriture du fichier .mcm XML avec uniquement les noms de fichiers (sans chemin)
    mcm_filename = f'{mcm_name}.mcm'
    mcm_filepath = os.path.join(temp_dir, mcm_filename)
    xml_content = f"""<?xml version="1.0"?>
<Model>
  <Weights>{mask_basename}</Weights>
  <Compartment>
    <Type>Tensor</Type>
    <FileName>{dti_basename}</FileName>
  </Compartment>
</Model>
"""
    with open(mcm_filepath, 'w') as f:
        f.write(xml_content)

    # Créer l'archive .mcmx contenant le .mcm et le sous-dossier mcm/ avec les fichiers
    mcmx_path = os.path.join(temp_dir, f'{mcm_name}.mcmx')
    with zipfile.ZipFile(mcmx_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(mcm_filepath, mcm_filename)
        zipf.write(os.path.join(subdir_path, dti_basename), os.path.join(mcm_name, dti_basename))
        zipf.write(os.path.join(subdir_path, mask_basename), os.path.join(mcm_name, mask_basename))

    base_entities = dti_file.get_full_entities()
    result = {
        mcmx_path: upt_dict(base_entities, model='MCM', extension='.mcmx')
    }
    copy_from_dict(subject, result, pipeline=pipeline)
    return True



    

# def mcm_to_recobundle_bundle(subject,pipeline,bundle,**kwargs):
#     """
#     Project MCM metrics to Recobundle bundle.

#     Parameters
#     ----------
#     subject : Subject
#         The subject object containing the data
#     pipeline : str
#         The pipeline to use for processing
#     bundle : str
#         The bundle name to convert
#     kwargs : dict
#         Additional keyword arguments for processing
#     """
#     mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
#     #Remove - and _ from the bundle name
#     bundle_short = bundle.replace('-','').replace('_','')
#     tracts= subject.get_unique(suffix='tracto', pipeline='recobundle_segmentation',space='HCP105Group1Clustered',desc='normalized', extension='vtk',desc='noslr', bundle=bundle_short)
#     reference = subject.get(space='B0', datatype='anat',extension='nii.gz')[0]

#     res_dict = add_mcm_to_tracts(mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
#     copy_from_dict(subject, res_dict, pipeline=pipeline)
#     return True


def process_mcm_pipeline(
    subject,
    pipeline='mcm_tensors',
    pipeline_list=None,
    estimation_options=None,
    bundle_pipeline='bundle_seg',
):
    """
    Process the MSMT-CSD pipeline on the given subject.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """

    if isinstance(subject, str):
        subject = Subject(subject)

    if pipeline_list is None:
    # Define processing steps
        pipeline_list = [
            # 'init',
        #    'mcm_estimation',
            # 'create_fake_mcm_from_dti',
            'mcm_to_bundleseg_tracts',
            # 'mcm_to_bundleseg_tracts_full',
            # 'get_mcm_metrics',
            #'mcm_to_hcp_bundles'

        ]

    estimation_options = estimation_options or {
        'R': True,
        'c': 3,
        'n': 3,
        'F': True,
        'ml_mode': CLIArg('ml-mode', 2),
        'opt': CLIArg('optimizer', 'levenberg'),
    }

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'mcm_estimation': lambda: process_mcm_estimation(
            subject, pipeline, **estimation_options
        ),
            'create_fake_mcm_from_dti': lambda: create_fake_mcm_from_dti(subject, pipeline),
        'mcm_to_bundleseg_tracts': lambda: mcm_to_bundleseg_tracts(
            subject,
            pipeline,
            bundle_name='ALL',
            overwrite=False,
            bundle_pipeline=bundle_pipeline,
        ),
        'mcm_to_bundleseg_tracts_full': lambda: mcm_to_bundleseg_tracts_full(subject, pipeline, bundle_name='ALL',overwrite=True),
        'mcm_to_hcp_bundles': lambda: mcm_to_hcp_bundles(subject, pipeline, overwrite=True),
        'get_mcm_metrics': lambda: get_mcm_metrics(subject, pipeline,bundle_name='ALL', overwrite=True),
    }

    for index, step in enumerate(pipeline_list):
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()

            if index < len(pipeline_list) - 1:
                subject = Subject(subject.sub_id, db_root=subject.db_root)



def process_subject(
    sub,
    dataset_path,
    pipeline_name,
    pipeline_list=None,
    estimation_options=None,
    bundle_pipeline='bundle_seg',
):
    """
    Process a single subject - worker function for multiprocessing.

    Parameters
    ----------
    sub : str
        Subject ID
    dataset_path : str
        Path to the BIDS dataset
    pipeline_name : str
        Pipeline name to use
    """
    # Initialize subject
    ds = Dataset(dataset_path)
    subject = ds.get_subject(sub)

    # Check if already processed
    # if len(subject.get(model='MCM', pipeline=pipeline_name)) > 0:
    #     print(f"Skipping subject {sub} as tractography already exists")
    #     return

    print(f"Processing subject: {sub}")
    try:
        process_mcm_pipeline(
            subject,
            pipeline=pipeline_name,
            pipeline_list=pipeline_list,
            estimation_options=estimation_options,
            bundle_pipeline=bundle_pipeline,
        )
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        # Optionally, you can log the error or take other actions here
        return None
    
    return sub

    
@click.command()
@click.argument('step', type=click.Choice(['estimation','projection'], case_sensitive=False))
@click.option('--subject', required=True, help='Subject ID to process.')
@click.option('--db-root', required=True, help='Root directory of the BIDS database.')
@click.option('--pipeline', default='mcm_tensors', show_default=True, help='Derivative pipeline name.')
@click.option('--bundle-pipeline', default='bundle_seg', show_default=True, help='Input bundle derivative pipeline.')
@click.option('--n-comparts', type=click.IntRange(1), default=3, show_default=True, help='Number of anisotropic compartments.')
@click.option('--tensor-model', 'tensor_model', type=click.IntRange(1), default=3, show_default=True, help='Anisotropic tensor model.')
@click.option('--free-water/--no-free-water', default=True, show_default=True, help='Estimate a free-water compartment.')
@click.option('--model-selection', is_flag=True, help='Select the MCM model with AIC instead of a fixed compartment count.')
@click.option('--ml-mode', type=click.IntRange(0), default=2, show_default=True, help='Anima maximum-likelihood fitting mode.')
@click.option('--optimizer', default='levenberg', show_default=True, help='Anima optimizer name.')
def cli(step, subject, db_root, pipeline, bundle_pipeline, n_comparts, tensor_model,
    free_water, model_selection, ml_mode, optimizer):
    """
    Command-line interface for processing a single subject.
    """
    dataset_path = db_root
    pipeline_name = pipeline
    if step == 'estimation':
        pipeline_list = ['mcm_estimation']
        estimation_options = {
            'R': True,
            'c': tensor_model,
            'F': free_water,
            'ml_mode': CLIArg('ml-mode', ml_mode),
            'opt': CLIArg('optimizer', optimizer),
        }
        if model_selection:
            estimation_options['M'] = True
        else:
            estimation_options['n'] = n_comparts
    elif step == 'projection':
        pipeline_list = ['mcm_to_bundleseg_tracts']
        estimation_options = None
    process_subject(
        subject,
        dataset_path,
        pipeline_name,
        pipeline_list=pipeline_list,
        estimation_options=estimation_options,
        bundle_pipeline=bundle_pipeline,
    )



if __name__ == "__main__":
    cli()
