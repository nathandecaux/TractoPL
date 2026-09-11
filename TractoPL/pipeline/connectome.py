import os
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, Dataset
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template, prepare_atlas_for_recobundle, process_tractosearch
from TractoPL.utils.tractography import get_tractogram_endings, filter_tracto_by_endings, get_fiber_density, compare_fiber_density
from TractoPL.utils.registration import ants_registration
from TractoPL.set_config import get_HCP_bundle_names
from TractoPL.configuration import load_atlas_config
from TractoPL.analysis.tractometry import process_projection
import multiprocessing
import tempfile
from glob import glob
import traceback
from subprocess import call
import json
from tqdm import tqdm
from time import sleep
import numpy as np
import nibabel as nib
import pandas as pd
import click

def _available_bundle_names(subject, pipeline, datatype='tracto', suffix='tracto', atlas_name=None):
    query = {'pipeline': pipeline, 'datatype': datatype, 'suffix': suffix}
    if atlas_name is not None:
        query['atlas'] = atlas_name
    bundles = subject.get(**query)
    return sorted({bundle.get_entities().get('bundle') for bundle in bundles if bundle.get_entities().get('bundle')})


def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MCM pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True

# def segment_bundle_in_atlas_space(subject, pipeline, bundle,atlas_name='HCP105Group1Clustered', **kwargs):
#     """
#     Segment a bundle from a whole brain tractography

#     Parameters
#     ----------
#     subject : Subject
#         Subject object to process
#     pipeline : str
#         Pipeline name
#     bundle : str
#         Name of the bundle to segment
#     """
#     # Load the whole brain tractography
#     whole_brain_tract = subject.get_unique(suffix='tracto', pipeline=pipeline, label='brain',extension='trk', space=atlas_name)

#     # Load the bundle template
#     # template_path = f"/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/695768/tracts/{bundle}.trk"
#     template_path = f"/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/summed_{bundle}.tck"

#     # Register the bundle template to the subject
#     bundle_tract = call_recobundle(whole_brain_tract, template_path,atlas_name=atlas_name,bundle_name=bundle,**kwargs)

#     # Save the segmented bundle
#     copy_from_dict(subject, bundle_tract, pipeline=pipeline, bundle=bundle, desc='noslr')
#     return True

def atlas_fiber_density(subject, pipeline, bundle='ALL', atlas_name=None, **kwargs):
    """
    Compute the fiber density of a bundle in the atlas space

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle : str
        Name of the bundle to compute the fiber density for
    atlas_name : str
        Name of the atlas space to use
    """

    if bundle == 'ALL':
        bundle = _available_bundle_names(subject, 'bundle_seg', atlas_name=atlas_name)
   # Load the reference image in atlas space
    reference_image = subject.get_unique(suffix='dwi', pipeline='preprocessing', metric='FA', extension='nii.gz')
    for b in bundle:
        print(f"Processing bundle {b} for subject {subject.sub_id}")
    # Load the segmented bundle in atlas space
        bundle_tract = subject.get_unique(suffix='tracto', pipeline='bundle_seg', bundle=b, extension='trk', atlas=atlas_name)

        # Compute the fiber density
        density_map = get_fiber_density(bundle_tract, reference_image)

        # Save the fiber density map
        copy_from_dict(subject, density_map, pipeline=pipeline, bundle=b, atlas=atlas_name,datatype='atlas')
    
    return True

def subject_fiber_density(subject, pipeline, bundle='ALL', **kwargs):
    """
    Compute the fiber density of a bundle in the subject space

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle : str
        Name of the bundle to compute the fiber density for
    """

    if bundle == 'ALL':
        bundle = _available_bundle_names(subject, 'bundle_seg')
   # Load the reference image in subject space
    reference_image = subject.get_unique(suffix='dwi', pipeline='preprocessing', metric='FA', extension='nii.gz')
    for b in bundle:
        print(f"Processing bundle {b} for subject {subject.sub_id}")
        # Load the segmented bundle in subject space
        bundle_tract = subject.get_unique(suffix='tracto', pipeline='bundle_seg', bundle=b, extension='trk', datatype='tracto',atlas=None)
        # Compute the fiber density
        density_map = get_fiber_density(bundle_tract, reference_image)
        # Save the fiber density map
        copy_from_dict(subject, density_map, pipeline=pipeline, bundle=b)

    return True

def compare_bundle_density(subject, pipeline, bundle='ALL', atlas_name=None, **kwargs):
    """
    Compare the fiber density of a bundle in the atlas space and in the subject space

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle : str
        Name of the bundle to compare the fiber density for
    """

    if bundle == 'ALL':
        bundle = _available_bundle_names(subject, pipeline, suffix='density', atlas_name=atlas_name)
   # Load the reference image in subject space
    reference_image = subject.get_unique(suffix='dwi', pipeline='preprocessing', metric='FA', extension='nii.gz')

    corr_res=[]
    temp_dir= tempfile.mkdtemp()

    # Accumulators for mean diff, NCC, and disconnection maps across bundles
    diff_accum   = None
    diff_count   = None
    ncc_accum    = None
    ncc_count    = None
    disco_accum  = None
    disco_count  = None
    combined_affine   = None
    combined_entities = None

    for b in bundle:
        print(f"Comparing bundle {b} for subject {subject.sub_id}")
        # Load the fiber density map in atlas space
        atlas_density_path = subject.get_unique(suffix='density', pipeline=pipeline, bundle=b, atlas=atlas_name).path
        # Load the fiber density map in subject space
        subject_density = subject.get_unique(suffix='density', pipeline=pipeline, bundle=b,datatype='map')
        subject_density_path=subject_density.path
        atlas_density_map = nib.load(atlas_density_path).get_fdata()
        subject_density_map = nib.load(subject_density_path).get_fdata()
        

        # Compare the two density maps (e.g. by computing a correlation coefficient)
        correlation = np.corrcoef(atlas_density_map.ravel(), subject_density_map.ravel())[0, 1]
        print(f"Correlation between atlas and subject density maps for bundle {b}: {correlation}")

        # Compute sliding window NCC
        ncc_results = compare_fiber_density(atlas_density_path, subject_density_path, output_dir=os.path.join(temp_dir, b),win=3)
        mean_ncc = ncc_results['mean_ncc']
        print(f"Mean NCC for bundle {b}: {mean_ncc}")

        # Save NCC map
        ncc_map_file = os.path.join(os.path.dirname(ncc_results['mean_ncc_path']), 'fiber_density_ncc.nii.gz')
        res_dict = {ncc_map_file: upt_dict(subject_density.get_full_entities(), datatype='metric', suffix='ncc', bundle=b)}
        copy_from_dict(subject, res_dict, remove_after_copy=False)

        #Compute difference map
        difference_map = subject_density_map - atlas_density_map
        subject_affine = nib.load(subject_density_path).affine

        diff_map_path=os.path.join(temp_dir, f"{subject.sub_id}_{b}_density_difference.nii.gz")
        nib.save(nib.Nifti1Image(difference_map, affine=subject_affine), diff_map_path)
        res_dict={diff_map_path: upt_dict(subject_density.get_full_entities(),datatype='metric',suffix='diff')}
        copy_from_dict(subject, res_dict,remove_after_copy=False)

        # Disconnection probability map: max(0, 1 - subject/atlas), defined where atlas > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            atlas_weight = atlas_density_map / atlas_density_map.max()
            disconnection_map = np.where(
                atlas_density_map > 0,
                np.clip(1.0 - subject_density_map / atlas_density_map, 0, 1) * atlas_weight,
                0
            ).astype(np.float32)

        # Binary disconnection: atlas has fibers (>0) but subject does not
        binary_disconnection_map = ((atlas_density_map > 0) & (subject_density_map == 0)).astype(np.uint8)
        disco_map_path = os.path.join(temp_dir, f"{subject.sub_id}_{b}_disconnection.nii.gz")
        nib.save(nib.Nifti1Image(disconnection_map, affine=subject_affine), disco_map_path)
        res_dict = {disco_map_path: upt_dict(subject_density.get_full_entities(), datatype='metric', suffix='disconnection', bundle=b)}
        copy_from_dict(subject, res_dict, remove_after_copy=False)

        binary_disco_path = os.path.join(temp_dir, f"{subject.sub_id}_{b}_disconnection_binary.nii.gz")
        nib.save(nib.Nifti1Image(binary_disconnection_map, affine=subject_affine), binary_disco_path)
        res_dict = {binary_disco_path: upt_dict(subject_density.get_full_entities(), datatype='metric', suffix='disconnectionbinary', bundle=b)}
        copy_from_dict(subject, res_dict, remove_after_copy=False)

        mean_disconnection = float(disconnection_map[atlas_density_map > 0].mean()) if (atlas_density_map > 0).any() else float('nan')
        corr_res.append({'bundle': b, 'correlation': correlation, 'mean_ncc': mean_ncc, 'mean_disconnection': mean_disconnection})

        # Accumulate for combined maps
        ncc_map = nib.load(ncc_map_file).get_fdata()
        if diff_accum is None:
            diff_accum  = np.zeros_like(difference_map,   dtype=np.float64)
            diff_count  = np.zeros_like(difference_map,   dtype=np.int32)
            ncc_accum   = np.zeros_like(ncc_map,          dtype=np.float64)
            ncc_count   = np.zeros_like(ncc_map,          dtype=np.int32)
            disco_accum        = np.zeros_like(disconnection_map,        dtype=np.float64)
            disco_count        = np.zeros_like(disconnection_map,        dtype=np.int32)
            binary_disco_accum = np.zeros_like(binary_disconnection_map, dtype=np.int32)
            combined_affine   = subject_affine
            combined_entities = subject_density.get_full_entities()
        nonzero_diff = difference_map != 0
        diff_accum[nonzero_diff] += difference_map[nonzero_diff]
        diff_count[nonzero_diff] += 1
        nonzero_ncc = ncc_map != 0
        ncc_accum[nonzero_ncc] += ncc_map[nonzero_ncc]
        ncc_count[nonzero_ncc] += 1
        # Accumulate disconnection where atlas expects fibers
        nonzero_disco = atlas_density_map > 0
        disco_accum[nonzero_disco] += disconnection_map[nonzero_disco]
        disco_count[nonzero_disco] += 1
        binary_disco_accum += binary_disconnection_map.astype(np.int32)

    # Save combined (mean across bundles) diff and NCC maps
    if diff_accum is not None:
        with np.errstate(invalid='ignore'):
            combined_diff = np.where(diff_count > 0, diff_accum / diff_count, 0).astype(np.float32)
            combined_ncc  = np.where(ncc_count  > 0, ncc_accum  / ncc_count,  0).astype(np.float32)

        combined_diff_path = os.path.join(temp_dir, f"{subject.sub_id}_all_bundles_density_diff.nii.gz")
        nib.save(nib.Nifti1Image(combined_diff, combined_affine), combined_diff_path)
        res_dict = {combined_diff_path: upt_dict(combined_entities, datatype='metric', suffix='diff', bundle='all')}
        copy_from_dict(subject, res_dict, remove_after_copy=False)

        combined_ncc_path = os.path.join(temp_dir, f"{subject.sub_id}_all_bundles_ncc.nii.gz")
        nib.save(nib.Nifti1Image(combined_ncc, combined_affine), combined_ncc_path)
        res_dict = {combined_ncc_path: upt_dict(combined_entities, datatype='metric', suffix='ncc', bundle='all')}
        copy_from_dict(subject, res_dict, remove_after_copy=False)

        combined_disco = np.where(disco_count > 0, disco_accum / disco_count, 0).astype(np.float32)
        combined_disco_path = os.path.join(temp_dir, f"{subject.sub_id}_all_bundles_disconnection.nii.gz")
        nib.save(nib.Nifti1Image(combined_disco, combined_affine), combined_disco_path)
        res_dict = {combined_disco_path: upt_dict(combined_entities, datatype='metric', suffix='disconnection', bundle='all')}
        copy_from_dict(subject, res_dict, remove_after_copy=False)

        combined_binary_disco_path = os.path.join(temp_dir, f"{subject.sub_id}_all_bundles_disconnection_binary.nii.gz")
        nib.save(nib.Nifti1Image(binary_disco_accum.astype(np.int32), combined_affine), combined_binary_disco_path)
        res_dict = {combined_binary_disco_path: upt_dict(combined_entities, datatype='metric', suffix='disconnectionbinary', bundle='all')}
        copy_from_dict(subject, res_dict, remove_after_copy=False)

    # Save the correlation results
    corr_df = pd.DataFrame(corr_res)
    corr_df_path=os.path.join(temp_dir, f"{subject.sub_id}_density_correlation.csv")
    corr_df.to_csv(corr_df_path, index=False)
    res_dict={corr_df_path: upt_dict(subject_density.get_full_entities(),datatype='metric',suffix='corr',extension='csv',bundle=None)}
    copy_from_dict(subject, res_dict)

    return True

def connectome_pipeline(subject, pipeline='connectome', atlas_name=None, steps=None, **kwargs):
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
    pipeline_list = steps or ('compare_bundle_density',)

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'atlas_fiber_density': lambda: atlas_fiber_density(subject, pipeline, atlas_name=atlas_name),
        'subject_fiber_density': lambda: subject_fiber_density(subject, pipeline),
        'compare_bundle_density': lambda: compare_bundle_density(subject, pipeline, atlas_name=atlas_name)
    }

    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()
            subject=Subject(subject.sub_id, db_root=subject.db_root)  # Reload subject to update database
    
    return True


def process_single_subject(arg):
    """Process a single subject with the given arguments"""

    try :
        sub, dataset_path, pipeline, atlas_name, steps = arg
        print(f"Processing subject {sub} with pipeline {pipeline}")
        ds=Dataset(dataset_path)
        # subject = Subject(sub, db_root=dataset_path)
        return connectome_pipeline(ds.get_subject(sub), pipeline=pipeline, atlas_name=atlas_name, steps=steps)
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

@click.command()
@click.option('--db-root', required=True, help='Root directory of the BIDS database.')
@click.option('--atlas', required=True, help='Path to the atlas manifest JSON file.')
@click.option('--subject', help='Subject ID. Processes all subjects when omitted.')
@click.option('--pipeline', default='connectome', show_default=True, help='Derivative pipeline name.')
@click.option('--step', 'steps', multiple=True, type=click.Choice(['init', 'atlas_fiber_density', 'subject_fiber_density', 'compare_bundle_density']))
@click.option('--n-proc', type=click.IntRange(min=1), default=1, show_default=True)
def cli(db_root, atlas, subject, pipeline, steps, n_proc):
    """Compute connectome density metrics for one or more subjects."""
    atlas_config = load_atlas_config(atlas)
    dataset = Dataset(db_root)
    subject_ids = [subject] if subject else dataset.subject_ids
    jobs = [(subject_id, db_root, pipeline, atlas_config.name, steps) for subject_id in subject_ids]
    with multiprocessing.Pool(processes=n_proc) as pool:
        pool.map(process_single_subject, jobs)


if __name__ == "__main__":
    cli()