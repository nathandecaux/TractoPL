import os
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, Dataset
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template, prepare_atlas_for_recobundle, process_tractosearch
from TractoPL.utils.tractography import get_tractogram_endings, filter_tracto_by_endings, get_fiber_density,filter_tracto_by_endings_dipy
from TractoPL.utils.registration import ants_registration
from TractoPL.set_config import get_HCP_bundle_names
from TractoPL.configuration import AtlasConfig, load_atlas_config
from TractoPL.analysis.tractometry import process_projection
import multiprocessing
import tempfile
from glob import glob
import traceback
from subprocess import call
import json
from tqdm import tqdm
from time import sleep
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Bundle segmentation pipeline")
    parser.add_argument('step',help="Processing step",choices=['segmentation'])
    parser.add_argument('--atlas', '-a', required=True, help='Path to the atlas manifest JSON file.')
    parser.add_argument('--subject','-s', required=True, help='Subject ID to process.')
    parser.add_argument('--db-root','-d', required=True, help='Path to the BIDS database root.')
    return parser.parse_args()

def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MCM pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True

def segment_bundle_in_atlas_space(subject, pipeline, bundle, atlas, **kwargs):
    """
    Segment a bundle from a whole brain tractography

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle : str
        Name of the bundle to segment
    """
    atlas_name = atlas.name
    # Load the whole brain tractography
    whole_brain_tract = subject.get_unique(suffix='tracto', pipeline=pipeline, label='brain',extension='trk', space=atlas_name)

    template_path = atlas.bundles_dir / f"summed_{atlas.bundle_name(bundle)}.tck"

    # Register the bundle template to the subject
    bundle_tract = call_recobundle(whole_brain_tract, str(template_path),atlas_name=atlas_name,bundle_name=bundle,**kwargs)

    # Save the segmented bundle
    copy_from_dict(subject, bundle_tract, pipeline=pipeline, bundle=bundle, desc='noslr')
    return True

def register_template_to_anat(subject, pipeline, atlas, **kwargs):
    """
    Register the atlas template to the subject's anatomical image.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    atlas_name : str
        Name of the atlas to use for registration. Default is 'HCP105Group1Clustered'.
    """

    atlas_name = atlas.name
    # Load the anatomical image
    anat = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')

    if len(subject.get(pipeline=pipeline, suffix='anat', desc='Warped', extension='nii.gz')) > 0:
        print('Registration already done, skipping')
        return True

    # Register the atlas template to the subject's anatomical image
    # def ants_registration(moving, fixed, outprefix='registered',transform_type='affine', **kwargs):
    print('Registering template to subject anatomical image', anat.path)
    res_dict=ants_registration(str(atlas.reference), anat.path, outprefix='to_anat', moving_space=atlas_name, fixed_space='subject', transform_type='synquick', **kwargs)

    #Get the entry in res_dict that contains desc=='1Warp'
    entry = [k for k,v in res_dict.items() if v.get('desc','') == 'Warped']
    if len(entry) == 0:
        raise ValueError('No 1Warp entry found in registration result')
    entry = entry[0]
    new_entities=res_dict[entry]
    res_dict[entry]=upt_dict(new_entities,atlas=atlas_name,suffix='anat')
    # Save the transformation matrix
    copy_from_dict(subject,res_dict,pipeline='bundle_seg')
    return True

def move_fa_to_template_space(subject, pipeline, atlas, **kwargs):
    #Use the inverse transform to move the subject FA to template space

    atlas_name = atlas.name
    inv_warp=subject.get_unique(suffix='xfm', pipeline='bundle_seg', desc='1InverseWarp', extension='nii.gz')
    affine=subject.get_unique(suffix='xfm', pipeline='bundle_seg', desc='0GenericAffine')

    fa_subject = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')
    if len(subject.get(pipeline=pipeline, suffix='anat', desc='WarpedToTemplate', extension='nii.gz')) > 0:
        print('FA already moved to template space, skipping')
        return True
    print('Moving subject FA to template space', fa_subject.path)

    output_path = os.path.join(tempfile.gettempdir(), f"{subject.sub_id}_fa_in_template.nii.gz")
    cmd=f"antsApplyTransforms -d 3 -i {fa_subject.path} -r {atlas.reference} -o {output_path} -t [{affine.path},1] -t {inv_warp.path} --interpolation Linear"
    call(cmd, shell=True)
    res_dict={output_path:upt_dict(fa_subject.get_full_entities(),atlas=atlas_name,suffix='dwi',space=atlas_name)}
    copy_from_dict(subject,res_dict,pipeline=f'on{atlas_name}_Space')


def apply_trans_to_HCP_bundles(subject, pipeline, atlas, **kwargs):
    """
    Apply the transformation matrix to the HCP bundles using apply_trans_to_vtk.py

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    moving_space : str
        Space of the moving image. Default is 'HCP'.
    """
    moving_space = atlas.name
    # Load the transformation matrix
    inv_warp=subject.get_unique(suffix='xfm', pipeline='bundle_seg', desc='1InverseWarp', extension='nii.gz')
    affine=subject.get_unique(suffix='xfm', pipeline='bundle_seg', desc='0GenericAffine')

    moving_image = atlas.reference
    fixed_image = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')

    hcp_bundles = glob(str(atlas.bundles_dir / 'summed_*.trk'))
    print(f"Looking for atlas bundles in {atlas.bundles_dir}")
    if len(hcp_bundles) == 0:
        hcp_bundles = glob(str(atlas.bundles_dir / 'summed_*.vtk'))
    if len(hcp_bundles) == 0:
        raise FileNotFoundError(f"No atlas bundles found in {atlas.bundles_dir}")

    print(f'Found {len(hcp_bundles)} HCP bundles to transform')
    f_type='trk' if hcp_bundles[0].endswith('.trk') else 'vtk'

    for bundle_path in tqdm(hcp_bundles, desc="Transforming HCP bundles"):
        res_dict = {}
        if len(subject.get(pipeline='bundle_seg', bundle=bundle_path.split('summed_')[-1].split('.'+f_type)[0].replace('_',''), datatype='atlas', space='subject', atlas=moving_space,suffix='tracto'))>0:
            print(f'Bundle {bundle_path} already exists, skipping')
            continue
        bundle_name = bundle_path.split('summed_')[-1].split('.'+f_type)[0]
        print(f'Transforming bundle {bundle_name}')
        output_entities = upt_dict(subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck').get_full_entities(),
                                   {'bundle': bundle_name.replace('_',''), "datatype":'atlas', 'space': 'subject', 'atlas': moving_space, 'desc': 'transformed', 'suffix': 'tracto', 'extension': 'trk'})
        
        output_path = os.path.join(tempfile.gettempdir(), f"{subject.sub_id}_transformed_{bundle_name}.trk")
        #Create folder if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        need_flip = True if f_type == 'vtk' else False
        cmd = f"tractopl-apply-trans-to-vtk {bundle_path} -o {output_path} -t {affine.path} --invert-affine -t {inv_warp.path} --moving-image {moving_image} --fixed-image {fixed_image}"
        # if need_flip:
        #     cmd += " --flip --ants-space RAS"
        print(cmd)
        call(cmd, shell=True)
        res_dict[output_path] = output_entities
        copy_from_dict(subject,res_dict,pipeline="bundle_seg",remove_after_copy=False)

    # Clean up temporary files
    temp_files = glob(os.path.join(tempfile.gettempdir(), f"{subject.sub_id}_transformed_*.{f_type}"))
    for temp_file in temp_files:
        os.remove(temp_file)

def apply_trans_to_HCP_clusters(subject, pipeline, atlas, overwrite=True, **kwargs):
    """
    Apply the transformation matrix to the HCP clusters
    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    moving_space : str
        Space of the moving image. Default is 'HCP'.
    """
    atlas.require('centroids')
    moving_space = atlas.name
    # Load the transformation matrix
    inv_warp=subject.get_unique(suffix='xfm', pipeline='bundle_seg', desc='1InverseWarp', extension='nii.gz')
    affine=subject.get_unique(suffix='xfm', pipeline='bundle_seg', desc='0GenericAffine')

    moving_image = atlas.reference
    fixed_image = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')

    hcp_bundles = glob(str(atlas.centroids_dir / '*_centroids.vtk'))
    if len(hcp_bundles) == 0:
        hcp_bundles = glob(str(atlas.centroids_dir / '*_centroids.trk'))
        if len(hcp_bundles) == 0:
            raise FileNotFoundError(f'No atlas clusters found in {atlas.centroids_dir}')
    print(f'Found {len(hcp_bundles)} HCP clusters to transform')
    cluster_type=atlas.centroids_dir.name.split('_')[-1]
    # hcp_bundles = [b for b in hcp_bundles if 'CST_left' in b]
    for bundle_path in tqdm(hcp_bundles, desc="Transforming HCP clusters"):
        res_dict = {}

        bundle_name = os.path.basename(bundle_path).split('summed_')[-1].split('_centroids.vtk')[0]
        
        if len(subject.get(pipeline='bundle_seg', bundle=bundle_name.replace('_',''), datatype='atlas', space='subject', atlas=moving_space, clustering=cluster_type))>0 and not overwrite:
            print(f'Cluster {bundle_name} already exists, skipping')
            continue
        print(f'Transforming cluster {bundle_name}')
        output_entities = upt_dict(subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck').get_full_entities(),
                                   {'bundle': bundle_name.replace('_',''), "datatype":'atlas', 'space': 'subject', 'atlas': moving_space, 'desc': 'transformed', 'suffix': 'centroids', 'extension': 'vtk','clustering': cluster_type})
        output_path = os.path.join(tempfile.gettempdir(), f"{subject.sub_id}_transformed_{bundle_name}.vtk")
        #Create folder if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = f"tractopl-apply-trans-to-vtk {bundle_path} -o {output_path} -t {affine.path} --invert-affine -t {inv_warp.path} --moving-image {moving_image} --fixed-image {fixed_image}"
        print(cmd)
        call(cmd, shell=True)
        res_dict[output_path] = output_entities
        copy_from_dict(subject,res_dict,pipeline="bundle_seg",remove_after_copy=False)


def run_tractosearch_on_registered_atlas(subject, atlas, pipeline='bundle_seg', **kwargs):
    if isinstance(subject, str):
        subject = Subject(subject)

    print(pipeline)
    already_done = subject.get(suffix='tracto', pipeline=pipeline, extension='trk',datatype='tracto')

    anat_subject=subject.get_unique(pipeline='preprocessing', metric='FA', extension='nii.gz')
    model_bundles = subject.get(pipeline=pipeline, suffix='tracto', atlas=atlas.name, extension='trk',datatype='atlas')

    print("Already done bundles:", [a.get_full_entities()['bundle'] for a in already_done])
    #Remove bundles that are already done
    model_bundles = [m for m in model_bundles if m.get_full_entities()['bundle'] not in [a.get_full_entities()['bundle'] for a in already_done]]
    if len(model_bundles) == 0:
        print('All bundles already done, skipping')
        return True
    print(f'Processing {len(model_bundles)} bundles with TractoSearch')
    model_dict={m.get_full_entities()['bundle']:m for m in model_bundles}

    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck')
    res_dict=process_tractosearch(tracto, model_dict, radius=5.0, in_nii=anat_subject.path,ref_nii=anat_subject.path)
    copy_from_dict(subject,res_dict,pipeline=pipeline, remove_after_copy=False)
    

def run_bundle_seg_selected_bundles(subject, pipeline, bundle_list, atlas, **kwargs):
    """
    Run the bundlesegmentation pipeline on the given subject for selected bundles.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    pipeline : str
        Pipeline name
    bundle_list : list of str
        List of bundles to segment.
    """

    if isinstance(subject, str):
        subject = Subject(subject)
    model_files = glob(str(atlas.bundles_dir / 'summed_*.trk'))
    model_files = [
        path for path in model_files
        if any(atlas.bundle_name(bundle) in path for bundle in bundle_list)
    ]

    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck')
    anat_subject=subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')
    model_dict = {bundle: path for bundle, path in zip(bundle_list, model_files)}
    res_dict=process_tractosearch(
        tracto,
        model_dict,
        radius=5.0,
        in_nii=anat_subject.path,
        ref_nii=anat_subject.path,
        **kwargs,
    )
    copy_from_dict(subject,res_dict,pipeline=pipeline)


def run_bundleseg_on_centroids(subject, pipeline, atlas, **kwargs):
    atlas.require('centroids')
    anat = str(atlas.reference)
    models = glob(str(atlas.centroids_dir / '*_centroids.trk'))
    print(f'Found {len(models)} centroid models')
    # if len(models) == 0 or len(models_vtk) != 0 :
    #     #Convert all vtk to trk
    #     for vtk in models_vtk:
    #         trk = vtk.replace('.vtk', '.trk')
    #         cmd=f'flip_tractogram --reference {anat} {vtk} {trk}'
    #         call(cmd, shell=True)
    #         models.append(trk)
            

    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck')
    anat_subject=subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')
    model_dict = {
        os.path.basename(model).replace('_centroids.trk', ''): model
        for model in models
    }
    res_dict=process_tractosearch(
        tracto,
        model_dict,
        radius=5.0,
        in_nii=anat_subject.path,
        ref_nii=anat_subject.path,
        **kwargs,
    )
    copy_from_dict(subject,res_dict,pipeline=pipeline+'_centroids')

def copy_bundleseg_result(subject, result_folder, pipeline, **kwargs):
    """
    Copy the bundlesegmentation result to the subject's directory.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle_name : str
        Name of the bundle to segment
    kwargs : dict
        Additional keyword arguments for processing
    """

    # Load the bundlesegmentation tractography
    sub_id = subject.sub_id
    # get the folder in result_folder that contains <sub_id>_<SOMETHING>
    sub_folder = [x for x in os.listdir(result_folder) if x.startswith(sub_id+'_')][0]
    #Get the subsubfolder in sub_folder that contains files with _cleaned.trk
    trk_paths = glob(f'{os.path.join(result_folder, sub_folder)}/*/*/*_cleaned.trk')
    bundle_name_mapping = {v:k for k, v in get_HCP_bundle_names().items()}
    
    entities= subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck').get_full_entities()
    entities['extension'] = 'trk'


    res_dict = {f:upt_dict(entities,{'bundle': bundle_name_mapping[f.split('summed_')[-1].split('_cleaned')[0]]}) for f in trk_paths}

    copy_from_dict(subject,res_dict,pipeline=pipeline)
    
def get_single_bundleseg_endings(subject, pipeline, bundle_name, **kwargs):
    """
    Get the endings segmentation of the streamlines in the bundlesegmentation tractography.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle_name : str
        Name of the bundle to segment
    kwargs : dict
        Additional keyword arguments for processing
    """

    # Load the bundlesegmentation tractography
    tracto = subject.get_unique(suffix='tracto', pipeline=pipeline, bundle=bundle_name,extension='trk')

    # Get the endings segmentation
    endings = get_tractogram_endings(tracto, reference=subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz'))

    # Save the endings segmentation
    copy_from_dict(subject, endings, pipeline=pipeline, bundle=bundle_name, suffix='mask', datatype='endings')

def get_hcp_endings(subject, pipeline):
    """
    Get the endings segmentation of the HCP atlas bundles registered on the subject.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    """
    tracto = subject.get(suffix='tracto', pipeline=pipeline, atlas='HCP', extension='trk',datatype='atlas')
    already_done = subject.get(suffix='mask', pipeline=pipeline, datatype='atlasendings')
    already_done = [x.get_full_entities()['bundle'] for x in already_done]
    print(f"Already done: {already_done}")
    print(f"Bundles to process: {tracto}")
    reference = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')
    for bundle in tracto:
        try :
            bundle_name = bundle.get_full_entities()['bundle']
            if bundle_name in already_done:
                print(f"Bundle {bundle_name} already processed, skipping")
                continue
            print(f"Processing bundle {bundle_name}")
            endings = get_tractogram_endings(bundle, reference=reference)
            copy_from_dict(subject, endings, pipeline=pipeline, bundle=bundle_name, suffix='mask', datatype='atlasendings')

        except Exception as e:
            print(f"Error processing bundle {bundle}: {e}")
            continue
    return True

def filter_by_endings(subject, pipeline, bundle_name, datatype='endings', **kwargs):
    #filter_tracto_by_endings(tracto, start_mask, end_mask,**kwargs):
    """
    Filter the streamlines in the bundlesegmentation tractography by their endings.
    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle_name : str
        Name of the bundle to segment
    datatype : str
        Datatype of the endings segmentation
    kwargs : dict
        Additional keyword arguments for processing
    """
    # Load the bundlesegmentation tractography
    tracto = subject.get_unique(suffix='tracto', pipeline=pipeline,extension='tck')
    
    
    # Load the endings segmentation
    beginnings = subject.get_unique(suffix='mask', pipeline='bundle_seg', bundle=bundle_name, datatype=datatype, label='start')
    endings = subject.get_unique(suffix='mask', pipeline='bundle_seg', bundle=bundle_name, datatype=datatype, label='end')
    
    # Filter the streamlines by their endings
    filtered_tracto = filter_tracto_by_endings_dipy(tracto, reference=subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz'), start_mask=beginnings, end_mask=endings, **kwargs)
    
    # Save the filtered tractography
    copy_from_dict(subject, filtered_tracto, pipeline='filtered_tracto', bundle=bundle_name, desc='filtered', suffix='tracto', extension='tck')
    

def filter_by_hcp_endings(subject, pipeline, **kwargs):
    """
    Filter the streamlines in the bundlesegmentation tractography by their endings using the HCP atlas endings.
    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    kwargs : dict
        Additional keyword arguments for processing
    """

    endings_pipeline = 'bundle_seg'
    tracto_pipeline = 'msmt_csd'
    ending_list = subject.get(suffix='mask', datatype='atlasendings', pipeline=endings_pipeline)
    bundle_set = set([x.get_full_entities()['bundle'] for x in ending_list])
    print(f"Bundles with endings: {bundle_set}")
    for bundle in ['STPREMleft']:#bundle_set:
        print(f"Filtering bundle {bundle} by its endings")
        filter_by_endings(subject, tracto_pipeline, bundle, datatype='atlasendings', **kwargs)

    return True

def get_bundleseg_endings(subject, pipeline):
    """
    call get_single_bundleseg_endings for all bundles in the bundlesegmentation tractography
    """

    bundle_list = subject.get(suffix='tracto',
                              pipeline=pipeline,
                              extension='trk')
    print(bundle_list)
    already_done = subject.get(suffix='mask',
                               pipeline=pipeline,
                               desc='endings')
    already_done = [x.get_full_entities()['bundle'] for x in already_done]
    print(f"Already done: {already_done}")

    # print(f"Bundles to process: {bundle_list}")
    for bundle in bundle_list:
        try :
            bundle_name = bundle.get_full_entities()['bundle']
            if False:# bundle_name in already_done:
                print(f"Bundle {bundle_name} already processed, skipping")
                continue
            print(f"Processing bundle {bundle_name}")
            get_single_bundleseg_endings(subject, pipeline, bundle_name)
        except Exception as e:
            print(f"Error processing bundle {bundle}: {e}")
            continue
    return True

def get_hcp_bundle_masks(subject, pipeline, bundle_names='ALL', **kwargs):
    """
    Get the binary mask of the HCP atlas bundle registered on the subject.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle_names : str or list
        Name(s) of the bundle(s) to segment
    kwargs : dict
        Additional keyword arguments for processing
    """

    bundle_names = get_HCP_bundle_names().keys() if bundle_names == 'ALL' else bundle_names
    anat = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')

    for bundle_name in bundle_names:
        print(f'Processing bundle mask for {bundle_name}')
        # Load the bundlesegmentation tractography
        tracto = subject.get_unique(suffix='tracto', pipeline=pipeline, atlas='HCP', bundle=bundle_name, extension='trk',datatype='atlas')

        # Get the reference 

        res_dict= get_fiber_density(tracto, reference=anat)

        copy_from_dict(subject, res_dict, pipeline=pipeline, bundle=bundle_name,datatype='atlasmap')

def get_subject_bundle_masks(subject, pipeline, bundle_names='ALL', **kwargs):
    """
    Get the binary mask of the HCP atlas bundle registered on the subject.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle_names : str or list
        Name(s) of the bundle(s) to segment
    kwargs : dict
        Additional keyword arguments for processing
    """

    bundle_names = get_HCP_bundle_names().keys() if bundle_names == 'ALL' else bundle_names
    anat = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')

    for bundle_name in bundle_names:
        print(f'Processing bundle mask for {bundle_name}')
        # Load the bundlesegmentation tractography
        tracto = subject.get_unique(suffix='tracto', pipeline=pipeline, atlas=None, bundle=bundle_name, extension='trk',datatype='tracto')

        # Get the reference 

        res_dict= get_fiber_density(tracto, reference=anat)

        copy_from_dict(subject, res_dict, pipeline=pipeline, bundle=bundle_name,desc='subject')

def project_metric_onto_bundleseg(subject, pipeline, metric_name, **kwargs):
    """
    Project a metric onto the bundlesegmentation tractography.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    metric_name : str
        Name of the metric to project
    kwargs : dict
        Additional keyword arguments for processing
    """

    # Load the bundlesegmentation tractography
    bundle_list = subject.get(suffix='tracto',
                              pipeline=pipeline,
                              extension='trk')

    # Load the metric to project
    metric_list = subject.get(pipeline='mcm_tensors_staniz',
                              extension='nii.gz',
                              metric=metric_name)
    metric_dict = {
        x.get_full_entities()['bundle']:x for x in metric_list
    }

    beginnings_dict = subject.get(suffix='mask', datatype='endings',
                                  label='start')
    beginnings_dict = {
        x.get_full_entities()['bundle']:x for x in beginnings_dict if x.get_full_entities()['bundle'] in metric_dict.keys()
    }

    #Remove bundle from bundle_list if not in metric_list
    bundle_dict = {
        x.get_full_entities()['bundle']:x for x in bundle_list
        if x.get_full_entities()['bundle'] in metric_dict.keys()
    }

    print(f'Starting projection of {len(bundle_dict)} bundles with metric {metric_name}')

    # Project the metric onto the bundlesegmentation tractography
    res_dict = process_projection(bundle_dict, metric_dict, beginnings_dict, **kwargs)

    copy_from_dict(subject,
                   res_dict,
                   pipeline=pipeline,
                   metric=metric_name,
                   desc='projected',
                   datatype='metric')

def project_all_metrics(subject, pipeline, **kwargs):
    """
    Project all metrics onto the bundlesegmentation tractography.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    kwargs : dict
        Additional keyword arguments for processing
    """

    metric_list = subject.get(pipeline='mcm_tensors_staniz',
                              extension='nii.gz')
    metric_names = set([x.get_full_entities()['metric'] for x in metric_list])

    for metric_name in metric_names:
        try :
            print(f"Projecting metric {metric_name}")
            project_metric_onto_bundleseg(subject, pipeline, metric_name, **kwargs)
        except Exception as e:
            print(f"Error projecting metric {metric_name}: {e}")
            continue
    return True

def list_missing_bundleseg(dataset, pipeline):
    """
    List the bundles that are missing in the bundlesegmentation tractography.

    Parameters
    ----------
    subject : Dataset
        Dataset dataset object
    pipeline : str
        Pipeline name
    """

    all_bundles = set(get_HCP_bundle_names().keys())
    missing_bundles = {}
    ds_bundles = dataset.get_global(suffix='tracto', pipeline=pipeline, extension='trk')
    for sub in dataset.subject_ids:
        sub_bundles = set([x.get_full_entities()['bundle'] for x in ds_bundles if x.get_full_entities()['subject'] == sub])
        missing = all_bundles - sub_bundles
        if len(missing) > 0:
            missing_bundles[sub] = list(missing)
            print(f"Subject {sub} is missing bundles: {missing}")
    #Save as json file
    with open(f'missing_bundles_{pipeline}.json', 'w') as f:
        json.dump(missing_bundles, f, indent=4)

    return missing_bundles

def segment_subject_bundleseg(subject, atlas, pipeline='recobundle_segmentation',pipeline_list=None, **kwargs):
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
    if not pipeline_list:
        pipeline_list = [
            # 'init',
            # 'register_template_to_anat',
            # 'apply_trans_to_HCP_bundles',
            'apply_trans_to_HCP_clusters',
            # 'move_fa_to_template_space',
            # 'run_tractosearch_on_registered_atlas',
            # 'get_hcp_endings',
            # 'get_hcp_bundle_masks',
            # 'get_subject_bundle_masks',
            # 'filter_by_hcp_endings',
            # 'run_bundle_seg_on_registered_atlas'
            # 'run_bundleseg',
            # 'run_bundleseg_on_SLF',
            # 'run_bundleseg_on_centroids',
            # 'run_bundleseg_autocalibration',
            # 'get_bundleseg_endings',
            # "project_metric_onto_bundleseg"
        ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'register_template_to_anat': lambda: register_template_to_anat(subject, pipeline, atlas, **kwargs),
        'apply_trans_to_HCP_bundles': lambda: apply_trans_to_HCP_bundles(subject, pipeline, atlas, **kwargs),
        'apply_trans_to_HCP_clusters': lambda: apply_trans_to_HCP_clusters(subject, pipeline, atlas, **kwargs),
        'move_fa_to_template_space': lambda: move_fa_to_template_space(subject, pipeline, atlas, **kwargs),
        'run_tractosearch_on_registered_atlas': lambda: run_tractosearch_on_registered_atlas(subject, atlas, pipeline, **kwargs),
        'get_hcp_endings': lambda: get_hcp_endings(subject, pipeline, **kwargs),
        'get_hcp_bundle_masks': lambda: get_hcp_bundle_masks(subject, pipeline, bundle_names='ALL', **kwargs),
        'get_subject_bundle_masks': lambda: get_subject_bundle_masks(subject, pipeline, bundle_names='ALL', **kwargs),
        'filter_by_hcp_endings': lambda: filter_by_hcp_endings(subject, pipeline, **kwargs),
        'run_bundleseg_on_centroids': lambda: run_bundleseg_on_centroids(subject, pipeline, atlas, **kwargs),
        'get_bundleseg_endings': lambda: get_bundleseg_endings(subject, pipeline, **kwargs),
        'project_metric_onto_bundleseg': lambda: project_metric_onto_bundleseg(subject, pipeline, metric_name='IFW', **kwargs),
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
        sub, dataset_path, pipeline, pipeline_list, atlas = arg
        print(f"Processing subject {sub} with pipeline {pipeline}")
        ds=Dataset(dataset_path)
        # subject = Subject(sub, db_root=dataset_path)
        return segment_subject_bundleseg(ds.get_subject(sub), atlas=atlas, pipeline=pipeline,pipeline_list=pipeline_list)
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

from pprint import pprint





def main():
    args=parse_args()
    atlas = load_atlas_config(args.atlas)
    dataset_path = args.db_root
    pipeline_name = 'bundle_seg'
    if args.step == 'segmentation':
        pipeline_list = ['register_template_to_anat','apply_trans_to_HCP_bundles','apply_trans_to_HCP_clusters','run_tractosearch_on_registered_atlas']
 
    process_single_subject((args.subject, dataset_path, pipeline_name, pipeline_list, atlas))


if __name__ == "__main__":
    main()