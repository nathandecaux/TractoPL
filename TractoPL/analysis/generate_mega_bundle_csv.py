from TractoPL.data.loader import Dataset,Subject,BIDSFile
from TractoPL.data.vtk_loader import load_vtk_streamlines, save_vtk
from TractoPL.set_config import get_HCP_bundle_names
import numpy as np
import vtk
import pandas as pd
import os
import multiprocessing
import gc
dataset='actidep'
name_mapping = {
    "Fractional anisotropy": "FA",
    "Mean diffusivity": "MD",
    "Parallel diffusivity": "AD",
    "Perpendicular diffusivity": "RD",
    "Isotropic restricted water fraction": "IRF",
    "Free water fraction": "IFW"
}

mcm_pipeline='mcm_tensors_staniz'
association_pipeline='hcp_association_new_100pts_mcm_tensors_staniz_frechetlong2'

def process_bundle(bundle_name):
    output_path = f'/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/mega_bundle_csv/{mcm_pipeline}_{association_pipeline}_{bundle_name}_metrics.feather'

    if os.path.exists(output_path):
        print(f'Dataset {dataset}, Bundle {bundle_name} with pipeline mcm {mcm_pipeline} and association {association_pipeline} already done')
        return
    db_root=f'/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids'
    ds=Dataset(db_root,restore='/tmp/mega_bundle_csv_cache_'+dataset)
    
    vtk_metrics=ds.get_global(pipeline=mcm_pipeline,
        datatype='tracto',
        extension='vtk',
        bundle=bundle_name)
    
    
    vtk_association=ds.get_global(pipeline=association_pipeline,
        datatype='tracto',
        extension='vtk',
        desc='associations',
        bundle=bundle_name)
    
    print(f"Processing bundle: {bundle_name}")
    metric_file=vtk_metrics
    association_file=vtk_association
    print(f"  Found {len(metric_file)} metric files and {len(association_file)} association files.")
    bundle_metric_list=[]
    
    #Order files by subject id to ensure matching, get common subjects only
    metric_file.sort(key=lambda x: x.subject)
    association_file.sort(key=lambda x: x.subject)
    common_subjects=set([f.subject for f in metric_file]).intersection(set([f.subject for f in association_file]))
    metric_file=[f for f in metric_file if f.subject in common_subjects]
    association_file=[f for f in association_file if f.subject in common_subjects]
    print(f"  After filtering, {len(metric_file)} metric files and {len(association_file)} association files remain for common subjects.")
    #Create a mega csv for this bundle
    all_metrics_df=pd.DataFrame()
    for mf, af in zip(metric_file, association_file):
        subject_id=mf.subject
        subject_scalars = {}
        print(f"  Subject: {subject_id}")
        streamlines, scalars = load_vtk_streamlines(mf.path, reshape_scalars=True)
        _, assoc_scalars = load_vtk_streamlines(af.path, reshape_scalars=True)

        del scalars['colors']  #Remove colors scalar if present
        #Rename scalar keys with their acronyms for easier identification
        for key,scalar_arrays in scalars.items():
            #metric_name = acronym of key
            metric_name= name_mapping.get(key, key.replace(" ","_"))
            subject_scalars[metric_name]=np.concatenate([arr.flatten() for arr in scalar_arrays]).astype(np.float16)
        
        for key, array in assoc_scalars.items():
            subject_scalars[key]= np.concatenate([arr.flatten() for arr in array]).astype(np.float16)

        df=pd.DataFrame(subject_scalars)
        df['subject_id']=subject_id

        bundle_metric_list.append(df)
        del streamlines, scalars, assoc_scalars, subject_scalars, df

    for subject_scalars in bundle_metric_list:
        all_metrics_df=pd.concat([all_metrics_df,subject_scalars], ignore_index=True)
    del bundle_metric_list

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_metrics_df.to_feather(output_path)
    print(f"  Saved metrics Feather to: {output_path}")
    del all_metrics_df
    gc.collect()

if __name__ == '__main__':
    bundle_names = get_HCP_bundle_names()
    db_root=f'/home/ndecaux/NAS_EMPENN/share/projects/{dataset}/bids'
    ds=Dataset(db_root,restore='/tmp/mega_bundle_csv_cache_'+dataset)

    n_proc = 8 if os.uname().nodename == 'calcarine' else 1
    with multiprocessing.Pool(n_proc) as pool:
        pool.map(process_bundle, bundle_names)