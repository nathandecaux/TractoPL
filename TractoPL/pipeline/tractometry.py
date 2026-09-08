import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, Dataset
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg,first_match
from TractoPL.utils.clustering import associate_subject_to_centroids, get_stat_from_association_file,associate_subject_to_parcellation
from TractoPL.set_config import get_HCP_bundle_names
from dipy.tracking.streamline import set_number_of_points
import tempfile
import glob
import shutil
from time import sleep
import time
import multiprocessing
import argparse

# Répertoire contenant les centroids HCP
HCP_CENTROIDS_DIR = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/flipped/centroids_longcentral"
HCP_FULL_BUNDLE_DIR = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/vtk/"
HCP_REFERENCE = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz"
HCP_PARC="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/radtract_parcellations"
# pipeline='hcp_association_multiclusters_umapendpoints'
pipeline='tractometry'

CLUSTERING='cortex'

def parse_args(step=None):
    parser = argparse.ArgumentParser(description="Tractometry pipeline")
    if step is None:
        parser.add_argument('step', choices=['association'], help='Processing step')
    parser.add_argument('--model-ref', '-r', required=True, help='Path to the model reference image.')
    parser.add_argument('--model-bundles', '-b', required=True, help='Path to the model bundles.')
    parser.add_argument('--model-centroids', '-c', required=True, help='Path to the model centroids.')
    parser.add_argument('--model-parcellations', '-p', help='Path to the model parcellations.')
    parser.add_argument('--subject', '-s', help='Subject ID. Processes all subjects when omitted.')
    parser.add_argument('--db-root', '-d', required=True, help='Path to the BIDS database root.')
    parser.add_argument('--pipeline', default='tractometry', help='Output pipeline name.')
    parser.add_argument('--n-pts', type=int, default=100, help='Number of points used for association.')
    parser.add_argument('--n-proc', type=int, help='Number of subjects processed in parallel.')
    return parser.parse_args()
# def get_bundle_mapping():
#     """
#     Créer un mapping entre les noms de bundles et les fichiers centroids correspondants.
    
#     Returns
#     -------
#     bundle_mapping : dict
#         Dictionnaire associant les noms de bundles aux chemins des centroids
#     """
#     bundle_mapping = {}
    
#     # Scanner les fichiers centroids disponibles
#     centroid_files = glob.glob(os.path.join(HCP_CENTROIDS_DIR, "*_centroids.vtk"))
    
#     for centroid_file in centroid_files:
#         filename = os.path.basename(centroid_file)
#         # Extraire le nom du bundle (ex: summed_CST_left_centroids.vtk -> CST_left)
#         bundle_name = filename.replace('summed_', '').replace('_centroids.vtk', '')
        
#         actidep_name = [k for k,v in get_HCP_bundle_names().items() if v==bundle_name]
        
#         bundle_mapping[actidep_name] = centroid_file
    
#     print(f"Bundle mapping créé : {bundle_mapping}")
#     return bundle_mapping

def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the HCP association pipeline"""
    if not isinstance(subject, Subject):
        subject = Subject(subject)
    print(subject)
    create_pipeline_description(
        pipeline, 
        layout=subject.layout,  # Utiliser subject.layout comme prévu
        registrationMethod='SLR', 
        registrationParams='streamline_linear_registration', 
        exactParams={'model': 'HCP_centroids'},
        **kwargs
    )
    return True

def process_bundle_associations(subject, pipeline, n_pts=50, **kwargs):
    """
    Traiter toutes les associations de bundles pour un sujet.
    
    Parameters
    ----------
    subject : Subject
        Objet Subject à traiter
    pipeline : str
        Nom du pipeline
    """
    # Obtenir le mapping des bundles
    bundle_mapping = get_HCP_bundle_names()
    
    # Obtenir tous les fichiers VTK de la pipeline mcm_to_hcp_space
    vtk_files = subject.get(
        pipeline='mcm_to_hcp_space',
        space='HCP', 
        extension='vtk',
        datatype='tracto'
    )
    #Sort les fichiers par nom de bundle
    vtk_files.sort(key=lambda x: x.get_entities().get('bundle', ''))
    


    print(f"Trouvé {len(vtk_files)} fichiers VTK pour le sujet {subject.sub_id}")
    # vtk_files = [vtk_file for vtk_file in vtk_files if vtk_file.get_entities().get('bundle', '') == 'CC1']
    for vtk_file in vtk_files:
        # Extraire le nom du bundle depuis les entités
        entities = vtk_file.get_entities()
        bundle_name = entities.get('bundle', '')
        
        if bundle_name in bundle_mapping:
            hcp_bundle_name = bundle_mapping[bundle_name]
            centroid_path = os.path.join(HCP_CENTROIDS_DIR, f'summed_{hcp_bundle_name}_centroids.vtk')
            
            model_bundle_path = os.path.join(HCP_FULL_BUNDLE_DIR, f'summed_{hcp_bundle_name}.vtk')
            print(f"Traitement du bundle {bundle_name} avec centroids {centroid_path}")
            
            try:
                # Appliquer l'association
                start_time = time.time()
                res_dict = associate_subject_to_centroids(
                    subject_bundle=vtk_file,
                    model_centroids_path=centroid_path,
                    model_full_bundle_path=model_bundle_path,
                    reference_nifti=HCP_REFERENCE,
                    n_pts=n_pts,
                    **kwargs
                )
                elapsed_time = time.time() - start_time
                print(f"associate_subject_to_centroids executed in {elapsed_time:.2f} seconds")
                
                # Copier les résultats dans le layout BIDS
                copy_from_dict(subject, res_dict, pipeline=pipeline)
                
                print(f"Association terminée pour {bundle_name}")
                
            except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
                import traceback
                print(f"Erreur lors du traitement de {bundle_name}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
        else:
            print(f"Pas de centroids trouvés pour le bundle {bundle_name}")

def bundle_association_multiclusters(subject, pipeline, n_pts='2mm', clustering_method='frechet',mcm_pipeline='mcm_tensors_staniz', model_clustering=5.0, **kwargs):
    """
    Traiter toutes les associations de bundles pour un sujet.
    
    Parameters
    ----------
    subject : Subject
        Objet Subject à traiter
    pipeline : str
        Nom du pipeline
    """
    # Obtenir le mapping des bundles
    bundle_mapping = get_HCP_bundle_names()

    vtk_files = subject.get(
        pipeline=mcm_pipeline,
        extension='vtk',
        datatype='tracto'
    )
    #Sort les fichiers par nom de bundle
    vtk_files.sort(key=lambda x: x.get_entities().get('bundle', ''))
    
    if len(vtk_files) > 71:
        print(f"Subject {subject.sub_id} has more than 71 VTK files in {mcm_pipeline}, skipping association to avoid duplicates.")
        return
    
    ref_anat = subject.get_unique(
        pipeline='preprocessing',
        metric='FA',
        extension='nii.gz'
    )

    pipeline="tractometry"#pipeline+'_'+mcm_pipeline+'_'+clustering_method

    print(f"Trouvé {len(vtk_files)} fichiers VTK pour le sujet {subject.sub_id}")
    # vtk_files = [vtk_file for vtk_file in vtk_files if vtk_file.get_entities().get('bundle', '') == 'CC1']
    for vtk_file in vtk_files:
        # Extraire le nom du bundle depuis les entités
        entities = vtk_file.get_entities()
        bundle_name = entities.get('bundle', '')
        
        if bundle_name in bundle_mapping:
            hcp_bundle_name = bundle_mapping[bundle_name]
            centroid = subject.get_unique(
                pipeline='bundle_seg',clustering=clustering_method,
                bundle=bundle_name)
            centroid_path = centroid.path
            model_bundle = subject.get_unique(
                pipeline='bundle_seg',suffix='tracto',datatype='atlas',
                bundle=bundle_name)
            model_bundle_path = model_bundle.path
            print(f"Traitement du bundle {bundle_name} avec centroids {centroid_path}, full bundle {model_bundle_path}")
            
            if len(subject.get(pipeline=pipeline, bundle=bundle_name, datatype='metric', suffix='mean'))>0:
                print(f'Bundle {bundle_name} already exists in {pipeline}, skipping')
                continue
            try:
                # Appliquer l'association
                start_time = time.time()
                res_dict = associate_subject_to_centroids(
                    subject_bundle=vtk_file,
                    model_centroids_path=centroid_path,
                    model_full_bundle_path=model_bundle_path,
                    reference_nifti=ref_anat.path,
                    n_pts=n_pts,
                    slr=False,
                    model_clustering=model_clustering,
                    **kwargs
                )
                elapsed_time = time.time() - start_time
                print(f"associate_subject_to_centroids executed in {elapsed_time:.2f} seconds")
                
                # Copier les résultats dans le layout BIDS
                copy_from_dict(subject, res_dict, pipeline=pipeline,clustering=clustering_method)
                
                print(f"Association terminée pour {bundle_name}")
                
            except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
                import traceback
                print(f"Erreur lors du traitement de {bundle_name}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
        else:
            print(f"Pas de centroids trouvés pour le bundle {bundle_name}")

def bundle_association_parcellation(subject, pipeline, parcellations_dir=HCP_PARC,mcm_pipeline='mcm_tensors_staniz', **kwargs):
    bundle_dict=get_HCP_bundle_names()
    bundle_parc_dict={first_match(f, bundle_dict.values()).replace('_',''):f for f in os.listdir(parcellations_dir) if f.endswith('.nii.gz')}
    print("Bundle parcellation dict:", bundle_parc_dict)
    vtk_files = subject.get(pipeline=mcm_pipeline,
        extension='vtk',
        datatype='tracto',
        desc='cleaned'
    )

    ref_img= subject.get_unique(
        pipeline='preprocessing',
        metric='FA',
        extension='nii.gz'
    )

    affine_trans=subject.get_unique(
        to='subject',
        suffix='xfm',
        desc='0GenericAffine',
    )

    nonrigid_trans=subject.get_unique(
        to='subject',
        suffix='xfm',
        desc='1Warp'
    )
    transform_list=[affine_trans.path,nonrigid_trans.path]


    for vtk_file in vtk_files:
        bundle=vtk_file.get_entities().get('bundle','')
        if bundle!='CSTleft':
            continue
        if bundle not in bundle_parc_dict:
            print(f"No parcellation found for bundle {bundle}, skipping")
            continue
        parcellation_path=os.path.join(parcellations_dir,bundle_parc_dict[bundle])
        print(f"Using parcellation {parcellation_path} for bundle {bundle}")
        try:
            res_dict=associate_subject_to_parcellation(
                subject_bundle=vtk_file,
                parcellation=parcellation_path,
                subject_reference_nifti=ref_img.path,
                transform_list=transform_list,
                **kwargs
            )
            copy_from_dict(subject, res_dict, pipeline=pipeline+'_'+mcm_pipeline)
            print(f"Association parcellation terminée pour le bundle {bundle}")
            
        except Exception as e:
            import traceback
            print(f"Erreur lors du traitement de {bundle}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            continue



# subject=Subject('01002',db_root='/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
# bundle_association_parcellation(subject, pipeline='hcp_association_parcellation',mcm_pipeline='mcm_tensors_staniz')



def combine_csv(subject, pipeline):
    """
    Combine les fichiers CSV de métriques pour un sujet donné dans le format utilisé par Tractseg : un csv par métrique, contenant les moyennes pour chaque bundle (une colonne par bundle).
    """
    if not isinstance(subject, Subject):
        subject = Subject(subject)
    # Obtenir tous les fichiers CSV de métriques
    csv_files = subject.get(
        pipeline=pipeline,
        extension='csv',
        suffix='mean',
        datatype='metric'
    )
    if not csv_files:
        print(f"Aucun fichier CSV trouvé pour le sujet {subject.sub_id} dans le pipeline {pipeline}")
        return
    print(f"Trouvé {len(csv_files)} fichiers CSV pour le sujet {subject.sub_id} dans le pipeline {pipeline}")

    temp_dir = tempfile.mkdtemp()

    csv_dfs = [(csv_file.get_entities().get('bundle', ''), pd.read_csv(csv_file.path, delimiter=',')) for csv_file in csv_files]
    # Créer un dictionnaire pour stocker les DataFrames
    df_dict = {}
    entities= csv_files[0].get_entities()

    for metric in ['AD','RD','MD','FA','IFW','IRF']:
        df = pd.DataFrame()
        for bundle_name, bundle_df in csv_dfs:
            if metric not in bundle_df.columns:
                continue
            df[bundle_name] = bundle_df[metric]
        
        #Save the DataFrame to a CSV file in a temporary directory
        output_path = os.path.join(temp_dir, f"{metric}_metrics.csv")
        #set nan to 0.0
        # df.fillna(0.0, inplace=True)
        df.to_csv(output_path, index=False,sep=';')

        entities_metric = entities.copy()

        entities_metric['metric'] = metric
        del entities_metric['bundle']
        df_dict[output_path] = entities_metric
        print(f"Fichiers CSV combinés enregistrés dans {output_path}")

    # Copier les fichiers CSV combinés dans le layout BIDS
    copy_from_dict(
        subject=subject,
        file_dict=df_dict,
        pipeline=pipeline,
        suffix='tractsegmean',
        datatype='metric'
    )



def get_central_line_displacement(subject, pipeline=pipeline):
    """
    Compute the MDF between the subject's bundles after HCP association and the HCP centroids.
    Parameters
    ----------
    subject : Subject
        Subject object to process
    
    pipeline : str
        Pipeline name
    Returns
    -------
    CSV file path
        Path to the CSV file containing the MDF values for each bundle
    """

    if not isinstance(subject, Subject):
        subject = Subject(subject)
    
    # Obtenir le mapping des bundles
    bundle_mapping = get_HCP_bundle_names()
    
    # Obtenir tous les fichiers VTK de la pipeline hcp_association
    vtk_files = subject.get(
        pipeline=pipeline,
        desc='centroids',
        extension='vtk',
        datatype='tracto'
    )
    vtk_files.sort(key=lambda x: x.get_entities().get('bundle', ''))
    
    mdf_values = {}
    
    for vtk_file in vtk_files:
        entities = vtk_file.get_entities()
        bundle_name = entities.get('bundle', '')
        
        if bundle_name in bundle_mapping:
            hcp_bundle_name = bundle_mapping[bundle_name]
            centroid_path = os.path.join(HCP_CENTROIDS_DIR, f'summed_{hcp_bundle_name}_centroids.vtk')
            
            try:
                from dipy.tracking.distances import bundles_distances_mdf
                from dipy.io.streamline import load_vtk_streamlines
                
                # Charger les streamlines
                subject_streamlines = load_vtk_streamlines(vtk_file.path)
                centroid_streamlines = load_vtk_streamlines(centroid_path)
                
                subject_streamlines = set_number_of_points(subject_streamlines, 24)
                centroid_streamlines = set_number_of_points(centroid_streamlines, 24)
                if len(subject_streamlines) == 0 or len(centroid_streamlines) == 0:
                    print(f"Streamlines vides pour le bundle {bundle_name}, skipping MDF calculation.")
                    continue
                
                # Calculer le MDF
                mdf = bundles_distances_mdf(subject_streamlines, centroid_streamlines)
                mean_mdf = np.mean(mdf)
                
                mdf_values[bundle_name] = mean_mdf
                print(f"MDF for {bundle_name}: {mean_mdf}")
                
            except Exception as e:
                print(f"Erreur lors du calcul du MDF pour {bundle_name}: {e}")
                continue
        else:
            print(f"Pas de centroids trouvés pour le bundle {bundle_name}")
    
    # Sauvegarder les valeurs MDF dans un fichier CSV
    if mdf_values:
        entities = {'pipeline': pipeline, 'desc': 'hcpdisp','datatype':'stats', 'extension': 'csv',"suffix":'mdf'}
        output_path = subject.build_path(**entities)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(list(mdf_values.items()), columns=['bundle', 'MDF'])
        df.to_csv(output_path, index=False, sep=';')
        print(f"MDF values saved to {output_path}")
        return output_path

def compute_stats_vtk(subject,pipeline):
    """
    Compute stats on the VTK files generated by the HCP association pipeline.
    
    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    """
    if not isinstance(subject, Subject):
        subject = Subject(subject)
    
    vtk_files = subject.get(
        pipeline=pipeline,
        desc='associations',
        extension='vtk',
        datatype='tracto'
    )

    print(f"Found {len(vtk_files)} association VTK files for subject {subject.sub_id}")

    # scalar_files = subject.get(
    #     pipeline='mcm_tensors_staniz',
    #     extension='vtk',
    #     datatype='tracto'
    # )
    
    for vtk_file in vtk_files:
        entities = vtk_file.get_entities()
        bundle_name = entities.get('bundle', '')
        print(f"Computing stats for subject {subject.sub_id}, bundle {bundle_name}")

        if 'multiclusters' in pipeline or 'singlecluster' in pipeline:
            scalar_file = subject.get_unique(
                pipeline='mcm_tensors_staniz',
                bundle=bundle_name,
                extension='vtk',
                datatype='tracto'
            )
        else:
            scalar_file = subject.get_unique(
                pipeline='mcm_to_hcp_space',
                bundle=bundle_name,
                extension='vtk',
                datatype='tracto'
            )

        stats= get_stat_from_association_file(
            vtk_file,
            scalar_file)
        
        entities=vtk_file.get_entities()
        copy_from_dict(subject,file_dict=stats,pipeline=pipeline)



def process_hcp_association(subject,pipeline=pipeline,n_pts=50):
    """
    Process the HCP association pipeline on the given subject.
    
    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """

    if isinstance(subject, str):
        subject = Subject(subject)
    if not 'pts' in pipeline and not 'mm' in pipeline:
        pipeline=pipeline+f'_{n_pts}pts'
    elif 'mm' in pipeline:
        n_pts=pipeline.split('_')[-1]
    else:
        n_pts=[int(x.replace('pts','')) for x in pipeline.split('_') if 'pts' in x][0]
    # Define processing steps
    pipeline_list = [
        # 'init',
        # 'bundle_associations',
        'bundle_association_multiclusters',
        # "bundle_association_parcellation",
        # 'compute_stats_vtk',
        # 'combine_csv',
        # 'central_line_displacement'
    ]
    
    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'bundle_associations': lambda: process_bundle_associations(subject, pipeline,n_pts=n_pts),
        'bundle_association_multiclusters': lambda: bundle_association_multiclusters(subject, pipeline,n_pts=n_pts,clustering_method=CLUSTERING,mcm_pipeline='mcm_tensors_staniz',model_clustering=5.0),
        'bundle_association_parcellation': lambda: bundle_association_parcellation(subject, pipeline='hcp_association_parcellation',mcm_pipeline='mcm_tensors_staniz_with_AIC'),
        'compute_stats_vtk': lambda: compute_stats_vtk(subject, pipeline),
        'combine_csv': lambda: combine_csv(subject, pipeline='hcp_association_100pts_mcm_tensors_staniz'),
        'central_line_displacement': lambda: get_central_line_displacement(subject, pipeline)
    }
    
    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()
            # Refresh the subject object to ensure it has the latest data
            subject = Subject(subject.sub_id, db_root=subject.db_root)


def process_one_subject(sub):
    try:
        subject = ds.get_subject(sub)
        print(f"Processing subject: {sub}")
        process_hcp_association(subject, pipeline=args.pipeline, n_pts=args.n_pts)
    except Exception as e:
        print(f"Erreur lors du traitement du sujet {sub}: {e}")
        import traceback
        traceback.print_exc()
        return

def association():
    args = parse_args('association')
    HCP_REFERENCE = args.model_ref
    HCP_FULL_BUNDLE_DIR = args.model_bundles
    HCP_CENTROIDS_DIR = args.model_centroids
    if args.model_parcellations:
        HCP_PARC = args.model_parcellations

if __name__ == "__main__":
    args = parse_args()

    HCP_REFERENCE = args.model_ref
    HCP_FULL_BUNDLE_DIR = args.model_bundles
    HCP_CENTROIDS_DIR = args.model_centroids
    if args.model_parcellations:
        HCP_PARC = args.model_parcellations

    # If hostname is calcarine, set tempdir to /local/ndecaux/tmp.
    n_proc = args.n_proc or 8
    if os.uname()[1] == 'calcarine':
        tempfile.tempdir = '/local/ndecaux/tmp'
        n_proc = args.n_proc or 20

    config, tools = set_config()
    print("HCP Association pipeline : ", args.pipeline)
    print("=====================================")
    print('Reading dataset')

    ds = Dataset(args.db_root)
    print(f"Found {len(ds.subject_ids)} subjects")
    print("=====================================")
    
    subject_ids = [args.subject] if args.subject else ds.subject_ids
    if args.step == 'association':
        # Use multiprocessing to process subjects in parallel.
        with multiprocessing.Pool(n_proc) as pool:
            pool.map(process_one_subject, subject_ids)
    print("HCP Association pipeline completed")

