import os 
from pprint import pprint
from os.path import join as opj
import tempfile
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import xml.etree.ElementTree as ET
from dipy.io.streamline import load_trk, save_trk, load_tractogram
from dipy.tracking.streamline import transform_streamlines

from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import read_mapping
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import MinimumAverageDirectFlipMetric,AveragePointwiseEuclideanMetric
import multiprocessing
from functools import partial
import pathlib
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import time
from dipy.segment.clustering import QuickBundles

sub_list = "/local/tdurante/atlas/groupe1/sujets.txt"
input_dir="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/"
bundle_list = "/local/tdurante/atlas/groupe1/bundles.txt"


with open(sub_list, 'r') as f:
    subjects = f.read().splitlines()

subjects = {k+1:v for k,v in enumerate(subjects)}

# Charger la liste des faisceaux
with open(bundle_list, 'r') as f:
    bundles = f.read().splitlines()

def get_average_anat(subjects, input_dir):
    """
    Compute the average anatomical image from a list of subjects using SimpleITK.
    """
    # Load the anatomical images
    #Get first subject
    sub_id, sub_name = list(subjects.items())[0]
    anat_path = opj(input_dir, str(sub_name), 'transformed_anat.nii.gz')    
    img= sitk.ReadImage(anat_path)

    for sub_id, sub_name in list(subjects.items())[1:]:
        anat_path = opj(input_dir, str(sub_name), 'transformed_anat.nii.gz')
        if os.path.exists(anat_path):
            img += sitk.ReadImage(anat_path)
    
    # Compute the average
    img /= len(subjects)
    # Save the average image
    average_anat_path = opj(input_dir,'Atlas', 'average_anat.nii.gz')
    pathlib.Path(opj(input_dir,'Atlas')).mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, average_anat_path)
    print(f"Average anatomical image saved at {average_anat_path}")
    return average_anat_path


def cluster_streamlines(streamlines,anat_path,skip=True):

    """
    Cluster streamlines using Dipy's clustering algorithm.
    """
    if skip:
        return streamlines
    # Perform clustering
    tempdir=tempfile.mkdtemp()
    cmd="tractosearch_clusters.py {} 3 {} --in_nii {} --output_format trk".format(streamlines,tempdir,anat_path)
    print("Running command: ", cmd)
    os.system(cmd)
    # Get the path of the clustered streamlines
    clustered_path = tempdir + "/tractosearch_components__centroids.trk"
    return clustered_path


def get_summed_bundle(subjects,input_dir,bundle_name):
    """
    Compute the average bundle from a list of subjects using Dipy.
    """

    #Create Atlas dir if not exist
    pathlib.Path(opj(input_dir,'Atlas')).mkdir(parents=True, exist_ok=True)

    if True:
 
        # Get anatomical reference from the first subject
        sub_id, sub_name = list(subjects.items())[0]
        # anat_path = nib.load(str(opj(input_dir, str(sub_name), 'transformed_anat.nii.gz')))
        anat_path= str(opj(input_dir, str(sub_name), 'transformed_anat.nii.gz'))
        flag=False
        # Load the rest of the subjects' bundles
        for sub_id, sub_name in list(subjects.items()):
            trk_path = str(opj(input_dir, str(sub_name), f'transformed_{bundle_name}.trk'))
            if not flag:
                if os.path.exists(trk_path):
                    # streamlines = cluster_streamlines(load_trk(trk_path,anat_path).streamlines)
                    streamlines= load_trk(cluster_streamlines(trk_path,anat_path),anat_path).streamlines
                    flag=True
            else:
                if os.path.exists(trk_path):
                    streamlines.extend(load_trk(cluster_streamlines(trk_path,anat_path),anat_path).streamlines)
        print(f"Total streamlines for bundle {bundle_name}: {len(streamlines)}")
        # Save the average bundle
        average_bundle_path = str(opj('/data/HCP105_Tracts','Atlas','full_summed', f'summed_{bundle_name}.trk'))
        average_bunde= StatefulTractogram(streamlines, anat_path, Space.RASMM)
        save_trk(average_bunde, average_bundle_path,bbox_valid_check=True)
        print(f"Average bundle saved at {average_bundle_path}")
        return True
    # except Exception as e:
    #     print(f"Error processing bundle {bundle_name}: {str(e)}")
    #     return False




def process_all_bundles_multiprocess(subjects, input_dir, bundles, n_processes=None):
    """
    Process all bundles in parallel using multiprocessing.
    
    Parameters:
    -----------
    subjects : dict
        Dictionary of subjects
    input_dir : str
        Input directory containing the subject data
    bundles : list
        List of bundle names to process
    n_processes : int, optional
        Number of processes to use. If None, uses all available CPUs.
    """
    if n_processes is None:
        n_processes = min(multiprocessing.cpu_count()//2, len(bundles))
    
    print(f"Processing {len(bundles)} bundles using {n_processes} processes...")

    filtered_bundles = []
    for bundle in bundles:
        if bundle != 'CC' and not os.path.exists(opj('/data/HCP105_Tracts','Atlas','full_summed', f'summed_{bundle}.trk')):
            filtered_bundles.append(bundle)
        else:
            print(f"Bundle {bundle} already exists, skipping...")

    # Create a partial function with fixed arguments
    process_bundle = partial(get_summed_bundle, subjects, input_dir)
    # filtered_bundles = ['AF_right']
    print(f"Filtered bundles to process: {filtered_bundles}")
    # Process bundles in parallel
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(process_bundle, filtered_bundles)
    
    # Count successful and failed bundles
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"Completed averaging {successful} bundles. Failed: {failed}.")
    return successful, failed

def process_all_bundles(subjects, input_dir, bundles, n_processes=None):
    """
    Process all bundles sequentially."
    """
    successful = 0
    failed = 0
    for bundle in bundles:
        print(f"Processing bundle: {bundle}")
        result = get_summed_bundle(subjects, input_dir, bundle)
        if result:
            successful += 1
         
    return successful, failed


def do_clustering(bundle_name='CC'):
    """
    Cluster streamlines using Dipy's clustering algorithm.
    """
    #Create the cluster dir if not exist
    try :
        atlas_name = 'Atlas'
        pathlib.Path(opj(input_dir,atlas_name,'clusters')).mkdir(parents=True, exist_ok=True)
        # Load the streamlines
        anat_path = str(opj(input_dir,'Atlas', 'average_anat.nii.gz'))
        trk_path = str(opj(input_dir,atlas_name, f'summed_{bundle_name}.trk'))
        trk= load_trk(trk_path, anat_path)
        streamlines = trk.streamlines
        # Resample streamlines to have only 2 points for simplified clustering
        # Perform clustering
        qb = QuickBundles(threshold=5.)
        clusters = qb.cluster(streamlines)
        # Save the clusters
        cluster_path = str(opj(input_dir,atlas_name,'clusters', f'clusters_{bundle_name}.trk'))
        trk.streamlines = clusters.centroids
        save_trk(trk, cluster_path, bbox_valid_check=True)
        print(f"Clusters saved at {cluster_path}")
        return True
    except Exception as e:
        print(f"Error processing clusters for bundle {bundle_name}: {str(e)}")
        return False

def process_all_clusters(bundles):
    """
    Process all clusters sequentially.
    """
    successful = 0
    failed = 0
    for bundle in bundles:
        if bundle != 'CC' and not os.path.exists(opj(input_dir,'Atlas', 'clusters', f'clusters_{bundle}.trk')):
            print(f"Processing clusters for bundle: {bundle}")
            result = do_clustering(bundle)
            if result:
                successful += 1
            else:
                failed += 1
        else:
            print(f"Clusters for bundle {bundle} already exist, skipping...")
            successful += 1
    print(f"Completed clustering {successful} bundles. Failed: {failed}.")
    return successful, failed

def process_all_clusters_multiprocess(bundles, n_processes=None):
    """
    Process all clusters in parallel using multiprocessing.
    
    Parameters:
    -----------
    bundles : list
        List of bundle names to process
    n_processes : int, optional
        Number of processes to use. If None, uses all available CPUs.
    """
    if n_processes is None:
        n_processes = min(multiprocessing.cpu_count(), len(bundles))
    
    print(f"Processing {len(bundles)} bundles using {n_processes} processes...")
    
    # Create a partial function with fixed arguments
    process_bundle = partial(cluster_streamlines)

    filtered_bundles = []
    for bundle in bundles:
        if bundle != 'CC' and not os.path.exists(opj(input_dir,'Atlas', 'clusters', f'clusters_{bundle}.trk')):
            filtered_bundles.append(bundle)
        else:
            print(f"Clusters for bundle {bundle} already exist, skipping...")

    
    # Process bundles in parallel
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(process_bundle, filtered_bundles)
    
    # Count successful and failed bundles
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"Completed clustering {successful} bundles. Failed: {failed}.")
    return successful, failed


def create_whole_brain(bundle_dir,ref_path,bundles=bundles):
    """
    Create a whole brain tractogram from the bundles.
    """
    # Load the streamlines
    streamlines = Streamlines()
    # Load the reference anatomical image
    print(f"Loading reference anatomical image: {ref_path}")
    ref_img = nib.load(ref_path)
    for bundle in bundles:
        if bundle != 'CC':
            print(f"Processing bundle: {bundle}")
            trk_path = [str(x) for x in pathlib.Path(bundle_dir).glob(f"**/*{bundle}.trk")][0]
            print(f"Loading {trk_path}")
            if os.path.exists(trk_path):
                trk = load_trk(trk_path, ref_img)
                streamlines.extend(trk.streamlines)

    
    
    # Save the whole brain tractogram
    whole_brain_path = str(opj(bundle_dir, 'whole_brain.trk'))
    whole_brain = StatefulTractogram(streamlines, ref_path, Space.RASMM)
    save_trk(whole_brain, whole_brain_path, bbox_valid_check=True)
    print(f"Whole brain tractogram saved at {whole_brain_path}")
    return True


if __name__ == "__main__":
    # Get the average anatomical image
    # average_anat_path = get_average_anat(subjects, input_dir)
    
    # # Process all bundles in parallel
    successful, failed = process_all_bundles_multiprocess(subjects, input_dir, bundles, n_processes=1)

    #Create high res whole brain tractogram
    # whole_brain_path = create_whole_brain(opj(input_dir,'Atlas'), str(opj(input_dir,'Atlas','average_anat.nii.gz')), bundles=bundles)

    #Create whole brain clusters
    # cluster_path = create_cluster(opj(input_dir,'Atlas','summed_CC_1.trk'), str(opj(input_dir,'Atlas','average_anat.nii.gz')))
    
    # print(f"\nSummary:")
    # # print(f"- Average anatomical image created: {os.path.exists(average_anat_path)}")
    # print(f"- Bundles successfully averaged: {successful}")
    # print(f"- Bundles that failed: {failed}")
    # print(f"- Total bundles processed: {len(bundles)}")
    # print(f"Results saved in: {opj(input_dir, 'Atlas')}")

    # Process all clusters
    # successful, failed = process_all_clusters(bundles)

    # process_all_clusters_multiprocess(bundles, n_processes=32)

    #Create whole brain tractogram from clusters
    # create_whole_brain(opj(input_dir,'Atlas','clusters'), str(opj(input_dir,'Atlas','average_anat.nii.gz')), bundles=bundles)