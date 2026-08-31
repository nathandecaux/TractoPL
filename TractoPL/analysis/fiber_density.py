import os
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
from TractoPL.data.loader import Subject, Dataset, BIDSFile
import tempfile
from subprocess import call

def average_with_ants(dataset, output_dir=None):
    """
    Find all subject files with desc-fixels2peak and suffix=density,
    transform them to common space and create template using antsMultivariateTemplateConstruction2.sh
    """

    # Define output directory
    if output_dir is None:
        output_dir = '/local/ndecaux/Data/actidep_atlas'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file prefix
    output_file = Path(output_dir) / 'fiber_density_'

    # Get files
    files = dataset.get_global(pipeline='preprocessing', metric='FA', extension='nii.gz')

    # Create a line separated string with all the files paths    
    file_list = '\n'.join([str(file.path) for file in files])
    print(file_list)
    
    # Create file list directly in the output directory
    file_list_path = Path(output_dir) / 'file_list.txt'
    with open(file_list_path, 'w') as f:
        f.write(file_list)
    
    # Run antsMultivariateTemplateConstruction2.sh
    cmd = ['antsMultivariateTemplateConstruction2.sh',
           '-d', '3',
           '-o', str(output_file),
           '-c', '2',
           '-n', '0',
           '-j', '32',
           '-t', 'Affine',
            str(file_list_path)]
    
    print(f"Running command: {' '.join(cmd)}")
    call(cmd)

    print(f"Template created at: {output_file}")

    

# Example usage
if __name__ == "__main__":
    dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
    ds = Dataset(dataset_path)
    average_with_ants(ds)