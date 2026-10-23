import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import check_call
from TractoPL.set_config import set_config
from TractoPL.utils.tools import upt_dict
from TractoPL.data.loader import Subject
from TractoPL.data.io import copy_from_dict
from TractoPL.utils.mat import save_flatten_bval
import tempfile
import glob
import shutil
import click


pipeline = 'preprocessing'
PREPROCESSING_STEPS = ['anima', 'dti', 'fa']


def _as_subject(subject, db_root=None):
    return subject if isinstance(subject, Subject) else Subject(subject, db_root)

def flip_bvec_y(bvec_path, output_path,dim2flip=1):
    """
    Flip bvec in y direction by multiplying y component by -1
    """
    bvec_data = np.loadtxt(bvec_path)
    if bvec_data.shape[0] == 3:  # Standard format: 3 rows (x,y,z), N columns
        bvec_data[dim2flip, :] *= -1  # Flip y component
    elif bvec_data.shape[1] == 3:  # Alternative format: N rows, 3 columns (x,y,z)
        bvec_data[:, dim2flip] *= -1  # Flip y component
    np.savetxt(output_path, bvec_data, fmt='%.6f')

def _flatten_bval(bval_path, output_path):
    """
    Flatten a bval file to a single row.
    """
    save_flatten_bval(bval_path, output_path)

def animaPreprocessing(
    subject,
    with_reversed_b0=True,
    db_root=None,
    flip_bvecs=True,
    temp_dir=None,
):
    """
    Calls the Anima diffusion preprocessing script on the given subject.
    """
    
    temp_dir = temp_dir or tempfile.mkdtemp(prefix="tractopl-preprocessing-")
    os.makedirs(temp_dir, exist_ok=True)
    subject = _as_subject(subject, db_root)

    # Preprocess diffusion data
    print("Preprocess diffusion data")
    config,tools = set_config()

    dwi = subject.get_unique(suffix='dwi',extension='nii.gz',scope='raw',desc=None)

    t1 = subject.get(suffix='T1w',scope='raw',extension='nii.gz')[0]
    b0_reversed= subject.get_unique(desc='b0reversed',scope='raw',extension='nii.gz')
    dicom_folder = subject.dicom_folder
    
    # Copy files to temp directory and flip bvec
    temp_dwi = os.path.join(temp_dir, os.path.basename(dwi.path))

    temp_t1 = os.path.join(temp_dir, os.path.basename(t1.path))
    
    shutil.copy2(dwi.path, temp_dwi)
    shutil.copy2(t1.path, temp_t1)


    bval = subject.get(suffix='dwi',scope='raw',extension='bval')[0]
    temp_bval = os.path.join(temp_dir, os.path.basename(bval.path))

    save_flatten_bval(bval.path, temp_bval)
    # shutil.copy2(bval.path, temp_bval)


    bvec = subject.get(suffix='dwi',scope='raw',extension='bvec')[0]
    temp_bvec = os.path.join(temp_dir, os.path.basename(bvec.path))

    if flip_bvecs:
        if isinstance(flip_bvecs,int):
            flip_bvec_y(bvec.path, temp_bvec, dim2flip=flip_bvecs)
        else:
            flip_bvec_y(bvec.path, temp_bvec)
    else:
        shutil.copy2(bvec.path, temp_bvec)
    
    if with_reversed_b0:
        temp_b0_reversed = os.path.join(temp_dir, os.path.basename(b0_reversed.path))
        shutil.copy2(b0_reversed.path, temp_b0_reversed)
    
    #Get directory and prefix of dwi file
    dwiPrefix = os.path.basename(dwi.path).split('.')[0] 
    
    #Get files with full path
    dicom_files = [f for f in glob.glob(os.path.join(dicom_folder, "**", "*"), recursive=True) if os.path.isfile(f)]
    print(dicom_files[:10])
    preprocCommand = [
        "python3",
        tools['animaDiffusionImagePreprocessing'],
        "-t",
        temp_t1,
        "-i",
        temp_dwi,
    ]


    if flip_bvecs:
        preprocCommand = preprocCommand + ["-g", temp_bvec]

    preprocCommand +=  ["-D"] + dicom_files

    preprocCommand = preprocCommand + ["-b", temp_bval]
    # preprocCommand = preprocCommand + ["--no-disto-correction"]
    # preprocCommand = preprocCommand + ["--no-eddy-correction","--no-denoising"]
    if with_reversed_b0:
        preprocCommand = preprocCommand + ["-r", temp_b0_reversed]

    print(preprocCommand)
    check_call(preprocCommand, cwd=temp_dir)
    
    print("Preprocess data finished")
    entities = {'suffix':'dwi','pipeline':pipeline}
    # pipeline = 'preprocessing'
    
    tensors = os.path.join(temp_dir, dwiPrefix + "_Tensors.nrrd")
    preprocessed_dwi = os.path.join(temp_dir, dwiPrefix + "_preprocessed.nrrd")
    preprocessed_bvec = os.path.join(temp_dir, dwiPrefix + "_preprocessed.bvec")
    brain_mask = os.path.join(temp_dir, dwiPrefix + "_brainMask.nrrd")


    entities=dwi.get_full_entities()

    res_dict= {
        tensors: upt_dict(entities, {'datatype': 'dwi', 'desc': 'tensors', 'model': 'DTI'}),
        preprocessed_dwi: upt_dict(entities, {'datatype': 'dwi', 'desc': 'preproc'}),
        preprocessed_bvec: upt_dict(entities, {'datatype': 'dwi', 'desc': 'preproc', 'extension': 'bvec'}),
        brain_mask: upt_dict(entities, {'label': 'brain', 'suffix': 'mask','space':'B0','datatype': 'anat', 'pipeline': pipeline})
    }
    

    
    print(res_dict)

    copy_from_dict(subject,res_dict,pipeline=pipeline,remove_after_copy=False)
    # move2nii(tensors, tensors_target)
    # move2nii(preprocessed_dwi, prepocessed_dwi_target)
    # shutil.move(preprocessed_bvec, preprocessed_bvec_target)
    # move2nii(brain_mask, brain_mask_target)
    
    # os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_B0.nrrd"))
    # os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_NoiseVariance.nrrd"))

    # Clean up temp directory
    # shutil.rmtree(temp_dir)


def compute_dti(subject, db_root=None,flip_bvec=True,flatten_bval=True):
    """
    Computes DTI tensors from the preprocessed diffusion data.
    """
    config,tools = set_config()
    
    subject = _as_subject(subject, db_root)
    
    preproc_dwi = subject.get_unique(suffix='dwi',pipeline=pipeline,desc='preproc',extension='nii.gz',model=None)
    preproc_bvec= subject.get_unique(suffix='dwi',pipeline=pipeline,desc='preproc',extension='bvec')

    bval= subject.get_unique(suffix='dwi',scope='raw',extension='bval')
    #create temporary directory
    temp_dir = tempfile.mkdtemp()
    #dtiEstimationCommand = [animaDTIEstimator, "-i", outputImage, "-o", dwiImagePrefix + "_Tensors.nrrd",
                        # "-O", dwiImagePrefix + "_Tensors_B0.nrrd", "-N", dwiImagePrefix + "_Tensors_NoiseVariance.nrrd",
                        # "-g", outputBVec, "-b", args.bval]

    if flip_bvec:
        
        if isinstance(flip_bvec,int):
            print(f'Flipping bvec file along the axis {flip_bvec}.')
            flip_bvec_y(preproc_bvec.path,temp_dir+'/preproc_bvec.bvec', dim2flip=flip_bvec)  # Assuming you have a function to flip the bvec file
        else:
            print('Flipping bvec file along the y-axis.')
            flip_bvec_y(preproc_bvec.path,temp_dir+'/preproc_bvec.bvec')
        bvec_path=temp_dir+'/preproc_bvec.bvec'
    else:
        bvec_path=preproc_bvec.path

    if flatten_bval:
        _flatten_bval(bval.path, temp_dir+'/preproc_bval.bval')
        bval_path=temp_dir+'/preproc_bval.bval'
    else:
        bval_path=bval.path
    dti_command= f'animaDTIEstimator -i {preproc_dwi.path} -o {temp_dir}/dti_tensors.nii.gz -g {bvec_path} -b {bval_path}'
    
    check_call(dti_command, shell=True)

    res_dict = {
        f'{temp_dir}/dti_tensors.nii.gz': upt_dict(preproc_dwi.get_full_entities(), model='DTI')
    }
    print(preproc_dwi.get_full_entities())
    copy_from_dict(subject, res_dict, pipeline=pipeline)
    
def compute_fa(subject, db_root=None):
    """
    Computes FA from the preprocessed diffusion data.
    """
    config,tools = set_config()
    
    subject = _as_subject(subject, db_root)
    
    tensors = subject.get_unique(suffix='dwi',pipeline=pipeline,model='DTI',metric=None)
    print(tensors.path)
    #create temporary directory
    temp_dir = tempfile.mkdtemp()

    fa_command = f'{tools["animaDTIScalarMaps"]} -i {tensors.path} -f {temp_dir}/fa.nii.gz'
    
    check_call(fa_command, shell=True)

    res_dict = {
        f'{temp_dir}/fa.nii.gz': upt_dict(tensors.get_full_entities(), metric='FA')
    }
    print(tensors.get_full_entities())
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_all_FA(db_root=None,subjects=None):
    """
    Process all subjects in the database to compute FA.
    """
    config,tools = set_config()
    
    from TractoPL.data.loader import Dataset
    ds = Dataset(db_root)
    
    subjects = ds.get_subjects() if subjects is None else subjects
    
    for subject in subjects:
        try:
            print(f"Processing subject {subject} for FA computation.")
            already = ds.get(subject, suffix='dwi', pipeline=pipeline, model='DTI',metric='FA')
            if len(already) > 0:
                print(f"Subject {subject} already has FA computed. Skipping.")
                continue
            compute_fa(subject, db_root=db_root)
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")

@click.command()
@click.option('--subject', required=True, help='Subject ID to preprocess.')
@click.option('--db-root', required=True, help='Root directory of the BIDS database.')
@click.option('--with-reversed-b0/--without-reversed-b0', default=True, show_default=True, help='Use reversed B0 images for distortion correction.')
@click.option('--flip-bvec', default=False, show_default=False, help='Flip the y component of the b-vectors. Either false, or an integer axis (0, 1, 2).')
@click.option('--step', 'steps', multiple=True, type=click.Choice(PREPROCESSING_STEPS),
              help='Step to run; repeat to select multiple steps. Runs every step when omitted.')
def cli(subject, db_root, with_reversed_b0, flip_bvec, steps):
    """
    Run the preprocessing pipeline: animaPreprocessing, compute_dti, and compute_fa.
    """
    steps = steps or PREPROCESSING_STEPS
    try:
        print(f"Starting preprocessing for subject {subject} ({', '.join(steps)}).")
        if 'anima' in steps:
            animaPreprocessing(subject, with_reversed_b0=with_reversed_b0, db_root=db_root)
        if 'dti' in steps:
            compute_dti(subject, db_root=db_root, flip_bvec=flip_bvec)
        if 'fa' in steps:
            compute_fa(subject, db_root=db_root)
        print(f"Finished preprocessing for subject {subject}.")
    except Exception as e:
        print(f"Error during preprocessing for subject {subject}: {e}", file=sys.stderr)
        sys.exit(1)


prepocessing = cli

if __name__ == '__main__':
    config,tools = set_config()
    cli()
