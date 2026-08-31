import os
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.tractseg_utils import tractseg, register_fa_on_MNI, move_peaks_to_mni
from TractoPL.utils.registration import apply_transforms
import tempfile
import glob
import ants
import shutil


def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MCM pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True

def register_fa(subject,pipeline='tractseg', **kwargs):
    """
    Register the FA image to MNI space using dipy.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    kwargs : dict
        Additional arguments to pass to the registration algorithm

    Returns
    -------
    dict
        Dictionary containing the output files and their paths
    """
    # Get the FA image from the subject
    fa = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')

    # Register the FA image to MNI space
    res_dict= register_fa_on_MNI(fa, **kwargs)
    copy_from_dict(subject=subject, file_dict=res_dict, pipeline=pipeline, **kwargs)
    return res_dict

def move_peaks(subject, pipeline='tractseg', **kwargs):
    """
    Move the peaks file to the TractSeg pipeline.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    kwargs : dict
        Additional arguments to pass to the registration algorithm

    Returns
    -------
    dict
        Dictionary containing the output files and their paths

    """
    # Get the peaks file from the subject
    peaks = subject.get_unique(suffix='peaks', pipeline='msmt_csd', extension='nii.gz', desc='preproc')
    affine_mat = subject.get_unique(suffix='xfm', pipeline='tractseg', extension='mat',to='MNI')
    moved_peaks = move_peaks_to_mni(peaks,affine_mat, **kwargs)
    entities = peaks.get_entities()
    entities = upt_dict(entities, pipeline=pipeline, space='MNI')
    print(subject.write_object(moved_peaks, **entities))

def process_tractseg(subject, pipeline, **kwargs):
    """
    Process all the TractSeg pipeline on the given subject.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """
    if isinstance(subject, str):
        subject = Subject(subject)

    # Get peaks
    peaks = subject.get_unique(suffix='peaks', pipeline=pipeline,space='MNI', extension='nii.gz',desc='normalized',**kwargs)

    tractseg(peaks, **kwargs)

def tractseg_pipeline(subject,pipeline='tractseg'):
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
        # 'init',
        # 'register_fa',
        'move_peaks',
        'process_tractseg'
    ]
    

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'register_fa': lambda: register_fa(subject, pipeline),
        'move_peaks': lambda: move_peaks(subject, pipeline),
        'process_tractseg': lambda: process_tractseg(subject, pipeline)
    }

    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()





    

from pprint import pprint
if __name__ == "__main__":
    config, tools = set_config()
    # subject = Subject('100206',db_root='/home/ndecaux/Data/HCP/')
    subject = Subject('03011')

    tractseg_pipeline(subject,pipeline='tractseg')