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
from TractoPL.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template
from TractoPL.utils.registration import apply_transforms
import tempfile
import glob
import ants
import shutil


def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the Classifyber pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True


def classifyber_register_to_mni(subject,template_path="/home/ndecaux/Git/app-classifyber-segmentation/MNI152_T1_1.25mm_brain.nii.gz", pipeline='classifyber', atlas_name='MNI152', **kwargs):
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
    anat = subject.get_unique(metric='FA', pipeline='preprocessing', extension='nii.gz')
    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',desc='normalized',algo='ifod2',extension='tck')

    print(template_path)
    # Register the anatomical image to the template space
    res_dict = register_anat_subject_to_template(anat, template_path=template_path,tractogram=tracto,atlas_name=atlas_name, **kwargs)

    # Copy the registered anatomical image to the subject's pipeline directory
    copy_from_dict(subject, res_dict, pipeline=pipeline)
    
    return True

if __name__ == "__main__":
    # Example usage
    subject = Subject('03011')
    pipeline = 'classifyber'
    init_pipeline(subject, pipeline)
    # classifyber_register_to_mni(subject, pipeline=pipeline)
    classifyber_register_to_mni(subject, pipeline='DeepWMA',template_path="/home/ndecaux/Git/DeepWMA_v2/SegModels/100HCP-population-mean-T2.nii.gz",atlas_name='100HCP')