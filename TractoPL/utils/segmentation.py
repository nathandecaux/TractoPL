import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from os.path import join as opj
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, move2nii,copy2nii,copy_from_dict
from TractoPL.utils.tools import add_kwargs_to_cli, del_key, upt_dict, create_pipeline_description, get_exact_file
import tempfile
import glob
import shutil
from pprint import pprint

def fast_segmentation(t1,**kwargs):
    """
    Segment the T1 image using FAST from FSL

    Parameters
    ----------
    t1 : BIDSFile
        The T1 image to segment
    kwargs : dict
        Additional arguments to pass to the FAST script
    """
    # Set the config
    config, tools = set_config()
    # Set the path to the T1 image
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)

    src_entities=t1.get_entities()
    #Copy the t1 image to the temporary folder
    t1_tmp = copy2nii(t1.path,opj(tmp_folder,'t1.nii.gz'))
    
    # Run the segmentation
    command = tools['fast']
    
    command = add_kwargs_to_cli(command, **kwargs)

    command = command + [t1_tmp]
    call(command)
    src_entities=upt_dict(src_entities,desc=src_entities['suffix'])

    res_dict = {
            't1_pveseg.nii.gz': upt_dict(src_entities,suffix='propseg',label='tissues'),
            't1_seg.nii.gz': upt_dict(src_entities,suffix='dseg',label='tissues'),
            't1_pve_0.nii.gz': upt_dict(src_entities,suffix='propseg',label='CSF'),
            't1_pve_1.nii.gz': upt_dict(src_entities,suffix='propseg',label='GM'),
            't1_pve_2.nii.gz': upt_dict(src_entities,suffix='propseg',label='WM'),
            't1_seg_0.nii.gz': upt_dict(src_entities,suffix='mask',label='CSF'),
            't1_seg_1.nii.gz': upt_dict(src_entities,suffix='mask',label='GM'),
            't1_seg_2.nii.gz': upt_dict(src_entities,suffix='mask',label='WM')
            }
    
    #Add the tmp_folder to the keys
    res_dict = {opj(tmp_folder,k):v for k,v in res_dict.items()}    
    return res_dict

def segment_t1_on_B0(subject,pipeline):
    config, tools = set_config()

    entities = {'suffix':'T1w','label':'brain'}
    t1 = get_exact_file(subject.get(**entities))
    res_dict = fast_segmentation(t1,g=True)
    copy_from_dict(subject,res_dict,pipeline=pipeline)
    return res_dict

from pprint import pprint
if __name__ == "__main__":
    config,tools = set_config()
    subject = Subject('03011')
    
    # res_dict=segment_t1_on_B0(subject,'msmt_csd')
    