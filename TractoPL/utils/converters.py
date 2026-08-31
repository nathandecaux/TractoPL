from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, move2nii, parse_filename, BIDSFile, copy2nii
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import dipy
from time import process_time


def call_dipy_converter(tracts, output_path=None,reference=None, **kwargs):
    """
    Call the dipy_convert_tractogram command with the specified arguments.
    
    Parameters
    ----------
    command_args : list
        List of command line arguments for dipy_convert_tractogram.
        
    Returns
    -------
    int
        Return code from the command execution.
    """
    # Construct the command
    command = ["dipy_convert_tractogram", tracts,'--force']

    if output_path is not None:
        command+= ["--out_tractogram", output_path]
    if reference is not None:
        command+= ["--reference", reference]
    
    add_kwargs_to_cli(command, **kwargs)

    # Execute the command
    return_code = call(command)
    
    return return_code


    