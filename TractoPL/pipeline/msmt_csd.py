#!/usr/bin/env python

import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject, Dataset
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.fod import get_tissue_responses, get_msmt_csd, get_peaks, normalize_fod, fod_to_fixels, get_peak_density, fixel_to_peaks
from TractoPL.utils.tractography import generate_ifod2_tracto,generate_trekker_tracto, generate_trekker_tracto_tck
import tempfile
import glob
import shutil
from time import sleep
import click

DEFAULT_STEPS = (
    'init',
    'response',
    'fod',
    'normalize',
    'fixels',
    'fixels2peaks',
    'fixel_density',
    'ifod2_tracto',
)


def validate_steps(steps):
    """Return requested pipeline steps after validating their names."""
    requested_steps = tuple(steps or DEFAULT_STEPS)
    supported_steps = set(DEFAULT_STEPS) | {'peaks', 'peak_density', 'trekker_tracto'}
    unknown_steps = sorted(set(requested_steps) - supported_steps)
    if unknown_steps:
        raise ValueError(f"Unsupported MSMT-CSD steps: {', '.join(unknown_steps)}")
    return requested_steps


def already_done(subject, pipeline, out_entities):
    """Check if all output entities already exist for the subject in the given pipeline"""
    if len(subject.get(**out_entities)) > 0:
        return True
    return False
    

def get_dwi_data(subject):
    """Helper function to get DWI data for a subject"""
    dwi = subject.get_unique(suffix='dwi', desc='preproc', pipeline='preprocessing', extension='nii.gz',model=None)
    bval = subject.get(extension='bval')[0]
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', space='B0')
    return dwi, bval, bvec, mask

def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MSMT-CSD pipeline"""
    create_pipeline_description(
        pipeline, 
        layout=subject.layout, 
        registrationMethod='antsRegistrationSyNQuick.sh', 
        registrationParams='default', 
        exactParams={'-d': '3'},
        **kwargs
    )
    return True

def process_response(subject, dwi_data, pipeline, **kwargs):
    """Calculate tissue response functions"""
    dwi, bval, bvec, mask = dwi_data
    force = kwargs.pop('force', False)
    if already_done(subject, pipeline, {'suffix':'response', 'label':'WM'}) and not force:
        print("Response functions already exist, skipping computation.")
        return
    res_dict = get_tissue_responses(dwi, bval, bvec, mask, inverse_bvec=True, **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_fod(subject, dwi_data, pipeline, **kwargs):
    """Run MSMT-CSD to calculate fiber orientation distributions"""
    force = kwargs.pop('force', False)
    if already_done(subject, pipeline, {'suffix':'fod', 'label':'WM'}) and not force:
        print("FODs already exist, skipping computation.")
        return
    dwi, bval, bvec, mask = dwi_data
    csf_response = subject.get_unique(label='CSF', suffix='response', pipeline=pipeline)
    gm_response = subject.get_unique(label='GM', suffix='response', pipeline=pipeline)
    wm_response = subject.get_unique(label='WM', suffix='response', pipeline=pipeline)

    res_dict = get_msmt_csd(
        dwi=dwi, 
        bval=bval, 
        bvec=bvec, 
        csf_response=csf_response, 
        gm_response=gm_response, 
        wm_response=wm_response, 
        mask=mask,
        inverse_bvec=True,
        **kwargs
    )
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_normalize(subject, pipeline, **kwargs):
    """Normalize FODs"""
    force = kwargs.pop('force', False)
    if already_done(subject, pipeline, {'suffix':'fod', 'label':'WM', 'desc':'normalized'}) and not force:
        print("Normalized FODs already exist, skipping computation.")
        return
    wm_fod = subject.get_unique(label='WM', suffix='fod', pipeline=pipeline, desc='preproc')
    gm_fod = subject.get_unique(label='GM', suffix='fod', pipeline=pipeline, desc='preproc')
    csf_fod = subject.get_unique(label='CSF', suffix='fod', pipeline=pipeline, desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain',  space='B0')
    res_dict = normalize_fod(wm_fod, gm_fod, csf_fod, mask, **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_fixels(subject, pipeline, **kwargs):
    """Perform fixel-based analysis"""
    force = kwargs.pop('force', False)
    if already_done(subject, pipeline, {'extension':'fixel', 'label':'WM'}) and not force:
        print("Fixels already exist, skipping computation.")
        return
    wm_fod = subject.get_unique(label='WM', model='fod', pipeline=pipeline, desc='preproc', suffix='fod')
    mask = subject.get_unique(suffix='mask', label='brain', space='B0')
    res_dict = fod_to_fixels(fod=wm_fod, mask=mask, max_peaks=CLIArg('maxnum', 3), **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline,remove_after_copy=False)

def process_peaks(subject, pipeline, **kwargs):
    """Perform peak extraction"""
    wm_fod = subject.get_unique(label='WM', model='fod', pipeline=pipeline, desc='preproc', suffix='fod')
    peaks_dict = get_peaks(wm_fod, **kwargs)
    copy_from_dict(subject, peaks_dict, pipeline=pipeline)

def process_peak_density(subject, pipeline, **kwargs):
    """Calculate peak density from peaks"""
    peaks = subject.get_unique(suffix='peaks', label='WM', desc='preproc', pipeline=pipeline)
    peak_density = get_peak_density(peaks, **kwargs)
    entities = peaks.get_entities()
    entities = upt_dict(entities, suffix='density', extension='nii.gz', pipeline=pipeline)
    subject.write_object(peak_density, **entities)

def process_fixels2peaks(subject, pipeline, **kwargs):
    """Convert fixels to peaks"""
    force = kwargs.pop('force', False)
    if already_done(subject, pipeline, {'suffix':'peaks', 'label':'WM', 'desc':'fixels2peaks'}) and not force:
        print("Fixel peaks already exist, skipping computation.")
        return
    fixels = subject.get_unique(extension='fixel', label='WM', pipeline=pipeline)
    peaks = fixel_to_peaks(fixels, **kwargs)
    print(peaks)
    copy_from_dict(subject, peaks, pipeline=pipeline)
    # entities = upt_dict(fixels.get_entities(), suffix='peaks', extension='nii.gz', pipeline=pipeline,desc='fixels2peaks')
    # subject.write_object(peaks, **entities)

def process_fixel_density(subject, pipeline, **kwargs):
    """Calculate density from fixel peaks"""
    force = kwargs.pop('force', False)
    if already_done(subject, pipeline, {'suffix':'density', 'label':'WM', 'desc':'fixel'}) and not force:
        print("Fixel density already exist, skipping computation.")
        return
    fixel_peaks = subject.get_unique(suffix='peaks', label='WM', pipeline=pipeline, desc='fixels2peaks')
    fixel_density = get_peak_density(fixel_peaks, **kwargs)
    entities = upt_dict(fixel_peaks.get_entities(), suffix='density', extension='nii.gz', pipeline=pipeline)
    subject.write_object(fixel_density, **entities)

def process_ifod2_tracto(subject, pipeline, **kwargs):
    """Run iFOD2 tractography"""
    force = kwargs.pop('force', False)
    if already_done(subject, pipeline, {'suffix':'tracto', 'algo':'ifod2'}) and not force:
        print("iFOD2 tractography already exist, skipping computation.")
        return
    odf = subject.get_unique(suffix='fod', label='WM', desc='preproc', pipeline=pipeline)
    seeds = subject.get_unique(suffix='mask', label='brain', space='B0')
    tracto = generate_ifod2_tracto(odf, seeds, **kwargs)
    copy_from_dict(subject, tracto, pipeline=pipeline,datatype='tracto',algo='ifod2',label='brain')

def process_trekker_tracto(subject, pipeline, **kwargs):
    """Run Trekker tractography"""
    odf = subject.get_unique(suffix='fod', label='WM', desc='preproc', pipeline=pipeline)
    seeds = subject.get_unique(suffix='mask', label='brain', space='B0')
    tracto = generate_trekker_tracto_tck(odf,seeds, **kwargs)
    copy_from_dict(subject, tracto, pipeline='trekker',datatype='tracto',algo='trekker')


def process_msmt_csd(
    subject,
    steps=None,
    pipeline='msmt_csd',
    tractography_algorithm='ifod2',
    n_streamlines=1000000,
    force=False,
    wait_seconds=0,
):
    """
    Process the MSMT-CSD pipeline on the given subject.
    
    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """

    if isinstance(subject, str):
        subject = Subject(subject)
    pipeline_list = validate_steps(steps)
    if tractography_algorithm not in {'ifod2', 'trekker'}:
        raise ValueError("tractography_algorithm must be 'ifod2' or 'trekker'")
    if n_streamlines <= 0:
        raise ValueError("n_streamlines must be greater than zero")

    if 'ifod2_tracto' in pipeline_list and tractography_algorithm == 'trekker':
        pipeline_list = tuple(
            'trekker_tracto' if step == 'ifod2_tracto' else step
            for step in pipeline_list
        )
    
    # Get DWI data that will be used across multiple steps
    dwi_data = get_dwi_data(subject) if {'response', 'fod'} & set(pipeline_list) else None
    
    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'response': lambda: process_response(subject, dwi_data, pipeline, force=force),
        'fod': lambda: process_fod(subject, dwi_data, pipeline, force=force),
        'normalize': lambda: process_normalize(subject, pipeline, force=force),
        'fixels': lambda: process_fixels(subject, pipeline, force=force, afd=True, peak=True, disp=True),
        'peaks': lambda: process_peaks(subject, pipeline),
        'peak_density': lambda: process_peak_density(subject, pipeline),
        'fixels2peaks': lambda: process_fixels2peaks(subject, pipeline, force=force),
        'fixel_density': lambda: process_fixel_density(subject, pipeline, force=force),
        'ifod2_tracto': lambda: process_ifod2_tracto(subject, pipeline, force=force, n_streams=CLIArg('-select', n_streamlines)),
        'trekker_tracto': lambda: process_trekker_tracto(subject, pipeline, n_seeds=n_streamlines)
    }
    
    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()
            if wait_seconds:
                sleep(wait_seconds)
            #Refresh the subject object to ensure it has the latest data
            subject = Subject(subject.sub_id, db_root=subject.db_root)
    
@click.command()
@click.option('--subject', required=True, help='Subject ID to process.')
@click.option('--db-root', required=True, help='Root directory of the BIDS database.')
@click.option('--pipeline', default='msmt_csd', show_default=True, help='Derivative pipeline name.')
@click.option('--step', 'steps', multiple=True, help='Step to run; repeat to select multiple steps.')
@click.option('--tractography-algorithm', type=click.Choice(['ifod2', 'trekker']), default='ifod2', show_default=True)
@click.option('--n-streamlines', type=click.IntRange(min=1), default=1000000, show_default=True)
@click.option('--force', is_flag=True, help='Recompute existing outputs.')
def process_subject(subject, db_root, pipeline, steps, tractography_algorithm, n_streamlines, force):
    """
    Process a single subject through the MSMT-CSD pipeline.
    
    Parameters
    ----------
    subject : str
        Subject ID to process
    db_root : str
        Root directory of the BIDS database
    """
    ds = Dataset(db_root)
    subject_obj = ds.get_subject(subject)
    process_msmt_csd(
        subject_obj,
        steps=steps,
        pipeline=pipeline,
        tractography_algorithm=tractography_algorithm,
        n_streamlines=n_streamlines,
        force=force,
    )

if __name__ == "__main__":
    process_subject()