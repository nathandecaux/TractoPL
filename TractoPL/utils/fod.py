import os
import numpy as np
from TractoPL.set_config import set_config
from TractoPL.data.io import FixelFile
from TractoPL.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import nibabel as nib


def do_inverse_bvec(bvec_tmp):
    print(f"Inverting bvec file {bvec_tmp}")
    bvec = np.loadtxt(bvec_tmp)
    bvec[0, :] = -bvec[0, :]
    np.savetxt(bvec_tmp, bvec, fmt='%.18e')


# Implémentation des fonctions spécifiques de MRtrix3

def get_tissue_responses(dwi, bval, bvec, mask=None, inverse_bvec=True, **kwargs):
    """
    Calls the MRtrix3 dwi2response dhollander script to estimate the tissue responses from the DWI data.

    Parameters
    ----------
    dwi : BIDSFile
        The DWI data to estimate the responses from
    bval : BIDSFile
        The bval file associated with the DWI data
    bvec : BIDSFile
        The bvec file associated with the DWI data
    mask : BIDSFile (optional)
        Brain mask to use for the estimation
    inverse_bvec : bool (optional)
        Whether to invert the y component of the bvec file. Default is True.
    """
    inputs = {
        "dwi": dwi,
        "bval": bval,
        "bvec": bvec,
        "mask": mask
    }

    def prepare_inputs(tmp_inputs):
        if inverse_bvec and tmp_inputs["bvec"]:
            do_inverse_bvec(tmp_inputs["bvec"])

    command_args = [
        'dhollander',
        inputs["dwi"].path,
        'response_sfwm.txt',
        'response_gm.txt',
        'response_csf.txt',
        '-fslgrad',
        '$bvec',
        inputs["bval"].path
    ]

    if mask:
        command_args.extend(['-mask',
                            mask.path])

    output_pattern = {
        'response_sfwm.txt': {"suffix": "response", "extension": "txt", "label": "WM"},
        'response_gm.txt': {"suffix": "response", "extension": "txt", "label": "GM"},
        'response_csf.txt': {"suffix": "response", "extension": "txt", "label": "CSF"}
    }

    return run_mrtrix_command(
        'dwi2response',
        inputs,
        output_pattern,
        dwi.get_entities(),
        prepare_inputs_fn=prepare_inputs,
        command_args=command_args,
        **kwargs
    )


def get_msmt_csd(dwi, bval, bvec, csf_response, gm_response, wm_response, mask=None, inverse_bvec=True, **kwargs):
    """
    Calls the MRtrix3 dwi2fod msmt_csd script to estimate the fiber orientation distributions from the DWI data.

    Parameters
    ----------
    dwi : BIDSFile
        The DWI data to estimate the responses from
    bval : BIDSFile
        The bval file associated with the DWI data
    bvec : BIDSFile
        The bvec file associated with the DWI data
    csf_response : BIDSFile
        The CSF response file
    gm_response : BIDSFile
        The GM response file
    wm_response : BIDSFile
        The WM response file
    mask : BIDSFile (optional)
        Brain mask to use for the estimation
    inverse_bvec : bool (optional)
        Whether to invert the y component of the bvec file. Default is True.
    """
    inputs = {
        "dwi": dwi,
        "bval": bval,
        "bvec": bvec,
        "wm_response": wm_response,
        "gm_response": gm_response,
        "csf_response": csf_response,
        "mask": mask
    }

    def prepare_inputs(tmp_inputs):
        if inverse_bvec and tmp_inputs["bvec"]:
            do_inverse_bvec(tmp_inputs["bvec"])

    command_args = [
        'msmt_csd',
        inputs["dwi"].path,
        '-fslgrad',
        "$bvec",
        inputs["bval"].path
    ]

    if mask:
        command_args.extend(['-mask',
                             mask.path])
    for tissue, resp_file in [("csf", csf_response), ("gm", gm_response), ("wm", wm_response)]:
        command_args.extend([resp_file.path, f'{tissue}_fod.nii.gz'])

    output_pattern = {
        'csf_fod.nii.gz': {"model": "fod", "extension": "nii.gz", "label": "CSF", "suffix": "fod"},
        'gm_fod.nii.gz': {"model": "fod", "extension": "nii.gz", "label": "GM", "suffix": "fod"},
        'wm_fod.nii.gz': {"model": "fod", "extension": "nii.gz", "label": "WM", "suffix": "fod"}
    }

    return run_mrtrix_command(
        'dwi2fod',
        inputs,
        output_pattern,
        dwi.get_entities(),
        prepare_inputs_fn=prepare_inputs,
        command_args=command_args,
        **kwargs
    )


def normalize_fod(wmfod, gmfod, csffod, mask, **kwargs):
    """
    Run the MRtrix3 mtnormalise script to normalize the fod data.
    """
    inputs = {
        "wmfod": wmfod,
        "gmfod": gmfod,
        "csffod": csffod,
        "mask": mask
    }

    command_args = [
        inputs["wmfod"].path,
        'norm_wmfod.nii.gz',
        inputs["gmfod"].path,
        'norm_gmfod.nii.gz',
        inputs["csffod"].path,
        'norm_csffod.nii.gz',
        '-mask',
        inputs["mask"].path]

    output_pattern = {
        'norm_wmfod.nii.gz': {"suffix": "fod", "extension": "nii.gz", "desc": "normalized", "label": "WM"},
        'norm_gmfod.nii.gz': {"suffix": "fod", "extension": "nii.gz", "desc": "normalized", "label": "GM"},
        'norm_csffod.nii.gz': {"suffix": "fod", "extension": "nii.gz", "desc": "normalized", "label": "CSF"}
    }

    return run_mrtrix_command(
        'mtnormalise',
        inputs,
        output_pattern,
        wmfod.get_entities(),
        command_args=command_args,
        **kwargs
    )


def fod_to_fixels(fod, mask=None, **kwargs):
    """
    Calls the MRtrix3 fod2fixel script to estimate the fixels from the fod data.

    Parameters
    ----------
    fod : BIDSFile
        The fod data to estimate the fixels from
    mask : BIDSFile (optional)
        Brain mask to use for the estimation
    """
    inputs = {
        "fod": fod,
        "mask": mask
    }

    # afd = "afd.nii.gz"
    # peak_amp = "peaks_amp.nii.gz"
    # disp = "disp.nii.gz"

    command_args = [
        inputs["fod"].path,
        'fixels',
        # '-afd', afd,
        # '-peak', peak_amp,
        # '-disp', disp,
        # '-nii'
    ]

    for k in list(kwargs.keys()):
        if k in ['afd','peak','disp','skew']:
            kwargs[k]=k+'.mif'
            command_args.extend(['-'+k, kwargs[k]])
            inputs[k] = kwargs[k]
            del kwargs[k]
    #         command_args.extend(add_kwargs_to_cli(k, kwargs[k]))
    #         del kwargs[k]
    # if any(['afd','peak', 'disp'] in kwargs.keys()):
    #     kwargs['afd']='afd.mif'
    #     command_args.extend(['-afd', kwargs['afd']])
    #     inputs['afd'] = kwargs['afd']
    #     del kwargs['afd']

    if mask:
        command_args.extend(['-mask',
                             mask.path])

    base_entities = fod.get_entities()
    output_pattern = {
        'fixels': upt_dict(base_entities, {"suffix": "fixels", 'is_dir': True,'extension': 'fixel'}),
    }

    for k in ['afd','peak','disp','skew']:
        if k in inputs:
            output_pattern[f'{k}.nii.gz'] = upt_dict(base_entities, suffix='fixels', desc=k, extension='nii.gz', is_dir=False)
    
    res_dict = run_mrtrix_command(
        'fod2fixel',
        inputs,
        output_pattern,
        fod.get_entities(),
        command_args=command_args,
        **kwargs
    )

    fixel_path = str([k for k,v in res_dict.items() if os.path.basename(k) == 'fixels'][0])

    fixel_file = FixelFile(fixel_path).write(fixel_path+'.fixel',compress=True)

    final_dict={f'{fixel_path}.fixel': upt_dict(base_entities, suffix='fixels', extension='fixel', is_dir=False)}
    for k in ['afd','peak','disp','skew']:
        if f'{k}.nii.gz' in res_dict:
            final_dict[res_dict[f'{k}.nii.gz'].path]=res_dict[f'{k}.nii.gz']
    # json_sidecar = {
    #     'fixel_files': {
    #         # 'afd.nii.gz': upt_dict(base_entities, suffix='fixels', desc='afd'),
    #         # 'peaks_amp.nii.gz': upt_dict(base_entities, suffix='fixels', desc='peaks_amp'),
    #         # 'disp.nii.gz': upt_dict(base_entities, suffix='fixels', desc='disp'),
    #         'directions.nii': upt_dict(base_entities, suffix='fixels', desc='directions', extension='nii'),
    #         'index.nii': upt_dict(base_entities, suffix='fixels', desc='index', extension='nii')
    #     }
    # }

    # #Write the json sidecar in the fixel parent directory
    # json_path = opj(pathlib.Path(fixel_path).parent, 'fixels.json')
    # print('JSON PATH:', json_path)
    # with open(json_path, 'w') as f:
    #     json.dump(json_sidecar, f)

    # res_dict[json_path] = upt_dict(res_dict[fixel_path], extension='json',is_dir=False)

    return final_dict


def get_peaks(fod, mask=None, **kwargs):
    """
    Calls the MRtrix3 sh2peaks script to estimate the peaks from the fod data.

    Parameters
    ----------
    fod : BIDSFile
        The fod data to estimate the peaks from
    mask : BIDSFile (optional)
        Brain mask to use for the estimation
    """
    inputs = {
        "fod": fod,
        "mask": mask
    }

    command_args = [
        inputs["fod"].path]

    if mask:
        command_args.extend(['-mask',
                             mask.path])

    command_args.append('peaks.nii.gz')

    output_pattern = {
        'peaks.nii.gz': {"suffix": "peaks", "extension": "nii.gz"}
    }

    return run_mrtrix_command(
        'sh2peaks',
        inputs,
        output_pattern,
        fod.get_entities(),
        command_args=command_args,
        **kwargs
    )


def get_peak_density(peaks_file):
    """
    Calculate peak density from peaks data.

    Parameters
    ----------
    peaks_file : BIDSFile
        The peaks data to estimate the peak density from
    """
    peaks_img = nib.load(peaks_file.path)
    peaks_data = peaks_img.get_fdata()

    # Count the number of peaks in each voxel (each peak has 3 components x,y,z)
    density = np.sum(np.sqrt(np.sum(peaks_data.reshape(
        *peaks_data.shape[:-1], -1, 3) ** 2, axis=-1)) > 0, axis=-1)

    # Create a new nifti image with the peak density
    density_img = nib.Nifti1Image(density.astype('uint8'), peaks_img.affine)

    return density_img


def fixel_to_peaks(fixels, **kwargs):
    """
    Calls the MRtrix3 fixel2sh script to estimate the peaks from the fixel data.

    Parameters
    ----------
    fixel_dir : BIDSFile
        The fixel data to estimate the peaks from
    """
    fixels_obj = FixelFile(fixels.path)

    fixels_obj = fixels_obj

    inputs = {
        "fixels": fixels
    }

    command_args = [
        fixels_obj.dir_path,
        'peaks.nii.gz'
    ]

    output_pattern = {
        'peaks.nii.gz': dict(suffix='peaks', extension='nii.gz',desc='fixels2peaks')
    }

    return run_mrtrix_command(
        'fixel2peaks',
        inputs,
        output_pattern,
        fixels.get_entities(),
        command_args=command_args,
        **kwargs
    )


def project_gfa_on_tracts(fixels, tracts, **kwargs):
    """
    Calls the MRtrix3 fixel2voxel script to project the GFA on tracts.

    Parameters
    ----------
    fixel_dir : BIDSFile
        The fixel data to estimate the peaks from
    tracts : BIDSFile
        The tracts to project the GFA on
    """
    fixels_obj = FixelFile(fixels.path)

    fixels_obj = fixels_obj

    inputs = {
        "fixels": fixels,
        "tracts": tracts
    }

    command_args = [
        fixels_obj.dir_path,
        inputs["tracts"].path,
        'gfa_on_tracts.nii.gz'
    ]

    output_pattern = {
        'gfa_on_tracts.nii.gz': dict(suffix='gfa', extension='nii.gz',desc='on_tracts')
    }

    return run_mrtrix_command(
        'fixel2voxel',
        inputs,
        output_pattern,
        fixels.get_entities(),
        command_args=command_args,
        **kwargs
    )

if __name__ == "__main__":
    config, tools = set_config()
    # dwi = subject.get_unique(suffix='dwi', desc='preproc',
    #                          pipeline='preprocessing', extension='nii.gz')
    # bval = subject.get_unique(extension='bval')
    # bvec = subject.get_unique(extension='bvec', desc='preproc')
    # mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi')

    # pprint(get_tissue_responses(dwi, bval, bvec, mask, 'test'))

    # fod = subject.get_unique(
    #     model='fod', pipeline='msmt_csd', label='WM', suffix='dwi')
    # print(fod)

    # peaks = subject.get_unique(suffix='peaks', label='WM', desc='preproc', pipeline='msmt_csd', extension='nii.gz')
    # print(peaks)
    # threshold = 0.5
    # peaks_filtered = filter_peaks(peaks,threshold=threshold)
    # entities = peaks.get_entities()
    # entities = upt_dict(entities, suffix='peaks', extension='nii.gz', desc='filtered')

    # subject.write_object(peaks_filtered, **entities)

    # #Add a fake file to the fixels
    # fixels.add_file('/home/ndecaux/Data/actidep_bids/derivatives/msmt_csd/sub-03011/dwi/sub-03011_desc-preproc_label-WM_response.txt')

    

