import sys
import argparse

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import os
import glob
import shutil
from subprocess import call


def set_config():
    # Anima config
    configFilePath = os.path.join(os.path.expanduser("~"), ".anima",
                                  "config.txt")
    if not os.path.exists(configFilePath):
        print(
            'Please create a configuration file for Anima python scripts. Refer to the README'
        )
        quit()

    configParser = ConfParser.RawConfigParser()
    configParser.read(configFilePath)

    animaDir = configParser.get("anima-scripts", 'anima')
    animaDataDir = configParser.get("anima-scripts", 'extra-data-root')
    animaScriptsDir = configParser.get("anima-scripts",
                                       'anima-scripts-public-root')
    animaPrivScriptsDir = configParser.get("anima-scripts",
                                           'anima-scripts-root')

    # TractSeg config
    configFilePath = os.path.join(os.path.expanduser("~"), ".tractseg-config",
                                  "config.txt")
    if not os.path.exists(configFilePath):
        print('Please create a configuration file for tractseg.')
        quit()

    configParser = ConfParser.RawConfigParser()
    configParser.read(configFilePath)

    tractsegDir = configParser.get("tractseg", 'tractseg-bin')
    mrtrixDir = configParser.get("tractseg", 'mrtrix-bin')

    config = {
        'animaDir': animaDir,
        'animaDataDir': animaDataDir,
        'animaScriptsDir': animaScriptsDir,
        'animaPrivScriptsDir': animaPrivScriptsDir,
        'tractsegDir': tractsegDir,
        'mrtrixDir': mrtrixDir
    }

    animaConvertImage = os.path.join(animaDir, "animaConvertImage")
    animaApplyTransformSerie = os.path.join(animaDir,
                                            "animaApplyTransformSerie")
    animaTensorApplyTransformSerie = os.path.join(
        animaDir, "animaTensorApplyTransformSerie")
    animaMCMApplyTransformSerie = os.path.join(animaDir,
                                               "animaMCMApplyTransformSerie")
    animaCropImage = os.path.join(animaDir, "animaCropImage")
    animaGMMT2RelaxometryEstimation = os.path.join(
        animaDir, "animaGMMT2RelaxometryEstimation")
    animaDTIScalarMaps = os.path.join(animaDir, "animaDTIScalarMaps")
    animaMCMScalarMaps = os.path.join(animaDir, "animaMCMScalarMaps")

    animaBrainExtraction = os.path.join(animaScriptsDir, "brain_extraction",
                                        "animaAtlasBasedBrainExtraction.py")
    myAnimaSubjectsMCMFiberPreparation = os.path.join(
        animaPrivScriptsDir, "diffusion", "mcm_fiber_atlas_comparison",
        "myAnimaSubjectsMCMFiberPreparation.py")
    animaRegister3DImageOnAtlas = os.path.join(
        animaScriptsDir, "registration", "animaRegister3DImageOnAtlas.py")
    antsRegister3DImageOnAtlas = os.path.join(animaPrivScriptsDir,
                                              "registration",
                                              "antsRegister3DImageOnAtlas.py")
    animaDiffusionImagePreprocessing = os.path.join(
        animaScriptsDir, "diffusion", "animaDiffusionImagePreprocessing.py")
    
    animaPreprocRenaud = "/home/ndecaux/Code/Renaud/myAnimaDiffusionImagePreprocessing.py"
    Tractometry = os.path.join(animaPrivScriptsDir, "tractometry_Julie.py")

    #Check if fast is in path
    fast = shutil.which("fast")
    if fast is not None:
        fast = ['fast']
    else:
        #Requires docker
        fast = ['fsl', 'fast']

    tools = {
        'animaConvertImage': animaConvertImage,
        'animaApplyTransformSerie': animaApplyTransformSerie,
        'animaTensorApplyTransformSerie': animaTensorApplyTransformSerie,
        'animaMCMApplyTransformSerie': animaMCMApplyTransformSerie,
        'animaCropImage': animaCropImage,
        'animaGMMT2RelaxometryEstimation': animaGMMT2RelaxometryEstimation,
        'animaDTIScalarMaps': animaDTIScalarMaps,
        'animaMCMScalarMaps': animaMCMScalarMaps,
        'animaBrainExtraction': animaBrainExtraction,
        'myAnimaSubjectsMCMFiberPreparation':
        myAnimaSubjectsMCMFiberPreparation,
        'animaRegister3DImageOnAtlas': animaRegister3DImageOnAtlas,
        'antsRegister3DImageOnAtlas': antsRegister3DImageOnAtlas,
        'Tractometry': Tractometry,
        'animaDiffusionImagePreprocessing': animaPreprocRenaud,
        'fast': fast
    }

    #Add all to PATH
    os.environ["PATH"] += os.pathsep + animaDir
    os.environ["PATH"] += os.pathsep + animaScriptsDir
    os.environ["PATH"] += os.pathsep + animaPrivScriptsDir  

    return config, tools

import os
import glob

def get_HCP_bundle_names(bundle_name=None,inverse=False):
    """
    Get the list of bundle names from the HCP dataset.
    """
    # # Define the directory containing the bundle files
    # bundle_dir = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas"

    # # Get a list of all .trk files in the directory
    # bundle_list = glob.glob(os.path.join(bundle_dir, "*.trk"))

    # # Extract bundle names from file paths
    # bundle_names = [f.split('summed_')[-1].split('.trk')[0] for f in bundle_list]

    # return bundle_names
    if bundle_name is None:
        if not inverse:
            return {
                "ORright": "OR_right",
                "ICPleft": "ICP_left",
                "ILFleft": "ILF_left",
                "ILFright": "ILF_right",
                "SCPleft": "SCP_left",
                "CSTright": "CST_right",
                "CSTleft": "CST_left",
                "CC1": "CC_1",
                "ORleft": "OR_left",
                "SCPright": "SCP_right",
                "CC3": "CC_3",
                "FPTleft": "FPT_left",
                "FPTright": "FPT_right",
                "IFOleft": "IFO_left",
                "POPTright": "POPT_right",
                "POPTleft": "POPT_left",
                "ATRleft": "ATR_left",
                "ATRright": "ATR_right",
                "IFOright": "IFO_right",
                "SLFIIIleft": "SLF_III_left",
                "STRleft": "STR_left",
                "CC7": "CC_7",
                "STRright": "STR_right",
                "CC6": "CC_6",
                "STFOleft": "ST_FO_left",
                "CC5": "CC_5",
                "SLFIleft": "SLF_I_left",
                "SLFIIIright": "SLF_III_right",
                "CC4": "CC_4",
                "SLFIright": "SLF_I_right",
                "STFOright": "ST_FO_right",
                "SLFIIright": "SLF_II_right",
                "MLFright": "MLF_right",
                "SLFIIleft": "SLF_II_left",
                "MCP": "MCP",
                "MLFleft": "MLF_left",
                "STOCCleft": "ST_OCC_left",
                "CGleft": "CG_left",
                "STPREMleft": "ST_PREM_left",
                "STOCCright": "ST_OCC_right",
                "CGright": "CG_right",
                "STPREMright": "ST_PREM_right",
                "CC2": "CC_2",
                "STPOSTCright": "ST_POSTC_right",
                "AFright": "AF_right",
                "STPOSTCleft": "ST_POSTC_left",
                "TOCCright": "T_OCC_right",
                "STPRECright": "ST_PREC_right",
                "STPRECleft": "ST_PREC_left",
                "TOCCleft": "T_OCC_left",
                "TPOSTCright": "T_POSTC_right",
                "AFleft": "AF_left",
                "TPOSTCleft": "T_POSTC_left",
                "STPARright": "ST_PAR_right",
                "TPREMleft": "T_PREM_left",
                "TPREMright": "T_PREM_right",
                "TPRECright": "T_PREC_right",
                "TPARright": "T_PAR_right",
                "STPREFright": "ST_PREF_right",
                "STPREFleft": "ST_PREF_left",
                "TPREFright": "T_PREF_right",
                "FXright": "FX_right",
                "FXleft": "FX_left",
                "CA": "CA",
                "ICPright": "ICP_right",
                "STPARleft": "ST_PAR_left",
                "TPRECleft": "T_PREC_left",
                "UFleft": "UF_left",
                "UFright": "UF_right",
                "TPARleft": "T_PAR_left",
                "TPREFleft": "T_PREF_left"
            }
        else:
            return get_HCP_bundle_names(inverse=False)
    else:
        if not inverse:
            mapping = get_HCP_bundle_names()
            return mapping.get(bundle_name,bundle_name)
        else:
            mapping = get_HCP_bundle_names()
            inv_mapping = {v: k for k, v in mapping.items()}
            return inv_mapping.get(bundle_name,bundle_name)

if __name__ == '__main__':
    set_config()
    print(get_HCP_bundle_names())
    print(get_HCP_bundle_names("OR_right",inverse=True))
    print(get_HCP_bundle_names("ORright",inverse=False))
    