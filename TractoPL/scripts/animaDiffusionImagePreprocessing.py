#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaDiffusionImagePreprocessing.py ..." has to be run

import sys
import argparse
import uuid
import pydicom
import numpy as np
import struct
from TractoPL.configuration import load_tool_config

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import os
import shutil
from subprocess import call

config,tools=load_tool_config()
animaDir = config["animaDir"]
animaDataDir = config["animaDataDir"]
animaScriptsDir = config["animaScriptsDir"]

# Argument parsing
parser = argparse.ArgumentParser(
    description="Prepares DWI for model estimation: gradients reworking on Siemens based on dicoms, denoising, brain masking, distortion correction.")
parser.add_argument('-b', '--bval', type=str, required=True, help="DWI b-values file")
parser.add_argument('-g', '--grad', type=str, default="", help="DWI gradients file")
parser.add_argument('-r', '--reverse', type=str, default="", help="Reversed PED B0 image")
parser.add_argument('-d', '--direction', type=int, default=1, help="PED direction (0: x, 1: y, 2: z)")
parser.add_argument('-D', '--dicom', type=str, nargs='+', default="",
                    help="Dicom file to put dcm2nii bvec file to real coordinates")
parser.add_argument('--no-disto-correction', action='store_true', help="Do not perform distortion correction")
parser.add_argument('--no-denoising', action='store_true', help="Do not perform NL-Means denoising")
parser.add_argument('-t', '--t1', type=str, default="", help="T1 image for brain masking (B0 used if not provided)")
parser.add_argument('--no-brain-masking', action='store_true', help="Do not perform any brain masking")
parser.add_argument('--no-eddy-correction', action='store_true',
                    help="Do not perform Eddy current distortion correction")
parser.add_argument('--register-t1-on-dwi', action='store_true',
                    help="T1 registration on DWI is needed as they were not acquired in the same session")
parser.add_argument('-i', '--input', type=str, required=True, help='DWI file to process')

parser.add_argument('-K', '--keep-intermediate-folder', action='store_true',
                    help='Keep intermediate folder after script end')

args = parser.parse_args()

tmpFolder = os.path.join(os.path.dirname(args.input), 'diff_preproc_' + str(uuid.uuid1()))

if not os.path.isdir(tmpFolder):
    os.mkdir(tmpFolder)

animaEddyCurrentCorrection = os.path.join(animaDir,"animaEddyCurrentCorrection")
animaConvertImage = os.path.join(animaDir,"animaConvertImage")
animaCropImage = os.path.join(animaDir,"animaCropImage")
animaMaskImage = os.path.join(animaDir,"animaMaskImage")
animaMorphologicalOperations = os.path.join(animaDir,"animaMorphologicalOperations")
animaTransformSerieXmlGenerator = os.path.join(animaDir,"animaTransformSerieXmlGenerator")
animaApplyTransformSerie = os.path.join(animaDir,"animaApplyTransformSerie")
animaPyramidalBMRegistration = os.path.join(animaDir,"animaPyramidalBMRegistration")
animaDenseSVFBMRegistration = os.path.join(animaDir,"animaDenseSVFBMRegistration")
animaDistortionCorrection = os.path.join(animaDir,"animaDistortionCorrection")
animaBMDistortionCorrection = os.path.join(animaDir,"animaBMDistortionCorrection")
animaApplyDistortionCorrection = os.path.join(animaDir,"animaApplyDistortionCorrection")
animaNLMeansTemporal = os.path.join(animaDir,"animaNLMeansTemporal")
animaDTIEstimator = os.path.join(animaDir,"animaDTIEstimator")

pythonExecutable = sys.executable
animaBrainExtraction = os.path.join(animaScriptsDir,"brain_extraction","animaAtlasBasedBrainExtraction.py")

dwiImage = args.input
dwiImagePrefix = os.path.splitext(dwiImage)[0]
if os.path.splitext(dwiImage)[1] == '.gz':
    dwiImagePrefix = os.path.splitext(dwiImagePrefix)[0]

tmpDWIImagePrefix = os.path.join(tmpFolder, os.path.basename(dwiImagePrefix))

outputImage = dwiImage
outputBVec = args.grad

if not (args.dicom == "") and not (args.grad == ""):
    # adapted from http://neurohut.blogspot.fr/2015/11/how-to-extract-bval-bvec-from-dicom.html
    # The goal here is to ensure the bvec file extracted from dcm2nii is well put
    # back in real coordinates. This assumes dcm2nii worked for gradient extraction which is not always the case.
    # If not, use the dicom folder option In any case, it works only for Siemens scanners though as far as I know

    img_plane_position = np.zeros(3)
    for dicom_file in args.dicom:
        image = pydicom.dcmread(dicom_file)
        dicomBaseFormat = [0x0020, 0x0012] in image

        if dicomBaseFormat:
            acq_number = image[0x0020, 0x0012].value
            if acq_number == 1:
                img_plane_position = image[0x0020, 0x0037].value
        else:
            acq_number = image[0x5200, 0x9230].value[0][0x0021, 0x1101].value[0][0x0020, 0x0012].value
            if acq_number == 1:
                img_plane_position = image[0x5200, 0x9230].value[0][0x0020, 0x9116].value[0][0x0020, 0x0037].value

    V1 = np.array([float(img_plane_position[0]), float(img_plane_position[1]), float(img_plane_position[2])])
    V2 = np.array([float(img_plane_position[3]), float(img_plane_position[4]), float(img_plane_position[5])])
    V3 = np.cross(V1, V2)

    orMatrix = np.array([V1, V2, V3])

    bvecs = np.loadtxt(args.grad)
    bvecs_corrected = np.dot(orMatrix.transpose(), bvecs)

    np.savetxt(tmpDWIImagePrefix + "_real.bvec", bvecs_corrected, fmt="%.12f")
    outputBVec = tmpDWIImagePrefix + "_real.bvec"

elif not (args.dicom == "") and (args.grad == ""):
    bvecs_corrected = [np.zeros(3)] * len(args.dicom)
    for dicom_file in args.dicom:
        image = pydicom.dcmread(dicom_file)
        dicomBaseFormat = [0x0019, 0x100c] in image

        if dicomBaseFormat:
            bval = float(image[0x0019, 0x100c].value)
            directionality = image[0x0019, 0x100d].value
        else:
            bval = image[0x5200, 0x9230].value[0][0x0018, 0x9117].value[0][0x0018, 0x9087].value
            directionality = image[0x5200, 0x9230].value[0][0x0018, 0x9117].value[0][0x0018, 0x9075].value

        if directionality == 'NONE' or bval == 0:
            bvec = [0, 0, 0]
        else:
            vec = []

            if dicomBaseFormat:
                vecData = image[0x0019, 0x100e].value
            else:
                vecData = image[0x5200, 0x9230].value[0][0x0018,0x9117].value[0][0x0018,0x9076].value[0][0x0018,0x9089].value

            if type(vecData) == type(list()):
                vec = np.array(vecData)
            else:
                buff = struct.unpack('ddd', vecData)
                vec = np.array(buff)

            bvec = np.zeros(3)
            bvec = vec

        if dicomBaseFormat:
            acq_number = image[0x0020, 0x0012].value
        else:
            acq_number = image[0x5200, 0x9230].value[0][0x0021, 0x1101].value[0][0x0020, 0x0012].value

        bvecs_corrected[acq_number - 1] = bvec

    bvecs_corrected = np.array(bvecs_corrected)
    np.savetxt(tmpDWIImagePrefix + "_real.bvec", bvecs_corrected.transpose(), fmt="%.12f")
    outputBVec = tmpDWIImagePrefix + "_real.bvec"

if outputBVec == "":
    sys.exit("Gradient file needs to be provided (either through Dicom folder or through dcm2nii)")

# Distortion correction first
# Eddy current first
if args.no_eddy_correction is False:
    eddyCorrectionCommand = [animaEddyCurrentCorrection, "-i", dwiImage, "-I", outputBVec, "-o",
                             tmpDWIImagePrefix + "_eddy_corrected.nrrd",
                             "-O", tmpDWIImagePrefix + "_eddy_corrected.bvec", "-d", str(args.direction)]
    call(eddyCorrectionCommand)

    outputImage = tmpDWIImagePrefix + "_eddy_corrected.nrrd"
    outputBVec = tmpDWIImagePrefix + "_eddy_corrected.bvec"

# Extract brain from T1 image if present (used for further processing)
if (args.no_disto_correction is False or args.no_brain_masking is False) and not args.t1 == "":
    brainExtractionCommand = [pythonExecutable, animaBrainExtraction, "-i", args.t1]
    call(brainExtractionCommand)

# Then susceptibility distortion
if args.no_disto_correction is False:
    if not (args.reverse == ""):
        b0ExtractCommand = [animaCropImage, "-i", outputImage, "-t", "0", "-T", "0", "-o",
                            tmpDWIImagePrefix + "_B0.nrrd"]
        call(b0ExtractCommand)

        idTrsfName = os.path.join(animaDataDir, "id.txt")
        idTrsfXmlName = os.path.join(tmpFolder, "id.xml")
        idGenCommand = [animaTransformSerieXmlGenerator, "-i", idTrsfName, "-o", idTrsfXmlName]
        call(idGenCommand)

        resampleB0PACommand = [animaApplyTransformSerie, "-i", args.reverse, "-t", idTrsfXmlName, "-o",
                               tmpDWIImagePrefix + "_B0_Reverse.nrrd", "-g", tmpDWIImagePrefix + "_B0.nrrd"]
        call(resampleB0PACommand)

        initCorrectionCommand = [animaDistortionCorrection, "-s", "2", "-d", str(args.direction),
                                 "-f", tmpDWIImagePrefix + "_B0.nrrd", "-b", tmpDWIImagePrefix + "_B0_Reverse.nrrd",
                                 "-o", tmpDWIImagePrefix + "_init_correction_tr.nrrd"]
        call(initCorrectionCommand)
        bmCorrectionCommand = [animaBMDistortionCorrection, "-f", tmpDWIImagePrefix + "_B0.nrrd",
                               "-b", tmpDWIImagePrefix + "_B0_Reverse.nrrd", "-o",
                               tmpDWIImagePrefix + "_B0_corrected.nrrd", "-i",
                               tmpDWIImagePrefix + "_init_correction_tr.nrrd",
                               "--bs", "3", "-s", "10", "-d", str(args.direction), "-O",
                               tmpDWIImagePrefix + "_B0_correction_tr.nrrd"]
        call(bmCorrectionCommand)

        applyCorrectionCommand = [animaApplyDistortionCorrection, "-f", outputImage, "-t",
                                  tmpDWIImagePrefix + "_B0_correction_tr.nrrd", "-o",
                                  tmpDWIImagePrefix + "_corrected.nrrd"]
        call(applyCorrectionCommand)

        outputImage = tmpDWIImagePrefix + "_corrected.nrrd"
    elif not (args.t1 == ""):
        b0ExtractCommand = [animaCropImage, "-i", outputImage, "-t", "0", "-T", "0", "-o",
                            tmpDWIImagePrefix + "_B0.nrrd"]
        call(b0ExtractCommand)

        T1Prefix = os.path.splitext(args.t1)[0]
        if os.path.splitext(args.t1)[1] == '.gz':
            T1Prefix = os.path.splitext(T1Prefix)[0]

        tmpT1Prefix = os.path.join(tmpFolder, os.path.basename(T1Prefix))

        correctionCommand = [animaPyramidalBMRegistration, "-r", tmpDWIImagePrefix + "_B0.nrrd",
                             "-m", T1Prefix + "_masked.nrrd", "-o", tmpT1Prefix + "_rig.nrrd",
                             "-O", tmpT1Prefix + "_rig_tr.txt", "-p", "4", "-l", "1", "--sp", "2"]
        if args.register_t1_on_dwi is True:
            correctionCommand += ["-I", "1"]
        else:
            correctionCommand += ["-I", "0"]
        call(correctionCommand)

        command = [animaTransformSerieXmlGenerator, "-i", tmpT1Prefix + "_rig_tr.txt", "-o",
                   tmpT1Prefix + "_rig_tr.xml"]
        call(command)

        command = [animaApplyTransformSerie, "-i", T1Prefix + "_brainMask.nrrd", "-t",
                   tmpT1Prefix + "_rig_tr.xml", "-o", tmpDWIImagePrefix + "_roughMask.nrrd", "-g",
                   tmpDWIImagePrefix + "_B0.nrrd", "-n", "nearest"]
        call(command)

        morphoCommand = [animaMorphologicalOperations, "-i", tmpDWIImagePrefix + "_roughMask.nrrd", "-a", "dil",
                         "-r", "4", "-o", tmpDWIImagePrefix + "_roughMask_dil.nrrd"]
        call(morphoCommand)

        maskCommand = [animaMaskImage, "-i", tmpDWIImagePrefix + "_B0.nrrd", "-o",
                       tmpDWIImagePrefix + "_B0_rough_masked.nrrd", "-m", tmpDWIImagePrefix + "_roughMask_dil.nrrd"]
        call(maskCommand)

        correctionCommand = [animaDenseSVFBMRegistration, "-r", tmpT1Prefix + "_rig.nrrd",
                             "-m", tmpDWIImagePrefix + "_B0_rough_masked.nrrd", "-o",
                             tmpDWIImagePrefix + "_B0_corrected.nrrd", "-d", str(args.direction),
                             "-O", tmpDWIImagePrefix + "_B0_correction_tr.nrrd", "-t", "3", "--sym-reg", "2"]
        call(correctionCommand)

        applyCorrectionCommand = [animaApplyDistortionCorrection, "-f", outputImage, "-t",
                                  tmpDWIImagePrefix + "_B0_correction_tr.nrrd", "-o",
                                  tmpDWIImagePrefix + "_corrected.nrrd"]
        call(applyCorrectionCommand)

        outputImage = tmpDWIImagePrefix + "_corrected.nrrd"

# Then re-orient image to be axial first
dwiReorientCommand = [animaConvertImage, "-i", outputImage, "-o", tmpDWIImagePrefix + "_or.nrrd", "-R",
                      "AXIAL"]
call(dwiReorientCommand)
outputImage = tmpDWIImagePrefix + "_or.nrrd"

# Then perform denoising
if args.no_denoising is False:
    denoisingCommand = [animaNLMeansTemporal, "-i", outputImage, "-b", "0.5", "-n", "3", "-o",
                        tmpDWIImagePrefix + "_nlm.nrrd"]
    call(denoisingCommand)
    outputImage = tmpDWIImagePrefix + "_nlm.nrrd"

# Finally, brain mask image
if args.no_brain_masking is False:
    brainImage = args.t1

    b0ExtractCommand = [animaCropImage, "-i", outputImage, "-t", "0", "-T", "0", "-o",
                        tmpDWIImagePrefix + "_forBrainExtract.nrrd"]
    call(b0ExtractCommand)

    if brainImage == "":
        brainImage = tmpDWIImagePrefix + "_forBrainExtract.nrrd"
        brainExtractionCommand = [pythonExecutable, animaBrainExtraction, "-i", brainImage]
        call(brainExtractionCommand)

    if args.t1 == "":
        shutil.move(tmpDWIImagePrefix + "_forBrainExtract_brainMask.nrrd", dwiImagePrefix + "_brainMask.nrrd")
    else:
        T1Prefix = os.path.splitext(args.t1)[0]
        if os.path.splitext(args.t1)[1] == '.gz':
            T1Prefix = os.path.splitext(T1Prefix)[0]

        tmpT1Prefix = os.path.join(tmpFolder, os.path.basename(T1Prefix))

        t1RegistrationCommand = [animaPyramidalBMRegistration, "-r",
                                 tmpDWIImagePrefix + "_forBrainExtract.nrrd", "-m", T1Prefix + "_masked.nrrd", "-o",
                                 tmpT1Prefix + "_rig.nrrd", "-O", tmpT1Prefix + "_rig_tr.txt", "-p", "4", "-l", "1",
                                 "--sp", "2"]

        if args.register_t1_on_dwi is True:
            t1RegistrationCommand += ["-I", "1"]
        else:
            t1RegistrationCommand += ["-I", "0"]

        call(t1RegistrationCommand)

        command = [animaTransformSerieXmlGenerator, "-i", tmpT1Prefix + "_rig_tr.txt", "-o",
                   tmpT1Prefix + "_rig_tr.xml"]
        call(command)

        command = [animaApplyTransformSerie, "-i", T1Prefix + "_brainMask.nrrd", "-t",
                   tmpT1Prefix + "_rig_tr.xml", "-o", dwiImagePrefix + "_brainMask.nrrd", "-g",
                   tmpDWIImagePrefix + "_forBrainExtract.nrrd", "-n", "nearest"]
        call(command)

    brainExtractionCommand = [animaMaskImage, "-i", outputImage, "-m", dwiImagePrefix + "_brainMask.nrrd",
                              "-o", tmpDWIImagePrefix + "_masked.nrrd"]
    call(brainExtractionCommand)

    outputImage = tmpDWIImagePrefix + "_masked.nrrd"

shutil.copy(outputImage, dwiImagePrefix + "_preprocessed.nrrd")
shutil.copy(outputBVec, dwiImagePrefix + "_preprocessed.bvec")

# Estimate tensors if files were provided
dtiEstimationCommand = [animaDTIEstimator, "-i", outputImage, "-o", dwiImagePrefix + "_Tensors.nrrd",
                        "-O", dwiImagePrefix + "_Tensors_B0.nrrd", "-N", dwiImagePrefix + "_Tensors_NoiseVariance.nrrd",
                        "-g", outputBVec, "-b", args.bval]

if args.no_brain_masking is False:
    dtiEstimationCommand += ["-m", dwiImagePrefix + "_brainMask.nrrd"]

call(dtiEstimationCommand)

if not args.keep_intermediate_folder:
    shutil.rmtree(tmpFolder)
