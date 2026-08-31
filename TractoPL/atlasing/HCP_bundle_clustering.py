import os
from pprint import pprint
from os.path import join as opj

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import xml.etree.ElementTree as ET
from dipy.io.streamline import load_trk, save_trk, load_tractogram
from dipy.tracking.streamline import transform_streamlines

from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import read_mapping
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import MinimumAverageDirectFlipMetric,AveragePointwiseEuclideanMetric
import multiprocessing
from functools import partial
import pathlib
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import time
from dipy.segment.clustering import QuickBundles


bundle_path = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/summed_CST_right.trk"