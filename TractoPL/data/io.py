# import bids
from os.path import join as opj
import os
import SimpleITK as sitk
import shutil
import pathlib
import os
import glob
import zipfile
import shutil
import tempfile
import json
from copy import deepcopy

class FixelFile:
    """
    Une classe pour gérer les fichiers au format Fixel de MRtrix.
    
    Le format Fixel est une structure de répertoire contenant:
    - directions.mif: orientations des fixels
    - index.mif: index des fixels dans chaque voxel
    - values.mif: mesures scalaires correspondant aux fixels
    - et d'autres fichiers optionnels
    
    Cette classe permet de manipuler ces fichiers soit directement depuis un répertoire,
    soit depuis une archive .fixel (zip contenant la structure de répertoire).
    """
    def __init__(self, path):
        """
        Initialise l'objet FixelFile à partir d'un chemin vers un répertoire ou un fichier .fixel.
        
        Parameters
        ----------
        path : str
            Chemin vers un répertoire fixel ou un fichier .fixel (archive zip)
        """
        # Stocker le chemin original
        self.original_path = path
        
        # Déterminer si c'est un fichier .fixel ou un répertoire
        self.is_archive = path.endswith('.fixel') and os.path.isfile(path)
        
        # Créer un répertoire temporaire si c'est une archive
        if self.is_archive:
            self.temp_dir = tempfile.mkdtemp()
            self._extract_archive(path, self.temp_dir)
            self.dir_path = self.temp_dir
        else:
            self.dir_path = path
            self.temp_dir = None
        
        # Charger les fichiers du répertoire fixel
        self._load_files()
        
        # Charger les métadonnées si disponibles
        self.metadata = {}
        metadata_file = opj(self.dir_path, 'fixel.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
    
    def _extract_archive(self, archive_path, extract_dir):
        """
        Extrait une archive .fixel dans un répertoire.
        
        Parameters
        ----------
        archive_path : str
            Chemin vers le fichier .fixel (archive zip)
        extract_dir : str
            Répertoire où extraire les fichiers
        """
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    def _load_files(self):
        """Charge les fichiers du répertoire fixel comme attributs de l'objet."""
        self.files = {}
        
        # Parcourir les fichiers du répertoire
        for filepath in glob.glob(opj(self.dir_path, '*')):
            filename = os.path.basename(filepath)
            name = os.path.splitext(filename)[0]  # Nom sans extension
            self.files[name] = filepath
            
            # Ajouter comme attribut pour un accès facile
            # Par exemple: fixel.directions, fixel.index, fixel.values
            setattr(self, name, filepath)
    
    def write(self, output_path=None, compress=True):
        """
        Écrit le FixelFile sur le disque, soit comme répertoire, soit comme archive .fixel.
        
        Parameters
        ----------
        output_path : str, optional
            Chemin de sortie. Si non fourni, utilise le chemin original avec .fixel ajouté.
        compress : bool, optional
            Si True, crée une archive .fixel (zip). Sinon, crée un répertoire.
        
        Returns
        -------
        str
            Chemin vers le fichier ou répertoire créé
        """
        # Déterminer le chemin de sortie
        if output_path is None:
            if not self.original_path.endswith('.fixel'):
                output_path = self.original_path + '.fixel'
            else:
                output_path = self.original_path
        
        # Écrire les métadonnées si disponibles
        if self.metadata and not os.path.exists(opj(self.dir_path, 'fixel.json')):
            with open(opj(self.dir_path, 'fixel.json'), 'w') as f:
                json.dump(self.metadata, f, indent=2)
        
        # Créer une archive .fixel ou copier le répertoire
        if compress:
            # S'assurer que le chemin se termine par .fixel
            if not output_path.endswith('.fixel'):
                output_path += '.fixel'
            
            # Créer l'archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.dir_path):
                    for file in files:
                        file_path = opj(root, file)
                        arcname = os.path.relpath(file_path, self.dir_path)
                        zipf.write(file_path, arcname)
        else:
            # Copier le répertoire
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            shutil.copytree(self.dir_path, output_path)
        
        return output_path
    
    def update_metadata(self, **kwargs):
        """
        Met à jour les métadonnées du fixel.
        
        Parameters
        ----------
        **kwargs
            Paires clé-valeur à ajouter aux métadonnées
        """
        self.metadata.update(kwargs)
    
    def add_file(self, filepath, name=None):
        """
        Ajoute un fichier à la structure fixel.
        
        Parameters
        ----------
        filepath : str
            Chemin vers le fichier à ajouter
        name : str, optional
            Nom à donner au fichier. Si non fourni, utilise le nom du fichier.
        """
        if name is None:
            name = os.path.basename(filepath)
        
        dest_path = opj(self.dir_path, name)
        shutil.copy2(filepath, dest_path)
        
        # Mettre à jour les attributs
        self._load_files()
    
    def __del__(self):
        """Nettoie les ressources temporaires à la destruction de l'objet."""
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __str__(self):
        """Représentation en chaîne de l'objet."""
        files_str = "\n  ".join([f"{name}: {os.path.basename(path)}" for name, path in self.files.items()])
        return f"FixelFile at {self.dir_path}:\n  {files_str}"



def parse_filename(filename):
    """
    Parse the filename to extract the entities
    """
    entities = {}
    parts = filename.split('_')
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)
            entities[key] = value
    return entities


def convertNRRDToNifti(nrrd_path, nifti_path):
    """Convertit un fichier NRRD en NIfTI."""
    # Lire l'image NRRD
    itk_image = sitk.ReadImage(nrrd_path)

    # Écrire l'image NIfTI
    sitk.WriteImage(itk_image, nifti_path)

def copy2nii(source, dest):
    """Copie un fichier, en convertissant en nifti si nécessaire."""
    print(f"Copying {source} to {dest}")
    pathlib.Path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)

    if source.endswith(".nrrd") and dest.endswith(".nii.gz"):
        convertNRRDToNifti(source, dest)
    else:
        if os.path.isdir(source):
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest)
    return dest

def symbolic_link(source, dest):
    """Crée un lien symbolique."""
    print(f"Creating symbolic link from {source} to {dest}")
    pathlib.Path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)
    os.symlink(source, dest)
    return dest


def move2nii(source, dest):
    """Déplace un fichier, en convertissant en nifti si nécessaire."""
    print(f"Moving {source} to {dest}")
    pathlib.Path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)

    if source.endswith(".nrrd") and dest.endswith(".nii.gz"):
        convertNRRDToNifti(source, dest)
        os.remove(source)
    else:
        shutil.move(source, dest)
    return dest

def copy_list(dest, file_list):
    """
    Copy a list of files to a new location after checking they exist.
    Files could contains BIDSFile, BIDSFile, Path or str objects.
    """
    for f in file_list:
        src_path = ""
        if isinstance(f, str):
            src_path = f
        elif isinstance(f, pathlib.Path):
            src_path = str(f)
        # elif isinstance(f, bids.layout.BIDSFile):
        #     src_path = f.path
        elif isinstance(f, BIDSFile):
            src_path = f.path
        else:
            # raise ValueError(f"Unknown type {type(f)}")
            print(f"Unknown type {type(f)}")
            continue
        
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue
        else:
            shutil.copy(src_path, dest)


    
def remove_temp_dir(res_dict):
    """
    Remove the temporary directory of the result dictionary if it exists and is not empty.
    """

    if not res_dict:
        return
    first_key = list(res_dict.keys())[0]
    temp_dir = str(pathlib.Path(first_key).parent)
    if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error removing temporary directory {temp_dir}: {e}")
            pass
    

def copy_from_dict(subject, file_dict, pipeline=None,dry_run=False,remove_after_copy=True, **kwargs):
    """
    Copy files from a dictionary to the BIDS dataset.
    eg file_dict : 
    {'/tmp/tmpkm3ebons/t1_pve_0.nii.gz': {'datatype': 'anat',
                                      'extension': '.nii.gz',
                                      'label': 'CSF',
                                      'space': 'B0',
                                      'subject': '03011',
                                      'suffix': 'propseg'}
    }
    """
    mapping={}

    #Drop original_name entitie if it is in kwargs
    #drop pipeline if in kwargs and is not None
    if 'pipeline' in kwargs and pipeline is not None:
        del kwargs['pipeline']

    global_pipeline = pipeline
    
    for src_file, src_entities in file_dict.items():
        pipeline = global_pipeline
        if src_file is None or not os.path.exists(src_file):
            print(f"Source file {src_file} does not exist. Skipping.")
            continue
        entities= deepcopy(src_entities)
        entities.update(kwargs)
        if 'pipeline' in entities.keys() and global_pipeline is not None:
            del entities['pipeline']
        
        if 'pipeline' in entities.keys() and global_pipeline is None:
            pipeline = deepcopy(entities['pipeline'])
            del entities['pipeline']

        dest_file = subject.build_path(
            original_name=os.path.basename(src_file), **entities, pipeline=pipeline)
        if not dry_run:
            copy2nii(src_file, dest_file)
        else:
            print(f"Copying {src_file} to {dest_file}")
        mapping[src_file] = dest_file
    if remove_after_copy:
        remove_temp_dir(file_dict)
    return mapping