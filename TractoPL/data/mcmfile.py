import os
from os.path import join as opj
import tempfile
import glob
import shutil
import xml.etree.ElementTree as ET
import zipfile

class MCMFile:
    """
    Une classe pour gérer les fichiers au format MCM d'Anima.
    Basiquement, un zip contenant un fichier .mcm à la base et des fichiers .nrrd dans un répertoire mcm.
    """
    def __init__(self, path):
        """
        Initialise l'objet MCMFile à partir d'un chemin vers un répertoire ou un fichier .mcm.
        
        Parameters
        ----------
        path : str
            Chemin vers un fichier .mcmx (archive zip) ou un fichier .mcm (fichier texte contenant les informations MCM, dont le nom de fichiers de compartiments et pondération)
        """
        # Stocker le chemin original
        self.original_path = path
        
        # Déterminer si c'est un fichier .mcmx ou un répertoire
        self.is_archive = path.endswith('.mcmx') and os.path.isfile(path)
        
        # Créer un répertoire temporaire si c'est une archive
        if self.is_archive:
            self.temp_dir = tempfile.mkdtemp()
            self._extract_archive(path, self.temp_dir)
            self.dir_path = self.temp_dir
        else:
            self.dir_path = os.path.dirname(path) if os.path.isfile(path) else path
            self.temp_dir = None
        
        # Charger les fichiers du répertoire MCM
        self._load_files()
        
        # Lire le fichier .mcm pour extraire les informations sur les compartiments
        self._parse_mcm_file()
    
    def _extract_archive(self, archive_path, extract_dir):
        """
        Extrait une archive .mcmx dans un répertoire.
        
        Parameters
        ----------
        archive_path : str
            Chemin vers le fichier .mcmx (archive zip)
        extract_dir : str
            Répertoire où extraire les fichiers
        """
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    def _load_files(self):
        """Charge les fichiers du répertoire MCM comme attributs de l'objet."""
        self.files = {}
        
        # Trouver le fichier .mcm principal
        mcm_files = glob.glob(opj(self.dir_path, '*.mcm'))
        if mcm_files:
            self.mcm_file_path = mcm_files[0]
            self.files['mcm'] = self.mcm_file_path
            setattr(self, 'mcmfile', self.mcm_file_path)
            
            # Déterminer le nom du sous-dossier à partir du nom du fichier mcm
            mcm_basename = os.path.basename(self.mcm_file_path)
            subdir_name = os.path.splitext(mcm_basename)[0]  # Nom sans extension
            
            # Chercher le répertoire contenant les fichiers associés
            mcm_subdir = opj(self.dir_path, subdir_name)
            
            if os.path.exists(mcm_subdir):
                # Parser le fichier MCM pour obtenir les noms des fichiers
                mcm_data = read_mcm_file(self.mcm_file_path)
                
                # Charger le fichier de poids
                weights_filename = mcm_data['weights']
                weights_path = opj(mcm_subdir, weights_filename)
                if os.path.exists(weights_path):
                    self.files['weights'] = weights_path
                    setattr(self, 'weights', weights_path)
                
                # Initialiser le dictionnaire pour les compartiments
                self.compartment_files = {}
                
                # Charger les fichiers de compartiments
                for comp in mcm_data['compartments']:
                    comp_filename = comp['filename']
                    comp_path = opj(mcm_subdir, comp_filename)
                    if os.path.exists(comp_path):
                        # Extraire le numéro du compartiment du nom de fichier
                        comp_parts = os.path.splitext(comp_filename)[0].split('_')
                        if len(comp_parts) > 1:
                            comp_num = comp_parts[-1]
                            self.compartment_files[comp_num] = comp_path
                            self.files[f"compartment_{comp_num}"] = comp_path
            
            # Pour la rétrocompatibilité, vérifier aussi le répertoire 'mcm'
            elif os.path.exists(opj(self.dir_path, 'mcm')):
                # Continuer avec le code existant pour le répertoire 'mcm'
                for filepath in glob.glob(opj(self.dir_path, 'mcm', '*')):
                    filename = os.path.basename(filepath)
                    basename, ext = os.path.splitext(filename)
                    
                    # Stocker le chemin complet dans le dictionnaire files
                    self.files[basename] = filepath
                    
                    # Le fichier de poids
                    if basename == 'mcm_weights':
                        setattr(self, 'weights', filepath)
                    
                    # Les compartiments
                    if basename.startswith('mcm_') and basename != 'mcm_weights':
                        compartment_num = basename.split('_')[1]
                        if not hasattr(self, 'compartment_files'):
                            self.compartment_files = {}
                        self.compartment_files[compartment_num] = filepath
    
    def _parse_mcm_file(self):
        """
        Analyse le fichier .mcm pour extraire les informations sur les compartiments.
        """
        if not hasattr(self, 'mcmfile') or not os.path.exists(self.mcmfile):
            print(f"Attention: fichier MCM introuvable à {getattr(self, 'mcmfile', 'non défini')}")
            self.compartments = {}
            return
        
        # Utiliser la fonction read_mcm_file existante
        mcm_data = read_mcm_file(self.mcmfile)
        
        # Préparer le dictionnaire des compartiments
        self.compartments = {}
        
        # Stocker le chemin du fichier de poids
        self.weights_filename = mcm_data['weights']
        if hasattr(self, 'compartment_files'):
            for i, comp in enumerate(mcm_data['compartments']):
                comp_num = comp['filename'].split('_')[-1].split('.')[0]
                self.compartments[comp_num] = {
                    'type': comp['type'],
                    'filename': comp['filename'],
                    'path': self.compartment_files.get(comp_num, None),
                    'index': i
                }
    
    def write(self, output_path=None, compress=True):
        """
        Écrit le MCMFile sur le disque, soit comme répertoire, soit comme archive .mcmx.
        
        Parameters
        ----------
        output_path : str, optional
            Chemin de sortie. Si non fourni, utilise le chemin original avec extension modifiée.
        compress : bool, optional
            Si True, crée une archive .mcmx (zip). Sinon, crée un répertoire.
        
        Returns
        -------
        str
            Chemin vers le fichier ou répertoire créé
        """
        # Déterminer le chemin de sortie
        if output_path is None:
            # Obtenir la base du chemin original sans extension
            base_path = os.path.splitext(self.original_path)[0]
            if compress:
                output_path = f"{base_path}.mcmx"
            else:
                output_path = f"{base_path}_mcm"
        else:
            # S'assurer que l'extension est correcte
            if compress and not output_path.endswith('.mcmx'):
                output_path += '.mcmx'
        
        # Déterminer le nom du sous-dossier des compartiments
        compartment_subdir = None
        if hasattr(self, 'mcmfile') and os.path.exists(self.mcmfile):
            mcm_basename = os.path.basename(self.mcmfile)
            subdir_name = os.path.splitext(mcm_basename)[0]
            compartment_subdir_path = opj(self.dir_path, subdir_name)
            if os.path.exists(compartment_subdir_path):
                compartment_subdir = subdir_name
        
        # Écrire les fichiers
        if compress:
            # Créer l'archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Ajouter le fichier .mcm principal
                if hasattr(self, 'mcmfile') and os.path.exists(self.mcmfile):
                    arcname = os.path.basename(self.mcmfile)
                    zipf.write(self.mcmfile, arcname)
                
                # Ajouter les fichiers du dossier des compartiments
                if compartment_subdir:
                    compartment_dir = opj(self.dir_path, compartment_subdir)
                    for file_path in glob.glob(opj(compartment_dir, '*')):
                        arcname = os.path.join(compartment_subdir, os.path.basename(file_path))
                        zipf.write(file_path, arcname)
                
                # Ajouter les fichiers du répertoire mcm (pour rétrocompatibilité)
                mcm_dir = opj(self.dir_path, 'mcm')
                if os.path.exists(mcm_dir):
                    for file_path in glob.glob(opj(mcm_dir, '*')):
                        arcname = os.path.join('mcm', os.path.basename(file_path))
                        zipf.write(file_path, arcname)
        else:
            # Créer un répertoire
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)
            
            # Copier le fichier .mcm principal
            if hasattr(self, 'mcmfile') and os.path.exists(self.mcmfile):
                shutil.copy2(self.mcmfile, opj(output_path, os.path.basename(self.mcmfile)))
            
            # Copier le dossier des compartiments
            if compartment_subdir:
                compartment_dir = opj(self.dir_path, compartment_subdir)
                output_compartment_dir = opj(output_path, compartment_subdir)
                os.makedirs(output_compartment_dir, exist_ok=True)
                for file_path in glob.glob(opj(compartment_dir, '*')):
                    shutil.copy2(file_path, opj(output_compartment_dir, os.path.basename(file_path)))
            
            # Copier les fichiers du répertoire mcm (pour rétrocompatibilité)
            mcm_dir = opj(self.dir_path, 'mcm')
            if os.path.exists(mcm_dir):
                os.makedirs(opj(output_path, 'mcm'), exist_ok=True)
                for file_path in glob.glob(opj(mcm_dir, '*')):
                    shutil.copy2(file_path, opj(output_path, 'mcm', os.path.basename(file_path)))
        
        return output_path
    
    # def __del__(self):
    #     """Nettoie les ressources temporaires à la destruction de l'objet."""
    #     if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
    #         shutil.rmtree(self.temp_dir)
    
    def __str__(self):
        """Représentation en chaîne de l'objet."""
        result = [f"MCMFile at {self.dir_path}:"]
        
        # Afficher le fichier .mcm
        if hasattr(self, 'mcmfile'):
            result.append(f"  MCM File: {os.path.basename(self.mcmfile)}")
        
        # Afficher le fichier de poids
        if hasattr(self, 'weights'):
            result.append(f"  Weights: {os.path.basename(self.weights)}")
        
        # Afficher les compartiments
        if hasattr(self, 'compartments') and self.compartments:
            result.append("  Compartments:")
            for comp_num, comp_info in self.compartments.items():
                result.append(f"    {comp_num}: {comp_info['type']} - {os.path.basename(comp_info['filename'])}")
        
        return "\n".join(result)
    
    def get_compartment(self, num=None):
        """
        Récupère les informations d'un compartiment spécifique.
        
        Parameters
        ----------
        num : str or int
            Numéro du compartiment à récupérer
        
        Returns
        -------
        dict
            Informations sur le compartiment, ou None si non trouvé
        """
            
        num = str(num)  # Convertir en chaîne pour la cohérence
        if hasattr(self, 'compartments') and num in self.compartments:
            return self.compartments[num]
        return None

    def get_weights(self):
        """
        Récupère le chemin du fichier de poids. Décompresse l'archive dans un répertoire temporaire si nécessaire.
        Returns
        -------
        str or None
            Chemin vers le fichier de poids, ou None si non trouvé
        """
        if hasattr(self, 'weights'):
            return self.weights
        
        # Si le fichier de poids n'est pas trouvé, retourner None
        return None
    

    def get_reference(self):
        """
        Récupère le chemin du fichier de référence (premier compartiment).
        
        Returns
        -------
        str or None
            Chemin vers le fichier de référence, ou None si non trouvé
        """
        if hasattr(self, 'compartments') and self.compartments:
            first_comp_num = sorted(self.compartments.keys())[0]
            first_comp = self.compartments[first_comp_num]
            return first_comp['path']
        
        return None

def read_mcm_file(mcm_file):
    """
    Read the MCM file and return dictionary containing model structure.
    
    Parameters
    ----------
    mcm_file : str
        Path to .mcm XML file
        
    Returns
    -------
    dict
        Dictionary containing weights file and compartment information
    """
    tree = ET.parse(mcm_file)
    root = tree.getroot()
    
    model = {
        'weights': root.find('Weights').text,
        'compartments': []
    }
    
    for comp in root.findall('Compartment'):
        model['compartments'].append({
            'type': comp.find('Type').text,
            'filename': comp.find('FileName').text
        })
        
    return model

