import os
import hashlib
import json
import time
from pathlib import Path

def calculate_folder_hash(folder_path, ignore_patterns=None):
    """
    Calcule un hash représentant l'état du dossier et de ses fichiers.
    
    Args:
        folder_path (str): Chemin du dossier à hacher
        ignore_patterns (list): Liste de patterns de fichiers à ignorer (ex: ['.git', '*.tmp'])
    
    Returns:
        str: Hash hexadécimal représentant l'état du dossier
    """
    if ignore_patterns is None:
        ignore_patterns = ['.git', '*.tmp', '.DS_Store', '__pycache__', '*.pyc']
    
    hasher = hashlib.md5()
    
    # Parcourir tous les fichiers et dossiers récursivement
    for root, dirs, files in os.walk(folder_path):
        # Ignorer les dossiers correspondants aux patterns
        dirs[:] = [d for d in dirs if not any(pat in d for pat in ignore_patterns)]
        
        # Ajouter le nom du dossier au hash
        hasher.update(root.encode())
        
        # Ajouter chaque fichier au hash (nom + date de modification)
        for file in sorted(files):
            if not any(pat in file for pat in ignore_patterns):
                file_path = os.path.join(root, file)
                try:
                    # Utiliser le nom du fichier et sa date de modification
                    file_info = f"{file_path}:{os.path.getmtime(file_path)}"
                    hasher.update(file_info.encode())
                except OSError:
                    # En cas d'erreur (fichier supprimé entre-temps, etc.), ignorer
                    pass
    
    return hasher.hexdigest()

def save_folder_hash(folder_path, hash_file_path=None, metadata=None):
    """
    Calcule le hash du dossier et le sauvegarde dans un fichier.
    
    Args:
        folder_path (str): Chemin du dossier à hacher
        hash_file_path (str): Chemin où sauvegarder le fichier de hash
        metadata (dict): Métadonnées supplémentaires à stocker
    
    Returns:
        str: Chemin du fichier de hash
    """
    folder_hash = calculate_folder_hash(folder_path)
    
    if hash_file_path is None:
        # Par défaut, stocker dans le répertoire personnel de l'utilisateur
        hash_file_path = os.path.join(os.path.expanduser('~'), '.actidep_folder_hash.json')
    
    data = {
        'folder_path': folder_path,
        'hash': folder_hash,
        'timestamp': time.time(),
    }
    
    if metadata:
        data.update(metadata)
    
    # Créer le dossier parent si nécessaire
    Path(os.path.dirname(hash_file_path)).mkdir(parents=True, exist_ok=True)
    
    with open(hash_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return hash_file_path

def check_folder_changed(folder_path, hash_file_path=None):
    """
    Vérifie si le dossier a été modifié depuis la dernière sauvegarde du hash.
    
    Args:
        folder_path (str): Chemin du dossier à vérifier
        hash_file_path (str): Chemin du fichier de hash
    
    Returns:
        bool: True si le dossier a été modifié, False sinon
        str: Ancien hash ou None si le fichier de hash n'existe pas
    """
    if hash_file_path is None:
        hash_file_path = os.path.join(os.path.expanduser('~'), '.actidep_folder_hash.json')
    
    # Si le fichier de hash n'existe pas, considérer que le dossier a été modifié
    if not os.path.exists(hash_file_path):
        return True, None
    
    try:
        with open(hash_file_path, 'r') as f:
            data = json.load(f)
        
        # Vérifier que le hash correspond au même dossier
        if data.get('folder_path') != folder_path:
            return True, None
        
        old_hash = data.get('hash')
        new_hash = calculate_folder_hash(folder_path)
        
        return old_hash != new_hash, old_hash
    
    except (json.JSONDecodeError, IOError):
        # En cas d'erreur de lecture du fichier de hash, considérer que le dossier a été modifié
        return True, None
