import os
import json
from os.path import join as opj
from pathlib import Path
import pathlib
from os.path import join as opj
from ..data.io import copy2nii, symbolic_link
from ..set_config import set_config
from subprocess import call
import shutil
import tempfile
import time
def run_cli_command(command_name, inputs, output_pattern, entities_template={}, 
                   prepare_inputs_fn=None, command_args=None,use_sym_link=True, **kwargs):
    """
    Fonction générique pour exécuter une commande CLI sur des fichiers d'entrée
    et retourner les résultats structurés.

    Parameters
    ----------
    command_name : str
        Nom de la commande à exécuter
    inputs : dict
        Dictionnaire associant un nom à chaque fichier d'entrée BIDSFile
        {"dwi": dwi_file, "bval": bval_file, ...}
    output_pattern : dict
        Dictionnaire associant les noms de fichiers de sortie à leurs entités
        {"output.nii.gz": {"label": "WM"}, ...}
    entities_template : dict ou BIDSFile
        Entités de base à utiliser pour construire les entités de sortie
    prepare_inputs_fn : callable, optional
        Fonction pour préparer les entrées avant l'exécution (ex: inverser les bvecs)
    command_args : list, optional
        Liste des arguments spécifiques à la commande (remplace la construction automatique)
        Les références symboliques sous forme de "$nom_input" seront remplacées par les chemins temporaires
    **kwargs : 
        Arguments additionnels à passer à la commande
    
    Returns
    -------
    dict
        Dictionnaire associant les chemins complets des fichiers de sortie à leurs entités
    """
    # Configuration et création du dossier temporaire
    config, tools = set_config()
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)
    
    if prepare_inputs_fn is not None:
        use_sym_link = False  # Si une fonction de préparation est fournie, on ne peut pas utiliser de liens symboliques

    # Copie des fichiers d'entrée
    tmp_inputs = {}
    for name, file_obj in inputs.items():
        if file_obj is None:
            tmp_inputs[name] = None
            continue
            
        if isinstance(file_obj, str):
            # Si c'est déjà un chemin de fichier
            tmp_inputs[name] = file_obj
        else:
            tmp_path = opj(tmp_folder, f'{name}{file_obj.extension}')
            if use_sym_link:
                tmp_inputs[name] = symbolic_link(file_obj.path, tmp_path)
            else:
                tmp_inputs[name] = copy2nii(file_obj.path, tmp_path)
   
    
    # Application de la fonction de préparation si nécessaire
    if prepare_inputs_fn:
        prepare_inputs_fn(tmp_inputs)
    
    # Construction de la commande
    command = [command_name]
    
    if command_args:
        # Remplacer les références symboliques par les chemins temporaires
        processed_args = []
        for arg in command_args:
            if isinstance(arg, str) and arg.startswith('$'):
                # Extraire le nom de l'entrée (sans le $)
                input_name = arg[1:]
                if input_name in tmp_inputs and tmp_inputs[input_name] is not None:
                    processed_args.append(tmp_inputs[input_name])
                else:
                    # Si l'entrée n'existe pas, on conserve l'argument tel quel
                    processed_args.append(arg)
            else:
                processed_args.append(arg)
        
        command.extend(processed_args)
    else:
        # Construction par défaut en ajoutant simplement les fichiers d'entrée
        for input_path in tmp_inputs.values():
            if input_path:  # Ne pas ajouter les entrées None
                command.append(input_path)
    
    # Ajout des kwargs à la commande
    command = add_kwargs_to_cli(command, **kwargs)
    print(command)
    print(f"Running command: {' '.join(command)}")
    
    # Exécution de la commande
    call(command)

    time.sleep(1)
    
    print(f"{command_name} done")
    
    # Préparation des entités de base pour les sorties
    if isinstance(entities_template, dict):
        base_entities = entities_template
    else:
        # Si c'est un BIDSFile, on récupère ses entités
        base_entities = entities_template.get_entities()
    
    # Construction du dictionnaire de résultats
    res_dict = {opj(tmp_folder, output_file): upt_dict(base_entities.copy(), **entity_updates) 
                for output_file, entity_updates in output_pattern.items()}
    
    return res_dict

def run_anima_command(command_name, inputs, output_pattern, entities_template, 
                      prepare_inputs_fn=None, command_args=None, **kwargs):
    """
    Fonction spécifique pour exécuter une commande Anima sur des fichiers d'entrée.

    Parameters
    ----------
    command_name : str
        Nom de la commande Anima à exécuter
    inputs : dict
        Dictionnaire associant un nom à chaque fichier d'entrée
    output_pattern : dict
        Dictionnaire associant les noms de fichiers de sortie à leurs entités
    entities_template : dict ou BIDSFile
        Entités de base à utiliser pour construire les entités de sortie
    prepare_inputs_fn : callable, optional
        Fonction pour préparer les entrées avant l'exécution
    command_args : list, optional
        Liste des arguments spécifiques à la commande Anima
    **kwargs : 
        Arguments additionnels à passer à la commande
    
    Returns
    -------
    dict
        Dictionnaire associant les chemins complets des fichiers de sortie à leurs entités
    """
    return run_cli_command(
        command_name,
        inputs,
        output_pattern,
        entities_template,
        prepare_inputs_fn=prepare_inputs_fn,
        command_args=command_args,
        **kwargs
    )



def run_mrtrix_command(command_name, inputs, output_pattern, entities_template, 
                      prepare_inputs_fn=None, command_args=None, **kwargs):
    """
    Fonction spécifique pour exécuter une commande MRtrix3 sur des fichiers d'entrée.
    Ajoute automatiquement le flag '-force' aux commandes MRtrix3.

    Parameters
    ----------
    command_name : str
        Nom de la commande MRtrix3 à exécuter
    inputs : dict
        Dictionnaire associant un nom à chaque fichier d'entrée
    output_pattern : dict
        Dictionnaire associant les noms de fichiers de sortie à leurs entités
    entities_template : dict ou BIDSFile
        Entités de base à utiliser pour construire les entités de sortie
    prepare_inputs_fn : callable, optional
        Fonction pour préparer les entrées avant l'exécution
    command_args : list, optional
        Liste des arguments spécifiques à la commande MRtrix3
    **kwargs : 
        Arguments additionnels à passer à la commande
    
    Returns
    -------
    dict
        Dictionnaire associant les chemins complets des fichiers de sortie à leurs entités
    """
    # Ajouter le flag -force automatiquement si pas déjà présent dans command_args
    if command_args and '-force' not in command_args:
        command_args = command_args + ['-force']
    elif not command_args:
        command_args = ['-force']
        
    return run_cli_command(
        command_name,
        inputs,
        output_pattern,
        entities_template,
        prepare_inputs_fn=prepare_inputs_fn,
        command_args=command_args,
        **kwargs
    )

def run_multiple_commands(command_list, inputs, output_pattern, entities_template,
                        prepare_inputs_fn=None, **kwargs):
    """
    Fonction pour exécuter plusieurs commandes sur les fichiers d'entrée.
    Chaque commande est une liste contenant le nom de la commande et ses arguments.
    
    Parameters
    ----------
    command_list : list of lists
        Liste des commandes à exécuter, où chaque commande est une liste [command_name, *args]
    inputs : dict
        Dictionnaire associant un nom à chaque fichier d'entrée
    output_pattern : dict
        Dictionnaire associant les noms de fichiers de sortie à leurs entités
    entities_template : dict ou BIDSFile
        Entités de base à utiliser pour construire les entités de sortie
    prepare_inputs_fn : callable, optional
        Fonction pour préparer les entrées avant l'exécution
    **kwargs :
        Arguments additionnels à passer à toutes les commandes
    Returns
    -------
    dict
        Dictionnaire associant les chemins complets des fichiers de sortie à leurs entités
    """
    # Configuration et création du dossier temporaire
    config, tools = set_config()
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)
    
    # Copier les fichiers d'entrée
    tmp_inputs = {}
    for name, file_obj in inputs.items():
        if file_obj is None:
            tmp_inputs[name] = None
            continue
            
        if isinstance(file_obj, str):
            # Si c'est déjà un chemin de fichier
            tmp_inputs[name] = file_obj
        else:
            tmp_path = opj(tmp_folder, f'{name}{file_obj.extension}')
            tmp_inputs[name] = copy2nii(file_obj.path, tmp_path)
    
    # Appliquer la fonction de préparation si nécessaire
    if prepare_inputs_fn:
        prepare_inputs_fn(tmp_inputs)
    
    # Exécuter chaque commande
    for cmd in command_list:
        if not cmd:  # Skip empty commands
            continue
            
        command_name = cmd[0]
        command_args = cmd[1:] if len(cmd) > 1 else []
        
        # Construction de la commande
        command = [command_name]
        
        # Remplacer les références symboliques par les chemins temporaires
        processed_args = []
        for arg in command_args:
            if isinstance(arg, str) and arg.startswith('$'):
                # Extraire le nom de l'entrée (sans le $)
                input_name = arg[1:]
                if input_name in tmp_inputs and tmp_inputs[input_name] is not None:
                    processed_args.append(tmp_inputs[input_name])
                else:
                    # Si l'entrée n'existe pas, on conserve l'argument tel quel
                    processed_args.append(arg)
            else:
                processed_args.append(arg)
        
        command.extend(processed_args)
        
        # Ajout des kwargs à la commande
        command = add_kwargs_to_cli(command, **kwargs)
        print(f"Running command: {' '.join(str(c) for c in command)}")
        
        # Exécution de la commande
        call(command)
        time.sleep(1)
        
        print(f"{command_name} done")
    
    # Préparer les entités de base pour les sorties
    if isinstance(entities_template, dict):
        base_entities = entities_template
    else:
        # Si c'est un BIDSFile, on récupère ses entités
        base_entities = entities_template.get_entities()
    
    # Construction du dictionnaire de résultats
    res_dict = {opj(tmp_folder, output_file): upt_dict(base_entities.copy(), **entity_updates)
                for output_file, entity_updates in output_pattern.items()}
    
    return res_dict

        
def del_key(dct, key):
    d = dct.copy()
    del d[key]
    return d

def upt_dict(dct, new_items=None, **kwargs):
    """
    Update a dictionary with new items and/or keyword arguments. Not sure if it's very compliant with Python 
    conventions, but it works.

    Parameters
    ----------
    dct : Initial dictionary
    new_items : dict
        Dictionary of new items to add or update in the dictionary
    **kwargs : named arguments
        Named arguments to add or update in the dictionary

    """
    d = dct.copy()
    
    # Traiter le dictionnaire positionnel s'il existe
    if new_items and isinstance(new_items, dict):
        d.update(new_items)


    # Traiter les arguments nommés comme avant
    for key, value in kwargs.items():
        if isinstance(value, dict):
            d.update(value)
        else:
            if key != '':
                d[key] = value
            else:
                raise ValueError("Empty key. Please provide only named arguments.")
    
    print(d)
    return d

def create_pipeline_description(pipeline, layout,**kwargs):
    """
    Create the dataset_description.json file for the given subject and pipeline.
    """
    dataset_description = {
        "Name": f"actiDep {pipeline} pipeline",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "PipelineDescription": {
            "Name": pipeline,
            "Version": "0.1"
        }
    }
    dataset_description['PipelineDescription'].update(kwargs)

    #If pipeline folder does not exist, create it (pathlib)
    Path(opj(layout.root,'derivatives',pipeline)).mkdir(parents=True, exist_ok=True)
    
    with open(opj(layout.root,'derivatives',pipeline,'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, indent=4)
    
def get_exact_file(list_of_files):
    """
    Return the file in the list if there is only one, else raise an error.
    """
    if len(list_of_files) != 1:
        raise ValueError(f"Found {len(list_of_files)} files")
    return list_of_files[0]

class CLIArg:
    def __init__(self, name, value=None, long_flag=True):
        self.name = name
        self.value = value
        self.long_flag = long_flag

        if long_flag:
            #Add the -- to the name if it is not already there
            self.name = self.name.lstrip('-')  # Remove any leading hyphens
            self.name = '--' + self.name
        else:
            #Add single - if not already there and remove any double --
            self.name = self.name.replace('--', '-')
            if not self.name.startswith('-'):
                self.name = '-'+self.name

    def get(self):
        if self.value:
            return [f"{self.name}", str(self.value)]
        else:
            return [f"{self.name}"]

def first_match(string_seq, list_of_keys):
    """
    Return the first matching key in the string sequence.
    """
    for key in list_of_keys:
        if key in string_seq:
            return key
    return None
    

def add_kwargs_to_cli(command,**kwargs):
    """
    Add the named arguments to the command line command.
    """
    if not kwargs:
        return command
    else:
        for key, value in kwargs.items():
            if isinstance(value,CLIArg):
                command=command+value.get()
            else:
                if isinstance(value,bool) and value:
                    command = command + [f"-{key.replace('-','')}"]
                else :
                    command = command + [f"-{key.replace('-','')}", str(value)]
        return command

