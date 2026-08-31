from bids import BIDSLayout
import bids
from os.path import join as opj
import SimpleITK as sitk
from TractoPL.utils.tools import create_pipeline_description
from TractoPL.data.io import copy2nii, move2nii, parse_filename
from TractoPL.utils.hash_utils import check_folder_changed, save_folder_hash
import ants
import json
from pprint import pprint
import nibabel as nib
import ants
import pathlib
import os

class BIDSFile(bids.layout.BIDSFile):
    def __init__(self, bids_file, subject=None):
        vars(self).update(vars(bids_file))
        self.bids_file = bids_file
        self.scope = 'derivatives' if 'derivatives' in self.path else 'raw'
        self.pipeline = self.path.split(
            'derivatives/')[1].split('/')[0] if self.scope == 'derivatives' else None
        self.extra_entities = parse_filename(self.filename)
        self.extra_entities.pop('sub', None)
        self.subject_obj = subject

        # Set entities as attributes
        for k, v in self.get_full_entities().items():
            setattr(self, k, v)

    def copy(self, dest):
        """Copy the file to a new location, converting if necessary."""
        return copy2nii(self.path, dest)

    def move(self, dest):
        """Move the file to a new location, converting if necessary."""
        return move2nii(self.path, dest)

    def get_full_entities(self, **kwargs):
        """
        Another wrapper around a BIDSLayout method.
        In addition to the 'official' entities, it also returns the custom entities, scopes and pipelines.
        """

        entities = self.bids_file.get_entities(**kwargs)
        entities['scope'] = self.scope
        entities['pipeline'] = self.pipeline
        entities.update(self.extra_entities)
        return entities

    def get_subject(self):
        """
        Return the Subject object associated with this file.
        """
        if self.subject_obj is not None:
            return self.subject_obj
        else:
            return Subject(self.entities['subject'])


class Subject:

    def __init__(self,
                 sub_id,
                 db_root='/home/ndecaux/Data/actidep_bids',
                 layout=None,
                 force_index=False):
        self.db_root = db_root
        self.sub_id = sub_id
        self.bids_id = f'sub-{sub_id}'
        self.db_cache_path = opj(os.path.expanduser('~'), '.actidep_cache')
        self.hash_file_path = opj(os.path.expanduser('~'), f'.actidep_hash_{sub_id}.json')
        
        # Vérifier si le dossier a été modifié en utilisant le hash
        folder_changed, _ = check_folder_changed(self.db_root, self.hash_file_path)
        
        if layout is None:
            self.layout = BIDSLayout(self.db_root,
                                    derivatives=True,
                                    validate=False)
            # ,
            #                         database_path=self.db_cache_path,
            #                         reset_database=force_index or folder_changed)
            
            # Sauvegarder le nouveau hash après indexation
            if folder_changed or force_index:
                save_folder_hash(self.db_root, self.hash_file_path, 
                                {'subject': sub_id, 'last_indexed': True})
        else:
            self.layout = layout
            
        self.dicom_folder = opj(self.db_root, self.bids_id, 'sourcedata',
                                f'{self.bids_id}_dicoms')
                                
    def refresh(self):
        """
        Force réindexation du layout BIDS pour prendre en compte les changements dans la base
        """
        folder_changed, old_hash = check_folder_changed(self.db_root, self.hash_file_path)
        
        # Si le dossier n'a pas changé, éviter la réindexation sauf si explicitement demandé
        if not folder_changed:
            print(f"Le dossier {self.db_root} n'a pas été modifié depuis la dernière indexation")
            return self.layout
            
        self.layout = BIDSLayout(self.db_root,
                               derivatives=True,
                               validate=False,
                               database_path=self.db_cache_path,
                               reset_database=True)
                               
        # Sauvegarder le nouveau hash après réindexation
        save_folder_hash(self.db_root, self.hash_file_path, 
                        {'subject': self.sub_id, 'last_indexed': True})
        
        return self.layout

    def get(self, **kwargs):
        """
        Recover files that match the given criteria from the BIDS dataset.
        eg: get(model='DTI',metric='FA') should return files with _model-DTI_ and _metric-FA_ in their name.
        Basically a wrapper around the BIDSLayout.get() method that also handles custom entities.
        """
        # Get all kwargs that are known entities
        # self.layout = BIDSLayout(self.db_root,
        #                          derivatives=True,
        #                          validate=False,
        #                          database_path=self.db_cache_path)
        official_entities = list(self.layout.get_entities().keys())
        other_entries = [
            'return_type', 'target', 'scope', 'regex_search', 'absolute_paths',
            'invalid_filters'
        ]

        # Séparer les entités connues et les entités personnalisées
        known_entities = {
            k: kwargs[k]
            for k in kwargs.keys() if k in official_entities + other_entries
        }
        custom_entities = {
            k: v
            for k, v in kwargs.items() if k not in known_entities.keys()
        }

        # Préparer les paramètres pour la requête BIDS
        query_params = {'subject': self.sub_id, 'regex_search': True}

        # Traiter les entités "None" et les négations (!) pour les entités connues
        for entity, value in list(known_entities.items()):
            if value is None:
                # Si la valeur est None, on n'ajoute pas ce critère à la requête
                custom_entities.update({entity: known_entities.pop(entity)})

            elif isinstance(value, str) and '!' in value:
                # Si on veut exclure une valeur, on construit une regex d'exclusion
                excluded_value = value.replace('!', '')
                known_entities[entity] = f'^((?!{excluded_value}).)*$'

        # Ajouter les entités connues restantes à la requête
        query_params.update(known_entities)

        # Récupérer la liste initiale de fichiers
        sub_list = self.layout.get(**query_params)

        # Filtrer par pipeline si spécifié
        if 'pipeline' in custom_entities:
            pipeline = custom_entities.pop('pipeline')
            # Filter files that don't contains derivative/<pipeline> in their path
            sub_list = [
                f for f in sub_list
                if f.path.find(f'derivatives/{pipeline}/') != -1
            ]

        # Filtrer par entités personnalisées
        files = []
        for f in sub_list:
            entities = parse_filename(f.filename)
            try:
                matched = True
                for k, v in custom_entities.items():
                    if v is None:
                        # Si la valeur est None, on vérifie que la clé n'existe pas dans les entités
                        if k in entities:
                            matched = False
                            break
                    elif isinstance(v, str) and '!' in v:
                        # Si la valeur commence par !, on vérifie que la clé n'existe pas avec une valeur spécifique
                        if entities.get(k) == v[1:]:
                            matched = False
                            break
                    else:
                        # Comportement habituel: vérification que la clé existe avec la valeur attendue
                        if entities.get(k) != v:
                            matched = False
                            break

                if matched:
                    actidep_file = BIDSFile(f, self)
                    files.append(actidep_file)
            except KeyError:
                # Si une clé n'existe pas et qu'on cherche une valeur spécifique (pas None)
                continue

        return files

    def get_unique(self, **kwargs):
        """
        Return the unique file that matches the given criteria.
        """
        files = self.get(**kwargs)
        if len(files) == 0:
            raise ValueError(f"No file found for {kwargs}")
        if len(files) > 1:
            raise ValueError(f"Multiple files found for {kwargs}")
        return files[0]

    def build_path(self,
                   suffix,
                   pipeline=None,
                   scope=None,
                   original_name=None,
                   is_dir=False,
                   **kwargs):
        """
        Construire le chemin d'un fichier dans le layout à partir d'entités et d'un pipeline donnés.

        Args:
            suffix (str): Suffixe du fichier (dwi, T1w, etc.)
            pipeline (str): Nom de la pipeline (preprocessing, etc.)
            original_name (str): Nom du fichier original
            **kwargs: Entités supplémentaires à ajouter au nom du fichier
        """
        entities = {'suffix': suffix}

        for k, v in kwargs.items():
            entities[k] = v

        root_dir = self.layout.root
        parent_dir = entities.pop(
            'datatype') if 'datatype' in entities else entities['suffix']
        if pipeline is None:
            target_dir = opj(root_dir, self.bids_id, parent_dir)
        else:
            if pipeline not in self.get_pipelines():
                create_pipeline_description(pipeline, self.layout)

            target_dir = opj(root_dir, 'derivatives', pipeline, self.bids_id,
                             parent_dir)

        fileradix = f'{self.bids_id}'

        if original_name is not None and 'extension' not in entities:
            # entities['desc'] = original_name.split('_')[-1].split('.')[0]
            extension = '.' + '.'.join(original_name.split('.')[1:])
            if extension == '.nrrd':
                extension = '.nii.gz'

        else:

            is_dir = entities.get('is_dir') or is_dir

            if 'extension' in entities:
                extension = '.' + entities['extension'].lstrip('.')
            else:

                extension = '' if is_dir else '.nii.gz'
                # Display a warning
                print(
                    f"Warning: No extension provided for {entities}, assuming {extension}"
                )

        remaining_entities = {
            k: v
            for k, v in entities.items()
            if k not in ['suffix', 'subject', 'extension']
        }

        for k, v in sorted(remaining_entities.items()):
            fileradix += f'_{k}-{v}'
        path = opj(target_dir,
                   fileradix + "_" + entities['suffix'] + extension)

        return path

    def get_pipelines(self):
        """
        Return the list of pipelines that have been run on this subject.
        """
        pipelines = set()
        for f in self.layout.get(subject=self.sub_id):
            if 'derivatives' in f.path:
                pipeline = f.path.split('derivatives/')[1].split('/')[0]
                pipelines.add(pipeline)
        return list(pipelines)

    def write_object(self,
                     obj,
                     suffix,
                     pipeline=None,
                     scope=None,
                     original_name=None,
                     refresh_layout=True,
                     **kwargs):
        """
        Write an object to a file in the layout.
        """
        path = self.build_path(suffix, pipeline, scope, original_name,
                               **kwargs)

        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        if isinstance(obj, sitk.Image):
            sitk.WriteImage(obj, path)

        elif isinstance(obj, ants.ANTsImage):
            ants.image_write(obj, path)

        elif isinstance(obj, dict):
            with open(path, 'w') as f:
                json.dump(obj, f)

        elif isinstance(obj, nib.Nifti1Image):
            nib.save(obj, path)

        elif isinstance(obj, str):
            copy2nii(obj, path)
        else:
            raise ValueError(
                f"Object type {type(obj)} not supported for writing.")
                
        # Mettre à jour le hash après l'écriture d'un nouveau fichier
        save_folder_hash(self.db_root, self.hash_file_path, 
                        {'subject': self.sub_id, 'last_write': True})
        
        # Réindexer le layout après l'écriture d'un nouveau fichier si demandé
        if refresh_layout:
            self.refresh()

        return path

    def get_entity_values(self):
        """
        Get all entities that exist for this subject and their possible values.
        
        Returns:
            dict: A dictionary where keys are entity names and values are lists of all possible values for those entities
        """
        # Initialize the result dictionary
        entity_values = {}

        # Get all files for this subject
        files = self.layout.get(subject=self.sub_id)

        #exclude .dcm files
        files = [f for f in files if not f.path.endswith('.dcm')]

        # Process each file
        for file in files:
            # Get file entities
            entities = parse_filename(file.filename)

            # Add standard BIDS entities
            standard_entities = file.get_entities()
            entities.update(standard_entities)

            # Process each entity
            for entity, value in entities.items():
                if entity == 'subject':  # Skip subject as it's constant for this Subject instance
                    continue

                # Initialize the list for this entity if it doesn't exist
                if entity not in entity_values:
                    entity_values[entity] = []

                # Add the value if it's not already in the list
                if value not in entity_values[entity]:
                    entity_values[entity].append(value)

        # Sort values for better readability
        for entity in entity_values:
            entity_values[entity].sort()

        return entity_values

class Dataset:
    def __init__(self, db_root='/home/ndecaux/Data/actidep_bids', force_index=False):
        __abstract__ = True  # Preserved from original code
        self.db_root = db_root
        #Home directory for the database
        self.db_cache_path = opj(os.path.expanduser('~'), '.actidep_cache')
        self.hash_file_path = opj(os.path.expanduser('~'), '.actidep_hash_global.json')
        
        # Vérifier si le dossier a été modifié en utilisant le hash
        folder_changed, _ = check_folder_changed(self.db_root, self.hash_file_path)
        
        self.layout = BIDSLayout(self.db_root, 
                                derivatives=True, 
                                validate=False)
                                # ,
                                # database_path=self.db_cache_path,
                                # reset_database=force_index or folder_changed)
        
        # Sauvegarder le nouveau hash après indexation
        if folder_changed or force_index:
            save_folder_hash(self.db_root, self.hash_file_path, {'global_db': True})

        # Get subject IDs from the layout
        self.subject_ids = self.layout.get_subjects()

    def refresh(self):
        """
        Force réindexation du layout BIDS pour prendre en compte les changements dans la base
        """
        folder_changed, old_hash = check_folder_changed(self.db_root, self.hash_file_path)
        
        # Si le dossier n'a pas changé, éviter la réindexation sauf si explicitement demandé
        if not folder_changed:
            print(f"Le dossier {self.db_root} n'a pas été modifié depuis la dernière indexation")
            return self.layout
            
        self.layout = BIDSLayout(self.db_root, 
                                derivatives=True, 
                                validate=False,
                                database_path=self.db_cache_path,
                                reset_database=True)
                                
        # Sauvegarder le nouveau hash après réindexation
        save_folder_hash(self.db_root, self.hash_file_path, {'global_db': True})
        
        self.subject_ids = self.layout.get_subjects()
        return self.layout

    def get(self, sub_id, **kwargs):
        # This method creates a new Subject instance on-demand.
        # It could be modified to retrieve from self.subjects if desired,
        # but the prompt focuses on __init__.
        subject = Subject(sub_id, self.db_root, self.layout)
        return subject.get(**kwargs)

    def get_subject(self, sub_id, force_index=False):
        """
        Return the Subject object associated with this ID.
        """
        if force_index:
            self.refresh()
            
        if sub_id in self.subject_ids:
            return Subject(sub_id, self.db_root, self.layout)
        else:
            raise ValueError(f"Subject {sub_id} not found in the dataset.")

    def save_db(self):
        """
        Save the BIDS dataset to the database root.
        This is a placeholder method, as BIDS datasets are typically not "saved" in the traditional sense.
        """
        # In practice, this might involve writing metadata or updating the layout.
        # Here we just print a message for demonstration purposes.

        self.layout.save(self.db_cache_path)
        return self.layout

if __name__ == "__main__":
    # Exemple d'utilisation avec force_index=True pour forcer la réindexation
    ds = Dataset('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids', force_index=True)
    # ou ds.refresh() pour réindexer après initialisation
    
    # Exemple avec un sujet
    # subject = ds.get_subject('03011', force_index=True)
    # ou subject.refresh() pour réindexer après initialisation
