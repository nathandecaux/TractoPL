from ancpbids import BIDSLayout, DatasetOptions
import os.path
from os.path import join as opj
import SimpleITK as sitk
from TractoPL.utils.tools import create_pipeline_description
from TractoPL.data.io import copy2nii, move2nii, parse_filename
import ants
import json
from pprint import pprint
import nibabel as nib
import pathlib
import os

class BIDSFile:
    def __init__(self, artifact, subject=None):
        self.artifact = artifact
        self.path = artifact.path
        self.filename = os.path.basename(self.path)
        
        self.scope = 'derivatives' if 'derivatives' in self.path else 'raw'
        
        if self.scope == 'derivatives':
            path_parts = self.path.split(os.sep)
            try:
                deriv_idx = path_parts.index('derivatives')
                if len(path_parts) > deriv_idx + 1:
                    self.pipeline = path_parts[deriv_idx + 1]
                else:
                    self.pipeline = None
            except ValueError:
                self.pipeline = None
        else:
            self.pipeline = None
            
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
        Returns all entities including BIDS entities, custom entities, scope and pipeline.
        """
        # Start with the entities from the artifact
        entities = {}
        if hasattr(self.artifact, 'entities'):
            entities.update(self.artifact.entities)
            
        # Add scope and pipeline
        entities['scope'] = self.scope
        entities['pipeline'] = self.pipeline
        
        # Add extra entities from filename parsing
        entities.update(self.extra_entities)
        
        return entities

    def get_subject(self):
        """
        Return the Subject object associated with this file.
        """
        if self.subject_obj is not None:
            return self.subject_obj
        else:
            sub_id = self.get_full_entities().get('sub')
            if sub_id:
                return Subject(sub_id)
            else:
                raise ValueError("No subject ID found for this file.")


class Subject:
    def __init__(self, sub_id, db_root='/home/ndecaux/Data/actidep_bids'):
        self.db_root = db_root
        self.sub_id = sub_id
        self.bids_id = f'sub-{sub_id}'
        # Initialize BIDSLayout with ancpbids
        self.layout = BIDSLayout(self.db_root)
        # Add root attribute for backward compatibility with pybids
        self.layout.root = self.layout.dataset_path()
        self.dicom_folder = opj(
            self.db_root, self.bids_id, 'sourcedata', f'{self.bids_id}_dicoms')

    def get(self, **kwargs):
        """
        Recover files that match the given criteria from the BIDS dataset.
        Using ancpbids.BIDSLayout.get() with post-filtering for custom entities.
        """
        # Refresh layout to ensure we have the latest data
        self.layout = BIDSLayout(self.db_root)
        self.layout.root = self.layout.dataset_path()  # Backward compatibility
        
        # Basic ancpbids query parameters
        query_params = {'subject': self.sub_id}
        
        # Sort criteria between ancpbids native params and custom entities
        custom_entities = {}
        excluded_entities = {}
        absent_entities = []
        
        # Handle datatype specially as it's not directly supported by ancpbids.layout.get
        if 'datatype' in kwargs:
            datatype = kwargs.pop('datatype')
            if isinstance(datatype, str) and not datatype.startswith('!'):
                # We'll filter by datatype in post-processing
                custom_entities['datatype'] = datatype
            else:
                # For negation or None, handle in post-filtering as usual
                custom_entities['datatype'] = datatype
        
        # Handle pipeline specially, mapping to scope if needed
        if 'pipeline' in kwargs:
            pipeline_val = kwargs.pop('pipeline')
            if isinstance(pipeline_val, str) and not pipeline_val.startswith('!'):
                query_params['scope'] = pipeline_val
            else:
                # For negation or None, handle in post-filtering
                custom_entities['pipeline'] = pipeline_val
        
        # Process other parameters
        for key, value in kwargs.items():
            # Special parameters for ancpbids.get()
            if key in ['suffix', 'extension', 'scope', 'target', 'return_type']:
                if isinstance(value, str) and value.startswith('!'):
                    excluded_entities[key] = value[1:]
                elif value is None:
                    absent_entities.append(key)
                else:
                    query_params[key] = value
            else:
                # Custom entities for post-filtering
                custom_entities[key] = value
        
        # Get initial list of artifacts from ancpbids
        artifacts = self.layout.get(**query_params)
        
        # Post-filter for custom entities, negations, and absent entities
        files = []
        for artifact in artifacts:
            actidep_file = BIDSFile(artifact, self)
            full_entities = actidep_file.get_full_entities()
            
            # Check if all custom entity criteria are satisfied
            match = True
            
            # Check custom entities
            for entity, value in custom_entities.items():
                if value is None:
                    # Entity should not exist or be None
                    if entity in full_entities and full_entities[entity] is not None:
                        match = False
                        break
                elif isinstance(value, str) and value.startswith('!'):
                    # Entity should not have specific value
                    if entity in full_entities and full_entities[entity] == value[1:]:
                        match = False
                        break
                else:
                    # Entity should have specific value
                    if entity not in full_entities or full_entities[entity] != value:
                        match = False
                        break
            
            # Check excluded entities (those with ! prefix)
            for entity, excluded_value in excluded_entities.items():
                if entity in full_entities and full_entities[entity] == excluded_value:
                    match = False
                    break
            
            # Check absent entities
            for entity in absent_entities:
                if entity in full_entities and full_entities[entity] is not None:
                    match = False
                    break
            
            if match:
                files.append(actidep_file)
        
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

    def build_path(self, suffix, pipeline=None, scope=None, original_name=None, is_dir=False, **kwargs):
        """
        Construire le chemin d'un fichier dans le layout à partir d'entités et d'un pipeline donnés.
        """
        entities = {'suffix': suffix}

        for k, v in kwargs.items():
            entities[k] = v

        root_dir = self.layout.dataset_path()  # Use dataset_path instead of root
        parent_dir = entities.pop(
            'datatype') if 'datatype' in entities else entities['suffix']
        if pipeline is None:
            target_dir = opj(root_dir, self.bids_id, parent_dir)
        else:
            if pipeline not in self.get_pipelines():
                create_pipeline_description(pipeline, self.layout)

            target_dir = opj(root_dir, 'derivatives',
                             pipeline, self.bids_id, parent_dir)

        fileradix = f'{self.bids_id}'

        if original_name is not None and 'extension' not in entities:
            extension = '.'+'.'.join(original_name.split('.')[1:])
            if extension == '.nrrd':
                extension = '.nii.gz'
        else:
            is_dir = entities.get('is_dir') or is_dir
            
            if 'extension' in entities:
                extension = '.' + entities['extension'].lstrip('.')
            else:
                extension = '' if is_dir else '.nii.gz'
                print(
                    f"Warning: No extension provided for {entities}, assuming {extension}")

        remaining_entities = {k: v for k, v in entities.items() if k not in [
            'suffix', 'subject', 'extension']}

        for k, v in sorted(remaining_entities.items()):
            fileradix += f'_{k}-{v}'
        path = opj(target_dir, fileradix+"_"+entities['suffix']+extension)

        return path

    def get_pipelines(self):
        """
        Return the list of pipelines that have been run on this subject.
        """
        pipelines = set()
        for artifact in self.layout.get(subject=self.sub_id, scope='derivatives'):
            path_parts = artifact.path.split(os.sep)
            try:
                deriv_idx = path_parts.index('derivatives')
                if len(path_parts) > deriv_idx + 1:
                    pipelines.add(path_parts[deriv_idx + 1])
            except ValueError:
                continue
        return list(pipelines)

    def write_object(self, obj, suffix, pipeline=None, scope=None, original_name=None, **kwargs):
        """
        Write an object to a file in the layout.
        """
        path = self.build_path(suffix, pipeline, scope,
                               original_name, **kwargs)
        
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

        return path

    def get_entity_values(self):
        """
        Get all entities that exist for this subject and their possible values.
        """
        entity_values = {}
        
        # Get all artifacts for this subject
        artifacts = self.layout.get(subject=self.sub_id)
        
        # Exclude .dcm files
        artifacts = [art for art in artifacts if not art.path.endswith('.dcm')]
        
        for artifact in artifacts:
            # Create an BIDSFile to get full entities
            actidep_file = BIDSFile(artifact, self)
            entities = actidep_file.get_full_entities()
            
            # Process each entity
            for entity, value in entities.items():
                if entity == 'subject':  # Skip subject as it's constant
                    continue
                
                # Initialize the list for this entity if it doesn't exist
                if entity not in entity_values:
                    entity_values[entity] = []
                
                # Add the value if it's not already in the list
                if value not in entity_values[entity]:
                    entity_values[entity].append(value)
        
        # Sort values for better readability
        for entity in entity_values:
            try:
                entity_values[entity].sort()
            except TypeError:
                # Handle cases where values might be of different types
                entity_values[entity] = [str(v) for v in entity_values[entity] if v is not None]
                entity_values[entity].sort()
            
        return entity_values


class Dataset:
    def __init__(self, db_root='/home/ndecaux/Data/actidep_bids'):
        __abstract__ = True
        self.db_root = db_root
        # Initialize BIDSLayout with ancpbids
        self.layout = BIDSLayout(self.db_root)
        # Add root attribute for backward compatibility with pybids
        self.layout.root = self.layout.dataset_path()
        
        # Get subject IDs from the layout
        self.subject_ids = self.layout.get_subjects()

    def get(self, sub_id, **kwargs):
        """
        Get files for a specific subject with given criteria.
        """
        subject = Subject(sub_id, self.db_root)
        return subject.get(**kwargs)

    def get_subject(self, sub_id):
        """
        Return the Subject object associated with this ID.
        """
        if sub_id in self.subject_ids:
            return Subject(sub_id, self.db_root)
        else:
            raise ValueError(f"Subject {sub_id} not found in the dataset.")
            
    def get_subjects(self):
        """
        Return the list of subject IDs in the dataset.
        """
        return self.layout.get_subjects()

if __name__ == "__main__":
    ds = Dataset('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
    print(ds.get_subjects())