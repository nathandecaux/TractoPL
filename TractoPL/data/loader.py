import os
import json
import pathlib
import nibabel as nib
import SimpleITK as sitk
import re
from os.path import join as opj
from TractoPL.data.io import copy2nii, move2nii, parse_filename
import pandas as pd
import pickle
from TractoPL.data.vtk_loader import load_vtk, save_vtk
from dipy.io.streamline import load_trk, save_trk, load_tck, save_tck
from dipy.io.streamline import load_tractogram, save_tractogram, StatefulTractogram, Space
from dipy.tracking.streamline import Streamlines
from concurrent.futures import ThreadPoolExecutor

_SENTINEL = object()  # Défini au niveau du module

def _resolve_db_root(db_root):
    """Resolve the dataset root from an argument or the public environment variable."""
    resolved_root = db_root or os.environ.get("TRACTOPL_DATASET_ROOT")
    if not resolved_root:
        raise ValueError(
            "db_root is required; pass it explicitly or set TRACTOPL_DATASET_ROOT"
        )
    return os.path.abspath(os.path.expanduser(os.fspath(resolved_root)))

class LayoutProxy:
    """
    Une classe proxy qui simule l'interface d'un layout PyBIDS.
    """
    def __init__(self, root, parent_obj=None):
        self.root = root
        self._parent_obj = parent_obj
    
    def get_subjects(self):
        """Retourner la liste des sujets depuis l'objet parent."""
        if hasattr(self._parent_obj, 'get_subjects'):
            return self._parent_obj.get_subjects()
        elif hasattr(self._parent_obj, 'subject_ids'):
            return self._parent_obj.subject_ids
        else:
            return []
    
    def get(self, **kwargs):
        """Déléguer la recherche à l'objet parent."""
        if hasattr(self._parent_obj, 'get_global'):
            return self._parent_obj.get_global(**kwargs)
        elif hasattr(self._parent_obj, 'get'):
            return self._parent_obj.get(**kwargs)
        else:
            return []

class BIDSFile:
    """
    Représentation minimale d'un fichier BIDS (allégée).
    """
    def __init__(self, path, row=None):
        self.path = path
        #Change into full path if not already
        if not os.path.isabs(self.path):
            self.path = os.path.abspath(self.path)
        self.filename = os.path.basename(path)
        if row is not None:
            # Utiliser directement les informations du DataFrame en filtrant valeurs non pertinentes
            def _is_nan(v):
                try:
                    return v != v  # NaN != NaN
                except Exception:
                    return False
            self.entities = {k: v for k, v in row.items()
                             if k not in ['path'] and v is not None and not _is_nan(v)}
        else:
            # Fallback: parsing basique
            self.entities = parse_filename(self.filename)
            if '.' in self.filename:
                self.entities['extension'] = self.filename[self.filename.find('.'):]
            match_sub = re.search(r'/sub-([A-Za-z0-9]+)/', path)
            if match_sub:
                self.entities['subject'] = match_sub.group(1)
            match_ses = re.search(r'/ses-([A-Za-z0-9]+)/', path)
            if match_ses:
                self.entities['session'] = match_ses.group(1)
        # Drapeaux dérivés
        self.entities['derivative'] = 'derivatives' in path
        if 'suffix' not in self.entities:
            stem = self.filename.split('.')[0]
            last = stem.split('_')[-1]
            if '-' not in last:
                self.entities['suffix'] = last

    def __getattr__(self, item):
        """Permet l'accès direct (file.model, file.extension, etc.) comme dans l'ancienne implémentation.
        Déclenché uniquement si l'attribut n'existe pas déjà sur l'objet.
        """
        if 'entities' in self.__dict__ and item in self.entities:
            return self.entities[item]
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{item}'")

    @property
    def extension(self):  # compat explicite si code externe attend une propriété
        return self.entities.get('extension')

    @property
    def subject(self):
        """Accès direct à l'identifiant sujet (pour compatibilité d'usage)."""
        return self.entities.get('subject')

    @property
    def session(self):
        """Accès direct à l'identifiant session si présent."""
        return self.entities.get('session')

    def get_full_entities(self):
        return dict(self.entities)

    def get_entities(self):
        return {k: v for k, v in self.entities.items() if k not in ['path', 'derivative']}
    def copy(self, dest):
        return copy2nii(self.path, dest)

    def move(self, dest):
        return move2nii(self.path, dest)

    def __eq__(self, other):
        return self.path == (other.path if isinstance(other, BIDSFile) else other)

    def __hash__(self):
        return hash(self.path)

    def __str__(self):
        return self.path
    
    @property
    def age(self):
        try:
            return os.path.getmtime(self.path)
        except Exception:
            return None
    @property
    def df(self):
        """Retourner un DataFrame pandas à une seule ligne avec les entités et propriétés."""
        data = dict(self.entities)
        
        for attr in ['path', 'age']:
            val = getattr(self, attr, None)
            if val is not None:
                data[attr] = val
        return pd.DataFrame([data])
    
    def update_entities(self, **new_entities):
        """Mettre à jour les entités du fichier, et recalculer le chemin si nécessaire et renommer le fichier."""
        # 1. Mettre à jour les entités
        for key, value in new_entities.items():
            if value is None:
                # Supprimer l'entité si la valeur est None
                self.entities.pop(key, None)
            else:
                self.entities[key] = value
        
        # 2. Reconstruire le nom de fichier basé sur les entités
        # Extraire le sujet depuis le chemin ou les entités
        subject = self.entities.get('subject')
        if not subject:
            match_sub = re.search(r'/sub-([A-Za-z0-9]+)/', self.path)
            if match_sub:
                subject = match_sub.group(1)
        
        if not subject:
            raise ValueError("Impossible de déterminer le sujet pour reconstruire le chemin")
        
        # Extraire les informations de base du chemin actuel
        session = self.entities.get('session')
        datatype = self.entities.get('datatype')
        pipeline = self.entities.get('pipeline')
        suffix = self.entities.get('suffix')
        extension = self.entities.get('extension', '')
        derivative = self.entities.get('derivative', False)
        
        # Déterminer le répertoire de base
        if derivative and pipeline:
            base_dir = opj(os.path.dirname(self.path).split('/derivatives/')[0], 
                          'derivatives', pipeline, f'sub-{subject}')
        else:
            base_dir = opj(os.path.dirname(self.path).split(f'/sub-{subject}')[0], 
                          f'sub-{subject}')
        
        # Ajouter session si présente
        if session:
            base_dir = opj(base_dir, f'ses-{session}')
        
        # Ajouter datatype
        if datatype:
            base_dir = opj(base_dir, datatype)
        
        # Construire le nouveau nom de fichier
        name_parts = [f'sub-{subject}']
        if session:
            name_parts.append(f'ses-{session}')
        
        # Ajouter les entités (sauf les réservées) triées alphabétiquement
        reserved = {'subject', 'session', 'datatype', 'pipeline', 'extension', 
                   'suffix', 'derivative', 'path'}
        entity_parts = []
        for key in sorted(self.entities.keys()):
            if key not in reserved and '_' not in key:
                value = self.entities[key]
                if value is not None:
                    try:
                        # Vérifier si ce n'est pas NaN
                        if value == value:
                            entity_parts.append(f'{key}-{value}')
                    except Exception:
                        entity_parts.append(f'{key}-{value}')
        
        name_parts.extend(entity_parts)
        
        # Ajouter le suffix
        if suffix:
            name_parts.append(suffix)
        
        # Construire le nom complet
        new_filename = '_'.join(name_parts) + extension
        new_path = opj(base_dir, new_filename)
        
        # 3. Renommer le fichier si le chemin a changé
        if new_path != self.path:
            # Créer le répertoire si nécessaire
            pathlib.Path(os.path.dirname(new_path)).mkdir(parents=True, exist_ok=True)
            
            # Renommer le fichier
            if os.path.exists(self.path):
                os.rename(self.path, new_path)
            
            # Mettre à jour le chemin interne
            self.path = new_path
            self.filename = os.path.basename(new_path)
        
        return self.path
    
class Subject:
    """
    Sujet léger basé uniquement sur le DataFrame global de Dataset.
    """
    def __init__(self, sub_id, db_root=None, layout=None, parent_actidep=None):
        self.sub_id = sub_id
        self.db_root = _resolve_db_root(db_root)
        self.bids_id = f"sub-{sub_id}"
        self.layout = layout or LayoutProxy(db_root, self)
        self.parent_actidep = parent_actidep
        self.dicom_folder = opj(self.db_root, self.bids_id, 'sourcedata', f'{self.bids_id}_dicoms')

    def get(self, to_dataframe=False, **kwargs):
        # Gestion du scope avant filtrage
        scope = kwargs.pop('scope', None)
        if self.parent_actidep is None:
            self.parent_actidep = Dataset(self.db_root)
        df = self.parent_actidep.build_dataframe()
        #if kwargs entities are not in df columns, add this column with all None values to avoid KeyError
        for k in kwargs.keys():
            if k not in df.columns:
                df[k] = None

        if df.empty:
            return pd.DataFrame() if to_dataframe else []
        sub_df = df[df['subject'] == self.sub_id]
        if sub_df.empty:
            return pd.DataFrame() if to_dataframe else []
        # Filtrage scope=raw => uniquement non-derivatives
        if scope == 'raw':
            sub_df = sub_df[sub_df['derivative'] != True]
            if sub_df.empty:
                return pd.DataFrame() if to_dataframe else []
        if 'subject' in kwargs:
            kwargs.pop('subject', None)
        filtered = self.parent_actidep._apply_filters(sub_df, kwargs)
        if filtered.empty:
            return pd.DataFrame() if to_dataframe else []
        if to_dataframe:
            return filtered.reset_index(drop=True)
        return [BIDSFile(row['path'], row) for _, row in filtered.iterrows()]

    def get_entity_values(self):
        if self.parent_actidep is None:
            return {}
        df = self.parent_actidep.build_dataframe()
        sub_df = df[df['subject'] == self.sub_id]
        out = {}
        for col in sub_df.columns:
            if col in ['path', 'subject']:
                continue
            vals = [v for v in sub_df[col].dropna().unique().tolist() if v != '']
            if vals:
                out[col] = sorted(vals)
        return out

    def get_unique(self, **kwargs):
        """
        Récupérer le fichier unique qui correspond aux critères donnés.
        """
        files = self.get(**kwargs)
        if len(files) == 0:
            raise ValueError(f"Aucun fichier trouvé pour {kwargs}")
        if len(files) > 1:
            raise ValueError(f"Plusieurs fichiers trouvés pour {kwargs}")
        return files[0]

    def build_path(self, suffix, datatype=None, pipeline=None, session=None,
                   extension='.nii.gz', original_name=None, is_dir=False, **entities):
        """Construire un chemin BIDS (raw ou derivative) avec:
        - entités triées alphabétiquement
        - extension déduite de original_name si extension reste par défaut
        - exclusion des valeurs None / NaN / 'nan'
        - suppression des clés contenant un underscore
        - support de is_dir (pas d'extension, création dossier)
        """
        # 1. Nettoyage des entités
        cleaned = {}
        for k, v in list(entities.items()):
            if v is None:
                continue
            try:
                if v != v:  # NaN
                    continue
            except Exception:
                pass
            if isinstance(v, str) and v.strip().lower() == 'nan':
                continue
            cleaned[k] = v

        # 2. Champs réservés
        reserved = {'subject', 'derivative', 'datatype', 'pipeline', 'extension', 'suffix', 'session', 'is_dir'}

        # Propagation
        if 'datatype' in cleaned and datatype is None:
            datatype = cleaned['datatype']
        if 'pipeline' in cleaned and pipeline is None:
            pipeline = cleaned['pipeline']
        if 'session' in cleaned and session is None:
            session = cleaned['session']
        if 'suffix' in cleaned and suffix is None:
            suffix = cleaned['suffix']
        if 'is_dir' in cleaned:
            try:
                is_dir = bool(cleaned['is_dir'])
            except Exception:
                pass

        # 3. Filtrage entités pour le nom
        entity_items = {k: v for k, v in cleaned.items() if k not in reserved and '_' not in k}

        # 4. Extension
        if is_dir:
            resolved_ext = ''
        else:
            if extension is None:
                resolved_ext = ''
            else:
                if original_name and (extension in ('.nii.gz', '.nii', '.gz') or extension in ('', None)):
                    parts = original_name.split('.')
                    if parts[-1]=='nrrd':
                        resolved_ext = '.nii.gz'
                    else:
                        if len(parts) > 2 and parts[-2].lower() == 'nii' and parts[-1].lower() == 'gz':
                            resolved_ext = '.nii.gz'
                        elif len(parts) > 1:
                            resolved_ext = '.' + parts[-1]
                        
                        else:
                            resolved_ext = extension if (extension.startswith('.') if extension else False) else (f'.{extension}' if extension else '')
                else:
                    resolved_ext = extension if extension.startswith('.') else f'.{extension}' if extension else ''

        # 5. Datatype
        datatype = datatype or suffix
        if not datatype:
            raise ValueError("datatype ou suffix requis pour construire le chemin")

        # 6. Base
        if pipeline:
            base = opj(self.db_root, 'derivatives', pipeline, self.bids_id)
        else:
            base = opj(self.db_root, self.bids_id)
        if session:
            base = opj(base, f"ses-{session}")
        base = opj(base, datatype)
        pathlib.Path(base).mkdir(parents=True, exist_ok=True)

        # 7. Nom
        name_parts = [self.bids_id]
        if session:
            name_parts.append(f"ses-{session}")
        for k in sorted(entity_items.keys()):
            name_parts.append(f"{k}-{entity_items[k]}")
        name = '_'.join(name_parts) + f"_{suffix}{resolved_ext}"
        full_path = opj(base, name)
        if is_dir:
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        return full_path

    def write_object(self, obj, suffix, **kwargs):
        path = self.build_path(suffix, **kwargs)
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if isinstance(obj, sitk.Image):
            sitk.WriteImage(obj, path)
        elif isinstance(obj, nib.Nifti1Image):
            nib.save(obj, path)
        elif isinstance(obj, dict):
            with open(path, 'w') as f:
                json.dump(obj, f)
        elif isinstance(obj, str) and os.path.exists(obj):
            copy2nii(obj, path)
        else:
            raise ValueError("Type non supporté pour write_object.")
        return path

class Dataset:
    """
    Gestion simplifiée de la base BIDS via un DataFrame unique.
    """
    def __init__(self, db_root=None,restore=False):
        self.db_root = _resolve_db_root(db_root)
        self.layout = LayoutProxy(db_root, self)
        self._df = None
        self._subjects_cache = {}
        self.subject_ids = []  # sera peuplé par build_dataframe()
        # Pré-construction pour exposer immédiatement les sujets
        self._NAN = float('nan')
        # Colonnes fixes : une entité du nom de fichier portant l'un de ces noms est ignorée
        self._RESERVED = frozenset(('path', 'subject', 'session', 'datatype', 'pipeline',
                            'derivative', 'extension', 'suffix'))
        if restore:
            if restore and os.path.exists(restore if isinstance(restore, str) else '/tmp/actidep.pkl'):
                self.restore(restore)
            else:
                print(Warning("Fichier de restauration introuvable, initialisation normale."))
                self.build_dataframe()
                self.save(restore if isinstance(restore, str) else '/tmp/actidep.pkl')

        else:
            try:
                self.build_dataframe()
            except Exception:
                # On ignore pour laisser une initialisation paresseuse si ça échoue
                pass





    def _list_dir(self,path):
        """(fichiers 'sub-*' [(nom, chemin)], sous-dossiers [(nom, chemin)]) — comme os.walk."""
        files, dirs = [], []
        try:
            it = os.scandir(path)
        except OSError:
            return None
        with it:
            for e in it:
                try:
                    is_dir = e.is_dir()
                except OSError:
                    is_dir = False
                if is_dir:
                    if not e.is_symlink():
                        dirs.append((e.name, e.path))
                elif e.name[:4] == 'sub-':
                    files.append((e.name, e.path))
        return files, dirs
    
    
    def _walk_subtree(self, path, subject, session, datatype, skip_derivs, out):
        """Parcours pré-ordre ; ajoute (subject, session, datatype, fichiers) dans `out`."""
        listed = self._list_dir(path)
        if listed is None:
            return out
        files, dirs = listed
        if files:
            out.append((subject, session, datatype, files))
        for name, p in dirs:
            if 'sourcedata' in name or (skip_derivs and name == 'derivatives'):
                continue
            s, se, dt = subject, session, datatype
            h = name[:4]
            if h == 'sub-':
                s = name[4:]
            elif h == 'ses-':
                se = name[4:]
            elif s and dt is None:
                dt = name
            self._walk_subtree(p, s, se, dt, skip_derivs, out)
        return out
 
 
# --- Méthode à placer dans la classe (remplace l'ancienne) ---
# n_threads : 8 par défaut sur 2 cœurs (cpu*4, plafonné à 32) ; sur un NFS/Lustre
# qui encaisse bien la concurrence, 16 à 32 peut encore accélérer.
    def build_dataframe(self, force=False, n_threads=None):
        if self._df is not None and not force:
            return self._df
 
        db_root = self.db_root
        if not os.path.isdir(db_root):
            self._df = pd.DataFrame()
            self.subject_ids = []
            return self._df
 
        n_threads = n_threads or getattr(self, 'n_threads', None) or min(32, (os.cpu_count() or 1) * 4)
 
        segments = []
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
 
            def plan(path, pipeline, derivative, skip_derivs):
                listed = self._list_dir(path)
                if listed is None:
                    return
                files, dirs = listed
                if files:
                    segments.append((pipeline, derivative, [(None, None, None, files)]))
                for name, p in dirs:
                    if 'sourcedata' in name or (skip_derivs and name == 'derivatives'):
                        continue
                    s = se = None
                    h = name[:4]
                    if h == 'sub-':
                        s = name[4:]
                    elif h == 'ses-':
                        se = name[4:]
                    segments.append((pipeline, derivative,
                                     pool.submit(self._walk_subtree, p, s, se, None, skip_derivs, [])))
 
            if not ('sourcedata' in db_root or '/derivatives/' in db_root
                    or db_root.endswith('/derivatives')):
                plan(db_root, None, False, True)
 
            derivatives_root = os.path.join(db_root, 'derivatives')
            if 'sourcedata' not in derivatives_root and os.path.isdir(derivatives_root):
                listed = self._list_dir(derivatives_root)
                if listed is not None:
                    for name, p in listed[1]:
                        if 'sourcedata' not in name:
                            plan(p, name, True, False)
 
            c_path, c_subj, c_ses, c_dt = [], [], [], []
            c_pipe, c_deriv, c_ext, c_suf = [], [], [], []
            ent_cols = {}
            a_path, a_subj, a_ses, a_dt = c_path.append, c_subj.append, c_ses.append, c_dt.append
            a_pipe, a_deriv, a_ext, a_suf = c_pipe.append, c_deriv.append, c_ext.append, c_suf.append
            reserved = self._RESERVED
            nan = self._NAN
            i = 0
 
            for pipeline, derivative, seg in segments:
                if not isinstance(seg, list):
                    seg = seg.result()
                for subject, session, datatype, files in seg:
                    for name, path in files:
                        dot = name.find('.')
                        if dot >= 0:
                            ext = name[dot:]
                            parts = name[:dot].split('_')
                        else:
                            ext = ''
                            parts = name.split('_')
                        subj = subject
                        ses = session
                        suffix_ent = None
                        for p in parts:
                            head = p[:4]
                            if head == 'sub-':
                                subj = p[4:]
                            elif head == 'ses-':
                                if not ses:
                                    ses = p[4:]
                            else:
                                k, sep, v = p.partition('-')
                                if sep:
                                    if k in reserved:
                                        if k == 'suffix':
                                            suffix_ent = v
                                        continue
                                    col = ent_cols.get(k)
                                    if col is None:
                                        col = ent_cols[k] = [nan] * i
                                    else:
                                        n = len(col)
                                        if n > i:
                                            col[i] = v
                                            continue
                                        if n < i:
                                            col.extend([nan] * (i - n))
                                    col.append(v)
                        last = parts[-1]
                        if '-' in last and suffix_ent is not None:
                            last = suffix_ent
                        a_path(path); a_subj(subj); a_ses(ses); a_dt(datatype)
                        a_pipe(pipeline); a_deriv(derivative); a_ext(ext); a_suf(last)
                        i += 1
 
        n = i
        if n == 0:
            self.subject_ids = []
            self._df = pd.DataFrame()
            return self._df
 
        data = {'path': c_path, 'subject': c_subj, 'session': c_ses, 'datatype': c_dt,
                'pipeline': c_pipe, 'derivative': c_deriv, 'extension': c_ext, 'suffix': c_suf}
        for k, col in ent_cols.items():
            if len(col) < n:
                col.extend([nan] * (n - len(col)))
            data[k] = col
 
        self._df = pd.DataFrame(data)
        self.subject_ids = sorted(set(c_subj))
        return self._df
    
    def _apply_filters(self, df, kwargs):
        if df.empty or not kwargs:
            return df
        import numpy as np
        import fnmatch
        
        def _is_null_value(x):
            """Vérifie si une valeur (convertie en string) est nulle/vide."""
            return x in ('nan', 'None', '', 'NaN', 'NAN')
        
        mask = np.ones(len(df), dtype=bool)
        for k, v in kwargs.items():
            # Normalisation spécifique pour l'extension: accepter sans point initial
            if k == 'extension' and isinstance(v, str):
                neg = v.startswith('!')
                core = v[1:] if neg else v
                if not core.startswith('.'):
                    core = '.' + core
                v = ('!' if neg else '') + core
            if k not in df.columns:
                if isinstance(v, str) and v.startswith('!'):
                    continue
                return df.iloc[0:0]
            # pandas >= 3 keeps NaN through astype(str); map it to 'nan' as pandas 2 did
            col = df[k].astype(object).where(df[k].notna(), 'nan').astype(str)
            if isinstance(v, str) and v.startswith('!'):
                target = v[1:]
                # Support wildcard * pour négation
                if '*' in target:
                    mask &= ~col.apply(lambda x: fnmatch.fnmatch(x, target) and not _is_null_value(x))
                else:
                    mask &= (col != target)
            elif v is None:
                mask &= (~df[k].notna())
            elif isinstance(v, (list, tuple)):
                # Support pour les listes : l'entité doit matcher un des éléments
                list_mask = np.zeros(len(df), dtype=bool)
                for item in v:
                    item_str = str(item)
                    # Support wildcard * dans les listes (exclut les valeurs nulles)
                    if '*' in item_str:
                        list_mask |= col.apply(lambda x, pat=item_str: fnmatch.fnmatch(x, pat) and not _is_null_value(x))
                    else:
                        list_mask |= (col == item_str)
                mask &= list_mask
            else:
                v_str = str(v)
                # Support wildcard * pour matching (exclut les valeurs nulles)
                if '*' in v_str:
                    mask &= col.apply(lambda x, pat=v_str: fnmatch.fnmatch(x, pat) and not _is_null_value(x))
                else:
                    mask &= (col == v_str)
            if not mask.any():
                return df.iloc[0:0]
        return df[mask]

    def get(self, sub_id, to_dataframe=False, **kwargs):
        # Intercepter scope
        scope = kwargs.pop('scope', None)
        df = self.build_dataframe()
        if df.empty:
            return pd.DataFrame() if to_dataframe else []
        sdf = df[df['subject'] == sub_id]
        if sdf.empty:
            return pd.DataFrame() if to_dataframe else []
        if scope == 'raw':
            sdf = sdf[sdf['derivative'] != True]
            if sdf.empty:
                return pd.DataFrame() if to_dataframe else []
        if 'subject' in kwargs:
            kwargs.pop('subject')
        sdf = self._apply_filters(sdf, kwargs)
        if sdf.empty:
            return pd.DataFrame() if to_dataframe else []
        if to_dataframe:
            return sdf.reset_index(drop=True)
        subj = self.get_subject(sub_id)
        return [BIDSFile(r['path'], r) for _, r in sdf.iterrows()]

    def get_global(self, to_dataframe=False, **kwargs):
        # Support scope=raw pour ignorer derivatives
        scope = kwargs.pop('scope', None)
        df = self.build_dataframe()
        if df.empty:
            return pd.DataFrame() if to_dataframe else []
        if scope == 'raw':
            df = df[df['derivative'] != True]
            if df.empty:
                return pd.DataFrame() if to_dataframe else []
        df = self._apply_filters(df, kwargs)
        if df.empty:
            return pd.DataFrame() if to_dataframe else []
        if to_dataframe:
            return df.reset_index(drop=True)
        # cache sujets
        cache = {}
        out = []
        for _, r in df.iterrows():
            sid = r['subject']
            if sid not in cache:
                cache[sid] = self.get_subject(sid)
            out.append(BIDSFile(r['path'], r))
        return out

    def get_subject(self, sub_id):
        if sub_id in self._subjects_cache:
            return self._subjects_cache[sub_id]
        self.build_dataframe()
        if self.subject_ids and sub_id not in self.subject_ids:
            raise ValueError(f"Sujet {sub_id} introuvable.")
        subj = Subject(sub_id, self.db_root, layout=self.layout, parent_actidep=self)
        self._subjects_cache[sub_id] = subj
        return subj

    def get_subjects(self):
        return self.subject_ids

    def describe(self):
        """Return a compact, serializable summary of the indexed BIDS dataset."""
        dataframe = self.build_dataframe()
        derivatives = []
        if not dataframe.empty and 'pipeline' in dataframe.columns:
            derivatives = sorted(
                pipeline
                for pipeline in dataframe.loc[dataframe['derivative'] == True, 'pipeline'].dropna().unique()
            )
        return {
            'root': self.db_root,
            'subjects': list(self.subject_ids),
            'file_count': len(dataframe),
            'derivatives': derivatives,
        }
    
    def to_dataframe(self):
        return self.build_dataframe()

    def refresh(self):
        self.build_dataframe(force=True)
        self._subjects_cache.clear()
        return True

    def save(self,filepath='/tmp/actidep.pkl'):
        print(f'Saving Dataset state to {filepath}')
        with open(filepath,'wb') as f:
            pickle.dump(self,f)
        print('Save complete')
        return filepath
    
    def restore(self,filepath='/tmp/actidep.pkl'):
        if filepath == True:
            filepath='/tmp/actidep.pkl'
        if not os.path.exists(filepath):
            self.save(filepath)
        with open(filepath,'rb') as f:
            obj=pickle.load(f)
        self.db_root=obj.db_root
        self.layout=obj.layout
        self._df=obj._df
        self._subjects_cache=obj._subjects_cache
        self.subject_ids=obj.subject_ids
        return True
# Test simplifié
def test():
    """
    Fonction de test pour vérifier le fonctionnement de la classe Dataset et Subject.
    """
    import sys
    from TractoPL.data.loader import Dataset, Subject

    test_path = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors/sub-03011/dwi/sub-03011_desc-preproc_model-MCM_dwi.mcmx"
    # Create an instance of MyLoader
    print(f"Recherche dans {test_path}")
    ds = Dataset('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
    
    # Afficher les sujets trouvés
    print(f"Sujets trouvés: {ds.subject_ids}")
    
    # Vérifier si le chemin de test existe
    print(f"Le fichier existe: {os.path.exists(test_path)}")
    
    # Tester la récupération du fichier
    try:
        results = ds.get('03011', model='MCM', extension='mcmx', pipeline='mcm_tensors')
        print(f"Nombre de fichiers trouvés: {len(results)}")
        
        if len(results) > 0:
            test_get = results[0]
            print(f"Fichier trouvé: {test_get.path}")
            print(f"Égal au test_path: {test_get == test_path}")
            
            subject = Subject('03011', db_root='/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
            
            try:
                test_get = subject.get_unique(model='MCM', extension='mcmx', pipeline='mcm_tensors')
                print(f"get_unique a trouvé: {test_get.path}")
                print(f"Égal au test_path: {test_get == test_path}")
                
                #Build path
                test_entities = test_get.get_full_entities()
                print(f"Entités: {test_entities}")     
                test_build_path=subject.build_path(**test_entities)
                print(f"Chemin construit: {test_build_path}")
                print(f"Égal au test_path: {test_build_path == test_path}")
            except Exception as e:
                print(f"Erreur avec get_unique: {e}")
        else:
            print("Aucun fichier trouvé, vérifions le chemin de la base de données")
            # Vérifier si le dossier existe
            bids_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
            print(f"Le dossier de la base BIDS existe: {os.path.exists(bids_path)}")
            
            # Chercher tous les fichiers .mcmx dans la base
            print("Recherche de fichiers .mcmx...")
            mcmx_files = []
            for root, dirs, files in os.walk(bids_path):
                for file in files:
                    if file.endswith('.mcmx'):
                        mcmx_files.append(os.path.join(root, file))
                        
            print(f"Fichiers .mcmx trouvés: {len(mcmx_files)}")
            for f in mcmx_files[:5]:  # Afficher les 5 premiers fichiers
                print(f"  - {f}")
    except Exception as e:
        print(f"Erreur pendant l'exécution: {e}")

def load_streamlines(file_path,reference=None,space='LPS'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if space not in ['LPS', 'RAS', 'LPSMM','RASMM']:
        raise ValueError(f"Unsupported space: {space}")
    else:
        if 'LPS' in space:
            space_enum = Space.LPSMM
        elif 'RAS' in space:
            space_enum = Space.RASMM
    if file_path.endswith('.vtk'):
        streamlines,_ = load_vtk(file_path)
        return Streamlines(streamlines)
    elif file_path.endswith('.trk'):
        if reference is None:
            reference = "same"
        return load_trk(file_path, reference=reference,to_space=space_enum).streamlines
    elif file_path.endswith('.tck'):
        if reference is None:
            raise ValueError("Reference must be provided for .tck files")
        return load_tck(file_path, reference=reference,to_space=space_enum).streamlines
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_streamlines(streamlines, file_path, reference=None, space='LPS',scalar_dict=None):
    if space not in ['LPS', 'RAS', 'LPSMM','RASMM']:
        raise ValueError(f"Unsupported space: {space}")
    else:
        if 'LPS' in space:
            space_enum = Space.LPSMM
        elif 'RAS' in space:
            space_enum = Space.RASMM

    if file_path.endswith(('.trk', '.tck')):
        if reference is None and file_path.endswith('.trk'):
            reference = "same"
        if reference is None and file_path.endswith('.tck'):
            raise ValueError("Reference must be provided for .tck files")
        tractogram=StatefulTractogram(streamlines, reference=reference,space=space_enum)
    if file_path.endswith('.vtk'):
        save_vtk(streamlines, file_path,scalar_dict=scalar_dict)
    elif file_path.endswith('.trk'):
        save_trk(tractogram, file_path)
  
    elif file_path.endswith('.tck'):
        save_tck(tractogram, file_path)
      

    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

if __name__ == "__main__":
    test()