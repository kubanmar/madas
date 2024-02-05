import os
import json
import random
import logging
from typing import Callable, List
from typing import Any
from traceback import format_exc as get_tb_string

import numpy as np
import pandas as pd

from .fingerprint import Fingerprint
from .similarity import SimilarityMatrix
from .material import Material
from .utils import tqdm

class MaterialsDatabase():
    """
    A database wrapper to simplify materials data download from online repositories and study the similarity of materials based on different measures.
    Materials in the database can be accessed via material identifiers (mid).

    **Keyword arguments**

    filename: *str*
        Name of database file.

        default: 'materials_database.db'

    filepath: *str*
        Path of database file.

        default: 'data'

    key_name: *str*
        Name of unique key used in the database backend.

        default: 'mid'

    api: *madas.apis.api_core.APIClass* object or *None*
        API object that provides an interface to web databases
        Default will use the NOMAD Encyclopedia API.

        default: None

    backend: *str* or *madas.backend.backend_core.Backend* object
        Name of database backend to use or backend object.
        Default is ASEs AtomsDatabase.

        default: 'ase'

    log_mode: *str*
        Logging mode: choose between:

        `"full"`: Write to screen and log file

        `"silent"` : Write to file only

        `"stream"` : Write to screen only

        `"None"` : Do not log

        default: `"full"`

    **Methods:**
    """

    def __init__(self, 
                 filename = 'materials_database.db', 
                 filepath = 'data',
                 key_name = 'mid',
                 api = None, 
                 backend = 'ase', 
                 log_mode = "full"):
        # initialize logs
        if  log_mode is not None and log_mode.lower() != "none":
            self._init_loggers(filepath, filename.split('.db')[0], log_mode = log_mode)
        else:
            self.log, self.api_logger = None, None

        # initialization of database backend
        from madas.backend import Backend
        if backend == 'ase':
            from madas.backend import ASEBackend
            self.backend = ASEBackend(filename, filepath, key_name = key_name, log = self.api_logger)
        elif isinstance(backend, Backend):
            self.backend = backend
            self.backend.set_logger(self.api_logger)
        else:
            raise AttributeError("Ivalid argument for backend: Choose from ['ase'] or pass a Backend object.")

        # initialization of API
        if api is None:
            from .apis.NOMAD_web_API import API
            self.api = API(logger = self.api_logger)
        else:
            self.api = api
            self.api.set_logger(self.api_logger)

        # import api doc strings
        self.add_material.__func__.__doc__ += f"\n{self.api.get_calculation.__func__.__doc__}" 
        self.fill_database.__func__.__doc__ += f"\n{self.api.get_calculations_by_search.__func__.__doc__}"
        self.add_property.__func__.__doc__ += f"\n{self.api.get_property.__func__.__doc__}"

        # miscellaneous
        self._iter_index = 0
        self.query_encoder = json.JSONEncoder

    @property
    def log_file_path(self):
        handlers = list(filter(lambda x: x.get_name().split('_')[-1] == 'error', self.log.handlers))
        return handlers[0].baseFilename

    def get_property(self, 
                     mid: str, 
                     property_name: str) -> Any:
        """
        Get a property of a single material specified by name or path in Material object from the database.

        **Arguments:**

        mid: *string*
            Material id of the requested material
        
        property_name: *str*
            Name or path in Material object of the requested property

            Example:

            "code_name" --> stored in Material().data["code_name"] retrieves the code used for the DFT calculation

            "electronic_dos/dos_values" --> Material().data["electronic_dos"]["dos_values"] retrieves the DOS values of the material

        **Returns:**
        
        property: *Any* --> property value if it exists in the database

        *None* --> else
        """
        material = self.backend.get_single(mid)
        prop = self._property_from_material(material, property_name)
        return prop

    def get_properties(self, 
                       property_name: str, 
                       output_mids: bool = False,
                       show_progress: bool = True) -> List[Any]:
        """
        Get a list of properties for all materials of the database.
        
        **Arguments:**
        
        property_name: *str*
            Name or path in Material object of the requested property

            Example:

            "code_name" --> stored in Material().data["code_name"] retrieves the code used for the DFT calculation

            "electronic_dos/dos_values" --> Material().data["electronic_dos"]["dos_values"] retrieves the DOS values of the material

        **Keyword arguments:**
        
        output_mids: *bool*
            Output the list of corresponding material ids together with the properties

            default: False

        show_progress: `bool`
            Show a progress bar.

            default: `True`

        **Returns:**
        
        tuple of:

        properties: *list*
            List of all properties of materials. If a property can not be retrieved, it will be set to *None*
        
        ,
        
        [mids]: *list*
            List of material ids
        """
        properties = []
        mids = []
        for material in tqdm(self, disable = not show_progress):
            prop = self._property_from_material(material, property_name)
            properties.append(prop)
            mids.append(material.mid)
        if output_mids:
            return properties, mids
        return properties

    def get_property_dataframe(self, property_paths: List[str]) -> pd.DataFrame:
        """
        Generate a pandas DataFrame object that contains a table of properties.

        **Arguments:**

        property_paths: *List[str]*
            List of property paths that point to the respective property.

        **Returns:**

        property_dataframe: *pandas.DataFrame*
            Dataframe, where the index contains the mids and the columns the respective properties.
        """
        _data = {}
        for material in self:
            new_entry = {}
            for prop_name in property_paths:
                new_entry[prop_name] = self._property_from_material(material, prop_name)
            _data[material.mid] = new_entry
        return pd.DataFrame(data=_data).T


    def get_fingerprint(self, 
                        fp_type: str, 
                        mid: str, 
                        name: str = None,
                        force_calculate = False,
                        similarity_function: Callable = None,
                        fingerprint_kwargs: dict = {}, **kwargs) -> Fingerprint:
        """
        Get a single fingerprint of type *fp_type* for a material with id *mid*.

        **Arguments:**

        fp_type: *str*
            Type of fingerprint X, must correspond to a XFingerprint() object defined elsewhere

        mid: *str*
            Material id of the requested material

        **Keyword arguments:**

        name: *str*
            Name of the fingerprint as used in the database, if name == None: name = fp_type. 
            This parameter is used to distinguish between fingerprints in the database, i.e. it must be unique for each unique fingerprint.
            
            default: None

        force_calculate: *bool*
            Force calculation of the fingerprint even if fingerprint data was stored in the database before.

            default: False

        similarity_function: *Callable*
            Similarity function that takes two fingerprints as arguments and calculates their similarity.

            default: None

        fingerprint_kwargs: *dict*
            Additional keyword arguments that are passed to Fingerprint().__init__.

            default: {}

        Additional keyword arguments are passed to Fingerprint().calculate().        

        **Returns:**
        
        Fingerprint() object

        None if calculation of Fingerprint failed

        **Raises:**

        KeyError: No material with specified id in the database.
        """
        material = self[mid]
        fingerprint = self._get_fingerprint(material, fp_type, name, fingerprint_kwargs, force_calculate, **kwargs)
        if similarity_function is not None:
            fingerprint.set_similarity_function(similarity_function)
        return fingerprint

    def get_fingerprints(self, 
                         fp_type: str, 
                         name: str = None, 
                         fingerprint_kwargs: dict = {},
                         force_calculate = False, 
                         similarity_function: Callable = None,
                         show_progress: bool = True, **kwargs) -> List[Fingerprint]:
        """
        Get fingerprints of type *fp_type* for all materials in the database.

                **Arguments:**

        fp_type: *str*
            Type of fingerprint X, must correspond to a XFingerprint() object defined elsewhere

        **Keyword arguments:**

        name: *str*
            Name of the fingerprint as used in the database, if name == None: name = fp_type. 
            This parameter is used to distinguish between fingerprints in the database, i.e. it must be unique for each unique fingerprint.
            
            default: None

        force_calculate: *bool*
            Force calculation of the fingerprint even if fingerprint data was stored in the database before.

            default: False

        fingerprint_kwargs: *dict*
            Additional keyword arguments that are passed to Fingerprint().__init__.

            default: {}

        similarity_function: *Callable*
            Similarity function that takes two fingerprints as arguments and calculates their similarity.

            default: None

        show_progress: *bool*
            Show a progress bar during calculation.

            default: True

        Additional keyword arguments are passed to Fingerprint().calculate().        

        **Returns:**
        
        List[Fingerprint() object or None]
        """
        fingerprints = []
        for material in tqdm(self, disable = not show_progress):
            fp = self._get_fingerprint(material, fp_type, name, fingerprint_kwargs, force_calculate, **kwargs)
            if similarity_function is not None:
                fp.set_similarity_function(similarity_function)
            fingerprints.append(fp) 
        return fingerprints

    def get_similarity_matrix(self, fp_type: str | type, name: str = None, dtype = np.float64, **kwargs) -> SimilarityMatrix:
        """
        Calculate a SimilarityMatrix() object from Fingerprints of type *fp_type* from all entries of the database.
        If fingerprints for some entries can not be calculated, they will be excluded from the database.

        **Arguments:**

        fp_type: *str* or `type`
            Type of fingerprint X, must correspond to a XFingerprint() object defined elsewhere

        **Keyword arguments:**

        name: *str*
            Name of the fingerprint as used in the database, if name == None: name = fp_type. 
            This parameter is used to distinguish between fingerprints in the database, i.e. it must be unique for each unique fingerprint.
            
            default: None

        dtype: *type*
            Data type used by SimilarityMatrix to store similarities

            default: numpy.float64

        Additional keyword arguments are passed to SimilarityMatrix().calculate().
                
        **Returns:**

        SimilarityMatrix() object

        **Raises:**

        ValueError: Fingerprints could not be obtained, no matrix can be calculated
        """
        fps = self.get_fingerprints(fp_type, name = name)
        fps = [fp for fp in fps if fp is not None]
        if len(fps) < 1:
            self.log.error('No fingerprints loaded (not generated?).')
            raise ValueError('No fingerprints fitting criteria.')
        simat = SimilarityMatrix(dtype = dtype).calculate(fps, **kwargs)
        return simat

    def add_fingerprint(self, 
                        fp_type: str | type, 
                        name: str = None, 
                        show_progress: bool = True, 
                        force_calculate: bool = True, 
                        fingerprint_kwargs: dict = {}, **kwargs) -> None:
        """
        Calculate fingerprints of all materials in the database and store them.

        **Arguments:**
        
        fp_type: *str* or *type*
            Type of fingerprint X, must correspond to a XFingerprint() object

            Fingerprint types can also be added as `type`s, then the data will be stored using
            the function `Fingerprint().serialize()`. Deserialized fingerprints from the database
            will be generic `Fingerprint` objects and the similarity function is not set.

        **Keyword arguments:**

        name: *str*
            Name of the fingerprint as used in the database, if name == None: name = fp_type. 
            This parameter is used to distinguish between fingerprints in the database, i.e. it must be unique for each unique fingerprint.
            
            default: None

        force_calculate: *bool*
            Force calculation of the fingerprint even if fingerprint data was stored in the database before.

            default: True

        fingerprint_kwargs: *dict*
            Additional keyword arguments that are passed to Fingerprint().__init__.

            default: {}

        show_progress: *bool*
            Show a progress bar during calculation.

            default: True

        Additional keyword arguments are passed to Fingerprint().calculate().        
        """
        if name is None:
            if isinstance(fp_type, type):
                name = fp_type.__name__
            else:
                name = str(fp_type)
        self.log.info(f'Generating {name} fingerprints...')
        fingerprints = self.get_fingerprints(fp_type, name = name, fingerprint_kwargs = fingerprint_kwargs, force_calculate = force_calculate, show_progress = show_progress, **kwargs)
        self.log.info(f'Writing {name} fingerprints to database...')
        # remove entries where calculation failed 
        fingerprints = list(filter(lambda x: x is not None, fingerprints))
        mid_list = [fingerprint.mid for fingerprint in fingerprints]
        if isinstance(fp_type, type):
            dictionary_list = [{fingerprint.name : fingerprint.serialize()} for fingerprint in fingerprints]
        else:
            dictionary_list = [{fingerprint.name : fingerprint.get_data_json()} for fingerprint in fingerprints]
        self.update_entries(mid_list, dictionary_list)
        self.log.info(f'Finished generation of {name} fingerprints.')
        self._update_metadata({'fingerprints' : [name]}) #TODO Adapt new metadata schema

    def add_fingerprints(self, 
                         fp_types: List[str], 
                         names: List[str] = [None], 
                         show_progress = False,
                         fingerprint_kwargs_list: List[dict] = None,
                         fingerprint_calculate_kwargs_list: List[dict] = None,
                         force_calculate = False):
        """
        Calculate several fingerprints of each material in the database and store them.
        Because storing data can be slow for large database, this method should be preferred over adding fingerprints one by one.

        **Arguments:**
        
        fp_types: *List[str]*
            List of types of fingerprint X, must correspond to a XFingerprint() object

        **Keyword arguments:**

        names: *List[str]*
            Names of the fingerprint as used in the database, if name == None: name = fp_type for any name in the list. 
            This parameter is used to distinguish between fingerprints in the database, i.e. it must be unique for each unique fingerprint.
            
            default: None

        force_calculate: *bool*
            Force calculation of the fingerprint even if fingerprint data was stored in the database before.

            default: False

        fingerprint_kwargs_list: *List[dict]*
            List of additional keyword arguments that are passed to each Fingerprint().__init__.

            default: None

        fingerprint_calculate_kwargs_list: *List[dict]*
            List of additional keyword arguments that are passed to each Fingerprint().calculate().

            default: None

        show_progress: *bool*
            Show a progress bar during calculation.

            default: True

        **Raises:**

        AssertionError: Number of kwargs passed to the __init__() and calculate() functions of all fingerprints are inconsitent. Thus, it is ambiguous which parameters correspond to which fingerprint.
        """
        while len(fp_types) < len(names):
            names.append(None)
        for idx, fp_type in enumerate(fp_types):
            if names[idx] is None:
                if isinstance(fp_type, type):
                    names[idx] = fp_type.__name__
                else:
                    names[idx] = str(fp_type)
        if fingerprint_kwargs_list is None:
            fingerprint_kwargs_list = [{} for _ in fp_types]
        if fingerprint_calculate_kwargs_list is None:
            fingerprint_calculate_kwargs_list = [{} for _ in fp_types]
        assert len(fp_types) == len(fingerprint_kwargs_list) == len(fingerprint_calculate_kwargs_list), "Inconsistent number of kwargs! Please provide arguments for each fingerprint type or None."   
        mid_list = []
        dictionary_list = []
        self.log.info(f"Generating {names} fingerprints...")
        for material in tqdm(self, disable = not show_progress):
            fps_data = {}
            for fp_type, name, fp_kwargs, calc_kwargs in zip(fp_types, names, fingerprint_kwargs_list, fingerprint_calculate_kwargs_list):
                new_fp = self._get_fingerprint(material, fp_type, name, fp_kwargs, force_calculate, **calc_kwargs)
                if new_fp is None:
                    fps_data[name] = None
                else:
                    if isinstance(fp_type, type):
                        fps_data[name] = new_fp.serialize()
                    else:                               
                        fps_data[name] = new_fp.get_data_json() if isinstance(new_fp, Fingerprint) else None
            mid_list.append(material.mid)
            dictionary_list.append(fps_data)
        self.log.info("Writing fingerprints to database...")
        self.update_entries(mid_list, dictionary_list)
        self.log.info('Finished for added fingerprints.')
        self._update_metadata({'fingerprints' : names}) #TODO Update metadata schema

    def add_material(self, *args, **kwargs) -> None:
        """
        Add a specified material to the database.
        Keyword arguments are passed to the api.
        Arguments passed to this function are used to construct the database mid.
        """
        mid = self.api._gen_mid(*args)
        if self.backend.has_entry(mid):
            self.log.info(f"Material already in db: {mid}")
        else:
            material = self.api.get_calculation(*args, **kwargs)
            self.backend.add_single(material)

    def fill_database(self, *args, repeat_query: bool = False, **kwargs):
        """
        Fills the database with all materials matching the query. Parameters depend on the API that is used. 
        To perform the query even though it has been performed before, set

        ``repeat_query = True``

        See below for the documentation of the API functions.
        """
        metadata = self.get_metadata()
        try:
            query_hash = self.api.hash_query(args)
        except TypeError:
            self.log.error(f"Query can not be hashed and therefore not logged. To avoid this, please provide a `query_hash_function` to your `API`\n Traceback: {get_tb_string()}")
            query_hash = None
        if 'search_queries' in metadata.keys() and not repeat_query:
            if query_hash in metadata['search_queries']:
                self.log.info("Query has already been performed.")
                return
        self.log.info('Retrieving data...')
        materials = list(set(self.api.get_calculations_by_search(*args, **kwargs)))
        self.log.info(f"Got data for {len(materials)} entries.")
        pop_indices = []
        for idx, material in enumerate(materials):
            if self.backend.has_entry(material.mid):
                pop_indices.append(idx)
        for idx in sorted(pop_indices, reverse=True):
            mat = materials.pop(idx)
            self.log.info(f"Material {mat.mid} already in database. Skipping.")
        self.backend.add_many(materials)
        if query_hash is not None:
            self._update_metadata({'search_queries':[query_hash]})

    def get_random(self, 
                   return_mid: bool = True) -> Material:
        """
        Returns a random material from the database.
        
        **Keyword arguments:**

        return_id: *bool*
            Return id of material instead of material

        **Returns:**

        Material id (*str*) of a random entry of the database: if return_id == True
        
        Material object of a random entry of the database: else
        """
        random_id = random.randint(1, len(self))
        material = self.backend.get_by_id(random_id)
        if return_mid:
            return material.mid
        else:
            return material

    def update_entry(self, 
                     mid: str, **kwargs) -> None:
        """
        Update a single entry of the database from a given dictionary.

        Usage:

            MaterialsDatabase().update_entry("a", key = value) --> updates parameter *key* of material (with id *mid*) with *value*
        
        **Arguments:**

        mid: *str*
            Mid of the corresponding database entry

        Additional keyword arguments are used to update the database entries.
        """
        self.backend.update_single(mid, **kwargs)


    def update_entries(self, 
                       mid_list: List[str], 
                       dictionary_list: List[dict]) -> None:
        """
        Update a list of entries in the database.

        Usage:

            MaterialsDatabase().update_entries(["a", "b"], [{"key1" : value1}, {"key2" : value2}]) 
                --> updates parameter *key1* of material "a" with *value1* and *key2* of material "b" with *value2* 
        
        **Arguments:**

        mid: *str*
            Mid of the corresponding database entry

        dictionary_list: *List[dict]*
            List of dictionaries htat contains data to update
        """
        self.backend.update_many(mid_list, dictionary_list)

    def add_property(self, mid, property_name, **kwargs):
        """
        Add a property to a spacific material of the database through the API.
        
        **Arguments:**

        mid: *str*
            Id of the material to update
            
        property_name: *str*
            Name of property for storage in the database

        Keyword arguments are passed to the API to retrieve the property.
        """
        property = self.api.get_property(**self.api.resolve_mid(mid), **kwargs)
        if property is not None:
            self.update_entry(mid, **{"data" : {property_name : property}})

    def get_metadata(self) -> dict:
        """
        Get the metadata of the database.
        """
        return self.backend.metadata

    def _property_from_material(self, material, property_name):
        try:
            prop = material.get_property_by_path(property_name)
        except KeyError:
            try:
                prop = material.get_data_by_path(property_name)
            except KeyError as e:
                self.log.error(f"No property of name: {str(e)}")
                prop = None
        return prop

    def _get_fingerprint(self, material, fp_type, name, fingerprint_kwargs, force_calculate, **kwargs):
        if name is None:
            if isinstance(fp_type, type):
                name = fp_type.__name__
            else:
                name = str(fp_type)
        try:
            if name in material.properties.keys() and not force_calculate:
                if isinstance(fp_type, type):
                    fp = fp_type.deserialize(material.properties[name])
                else:
                    fp = Fingerprint(fp_type, name=name).from_data(json.loads(material.properties[name]))
                fp.set_mid(material)
            else:
                fp = Fingerprint(fp_type, name = name, **fingerprint_kwargs).from_material(material, **kwargs)
            fp.set_name(name)
        except Exception as e:
            message = f"Could not get {name} fingerprint of type {fp_type} because of {str(e)}.\nTraceback: {get_tb_string()}"
            self.log.error(message)
            fp = None
        return fp

    def _update_metadata(self, update_dict):
        metadata = self.backend.metadata
        for key in update_dict.keys():
            if key in metadata.keys():
                if isinstance(update_dict[key], list):
                    if not isinstance(metadata[key], list):
                        metadata[key] = [metadata[key]]
                    for item in update_dict[key]:
                        if item not in metadata[key]:
                            metadata[key].append(item)
                else: 
                    metadata[key] = update_dict[key]
            else:
                metadata.update({key:update_dict[key]})
        self.backend.update_metadata(**metadata)

    def _init_loggers(self, path, db_filename, log_mode):

        log_mode = log_mode.strip().lower()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        log = logging.getLogger(db_filename +'_log')
        log.setLevel(logging.DEBUG)

        api = logging.getLogger(db_filename + '_api')
        api.setLevel(logging.DEBUG)

        handlers = []

        if log_mode != "stream":
            log_folder_name = '.' + db_filename + '_logs'
            self._log_folder_name = log_folder_name

            if not os.path.exists(os.path.join(path, log_folder_name)):
                os.makedirs(os.path.join(path, log_folder_name))

            log_file = logging.FileHandler(os.path.join(path, log_folder_name,db_filename+'_errors.log'))
            log_file.setLevel(logging.INFO)
            log_file.setFormatter(formatter)
            log_file.set_name(db_filename + '_error')
            handlers.append(log_file)
        
        if log_mode != "silent":
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            console.setFormatter(formatter)
            console.set_name(db_filename + '_console')
            handlers.append(console)

        for logger in [log, api]:
            for handler in handlers:
                if handler.get_name() not in [h.get_name() for h in logger.handlers]:
                    logger.addHandler(handler)
        self.log = log
        self.api_logger = api

    def __len__(self):
        return self.backend.get_length()

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= len(self):
            self._iter_index = 0
            raise StopIteration
        else:
            material = self.backend.get_by_id(self._iter_index)
            self._iter_index += 1
            return material

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.backend.get_single(key)
        elif isinstance(key, int):
            return self.backend.get_by_id(key)
        else:
            raise KeyError(f'Key {key} can not be interpreted as database key.')

    def __repr__(self) -> str:
        return f"MaterialsDatabase(filename = {self.backend.filename}, len = {len(self)})"