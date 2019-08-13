import os, sys
import json, requests
import time, datetime
import random, time
import logging, multiprocessing

import numpy as np
from ase.db import connect

from fingerprint import Fingerprint
from similarity import SimilarityMatrix
from utils import electron_charge, get_lattice_description
from NOMAD_enc_API import API

class MaterialsDatabase():

    def __init__(self, filename = 'materials_database.db', db_path = 'data', api = None, path_to_api_key = '.', silent_logging = False, connect_db = True, lock_db = True):
        self.atoms_db_name = filename
        self.atoms_db_path = os.path.join(db_path, self.atoms_db_name)
        self.db_path = db_path
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        self._init_loggers(db_path, filename.split('.db')[0], silent_logging = silent_logging)
        if api == None:
            self.api = API(key_path = path_to_api_key, logger = self.api_logger)
        else:
            self.api = api
            self.api.set_logger(self.api_logger)
        self.add_material.__func__.__doc__ = self.api.get_calculation.__func__.__doc__
        self.fill_database.__func__.__doc__ = self.api.get_calculations_by_search.__func__.__doc__
        if connect_db:
            self._connect_db(lock_db = lock_db)
        self._iter_index = 0

    def get_n_entries(self):
        """
        Returns the number of entries in the database. No arguments.
        """
        return len(self)

    def get_property(self, mid, property_name):
        """
        Get a property of a material by name.
        Args:
            * mid: string; material id of the requested material
            * property_name: string; name of the requested property as used in the database
        Returns:
            * property: if it exists in the database
            * None: else
        """
        row = self._get_row_by_mid(mid)
        if property_name in ['dos_values', 'dos_energies']:
            return row.data['dos'][property_name]
        elif property_name in row.data.keys():
            return row.data[property_name]
        elif property_name in row.keys():
            return row[property_name]
        else:
            return None

    def get_properties(self, property_name, output_mids = False):
        """
        Get a list of properties for all materials of the database.
        Args:
            * property_name: string; name of the property
        Kwargs:
            * output_mids: bool; detault: False; output the list of corresponding material ids with the properties
        Returns:
            * properties: list; properties of materials
            * [mids]: list; list of material ids
        """
        properties = []
        mids = []
        for db_id in range(1, self.get_n_entries()+1):
            row = self.atoms_db.get(db_id)
            mid = row.mid
            try:
                properties.append(row.data[property_name])
                mids.append(mid)
            except KeyError:
                self.log.error('No property "' + property_name + '" for material ' + mid)
        if output_mids:
            return properties, mids
        return properties

    def get_row_properties(self, property_name, output_mids = False):
        properties = []
        mids = []
        for db_id in range(1, self.get_n_entries()+1):
            row = self.atoms_db.get(db_id)
            mid = row.mid
            try:
                properties.append(row[property_name])
                mids.append(mid)
            except KeyError:
                self.log.error('No property "' + property_name + '" for material ' + mid)
        if output_mids:
            return properties, mids
        return properties

    def get_fingerprint(self, fp_type, mid = None, name = None, db_id = None, log = True, **kwargs):
        """
        Get a given type of fingerprint object of a given material.
        Args:
            * fp_type: string; type of fingerprint X, must correspond to a XFingerprint() object
        Kwargs:
            * mid: string; material id of the requested material
            * name: string; name of the fingerprint as used in the database, if name != fp_type
            * db_id: int; id of the material in the ASE AtomsDatabase()
            * log: bool; defines if the returned Fingerprint() object should contain a log (which makes it unsuitable for multiprocessing)
        Returns:
            * Fingerprint() object: if it exists in the database
            * None: else
        Additional kwargs are passed to Fingerprint().__init__() .
        """
        try:
            if db_id != None:
                row = self.atoms_db.get(db_id)
                mid = row.mid
            else:
                row = self.atoms_db.get(mid = mid)
        except KeyError:
            self.log.error('No material with mid %s.' %(mid))
            return None
        try:
            if row[fp_type] == None:
                self.log.error('Error in get_fingerprint. Got "None" for material %s' %(mid))
                return None
        except AttributeError:
            self.log.error('Fingerprint %s is not calculated for material %s.' %(fp_type, mid))
            return None
        logger = self.log if log else None
        return Fingerprint(fp_type = fp_type, db_row = row, logger = logger, name = name, **kwargs)

    def get_fingerprints(self, fp_type, name = None, log = True, **kwargs):
        """
        Return a list of fingerprints for a given fingerprint type.
        Identical to calling MaterialsDatabase().get_fingerprint() for all materials in the database.
        """
        fingerprints = []
        for db_id in range(1, self.get_n_entries()+1):
            fingerprints.append(self.get_fingerprint(fp_type, name = name, db_id = db_id, log = log, **kwargs))
        return fingerprints

    def get_similarity_matrix(self, fp_type, root = '.', data_path = 'data', large = False, **kwargs):
        """
        Calculate a SimilarityMatrix() object.
        Args:
            * fp_type: string; type of fingerprint X, must correspond to a XFingerprint() object
        Kwargs:
            * root: string; path of the SimilarityMatrix()
            * data_path: string; relative location of the folder, where the SimilarityMatrix() shall be created
            * large: bool; option to create a large SimilarityMatrix() object
        Returns:
            * SimilarityMatrix() object
        Additional kwargs are passed to SimilarityMatrix().calculate().
        """
        simat = SimilarityMatrix(root = root, data_path = data_path, large = large, log = False)
        simat.calculate(fp_type, self, **kwargs)
        return simat

    def get_formula(self, mid):
        """
        Return formula of material with given materia id.
        Args:
            * mid: string; material id of the requested material
        Returns:
            * formula: string; formula of the materials according to ASE AtomsDatabase()
        """
        row = self._get_row_by_mid(mid)
        return row.formula

    def get_material(self,mid):
        """
        Return the AtomsRow() object for a material with given material id.
        Args:
            * mid: string; material id of the requested material
        Returns:
            * ASE AtomsRow() object
        """
        return self._get_row_by_mid(mid)

    def get_atoms(self, mid):
        """
        Return the ASE Atoms object for a material with given material id.
        Args:
            * mid: string; material id of the requested material
        Returns:
            * ASE Atoms() object
        """
        row = self._get_row_by_mid(mid)
        return row.toatoms()

    def add_fingerprint(self, fp_type, name = None, show_progress = False, overwrite_entries = False, **kwargs):
        """
        Calculate fingerprints of all materials in the database and store them.
        Args:
            * fp_type: string; type of fingerprint X, must correspond to a XFingerprint() object
        Kwargs:
            * name: string, None; name of the fingerprint as used in the database, if name != fp_type, else None
        Returns:
            * None
        Additional kwargs are passed to Fingerprint().__init__().
        """
        self.log.info('Starting fingerprint generation for fp_type: ' + str(fp_type))
        fingerprints = self.gen_fingerprints_list(fp_type, name = name, overwrite_entries = overwrite_entries, **kwargs)
        self.log.info('Writing %s fingerprints to database.' %(fp_type))
        mid_list = [fingerprint.mid for fingerprint in fingerprints]
        dictionary_list = [{fingerprint.name:fingerprint.get_data_json()} for fingerprint in fingerprints]
        self.update_entries(mid_list, dictionary_list, show_progress = show_progress)
        self.log.info('Finished for fp_type: ' + str(fp_type))
        self._update_metadata({'fingerprints' : [fp_type]})

    def add_fingerprints(self, fp_types, names = [None], show_progress = False, overwrite_entries = False, **kwargs):
        """
        Calculate fingerprints of all materials in the database and store them.
        Args:
            * fp_types: list of strings; type of fingerprint X, must correspond to a XFingerprint() object
        Kwargs:
            * name: list of string, None; name of the fingerprint as used in the database, if name != fp_type, else None
        Returns:
            * None
        Additional kwargs are passed to _each_ instance of Fingerprint().__init__().
        """
        fingerprint_dict = {}
        while len(names) < len(fp_types):
            names.append(None)
        for fp_type, name in zip(fp_types, names):
            self.log.info('Starting fingerprint generation for fp_type: ' + str(fp_type))
            fingerprints = self.gen_fingerprints_list(fp_type, name = name, overwrite_entries = overwrite_entries, **kwargs)
            for fingerprint in fingerprints:
                if not fingerprint.mid in fingerprint_dict.keys():
                    fingerprint_dict[fingerprint.mid] = {fingerprint.name:fingerprint.get_data_json()}
                else:
                    fingerprint_dict[fingerprint.mid].update({fingerprint.name:fingerprint.get_data_json()})
        dictionary_list = []
        mid_list = []
        for mid, data in fingerprint_dict.items():
            mid_list.append(mid)
            dictionary_list.append(data)
        self.log.info('Writing fingerprints to database.')
        self.update_entries(mid_list, dictionary_list, show_progress = show_progress)
        self.log.info('Finished for fp_types: ' + str(fp_types))
        self._update_metadata({'fingerprints' : fp_types})


    def gen_fingerprints_list(self, fp_type, name = None, log = True, overwrite_entries = False, **kwargs):
        """
        Generate a list of fingerprints.
        Args:
            * fp_type: string; type of fingerprint X, must correspond to a XFingerprint() object
        Kwargs:
            * name: string; default: None; name of the fingerprint as used in the database, if name != fp_type, else None
            * log: bool; default: True; generate fingerprints with a logger
            * overwrite_entries: bool; default: False; always generate new fingerprints
        Additional keyword arguments are passed to Fingerprint().__init__().
        """
        fingerprints = []
        logger = self.log if log else None
        with self.atoms_db as db:
            for row_id in range(1,db.count()+1):
                row = db.get(row_id)
                try:
                    if overwrite_entries:
                        fingerprint = Fingerprint(fp_type = fp_type, name = name, db_row = None, logger = logger, **kwargs)
                        fingerprint.mid = row.mid
                        fingerprint.calculate(row, **kwargs)
                    else:
                        fingerprint = Fingerprint(fp_type = fp_type, name = name, db_row = row, logger = logger, **kwargs)
                except:
                    self.log.error('Fingerprint is not generated for material '+str(row.mid)+', because of: '+ str(sys.exc_info()[0].__name__)+': '+str(sys.exc_info()[1]))
                    continue
                fingerprints.append(fingerprint)
        return fingerprints

    def delete_keys(self, data_key):
        for row_id in range(1,self.atoms_db.count()+1):
            with self.atoms_db as db:
                db.update(row_id, delete_keys = [data_key])

    def add_material(self, *args, **kwargs):
        """
        Add a specified material to the database.
        Keyword arguments are passed to the api. The default is the NOMAD Encyclopedia API.
        Corresponding keywords are `nomad_material_id` and `nomad_calculation_id`.
        If those keywords are present, they will be used to construct the database mid.
        """
        mid = self.api.gen_mid(*args)
        try: #This is a not-so-nice hack. There must be a better solution.
            self.atoms_db.get(mid=mid)
            print('already in db: %s' %(mid))
        except KeyError:
            material = self.api.get_calculation(*args, **kwargs)
            self.atoms_db.write(material.atoms, data = material.data, mid = material.mid)

    def fill_database(self, *args, **kwargs):
        """
        Fills the database with one calculation per material for a given NOMAD search query.
        """
        self.log.info('Filling database with calculations matching the following query: ' + json.dumps(args) )
        materials = self.api.get_calculations_by_search(*args, **kwargs)
        ids = []
        for material in materials:
            ids.append(self.atoms_db.reserve(mid = material.mid))
        self._write_materials(materials, ids = ids)

    def get_random(self, return_id = True):
        """
        Returns a random entry from the database.
        Kwargs:
            * return_id: bool; True by default, return id of material instead of ASE AtomsRow() object
        Returns:
            * material id (string) of a random entry of the database: if return_id == True
            * ASE AtomsRow() object of a random entry of the database: else
        """
        n_rows = self.atoms_db.count()
        row = self.atoms_db.get(random.randint(0, n_rows))
        if return_id:
            return row.mid
        else:
            return row

    def update_entry(self, mid, dictionary):
        """
        Update a entry of the database from a given dictionary.
        Args:
             * mid: string; mid of the corresponding db entry
             * dictionary: dict; dictionary with the key-value pairs to be updated in the database
        """
        self._update_atoms_db(self.atoms_db, mid, dictionary)

    def update_entries(self, mid_list, dictionary_list, show_progress = False):
        """
        Update a list of entries in the database.
        Args:
            * mid_list: list of strings; list of mids of materials in the database
            * dictionary_list: list of dicts; list of dictionaries with key-value pairs to be updated
        Kwargs:
            * show_progress: bool; default = False; display percentage of progress to screen
        """
        max_len = len(mid_list)
        current_entry = 0
        for mid, dictionary in zip(mid_list, dictionary_list):
            if show_progress:
                print('progress', current_entry / max_len * 100, end = '\r')
            with self.atoms_db as db:
                self._update_atoms_db(db, mid, dictionary)
            current_entry += 1

    def add_property(self, mid, property_name):
        """
        Add a property to a spacific material of the database through the API.
        Args:
            * mid: string; mid of the material to update
            * property_name: string; name of property to be downloaded from recource
        """
        nomad_material_id, nomad_calculation_id = mid.split(':')
        property = self.api.get_property(nomad_material_id = nomad_material_id, nomad_calculation_id = nomad_calculation_id, property_name = property_name)
        if property != None:
            self.update_entry(mid, {property_name : property})

    def _write_materials(self, materials, ids=None):
        for id, material in zip(ids, materials):
            with self.atoms_db as db:
                mid = material.mid
                self.log.debug("Writing material with mid "+mid)
                if id != None:
                    db.update(id, atoms = material.atoms, data = material.data, mid = material.mid)
                else:
                    self.log.info('Material with mid ' + mid + ' already in db.')

    def _update_metadata(self, update_dict):
        metadata = self.atoms_db.metadata
        for key in update_dict.keys():
            try:
                metadata[key]
                if isinstance(update_dict[key], list):
                    if not isinstance(metadata[key], list):
                        metadata[key] = [metadata[key]]
                    for item in update_dict[key]:
                        if not item in metadata[key]:
                            metadata[key].append(item)
                else: metadata[key] = update_dict[key]
            except KeyError:
                metadata.update({key:update_dict[key]})
        self.atoms_db.metadata = metadata

    def _connect_db(self, lock_db):
        self.atoms_db = connect(self.atoms_db_path, use_lock_file = lock_db)

    def _init_loggers(self, path, db_filename, silent_logging):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        log = logging.getLogger(db_filename +'_log')
        log.setLevel(logging.DEBUG)

        api = logging.getLogger(db_filename + '_api')
        api.setLevel(logging.DEBUG)

        api_file = logging.FileHandler(os.path.join(path, db_filename + '_api.log'))
        api_file.setLevel(logging.INFO)
        api_file.setFormatter(formatter)

        error_file = logging.FileHandler(os.path.join(path,db_filename+'_errors.log'))
        error_file.setLevel(logging.ERROR)
        error_file.setFormatter(formatter)

        performance_file = logging.FileHandler(os.path.join(path,db_filename+'_perf.log'))
        performance_file.setLevel(logging.DEBUG)
        performance_file.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)

        log.addHandler(error_file)
        log.addHandler(performance_file)
        api.addHandler(api_file)
        if not silent_logging:
            log.addHandler(console)
            api.addHandler(console)

        self.log = log
        self.api_logger = api

    def _get_row_by_mid(self, mid):
        try:
            row = self.atoms_db.get(mid = mid)
            return row
        except KeyError:
            self.log.error("not in db: %s" %(mid))

    def _get_row_by_db_id(self, db_id): #TODO catch an error, maybe?
        return self.atoms_db.get(db_id)

    def _make_mid(self, nmid, ncid):
        return str(nmid)+':'+str(ncid)

    def __len__(self):
        return self.atoms_db.count()

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index + 1 > len(self):
            self._iter_index = 0
            raise StopIteration
        else:
            self._iter_index += 1
            return self._get_row_by_db_id(self._iter_index)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._get_row_by_mid(key)
            except KeyError:
                try:
                    return self._get_row_by_db_id(int(key) + 1)
                except:
                    raise KeyError('Key can not be interpreted as database key.')
        elif isinstance(key, int):
            try:
                return self._get_row_by_db_id(key + 1)
            except KeyError:
                raise KeyError('No entry with id = ' + str(key) + '.')
        else:
            raise KeyError('Key can not be interpreted as database key.')

    @staticmethod
    def _update_atoms_db(atoms_db, mid, dictionary):
        row = atoms_db.get(mid = mid)
        id = row.id
        atoms_db.update(id, **dictionary)
        """
        for key, value in dictionary.items():
            if isinstance(value, (list,np.ndarray)):
                value = json.dumps(value) #TODO Catch error.
            atoms_db.update(id, **{key:value})
        """
