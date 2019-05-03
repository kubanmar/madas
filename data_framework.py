import os, sys
import json, requests
import time, datetime
import random
import logging

import numpy as np
from ase.db import connect

from fingerprint import Fingerprint
from similarity import SimilarityMatrix
from utils import electron_charge, get_lattice_description
from NOMAD_enc_API import API

class MaterialsDatabase():

    def __init__(self, filename = 'materials_database.db', db_path = 'data', api = None, path_to_api_key = '.', silent_logging = False, connect_db = True):
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
        if connect_db:
            self._connect_db()

    def get_n_entries(self):
        return self.atoms_db.count()

    def get_property(self, mid, property_name):
        row = self._get_row_by_mid(mid)
        if property_name in ['dos_values', 'dos_energies']:
            return row.data.properties['dos'][property_name]
        elif property_name in row.data.properties.keys():
            return row.data.properties[property_name]
        else:
            return None

    def get_fingerprint(self, fp_type, mid = None, fp_name = None, db_id = None, log = True):
        try:
            if db_id != None:
                row = self.atoms_db.get(db_id)
                mid = row.mid
            else:
                row = self.atoms_db.get(mid = mid)
        except KeyError:
            self.log.error('Fingerprint %s is not calculated for material %s.' %(fp_type, mid))
            return None
        if row[fp_type] == None:
            self.log.error('Error in get_fingerprint. Got "None" for material %s' %(mid))
            return None
        return Fingerprint(fp_type, mid = mid, db_row = row, log = log, fp_name = fp_name)

    def get_similarity_matrix(self, fp_type, root = '.', data_path = 'data', large = False, **kwargs):
        simat = SimilarityMatrix(root = root, data_path = data_path, large = large)
        simat.calculate(fp_type, self.atoms_db, **kwargs)
        return simat

    def get_formula(self, mid):
        row = self._get_row_by_mid(mid)
        return row.formula

    def get_material(self,mid):
        return self._get_row_by_mid(mid)

    def get_atoms(self, mid):
        try:
            return self.atoms_db.get_atoms(mid = mid)
        except KeyError:
            self.log.error("not in db: %s" %(mid))
            return None

    def add_fingerprint(self, fp_type, start_from = None, fp_name = None, **kwargs):
        """
        i.e. use fp_function to calculate fingerprint based on properties and store in db using fp_name
        """
        fp_name = fp_type if fp_name == None else fp_name
        fingerprints = []
        ids = []
        self.log.info('Number of entries in db: ' + str(self.atoms_db.count()))
        self.log.info('Starting fingerprint generation for fp_type: ' + str(fp_type))
        if start_from == None:
            start_from = 1
        with self.atoms_db as db:
            for row_id in range(start_from,db.count()+1):
                row = db.get(row_id)
                try:
                    fingerprint = Fingerprint(fp_type, mid = row.mid, properties = row.data.properties, atoms = row.toatoms(), **kwargs)
                except AttributeError:
                    fingerprint = Fingerprint(fp_type, mid = row.mid, properties = row.data, atoms = row.toatoms(), **kwargs)
                except:
                    self.log.error('Fingerprint is not generated for material '+str(row.mid)+', because of: '+ str(sys.exc_info()[0].__name__)+': '+str(sys.exc_info()[1]))
                    continue
                fingerprints.append([row.id, fingerprint.get_data_json()])
            self.log.info('Writing %s fingerprints to database.' %(fp_type))
        for data in fingerprints:
            self.atoms_db.update(data[0], **{fp_name:data[1]})
            self.log.debug('db update for id '+str(data[0])+' with fingerprint '+str(fp_type))
        self.log.info('Finished for fp_type: ' + str(fp_type))
        self._update_metadata({'fingerprints' : [fp_type]})

    def put_data_to_none(self, data_key):
        ids = []
        from utils import list_chunks
        with self.atoms_db as db:
            for row_id in range(1,db.count()+1):
                row = db.get(row_id)
                ids.append(row.id)
        self.log.info('Changing value of key %s to "none"' %(data_key))
        for index, idx in enumerate(ids):
            self.atoms_db.update(idx, **{data_key:json.dumps(None)})

    def add_material(self, nomad_material_id, nomad_calculation_id):
        """
        This thing should download the data and construct the infrastructure that is necessary
        to create the different fingerprints.
        """
        mid = self._make_mid(nomad_material_id, nomad_calculation_id)
        try: #This is a not-so-nice hack. There must be a better solution.
            self.atoms_db.get(mid=mid)
            print('already in db: %s' %(mid))
        except KeyError:
            new_material = self.api.get_calculation(nomad_material_id = nomad_material_id, nomad_calculation_id = nomad_calculation_id)
            atoms = self._make_atoms(new_material)
            self.atoms_db.write(atoms, data = new_material, mid = mid)

    def fill_database(self, json_query):
        """
        Fills the database with one calculation per material for a given NOMAD search query.
        """
        self.log.info('Filling database with calculations matching the following query: ' + json.dumps(json_query) )
        materials = self.api.get_calculations_by_search(json_query)
        self._write_materials_chunks(materials)
        """
        with self.atoms_db as db:
            for material in materials:
                mid = material['mid']
                self.log.debug("Writing material with mid "+mid)
                try:
                    self.atoms_db.get(mid=mid)
                    print('already in db: %s' %(mid))
                except KeyError:
                    atoms = self._make_atoms(material)
                    db.write(atoms, data = material, mid = mid)
        """

    def _write_materials_chunks(self, materials, chunk_size = 10):
        chunk = []
        for material in materials:
            chunk.append(material)
            if len(chunk) >= chunk_size:
                with self.atoms_db as db:
                    for mat in chunk:
                        mid = mat['mid']
                        self.log.debug("Writing material with mid "+mid)
                        try:
                            self.atoms_db.get(mid=mid)
                            print('already in db: %s' %(mid))
                        except KeyError:
                            atoms = self._make_atoms(material)
                            db.write(atoms, data = mat, mid = mid)
                chunk = []

    def get_random(self, return_id = True):
        """
        Returns a random entry from the database.
        """
        n_rows = self.atoms_db.count()
        row = self.atoms_db.get(random.randint(0, n_rows))
        if return_id:
            return row.mid
        else:
            return row

    def update_entry(self, mid, dictionary):
        self._update_atoms_db(self.atoms_db, mid, dictionary)

    def update_entries(self, mid_list, dictionary_list):
        chunk = []
        for mid, dictionary in zip(mid_list, dictionary_list):
            chunk.append((mid,dictionary))
            if len(chunk) >= 10:
                for chunk_mid, chunk_dictionary in chunk:
                    with self.atoms_db as db:
                            self._update_atoms_db(db, chunk_mid, chunk_dictionary)
                chunk = []

    def add_property(self, mid, property_name):
        nomad_material_id, nomad_calculation_id = mid.split(':')
        property = self.api.get_property(nomad_material_id = nomad_material_id, nomad_calculation_id = nomad_calculation_id, property_name = property_name)
        if property != None:
            self.update_entry(mid, {property_name : property})

    def _update_metadata(self, update_dict):
        metadata = self.atoms_db.metadata
        for key in update_dict.keys():
            try:
                metadata[key]
                if isinstance(update_dict[key], list):
                    if not isinstance(metadata[key], list):
                        metadata[key] = [metadata[key]]
                    for item in update_dict[key]:
                        metadata[key].append(item)
                else: metadata[key] = update_dict[key]
            except KeyError:
                metadata.update({key:update_dict[key]})
        self.atoms_db.metadata = metadata

    def _connect_db(self):
        self.atoms_db = connect(self.atoms_db_path)

    def _init_loggers(self, path, db_filename, silent_logging):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        log = logging.getLogger('log')
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
            self.log("not in db: %s" %(mid))

    def _get_row_by_db_id(self, db_id): #TODO catch an error, maybe?
        return self.atoms_db.get(db_id)

    def _make_mid(self, nmid, ncid):
        return str(nmid)+':'+str(ncid)

    def _make_atoms(self, new_material):
        """
        creates the atoms object and links to the file and index where this thing is stored in the ase db
        """
        try:
            lattice_parameters = new_material['lattice_parameters']
        except KeyError:
            lattice_parameters = new_material['properties']['lattice_parameters']
        atoms = get_lattice_description(new_material['elements'], lattice_parameters)
        return atoms

    @staticmethod
    def _update_atoms_db(atoms_db, mid, dictionary):
        row = atoms_db.get(mid = mid)
        id = row.id
        for key, value in dictionary.items():
            if isinstance(value, (list,np.ndarray)):
                value = json.dumps(value) #TODO Catch error.
            atoms_db.update(id, **{key:value})
