import os, sys
import json, requests
import time, datetime
import random
import logging

import numpy as np
from ase.db import connect

from fingerprints import Fingerprint
from similarity import SimilarityMatrix
from utils import electron_charge, get_lattice_description

class MaterialsDatabase():

    def __init__(self, filename = 'materials_database.db', db_path = 'data', path_to_api_key = '.', silent_logging = False, connect_db = True):
        self.atoms_db_name = filename
        self.atoms_db_path = os.path.join(db_path, self.atoms_db_name)
        self.db_path = db_path
        self.api_key = self._read_api_key(path_to_api_key)
        self.api_url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials'
        if connect_db:
            self._connect_db()
        self._init_loggers(db_path, filename.split('.db')[0], silent_logging = silent_logging)

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

    def add_material(self, nomad_material_id, nomad_calculation_id, tags = None):
        """
        This thing should download the data and construct the infrastructure that is necessary
        to create the different fingerprints.
        """
        mid = self._make_mid(nomad_material_id, nomad_calculation_id)
        try: #This is a not-so-nice hack. There must be a better solution.
            self.atoms_db.get(mid=mid)
            print('already in db: %s' %(mid))
        except KeyError:
            new_material = {}
            new_material['properties'] = self._get_properties_NOMAD(nomad_material_id, nomad_calculation_id)
            new_material['elements'] = self._get_elements_NOMAD(nomad_material_id)
            if tags != None:
                if isinstance(tags,list):
                    new_material['tags'] = tags
                else:
                    new_material['tags'] = [tags]
            atoms = self._make_atoms(new_material)
            self.atoms_db.write(atoms, data = new_material, mid = mid)

    def fill_database(self, json_query, tags = None):
        """
        Fills the database with one calculation per material for a given NOMAD search query.
        """
        if tags == None:
            tags = str(datetime.datetime.now())
        materials_list = self._get_all_materials(json_query)
        if len(materials_list) == 0:
            sys.exit("Empty list returned. Change search query.")
        for index, material in enumerate(materials_list):
            trys = 0
            success = False
            while trys < 10 and not success:
                try:
                    #ERROR! Lost keyword 'calculations_list_matching_criteria'
                    [mid, cid] = self._choose_calculation(material['id'], material['calculations_list_matching_criteria'])
                    self.add_material(mid, cid, tags = tags)
                    success = True
                except KeyError:
                    trys += 1
                    self.netlog.info('Failed to load material '+str(material['id'])+' at try '+str(trys))
                    continue
            if not success:
                self.netlog.error('Failed to add material '+str(material['id']))
            print('Processed {:.3f} %'.format( (index + 1) / len(materials_list) * 100), end = '\r')
        print('\nFinished processing.')

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
        with self.atoms_db as db:
            for mid, dictionary in zip(mid_list, dictionary_list):
                self._update_atoms_db(db, mid, dictionary)

    def add_property_NOMAD(self, mid, property, is_materials_property = False):
        nomad_mid, nomad_cid = mid.split(':')
        if is_materials_property:
            json_answer = self._get_NOMAD_data(nomad_mid, None, self.api_key, property = property)
        else:
            json_answer = self._get_NOMAD_data(nomad_mid, nomad_cid, self.api_key, property = property)
        try:
            prop = json_answer[property]
        except KeyError:
            self.log.error("Failed to write property %s to db entry %s." %(property, mid))
        self.update_entry(mid, {property : prop})

    def set_api_url(self, new_url):
        self.api_url = new_url

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

        network = logging.getLogger('network')
        network.setLevel(logging.DEBUG)

        log = logging.getLogger('log')
        log.setLevel(logging.DEBUG)

        network_file = logging.FileHandler(os.path.join(path,db_filename+'_network.log'))
        network_file.setLevel(logging.INFO)
        network_file.setFormatter(formatter)

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
        network.addHandler(network_file)
        network.addHandler(error_file)
        if not silent_logging:
            log.addHandler(console)
            network.addHandler(console)

        self.log = log
        self.netlog = network

    def _get_row_by_mid(self, mid):
        try:
            row = self.atoms_db.get(mid = mid)
            return row
        except KeyError:
            self.log("not in db: %s" %(mid))

    def _get_row_by_db_id(self, db_id): #TODO catch an error, maybe?
        return self.atoms_db.get(db_id)

    def _get_all_materials(self, json_query):
        auth = (self.api_key, '')
        try:
            json_query['search_by']['per_page'] = 1000
            post = requests.post(self.api_url, auth = auth, json=json_query)
            pages = post.json()['pages']
        except KeyError:
            sys.exit("DB initialization error. Please retry.")
        if pages == None:
            return post.json()['results']
        else:
            results = []
            for result in post.json()['results']:
                results.append(result)
            to_load = [x for x in range(2,pages['pages']+1)]
            while len(to_load) > 0:
                for page in to_load:
                    json_query['search_by']['page'] = page
                    post = requests.post(self.api_url, auth = auth, json=json_query)
                    if post.status_code == 200:
                        for result in post.json()['results']:
                            results.append(result)
                        to_load.remove(page)
                    else:
                        continue
        return results

    def _choose_calculation(self,mid, cid_list):
        score_list = []
        for cid in cid_list:
            data = self._get_dos(mid, cid)
            energies = data['dos_energies']
            score = self._calc_dos_score(energies=energies)
            score_list.append([score, [mid, cid]])
        score_list.sort()
        return score_list[0][1]

    def _calc_dos_score(self,energies):
        point_density = (abs(max(energies) - min(energies)) / len(energies)) / electron_charge
        return point_density

    def _get_properties_NOMAD(self, mid, cid):
        json_answer = self._get_NOMAD_data(mid, cid, self.api_key)
        properties = {}
        #calculation properties
        for keyword in ['atomic_density', 'cell_volume', 'lattice_parameters', 'mass_density', 'mainfile_uri', 'code_name']:
            properties[keyword] = json_answer[keyword]
        for x in json_answer['energy']:
            if x['e_kind'] == 'Total E':
                totengy = x['e_val']
                break
        properties['energy'] = [json_answer['code_name'], totengy]
        #get also material properties
        json_answer = self._get_NOMAD_data(mid, None, self.api_key)
        for keyword in ['formula', 'point_group', 'space_group_number']:
            properties[keyword] = json_answer[keyword]
        properties['dos'] = self._get_dos(mid, cid)
        return properties

    def _get_elements_NOMAD(self, mid):
        auth = (self.api_key, '')
        #url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s/elements?pagination=off' %(str(mid))
        url = os.path.join(self.api_url,str(mid), 'elements?pagination=off')
        json_answer = requests.get(url, auth = auth).json()
        atom_list = []
        for atom in json_answer['results']:
            position = [float(x) for x in atom['position'][1:-1].split(',')]
            atom_list.append([atom['label'], position, atom['wyckoff']])
        return atom_list

    def _get_dos(self, mid, cid):
        auth = (self.api_key, '')
        #url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s/calculations/%s?property=dos' %(str(mid), str(cid))
        url = os.path.join(self.api_url,str(mid), 'calculations', str(cid)) + '?property=dos'
        print(url)
        json_answer = requests.get(url, auth = auth).json()
        return json_answer['dos']

    def _make_mid(self, nmid, ncid):
        return str(nmid)+':'+str(ncid)

    def _make_atoms(self, new_material):
        """
        creates the atoms object and links to the file and index where this thing is stored in the ase db
        """
        atoms = get_lattice_description(new_material['elements'], new_material['properties']['lattice_parameters'])
        return atoms

    @staticmethod
    def _get_NOMAD_data(mid, cid, api_key, property = None):
        """
        Gets the results of a certain calculation with material id ``mid`` and calculation id ``cid`` from NOMAD.
        if ``cid`` == None: get materials properties instead
        if property != None: search for single property instead
        """
        auth = (api_key, '')
        if cid == None:
            url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s' %(str(mid))
        else:
            url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s/calculations/%s' %(str(mid), str(cid))
        if property != None:
            url += '?property=%s' %(str(property))
        json_answer = requests.get(url, auth = auth).json()
        return json_answer

    @staticmethod
    def _update_atoms_db(atoms_db, mid, dictionary):
        row = atoms_db.get(mid = mid)
        id = row.id
        for key in dictionary.keys():
            value = dictionary[key]
            if isinstance(value, (list,np.ndarray)):
                value = json.dumps(value) #TODO Catch error.
            atoms_db.update(id, **{key:value})

    @staticmethod
    def _read_api_key(path_to_api_key):
        key_data = open(os.path.join(path_to_api_key,'api_key'),'r').readline()
        api_key = key_data[:-1] if key_data[-1] == '\n' else key_data
        return api_key
