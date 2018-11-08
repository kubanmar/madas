import os, sys
import json, requests
import time, datetime
import random

import numpy as np
from ase.db import connect

from fingerprints import Fingerprint
from utils import electron_charge, get_lattice_description

class MaterialsDatabase():

    def __init__(self, filename = 'materials_database.db', db_path = 'data', path_to_api_key = '.'):
        self.atoms_db_name = filename
        self.atoms_db_path = os.path.join(db_path, self.atoms_db_name)
        key_data = open(os.path.join(path_to_api_key,'api_key'),'r').readline()
        self.api_key = key_data[:-1] if key_data[-1] == '\n' else key_data
        self.api_url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials'
        self.atoms_db = connect(self.atoms_db_path)

    def get_property(self, mid, property_name):
        row = self._get_row_by_mid(mid)
        if property_name in ['dos_values', 'dos_energies']:
            return row.data.properties['dos'][property_name]
        elif property_name in row.data.properties.keys():
            return row.data.properties[property_name]
        else:
            return None

    def get_fingerprint(self, mid, fp_type, multiprocess = False):
        try:
            row = self.atoms_db.get(mid = mid)
        except KeyError:
            sys.exit('Fingerprint %s is not calculated for material %s.' %(fp_type, mid))
        if multiprocess:
            return Fingerprint(fp_type, db_row = row, database = self)
        else:
            return Fingerprint(fp_type, db_row = row)

    def get_formula(self, mid):
        row = self._get_row_by_mid(mid)
        return row.formula

    def get_atoms(self, mid):
        try:
            return self.atoms_db.get_atoms(mid = mid)
        except KeyError:
            print("not in db: ", mid)

    def add_fingerprint(self, fp_type):
        """
        i.e. use fp_function to calculate fingerprint based on propterties and store in db using fp_name
        """
        for row in self.atoms_db.select():
            fingerprint = Fingerprint(fp_type, row.data.properties, row.toatoms())
            #to_store =  json.dumps(fingerprint.calculate())
            #self.atoms_db.update(row.id, DOS = to_store)
            fingerprint.write_to_database(row.id, self.atoms_db)
        #self.update_database_file()
        #            db.update([i+1], **{property_name:pval}) ##stolen from Santiago ##TODO try!


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
                    [mid, cid] = self._choose_calculation(material['id'], material['calculations_list_matching_criteria'])
                    self.add_material(mid, cid, tags = tags)
                    success = True
                except KeyError:
                    trys += 1
                    #time.sleep(1)
                    continue
            if not success:
                print("Failed to add ", str(material),'.')
            if (index+1) % 10 == 0:
                print('Processed {:.3f} %'.format( (index + 1) / len(materials_list) * 100))
        print('Finished processing.')

    def get_random(self, return_id = True):
        n_rows = self.atoms_db.count()
        row = self.atoms_db.get(random.randint(0, n_rows))
        if return_id:
            return row.mid
        else:
            return row

    def _get_row_by_mid(self, mid):
        try:
            row = self.atoms_db.get(mid = mid)
            return row
        except KeyError:
            print("not in db: ", mid)

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
        auth = (self.api_key, '')
        mat_url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s' %(str(mid))
        calc_url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s/calculations/%s' %(str(mid), str(cid))
        #calculation properties
        json_answer = requests.get(calc_url, auth = auth).json()
        properties = {}
        for keyword in ['atomic_density', 'cell_volume', 'lattice_parameters', 'mass_density']:
            properties[keyword] = json_answer[keyword]
        for x in json_answer['energy']:
            if x['e_kind'] == 'Total E':
                totengy = x['e_val']
                break
        properties['energy'] = [json_answer['code_name'], totengy]
        #get also material properties
        json_answer = requests.get(mat_url, auth = auth).json()
        for keyword in ['formula', 'point_group', 'space_group_number']:
            properties[keyword] = json_answer[keyword]
        properties['dos'] = self._get_dos(mid, cid)
        return properties

    def _get_elements_NOMAD(self, mid):
        auth = (self.api_key, '')
        url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s/elements?pagination=off' %(str(mid))
        json_answer = requests.get(url, auth = auth).json()
        atom_list = []
        for atom in json_answer['results']:
            position = [float(x) for x in atom['position'][1:-1].split(',')]
            atom_list.append([atom['label'], position, atom['wyckoff']])
        return atom_list

    def _get_dos(self, mid, cid):
        auth = (self.api_key, '')
        url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials/%s/calculations/%s?property=dos' %(str(mid), str(cid))
        json_answer = requests.get(url, auth = auth).json()
        return json_answer['dos']

    def _SI_to_Angstom(self, length):
        return np.power(length,10^10)

    def _make_mid(self, nmid, ncid):
        return str(nmid)+':'+str(ncid)

    def _make_atoms(self, new_material):
        """
        creates the atoms object and links to the file and index where this thing is stored in the ase db
        """
        atoms = get_lattice_description(new_material['elements'], new_material['properties']['lattice_parameters'])
        return atoms
