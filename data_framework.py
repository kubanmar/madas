import os, sys
import json, requests
import time, datetime

import numpy as np

species_list = 'H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Nh,Fl,Mc,Lv,Ts,Og'.split(',')
electron_charge = 1.602176565e-19

from fingerprints import Fingerprint

class MaterialsDatabase():

    def __init__(self, filename = 'materials_database.json', db_path = 'data'):
        self.filename = filename
        self.filepath = os.path.join(db_path, self.filename)
        self.atoms_db_name = 'ase_atoms_' + filename
        self.atoms_db_path = os.path.join(db_path, self.atoms_db_name)
        self.api_key = 'eyJhbGciOiJIUzI1NiIsImlhdCI6MTUyMzg4MDE1OSwiZXhwIjoxNjgxNTYwMTU5fQ.eyJpZCI6ImVuY2d1aSJ9.MsMWQa3IklH7cQTxRaIRSF9q8D_2LD5Fs2-irpWPTp4'
        self.api_url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials'
        if not os.path.exists(db_path):
            os.mkdir(db_path)
        if not os.path.exists(self.filepath):
            with open(self.filepath,'w') as f:
                json.dump({}, f)
            self.materials_dict = {}
        else:
            with open(self.filepath,'r') as f:
                self.materials_dict = json.load(f)
        #check for atoms_db

    def get_property(self, mid, property_name):
        if property_name in ['dos_values', 'dos_energies']:
            return self.materials_dict[mid][property_name]
        elif property_name in self.materials_dict[mid]['properties'].keys():
            return self.materials_dict[mid]['properties'][property_name]
        else:
            return None

    def get_fingerprint(self, mid, fp_type):
        try:
            fp_data = self.materials_dict[mid]['fingerprints'][fp_type]
        except KeyError:
            sys.exit('Fingerprint %s is not calculated.' %(fp_type))
        return Fingerprint(fp_type, data = fp_data)

    def get_formula(self, mid):
        return self.materials_dict[mid]['properties']['formula']

    def update_database_file(self):
        with open(self.filepath,'r') as f:
            db_from_file = json.load(f)
        db_from_file.update(self.materials_dict)
        with open(self.filepath,'w') as f:
            json.dump(db_from_file, f, indent = 4)

    def add_atoms(self, mat_id):
        """
        creates the atoms object and links to the file and index where this thing is stored in the ase db
        """
        pass #TODO unwritten

    def add_fingerprint(self, fp_type, fp_data = None):
        """
        i.e. use fp_function to calculate fingerprint based on propterties and store in db using fp_name
        """
        for mid in self.materials_dict.keys():
            if 'fingerprints' not in self.materials_dict[mid].keys():  
                self.materials_dict[mid]['fingerprints'] = {}
            fingerprint = Fingerprint(fp_type, self.materials_dict[mid]['properties'], fp_data)
            self.materials_dict[mid]['fingerprints'][fp_type] = fingerprint.calculate()
        self.update_database_file()

    def add_material(self, nomad_material_id, nomad_calculation_id, tags = None):
        """
        This thing should download the data and construct the infrastructure that is necessary
        to create the different fingerprints.
        """
        mat_id = self._make_mid(nomad_material_id, nomad_calculation_id)
        if not mat_id in self.materials_dict.keys():
            new_material = {}
            new_material['properties'] = self._get_properties_NOMAD(nomad_material_id, nomad_calculation_id)
            new_material['elements'] = self._get_elements_NOMAD(nomad_material_id)
            if tags != None:
                if isinstance(tags,list):
                    new_material['tags'] = tags
                else:
                    new_material['tags'] = [tags]
            self.materials_dict[mat_id] = new_material
        else:
            print('already in db: %s' %(mat_id))

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
            if index % 10 == 0:
                print('Processed {:.3f} %'.format( index / len(materials_list) * 100))
        self.update_database_file()


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
