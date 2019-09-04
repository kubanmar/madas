import os, json, requests, sys, time
from ase import Atoms
from api_core import APIClass, Material
from utils import species_list
import numpy as np

API_base_url ='https://encyclopedia.nomad-coe.eu/api/v1.0/materials'
API_base_url_dev = 'http://enc-staging-nomad.esc.rzg.mpg.de/v1.0/materials'

calculation_properties_keyword_list = ['atomic_density', 'cell_volume', 'lattice_parameters', 'mass_density', 'mainfile_uri', 'code_name']
material_properties_keyword_list = ['formula', 'point_group', 'space_group_number']

class API(APIClass):

    def __init__(self, base_url = API_base_url, access_token = None, key_path = '.', logger = None):
        self.set_url(base_url)
        self.auth = (self._read_api_key(key_path), '') if access_token == None else (access_token, '')
        self.set_logger(logger)

    def get_calculation(self, nomad_material_id, nomad_calculation_id, calculation_data = None):
        """
        Get a calculation from the NOMAD Encyclopedia.
        Args:
            * nomad_material_id: int,string; NOMAD material id
            * nomad_calculation_id: int,string; NOMAD calculation id
        Kwargs:
            * calculation_data: dict; default: None; dictionary of already downloaded materials properties
        """
        mid = self.gen_mid(nomad_material_id, nomad_calculation_id)
        calculation = {'mid':mid}
        #calculation properties
        if calculation_data == None:
            url = self._construct_url(self.base_url, material_id = nomad_material_id, api_endpoint = 'calculations', calculation_id = nomad_calculation_id)
            failure_message = 'calculation ' + str(nomad_material_id) + ':' + str(nomad_calculation_id)
            answer = self._api_call(url, failure_message = failure_message)
            answer = answer.json()
        else:
            answer = calculation_data
        try:
            get_dos = True if answer['has_dos'] in [True, 'Yes'] else False
        except KeyError:
            get_dos = True if 'dos_energies' in answer['diagram_data'][0]['data_content'] else False
        for keyword in calculation_properties_keyword_list:
            calculation[keyword] = answer[keyword]
        totengy = None
        for x in answer['energy']:
            if x['e_kind'] == 'Total E':
                totengy = x['e_val']
                break
        calculation['energy'] = [answer['code_name'], totengy]
        #material properties
        #if calculation_data == None:
        url = self._construct_url(self.base_url, material_id = nomad_material_id)
        failure_message = 'material ' + str(nomad_material_id)
        answer = self._api_call(url, failure_message = failure_message).json()
        for keyword in material_properties_keyword_list:
            calculation[keyword] = answer[keyword]
        #DOS
        if get_dos:
            url = self._construct_url(self.base_url, material_id = nomad_material_id, api_endpoint = 'calculations', calculation_id = nomad_calculation_id, property = 'dos')
            failure_message = 'dos ' + str(nomad_material_id) + ':' + str(nomad_calculation_id)
            answer = self._api_call(url, failure_message = failure_message).json()
            try:
                calculation['dos'] = answer['dos']
            except KeyError:
                error_message = 'No property DOS for material ' + mid + '. \n Tried with url: ' + url
                self._report_error(error_message)
        #atomic positions
        url = self._construct_url(self.base_url, material_id = nomad_material_id, api_endpoint = 'elements', property = 'dos')
        url = self._no_pagination(url)
        failure_message = 'elements ' + str(nomad_material_id)
        answer = self._api_call(url, failure_message = failure_message).json()
        atom_list = []
        for atom in answer['results']:
            position = [float(x) for x in atom['position'][1:-1].split(',')]
            atom_list.append([atom['label'], position, atom['wyckoff']])
        calculation['elements'] = atom_list
        atoms = self._make_atoms(calculation)
        return Material(mid, atoms, calculation)

    def get_calculations_by_search(self, search_query, show_progress = True, force_dos = False, scroll = False):
        """
        Get a set of calculations from the NOMAD Encyclopedia.
        Args:
            * search_query: json dictionary; search query as used by the Encyclopedia web API
        Kwargs:
            * show_progress: bool; default = True; print progress of downloading data to screen
            * force_dos: bool; default = False; force to use materials with DOS data
        """
        materials_list = self._get_materials_list(search_query, scroll = scroll) #DEBUG
        list_filter = {key : value for key,value in search_query.items() if key != 'search_by'}
        if len(materials_list) == 0:
            self._report_error('Empty materials list returned. Try different search query.')
            sys.exit()
        materials = []
        if self.log != None:
            self.log.info('Got materials list.')
        else:
            print('Got materials list.')
        for index, item in enumerate(materials_list):
            url = self._construct_url(base_url = self.base_url, material_id = item['id'], api_endpoint = 'calculations')
            failure_message = 'Could not obtain list of calculations for material ' + str(item['id'])
            calculations = self._api_call(url, failure_message = failure_message)
            calc_list = calculations.json()["results"]
            for calc in calc_list:
                for key, value in item.items():
                    if key == 'id':
                        calc.update({'mat_id':value})
                    else:
                        calc.update({key:value})
            calc_list = self._filter_calculation_list(calc_list, list_filter)
            if calc_list == []:
                error_message = 'No calculations with matching criteria for material ' + str(item['id'])
                self._report_error(error_message)
                continue
            calc = self._select_calc(calc_list, force_dos = force_dos, return_calc = True)
            materials.append(self.get_calculation(item['id'], calc['id'], calculation_data = calc))
            if show_progress:
                print('Fetching materials {:.3f} %'.format( (index + 1) / len(materials_list) * 100), end = '\r')
        if show_progress:
            print('\n')
        return materials

    def get_property(self, nomad_material_id = 1, nomad_calculation_id = 1, property_name = 'cell_volume'):
        property = None
        url = self._construct_url(self.base_url, material_id = nomad_material_id, api_endpoint = 'calculations', calculation_id = nomad_calculation_id, property = property_name)
        failure_message = 'calculation property' + str(nomad_material_id) + ':' + str(nomad_calculation_id)
        answer = self._api_call(url, failure_message = failure_message)
        if property_name in answer.json().keys():
            property = answer.json()[property_name]
        else:
            url = self._construct_url(self.base_url, material_id = nomad_material_id) + '?property=' + property_name
            failure_message = 'material property' + str(nomad_material_id) + ':' + str(nomad_calculation_id)
            answer = self._api_call(url, failure_message = failure_message)
            if property_name in answer.json().keys():
                property = answer.json()[property_name]
        if property == None:
            error_message = 'No property of type ' + property_name + ' found for material ' + str(nomad_material_id) + ':' + str(nomad_calculation_id) + '.'
            self._report_error(error_message = error_message)
        return property

    def set_url(self, url):
        self.base_url = url

    def set_logger(self, logger):
        self.log = logger

    def set_auth(self, api_key):
        self.auth = (api_key, '')

    def gen_mid(self, *args):
        return ':'.join([str(arg) for arg in args])

    def print_api_keys(self):
        url = self._construct_url(self.base_url, 1, 'calculations', 1)
        for key in self._api_call(url).json().keys():
            print(key)
        url = self._construct_url(self.base_url, 1)
        for key in self._api_call(url).json().keys():
            print(key)

    def _make_atoms(self, new_material):
        """
        creates the atoms object and links to the file and index where this thing is stored in the ase db
        """
        try:
            lattice_parameters = new_material['lattice_parameters']
        except KeyError:
            lattice_parameters = new_material['properties']['lattice_parameters']
        atoms = self._get_lattice_description(new_material['elements'], lattice_parameters)
        return atoms

    def _get_materials_list(self, search_query, per_page = 1000, show_progress = False, scroll = False):
        if scroll: # Quick patch for scrolling elasticsearch database
            results = []
            search_query['search_by']['pagination'] = 'scroll'
            post = requests.post(self.base_url, auth = self.auth, json = search_query)
            for result in post.json()['results']:
                results.append(result['id'])
            n_materials = post.json()['total_results']
            scroll_id = post.json()['scroll_id']
            counter_max = int(np.ceil(n_materials / per_page))
            counter = 0
            while len(np.unique(results)) < n_materials and counter < counter_max:
                search_query['search_by']['scroll_id'] = str(scroll_id)
                post = requests.post(self.base_url, auth = self.auth, json = search_query)
                if str(post.status_code) != "200":
                    counter += 1
                    continue
                for result in post.json()['results']:
                    results.append(result)
                scroll_id = post.json()['scroll_id']
                counter += 1
            return results
        try:
            search_query['search_by']['per_page'] = per_page
            post = requests.post(self.base_url, auth = self.auth, json = search_query)
            pages = post.json()['pages']
        except KeyError:
            self._report_error('Could not post request to NOMAD API.')
            sys.exit("DB initialization error. Please retry.")
        if pages == None:
            return post.json()['results']
        else:
            results = []
            for result in post.json()['results']:
                results.append(result)
            to_load = [x for x in range(2,pages['pages']+1)]
            max_len = len(to_load)
            while len(to_load) > 0:
                for page in to_load:
                    search_query['search_by']['page'] = page
                    try:
                        post = requests.post(self.base_url, auth = self.auth, json = search_query)
                    except ConnectionError:
                        print('ConnectionError for', search_query, end = '\n')
                        continue
                    if post.status_code == 200:
                        for result in post.json()['results']:
                            results.append(result)
                        to_load.remove(page)
                    else:
                        break
                if show_progress:
                    print('Fetching materials list {:.3f} %'.format( (max_len - len(to_load)) / max_len * 100), end = '\r')
            if show_progress:
                print('\n')

        return results

    def _api_call(self, url, failure_message = 'Undefined error'):
        """
        Tries 10 times to connect to the NOMAD encyclopedia staging API and reports an error if it is not successful.
        In case of total failure, throws AssertionError.
        """
        trials = 0
        success = False
        while trials < 10 and not success:
            try:
                answer = requests.get(url, auth = self.auth)
                if int(answer.status_code) != 200:
                    assert False
                success = True
            except (AssertionError, ConnectionError) as error:
                error_message = 'Failed connection to server because of ' + str(error) + '\nThis was attempt number ' + str(trials) + '.'
                self._report_error(error_message)
                trials += 1
        if not success:
            error_message = 'Could not connect to NOMAD API:' + failure_message
            self._report_error(error_message)
            raise AssertionError(error_message)
        return answer

    def _filter_calculation_list(self, calculation_list, filter_json):
        filtered_list = []
        for calc in calculation_list:
            for key, value in filter_json.items():
                if key in calc.keys():
                    calc_value = calc[key]
                    if isinstance(value, list):
                        if isinstance(value[0], str):
                            calc_value = str(calc_value)
                        if (calc_value in value) or (value == 'Yes' and calc_value == 'True'):
                            filtered_list.append(calc)
                    else:
                        if isinstance(value, str):
                            calc_value = str(calc_value)
                        if (calc_value == value) or (value == 'Yes' and calc_value == 'True'):
                            filtered_list.append(calc)
        return filtered_list

    def _evaluate_calc(self, calc):
        value = 0
        if calc['functional_type'] == 'GGA': value += 100
        if calc['has_band_structure'] and calc['has_dos']: value += 10
        code_name = calc['code_name'].strip()
        if code_name == 'FHI-aims': value += 3
        elif code_name == 'VASP': value += 2
        elif code_name == 'Quantum Espresso': value += 1
        return value

    def _select_calc(self, calc_list, force_dos = True, return_calc = True):
        if force_dos:
            calc_list = self._filter_calculation_list(calc_list, filter_json={'has_dos':True})
        if return_calc:
            scored_list = [(self._evaluate_calc(calc), calc) for calc in calc_list]
        else:
            scored_list = [(self._evaluate_calc(calc), calc['id']) for calc in calc_list]
        scored_list.sort(reverse = True, key= lambda x: x[0])
        website_sorted = [x for x in scored_list if x[0] == scored_list[0][0]]
        return website_sorted[-1][1]
        #return scored_list[0][1]

    def _report_error(self, error_message):
        if self.log != None:
            self.log.error(error_message)
        else:
            print(error_message)

    def _add_backslash(self, string):
        if string[-1] != '/':
            string += '/'
        return string

    def _no_pagination(self, url):
        url += '?pagination=off'
        return url

    def _construct_url(self, base_url, material_id = None, api_endpoint = None, calculation_id = None, property = None):
        url = base_url
        if material_id != None:
            url = self._add_backslash(url)
            url += str(material_id)
        if api_endpoint != None:
            url = self._add_backslash(url)
            url += api_endpoint
        else:
            return url
        if calculation_id != None:
            url = self._add_backslash(url)
            url += str(calculation_id)
        else:
            return url
        if property != None:
            url += '?property='
            url += property
        return url

    def _get_lattice_parameters_from_string(self, string):
        lcs=string.split(',')
        lcs[0]=lcs[0][1:]
        lcs[-1]=lcs[-1][:-1]
        lcs_float=[]
        for index,item in enumerate(lcs):
            if index<3:
                lcs_float.append(float(item)*10**(10))
            else:
                lcs_float.append(float(item)/np.pi*180)
        return lcs_float

    def _get_lattice_description(self, elements, lattice_parameters):
        labels=[species_list[x[0]] for x in elements]
        positions = [x[1] for x in elements]
        cell=self._get_lattice_parameters_from_string(lattice_parameters)
        scaled_positions=[[x[0] * cell[0], x[1] * cell[1], x[2] * cell[2]] for x in positions]
        structure=Atoms(symbols=labels,positions=scaled_positions,cell=cell, pbc = True)
        return structure

    @staticmethod
    def _read_api_key(path_to_api_key):
        try:
            key_data = open(os.path.join(path_to_api_key,'api_key'),'r').readline()
            api_key = key_data[:-1] if key_data[-1] == '\n' else key_data
        except FileNotFoundError:
            api_key = None
        return api_key
