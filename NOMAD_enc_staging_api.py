
API_base_url = 'http://enc-staging-nomad.esc.rzg.mpg.de/v1.0/materials'

class API():

    def __init__(self, base_url = API_base_url, access_token = None, key_path = '.'):
        self.base_url = base_url
        self.auth = (access_token, '')

    def _filter_calculation_list(self, calculation_list, filter_json):
        filtered_list = []
        for calc in calculation_list:
            for key, value in filter_json.items():
                if key in calc.keys():
                    if calc[key] == value:
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

    def _select_calc(self, calc_list):
        calc_list = self._filter_calculation_list(calc_list, filter_json={'has_dos':True})
        scored_list = [(self._evaluate_calc(calc), calc['id']) for calc in calc_list]
        scored_list.sort(reverse = True)
        print(scored_list)
        return scored_list[0][1]

    @staticmethod
    def _construct_url(base_url, material_id = None, api_endpoint = None, calculation_id = None, property = None):
        url = base_url
        if material_id != None:
            url = _add_backslash(url)
            url += str(material_id)
        if api_endpoint != None:
            url = _add_backslash(url)
            url += api_endpoint
        else:
            return url
        if calculation_id != None:
            url = _add_backslash(url)
            url += str(calculation_id)
        else:
            return url
        if property != None:
            url += '?property='
            url += property
        return url

    @staticmethod
    def _add_backslash(string):
        if string[-1] != '/':
            string += '/'
        return string

    @staticmethod
    def _no_pagination(url):
        url += '?pagination=off'
        return url

    @staticmethod
    def _read_api_key(path_to_api_key):
        key_data = open(os.path.join(path_to_api_key,'api_key'),'r').readline()
        api_key = key_data[:-1] if key_data[-1] == '\n' else key_data
        return api_key
