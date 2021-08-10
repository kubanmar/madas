import os, json
import pytest, logging
from time import perf_counter

from simdatframe.data_framework import MaterialsDatabase
from simdatframe.apis.NOMAD_enc_API import API

TEST_TIMING = False

test_json = {"search_by":{"element":"Al,Si,P","exclusive":"0","page":1,"per_page":10},"has_dos":"Yes", "code_name":["VASP"]}
test_json_2 = {"search_by":{"element":"Al,P","exclusive":"0","page":1,"per_page":10},"has_dos":"Yes","code_name":["FHI-aims"]}
test_json_3 = {"search_by":{"element":"Cu,Ir","exclusive":"0","page":1,"per_page":10},"crystal_system":["cubic"],"system_type":["bulk"],"has_dos":"Yes"}
db_filename = 'unit_test_db'
file_path = '.'

# Initialize log for the tests
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(os.path.join('.' + db_filename + '_logs' , db_filename + '_api'))
log.setLevel(logging.DEBUG)

api_file = logging.FileHandler(os.path.join(file_path, db_filename + '_api.log'))
api_file.setLevel(logging.INFO)
api_file.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

log.addHandler(api_file)
log.addHandler(console)

@pytest.mark.skip()
def test_API_class():
    try:
        api = API(key_path = '..', logger  = log)
    except:
        assert False
    calc = api.get_calculation(665886,1880033)
    for key, value in calc.data.items():
        print(key, value)
    materials_list = api._get_materials_list(search_query = test_json, per_page = 10)
    for item in materials_list:
        print(item)

    if TEST_TIMING:
        t1 = perf_counter()
        mats = api.get_calculations_by_search(test_json_3)
        t2 = perf_counter()
        time_serial = t2-t1
        t1 = perf_counter()
        mats = api.get_calculations_by_search(test_json_3, parallel = True)
        t2 = perf_counter()
        time_parallel = t2-t1
        print(f"Serial execution of get_calculations_by_search takes {time_serial} seconds.")
        print(f"Parallel execution of get_calculations_by_search takes {time_parallel} seconds.")
    else:
        mats = api.get_calculations_by_search(test_json_3, parallel = True)
        for item in mats:
            print(item.mid)
    print(api.get_property(665886,1880033, 'formula'))

    url = api._construct_url(api.base_url, 1, 'calculations', 1)
    print(api._api_call(url).json())

    api.print_api_keys()

@pytest.mark.skip()
def test_filter_calculation_list():
    try:
        api = API(key_path = '..', logger  = log)
    except:
        assert False
    list_to_test = [{"value":1, "is_true":True, "rating":1},
                    {"value":2, "is_true":False, "rating":2},
                    {"value":3, "is_true":True, "rating":3},
                    {"value":4, "is_true":True, "rating":4},
                    {"value":5, "is_true":False, "rating":5}]
    filtered_list = api._filter_calculation_list(list_to_test, {"is_true":True})
    assert sorted(filtered_list, key = lambda x: x["rating"])[-1]["value"] == 4

@pytest.mark.skip()
def test_setup():

    if os.path.exists('enc_staging_test.db'):
        os.remove('enc_staging_test.db')
        os.remove(os.path.join('.enc_staging_test_logs', 'enc_staging_test_errors.log'))
        os.remove(os.path.join('.enc_staging_test_logs', 'enc_staging_test_api.log'))
        os.remove(os.path.join('.enc_staging_test_logs', 'enc_staging_test_perf.log'))

    db = MaterialsDatabase(filename = 'enc_staging_test.db', path_to_api_key = '..', db_path = '.')
    db.add_material(665886,1880033)
    db.fill_database(test_json_2)

    db.add_property("665886:1880033",'code_version')

@pytest.mark.skip()
def test_production_server():
    production_url = 'https://encyclopedia.nomad-coe.eu/api/v1.0/materials'
    try:
        api = API(key_path = '..', logger  = log, base_url = production_url)
    except:
        assert False
    calc = api.get_calculation(665886,1880033)
    for key, value in calc.data.items():
        print(key, value)
    materials_list = api._get_materials_list(search_query = test_json, per_page = 10)
    for item in materials_list:
        print(item)

    mats = api.get_calculations_by_search(test_json_2)
    for item in mats:
        print(item.mid)
    api.print_api_keys()
    print('formula of 665886:1880033', api.get_property(665886,1880033, 'pressure'))

    url = api._construct_url(api.base_url, 665886, 'calculations', 1880033)
    print(api._api_call(url).json())

    api.print_api_keys()
