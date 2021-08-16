import os, json
import pytest, logging
from time import perf_counter

from simdatframe.data_framework import MaterialsDatabase
from simdatframe.apis.NOMAD_enc_API import API

TEST_TIMING = False

#test_json = {"search_by":{"element":"Al,Si,P","exclusive":True,"page":1,"per_page":10},"has_dos":True, "code_name":["VASP"]}
#test_json_2 = {"search_by":{"element":"Al,P","exclusive":True,"restricted":False,"page":1,"per_page":10},"has_dos":True, "code_name":["VASP"]}
test_json_3 = {"search_by":{"element":"Cu,Ir","exclusive":True,"page":1,"per_page":10},"crystal_system":["cubic"],"system_type":["bulk"],"has_dos":True}
test_json = {"search_by":{"elements":["Cu","Pt"],"exclusive":True,"restricted":True,"page":1,"per_page":10},"has_dos":True}
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
        api = API(logger  = log)
    except:
        assert False
    calc = api.get_calculation('DWTe4soY7aCs8JcP2ty8-Dpow5to','qrXfGffK6LLT-5WI3YuNx662BUKx')
    for key, value in calc.data.items():
        print(key, value)
    materials_list = api._get_materials_list(search_query = test_json)
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
        mats = api.get_calculations_by_search(test_json, parallel = False)
        for item in mats:
            print(item.mid)
    print(api.get_property('DWTe4soY7aCs8JcP2ty8-Dpow5to','qrXfGffK6LLT-5WI3YuNx662BUKx', property_path = 'section_metadata/encyclopedia/status') )

@pytest.mark.skip()
def test_filter_calculation_list():
    try:
        api = API(logger  = log)
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
    api = API(logger  = log)
    if os.path.exists('enc_staging_test.db'):
        os.remove('enc_staging_test.db')
        os.remove(os.path.join('.enc_staging_test_logs', 'enc_staging_test_errors.log'))
        os.remove(os.path.join('.enc_staging_test_logs', 'enc_staging_test_api.log'))
        os.remove(os.path.join('.enc_staging_test_logs', 'enc_staging_test_perf.log'))

    db = MaterialsDatabase(filename = 'enc_staging_test.db', api = api, db_path = '.')
    db.add_material('DWTe4soY7aCs8JcP2ty8-Dpow5to','qrXfGffK6LLT-5WI3YuNx662BUKx')
    db.fill_database(test_json)

    db.add_property('DWTe4soY7aCs8JcP2ty8-Dpow5to:qrXfGffK6LLT-5WI3YuNx662BUKx','parsing_status', property_path = 'section_metadata/encyclopedia/status')
