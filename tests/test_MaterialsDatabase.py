import pytest
from simdatframe.data_framework import MaterialsDatabase
import os

test_json_old = {"search_by":{"element":"Al,Si,P","exclusive":"0","page":1,"per_page":10},"has_dos":"Yes", "code_name":["VASP"]}
test_json_also_old = {"search_by":{"element":"Cu,Ir","exclusive":True,"page":1,"per_page":10},"crystal_system":["cubic"],"system_type":["bulk"],"has_dos":True}
test_json = {"search_by":{"elements":["Cu","Pt"],"exclusive":True,"restricted":True,"page":1,"per_page":10},"has_dos":True}
#@pytest.mark.skip()
def test_database():
    if os.path.exists('data/test.db'):
        os.remove('data/test.db')
        os.remove('data/.test_logs/test_api.log')
        os.remove('data/.test_logs/test_errors.log')
        os.remove('data/.test_logs/test_perf.log')
        os.rmdir('data/.test_logs')
        os.rmdir('data')
    db = MaterialsDatabase(filename = 'test.db', db_path = 'data', path_to_api_key = '..')
    #db2 = MaterialsDatabase(filename = 'test.db', db_path = 'data', api = db.api)
    db.fill_database(test_json)
    db.fill_database(test_json)
#    db.add_material(32042,100490) #Needs update to new API

#    db.add_property(db[0].mid, 'code_version') #Needs update to new API

    db.add_fingerprints(['DOS', 'NMDDOS', 'SOAP', 'IAD', 'SYM', 'IAOD', 'AAO', 'AVEC', 'ChEnv', 'DUMMY', 'PROP', 'PWD'])

    assert db.get_property(db[1].mid, 'mid') == db[1].mid
    assert db.get_property(db[0].mid, 'atomic_density') == db[0].data['atomic_density']

    props, mids = db.get_properties('mid', output_mids = True)

    assert db.get_properties('id') == [entry.id for entry in db]

    assert db.get_properties("atomic_density") == [entry.data['atomic_density'] for entry in db]

    assert props == mids

    assert db._get_row_by_mid(db[0].mid).toatoms() == db[0].toatoms()

    assert db.get_n_entries() == sum([1 for entry in db])
