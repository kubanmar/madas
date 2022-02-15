import os
from simdatframe.fingerprint import Fingerprint
import pytest
import simdatframe
from simdatframe.data_framework import MaterialsDatabase, Material
from ase.build import bulk
import re, json

@pytest.fixture
def test_atoms():
    return bulk('Cu', 'fcc', a=3.6)

@pytest.fixture
def test_query():
    return {
        "search_by":
            {
                "elements": ["Cu","Pt"], 
                "exclusive":True, 
                "restricted":True,
                "page":1,
                "per_page":10
            },
        "has_dos":True,
        "testquery" : "query"
    }

@pytest.fixture
def test_material(test_atoms):
    return Material("a:b", atoms=test_atoms, data = {"test" : "data"})

class MockAPI():

    def __init__(self, test_material) -> None:
        self._test_material = test_material

    def get_calculations_by_search(self, *args, **kwargs):
        """
        Test doc string
        """
        return [self.test_material]

    @property
    def test_material(self):
        return self._test_material

    def get_calculation(self, return_id, *args, **kwargs):
        """
        Test doc string
        """
        new_material = self.test_material
        if return_id == 2:
            new_material.mid = "c:d"
        return new_material

    def get_property(self, *args, **kwargs):
        """
        Test doc string
        """
        return 1

    def set_logger(self, logger):
        self.log = logger

    def gen_mid(self, return_id):
        if return_id == 1:
            return "a:b"
        else:
            return "c:d"

    def resolve_mid(self, *args):
        return {"ignore" : "this"}

class MockFingerprint():

    def __init__(self, *args, name = None, **kwargs) -> None:
        self.calculated = False
        self.fp_type = "Mock"
        self.name = name
        self.mid = "a:b"

    def calculate(self, *args, **kwargs) -> object:
        self.calculated = True
        return self

    def get_data_json(self, *args, **kwargs):
        return json.dumps({"test" : "data"})

@pytest.mark.skip()
def test_Material():
    pass

def test_database(tmp_path, test_material, test_query, caplog, monkeypatch, test_atoms):

    """
    Test database behaviour. TODO: split into separate tests by using a fixture of the DB
    """

    def mock_gen_fingerprints_list(fp_type, *args, name = None, **kwargs):
        return [MockFingerprint(name = name)]

    monkeypatch.setattr(MaterialsDatabase, "gen_fingerprints_list", mock_gen_fingerprints_list)

    db = MaterialsDatabase(filename = 'test.db', db_path = str(tmp_path), api = MockAPI(test_material))

    db.fill_database(test_query)

    assert 'testquery' in str(caplog.text), "Did not write query to logs." 

    assert len(db) == 1, "Not exactly one material added in fill_database"

    assert db[0].mid == test_material.mid, "wrote wrong material to database"

    db.fill_database(test_query)

    assert len(db) == 1, "fill_database executed same query twice"

    assert db.get_property(test_material.mid, "test") == "data", "could not get property data from material"

    assert db.get_properties("test") == ["data"], "could not get properties from database"

    assert db.get_fingerprint("XX", mid = "XX") == None, "Found non-existing fingeprint"
    assert db.get_fingerprint("XX", mid = "XX", db_id = 100) == None, "Found non-existing fingerprint by db_id"

    assert "No material with mid XX." in str(caplog.text), "Did not log missing material in get_fingerprint"

    db.add_fingerprint("Mock", name = "test_name")

    assert "Finished for fp_type: Mock" in str(caplog.text), "Did not confirm fingerprint writing"

    assert hasattr(db[0], "test_name"), "Fingerprint was not stored for db entry"

    assert db.get_formula(db[0].mid) == "Cu", "Material has wrong formula"

    assert db.get_atoms(db[0].mid) == test_atoms, "Wrong atoms from db entry"

    assert db.get_random() == "a:b", "Random only entry has wrong mid"

    db.add_material(1)

    assert len(db) == 1, "Added material with same mid twice"

    db.add_material(2)

    assert len(db) == 2, "Did not add new material"

    db.update_entries(["a:b"], [{"delete":"this"}])

    assert db["a:b"]["delete"] == "this", "Failed updating entries"

    db.delete_keys("delete")

    assert not hasattr(db[0], "delete"), "DB keys did not get removed"

    db.add_property(db[0].mid, "test_property")

    assert db[0]["test_property"] == 1, "Failed adding properties"

    mat = db.get_material(db[0].mid)

    assert mat.mid == "a:b", "Did not get correct material from get_material"
