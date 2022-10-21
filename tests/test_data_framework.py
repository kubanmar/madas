import os
from simdatframe.apis.api_core import APIClass
from simdatframe.backend.backend_core import Backend
from simdatframe.fingerprint import Fingerprint
import pytest
from simdatframe import Material, MaterialsDatabase
from ase.build import bulk

from simdatframe.similarity import SimilarityMatrix

class MockAPI():

    def __init__(self, test_material) -> None:
        self._test_material = test_material
        self._called_get_calculation = 0
        self._called_get_property = 0
        self._called_get_calculations_by_search = 0

    def get_calculations_by_search(self, some_string, **kwargs):
        """
        Test doc string
        """
        if some_string == "duplicates":
            self._called_get_calculations_by_search += 1
            return [self.test_material, self.test_material]
        self._called_get_calculations_by_search += 1
        return [self.test_material]

    @property
    def test_material(self):
        return self._test_material

    def get_calculation(self, return_id = 1, *args, **kwargs):
        """
        Test doc string
        """
        self._called_get_calculation += 1
        new_material = self.test_material
        if return_id == 2:
            new_material.mid = "c:d"
        return new_material

    def get_property(self, *args, **kwargs):
        """
        Test doc string
        """
        self._called_get_property += 1
        return 1

    def set_logger(self, logger):
        self.log = logger

    def gen_mid(self, return_id = 1):
        if return_id == 1:
            return "a:b"
        else:
            return "c:d"

    def resolve_mid(self, *args):
        return {"ignore" : "this"}

class MockFingerprint():

    def __init__(self, *args, name = None, **kwargs) -> None:
        self.calculated = False
        self.set_fp_type("Mock")
        self.set_name(name)
        self._mid = "a:b"
        self._data = {"test" : "data"}

    def calculate(self, *args, **kwargs) -> object:
        self.calculated = True
        return self

    def from_data(self, *args, **kwargs) -> object:
        return MockFingerprint().calculate()

class MockBackend(Backend):

    def __init__(self, filename="materials_database.db", filepath="data", rootpath=".", make_dirs=True, key_name="mid", log=None):
        super().__init__(filename, filepath, rootpath, make_dirs, key_name, log)
        self._added_single = 0
        self._added_many = 0
        self._update_buffer = []

    def add_single(self, *args, **kwargs):
        self._added_single += 1

    def add_many(self, entries, **kwargs):
        if len(set(entries)) < len(entries):
            raise ValueError("Tried to write same entry to db twice!") 
        self._added_many += 1

    def get_by_id(self, db_id):
        return Material("a:b", atoms=bulk('Cu', 'fcc', a=3.6), data = {"test" : "data", "dos" : {"a" : "a", "b" : "b"}}, properties = {"test_prop" : "property"})

    def get_single(self, mid=None, **kwargs):
        return self.get_by_id(None)

    def has_entry(self, entry_id):
        if self._added_single or self._added_many:
            return True
        return False

    def get_length(self):
        return self._added_single + self._added_many

    def update_many(self, mids, kwargs_list):
        self._update_buffer.append([mids, kwargs_list])

    def update_single(self, mid, **kwargs):
        self._update_buffer.append([mid, kwargs])

    def update_metadata(self, *args, **kwargs):
        self._metadata.update(**kwargs)

class MockSimilarityMatrix():

    def __init__(self, calculated = False, **kwargs) -> None:
        self._calculated = calculated

    def calculate(self, *args, **kwargs):
        return MockSimilarityMatrix(True)

@pytest.fixture()
def test_atoms():
    return bulk('Cu', 'fcc', a=3.6)

@pytest.fixture()
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

@pytest.fixture()
def test_material(test_atoms):
    return Material("a:b", atoms=test_atoms, data = {"test" : "data"})

@pytest.fixture()
def materials_database(tmpdir, test_material):
    db = MaterialsDatabase(api = MockAPI(test_material), backend=MockBackend(), filepath=tmpdir)
    return db

def test_setup(materials_database):

    assert materials_database.log.name == "materials_database_log"
    assert materials_database.api_logger.name == "materials_database_api"

def test_API_missing_docstrings(tmpdir):

    class EmptyAPI(APIClass):

        def get_calculation(self, *args, **kwargs) -> Material:
            return Material("a")

        def get_calculations_by_search(self, *args, **kwargs):
            return [Material("a")]

        def get_property(self, **kwargs):
            return "a"

    db = MaterialsDatabase(filepath=tmpdir, api=EmptyAPI())

def test_add_material(materials_database):
    
    materials_database.add_material()

    assert materials_database.api._called_get_calculation == 1, "Did not make correct API function call"
    assert materials_database.backend._added_single == 1, "Did not call Backend.add_single function"

    materials_database.add_material()

    assert materials_database.api._called_get_calculation == 1, "Tried to get materials existing material"
    assert materials_database.backend._added_single == 1, "Tried to store existing material"

def test_get_property(materials_database, test_material):
    
    materials_database.add_material()

    assert materials_database.get_property(test_material.mid, "test") == "data", "Unable to get property from Material().data"

    assert materials_database.get_property(test_material.mid, "test_prop") == "property", "Unable to get property from Material().properties"

    assert materials_database.get_property(test_material.mid, "dos/a") == "a", "Unable to get nested property from Material().data"

def test_iteration(materials_database):

    assert len(materials_database) == len(list(materials_database)), "Returned more materials than available in database"

def test_get_properties(materials_database, test_material):

    materials_database.add_material()

    properties, mids = materials_database.get_properties("dos/a", True)
    assert properties == ["a"], "Did not recieve correct list of properties"
    assert mids == [test_material.mid], "Did not recieve correct list of mids"

def test_getitem(materials_database):
    materials_database.add_material()
    assert materials_database["a"].mid == "a:b", "Did not recieve correct material by mid"
    assert materials_database[0].mid == "a:b", "Did not recieve correct material by integer"

def test_get_fingerprint(materials_database, monkeypatch):
    
    monkeypatch.setattr(Fingerprint, "__init__", MockFingerprint.__init__)
    monkeypatch.setattr(Fingerprint, "calculate", MockFingerprint.calculate)
    monkeypatch.setattr(Fingerprint, "from_data", MockFingerprint.from_data)

    fp = materials_database.get_fingerprint("Mock", "a")

    assert fp.fp_type == "Mock", "Calulated wrong fingerprint"
    assert fp.mid == "a:b", "Wrong mid for fingerprint"

def test_get_fingerprints(materials_database, monkeypatch):

    materials_database.add_material()

    monkeypatch.setattr(Fingerprint, "__init__", MockFingerprint.__init__)
    monkeypatch.setattr(Fingerprint, "calculate", MockFingerprint.calculate)
    monkeypatch.setattr(Fingerprint, "from_data", MockFingerprint.from_data)

    fps = materials_database.get_fingerprints("Mock", "a")

    assert len(fps) == 1, "Wrong length of fingerprint list"
    assert fps[0].fp_type == "Mock", "Calulated wrong fingerprint"
    assert fps[0].mid == "a:b", "Wrong mid for fingerprint"

def test_get_similarity_matrix(materials_database, monkeypatch):
    
    monkeypatch.setattr(SimilarityMatrix, "__init__", MockSimilarityMatrix.__init__)
    monkeypatch.setattr(SimilarityMatrix, "calculate", MockSimilarityMatrix.calculate)

    monkeypatch.setattr(Fingerprint, "__init__", MockFingerprint.__init__)
    monkeypatch.setattr(Fingerprint, "calculate", MockFingerprint.calculate)
    monkeypatch.setattr(Fingerprint, "from_data", MockFingerprint.from_data)

    materials_database.add_material()

    simat = materials_database.get_similarity_matrix(None)

    assert simat._calculated == True, "Similarity matrix was not calculated"

def test_add_fingerprint(materials_database, monkeypatch):

    monkeypatch.setattr(Fingerprint, "__init__", MockFingerprint.__init__)
    monkeypatch.setattr(Fingerprint, "calculate", MockFingerprint.calculate)
    monkeypatch.setattr(Fingerprint, "from_data", MockFingerprint.from_data)

    materials_database.add_material()

    materials_database.add_fingerprint("Mock", name = "Mock1")

    assert materials_database.backend._update_buffer == [[['a:b'], [{'Mock1': '{"test": "data"}'}]]], "Tried to update with wrong data"

    assert materials_database.get_metadata()["fingerprints"] == ["Mock1"], "Did not write correct metadata"

def test_add_fingerprint_by_type(materials_database):

    class TestFingerprint(Fingerprint):

        def calculate(self, material, *args, **kwargs):
            self.set_mid(material)
            self.set_data("calculated", {"test" : 1})
            return self

    def Test_similarity(fp1, fp2):
        return fp1.data["calculated"]["test"] + fp2.data["calculated"]["test"]

    materials_database.add_material()

    materials_database.add_fingerprint(TestFingerprint)

    fps = materials_database.get_fingerprints(TestFingerprint, similarity_function = Test_similarity)

    assert fps[0].mid == "a:b", "Did not load fingerprint mid"

    assert fps[0].get_similarities(fps) == [2], "Did not calculate similarities for deserialized fingerprint."

def test_add_fingerprints(materials_database, monkeypatch):

    monkeypatch.setattr(Fingerprint, "__init__", MockFingerprint.__init__)
    monkeypatch.setattr(Fingerprint, "calculate", MockFingerprint.calculate)
    monkeypatch.setattr(Fingerprint, "from_data", MockFingerprint.from_data)

    materials_database.add_material()

    materials_database.add_fingerprints(["Mock", "Mock"], ["m1", "m2"])

    assert materials_database.backend._update_buffer == [[['a:b'], [{'m1': '{"test": "data"}', 'm2' : '{"test": "data"}'}]]], "Tried to update with wrong data"

    assert materials_database.get_metadata()["fingerprints"] == ["m1", "m2"], "Did not write correct metadata"

def test_fill_database(materials_database, caplog):
    
    materials_database.fill_database({"a" : "a"})

    assert materials_database.backend._added_many == 1, "Did not add many to backend"
    assert materials_database.api._called_get_calculations_by_search == 1, "Did not call API"
    assert materials_database.get_metadata()["search_queries"] == ['[{"a": "a"}]']

    materials_database.fill_database({"a" : "a"})

    assert materials_database.backend._added_many == 1, "Tried to add redundant materials"
    assert materials_database.api._called_get_calculations_by_search == 1, "Queried API twice"

    materials_database.fill_database({"b" : "b"})

    assert materials_database.backend._added_many == 2, "Tried to add redundant materials"
    assert materials_database.api._called_get_calculations_by_search == 2, "Queried API twice"

    assert "Material a:b already in database. Skipping." in str(caplog.text), "Did not avoid adding duplicates"

def test_fill_database_no_duplicates_from_api(materials_database):

    materials_database.fill_database("duplicates")

def test_get_random(materials_database):

    materials_database.add_material()

    assert materials_database.get_random(return_mid = True) == "a:b", "Did not get material from get_random" 

def test_update_entry(materials_database):

    materials_database.update_entry("a", b = "c")

    assert materials_database.backend._update_buffer == [["a", {"b" : "c"}]], "Wrong data passed to update function"

def test_update_entries(materials_database):

    materials_database.update_entries(["a"], [{"b" : "c"}])

    assert materials_database.backend._update_buffer == [[["a"], [{"b" : "c"}]]], "Wrong data passed to update function"

def test_add_property(materials_database):

    materials_database.add_property("a", "b")

    assert materials_database.api._called_get_property == 1, "Did not call API"
    assert materials_database.backend._update_buffer == [['a', {"data" : {"b" : 1}}]], "Did not update data correctly"

def test_update_metadata(tmpdir):

    db = MaterialsDatabase(filepath=tmpdir, rootpath="")

    db._update_metadata({"test":"this"})

    del db

    db = MaterialsDatabase(filepath=tmpdir, rootpath="")

    assert db.get_metadata() == {"test":"this"}, "Did not recover metadata"
