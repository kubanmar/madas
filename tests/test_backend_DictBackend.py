import pytest

from madas.backend.dictionary_backend import DictBackend
from madas import Material

from ase.build import bulk
from ase import Atoms

@pytest.fixture()
def backend():
    _backend = DictBackend()
    return _backend

@pytest.fixture()
def material():
    return Material("Si:test", atoms = bulk("Si"), data = {"test" : "data"}, properties={"test" : "me"})

@pytest.fixture()
def materials():
    return [Material(f"Si:test{idx}", atoms = bulk("Si"), data = {"test" : f"data{idx}"}, properties={"test" : "me"}) for idx in range(1000)]

def test_add_single(backend, material, capsys):
    backend.add_single(material)
    captured = capsys.readouterr()
    assert Atoms.fromdict(backend._dict["Si:test"]["atoms"]) == material.atoms, "Wrong atoms are written"
    assert captured.out.strip() == backend._log_write_message("Si:test"), "Logging to STDOUT failed."

def test_get_single(backend, material):
    backend.add_single(material)
    mat = backend.get_single(mid = material.mid)
    assert material == mat, "Did not return correct (same) material from database"

def test_add_many(backend, materials):
    backend.add_many(materials)
    for ref in materials:
        mat = backend.get_single(mid = ref.mid)
        assert ref == mat, f"Wrong data for entry with mid {ref.mid}"

def test_get_many(backend, materials):
    backend.add_many(materials)
    mid_list = [mat.mid for mat in materials]
    mats = backend.get_many(mids = mid_list)
    for mat in mats:
        assert mat in materials, "Got material that was not in initial list"
    for mat in materials:
        assert mat in mats, "Material was not returned by database"

def test_get_by_id(backend, material):
    backend.add_single(material)
    mat = backend.get_by_id(0)
    assert material == mat, "Did not return correct material from database"

def test_update_single(backend, material):
    backend.add_single(material)
    backend.update_single(material.mid, something = "new")
    backend.update_single(material.mid, different = 1)
    mat = backend.get_single(material.mid)
    assert mat.properties["something"] == "new", "Failed to update property of single material"
    assert mat.properties["different"] == 1, "Failed to update integer property of single material"

def test_update_many(backend, materials):
    backend.add_many(materials)
    mids = [mat.mid for mat in materials]
    backend.update_many(mids, [{"something" : f"different{idx}"} for idx in range(len(materials))])
    mats = backend.get_many(mids)
    for idx, mat in enumerate(mats):
        print(mat)
        assert mat.properties["something"] == f"different{idx}", f"Failed to update property of many materials: {mat}"

def test_metadata():
    backend = DictBackend()
    backend.update_metadata(something = "new")
    assert backend.metadata == {"something" : "new"}, "Failed to update metadata"


def test_has_entry(backend, material):
    backend.add_single(material)
    assert not backend.has_entry("a"), "Non existing entry reported"
    assert backend.has_entry(material.mid), "Exisiting entry not found"

def test_get_length(backend, material):
    assert backend.get_length() == 0, "Finite length for empty database"
    backend.add_single(material)
    assert backend.get_length() == 1, "Wrong length for one-entry database"

def test_to_json(backend, material):
    backend.add_single(material)
    data = backend.to_json()
    restored = DictBackend.from_json(data)
    assert backend.get_single(material.mid) == restored.get_single(material.mid), "Serialized database did not contain same data"