import pytest

import os

import pandas as pd

from ase.build import bulk
from ase.io import write

from madas.apis.local_data_API import FileReaderASE, CSVPropertyReader, API
from madas import Material, MaterialsDatabase

@pytest.fixture()
def atoms_test_data(tmpdir):
    atoms = bulk("Si")
    for folder_name in ["a", "b"]:
        os.makedirs(os.path.join(tmpdir, folder_name))
        filepath = os.path.join(tmpdir, folder_name, "geometry.in")
        write(filepath, atoms)
    return tmpdir, atoms

@pytest.fixture()
def csv_test_data(tmpdir):
    df = pd.DataFrame([[1, 2 ], [3, 4]], index = ["a", "b"], columns = ["property1", "property2"])
    filepath = os.path.join(tmpdir, "properties.csv")
    df.to_csv(filepath)
    return tmpdir, df.transpose().to_dict()

def test_FileReaderASE(atoms_test_data):

    tmpdir, atoms = atoms_test_data

    data_dir = os.path.join(tmpdir, "a")

    reader = FileReaderASE(file_path=data_dir)

    read_atoms = reader.read("geometry.in")

    assert read_atoms == atoms, "Atoms from file are not identical with original"

def test_CSVPropertyReader(csv_test_data):

    dir_, data = csv_test_data

    reader = CSVPropertyReader(dir_)

    read_data = reader.read("properties.csv")

    assert data == read_data, "Did not read correct data from CSV file"

def test_API_get_calculation(atoms_test_data, csv_test_data):

    tmpdir, atoms = atoms_test_data
    _, data = csv_test_data

    api = API(root=tmpdir)

    ref_material = Material(mid = api._gen_mid("a", "geometry.in", ".", "properties.csv"),
                            atoms=atoms,
                            data=data["a"])

    mat = api.get_calculation("a", "geometry.in", ".", "properties.csv", property_file_id="a")

    assert ref_material == mat, "Did not read correct material from test data"

def test_API_get_calculations_by_search(atoms_test_data, csv_test_data):
    
    tmpdir, atoms = atoms_test_data
    _, data = csv_test_data

    api = API(root=tmpdir)

    ref_material_a = Material(mid = api._gen_mid(os.path.join(tmpdir, "a"), 
                                                "geometry.in", 
                                                ".", 
                                                "properties.csv"),
                            atoms=atoms,
                            data=data["a"])
    ref_material_b = Material(mid = api._gen_mid(os.path.join(tmpdir, "b"), 
                                                "geometry.in", 
                                                ".", 
                                                "properties.csv"),
                            atoms=atoms,
                            data=data["b"])

    read_data = api.get_calculations_by_search(tmpdir, "geometry.in", ".", "properties.csv")

    assert len(read_data) == 2, "wrong number of data points read"
    assert ref_material_a in read_data, "did not read material a"
    assert ref_material_b in read_data, "did not read material b"

def test_API_get_materials_from_different_locations(tmpdir, monkeypatch):

    def mock_calculations_by_search(folder_path: str, 
                                   file_name: str, 
                                   property_file_path: str, 
                                   property_file_name: str, 
                                   file_reader_kwargs: str = {}, 
                                   property_reader_kwargs: str = {}):
        if folder_path == "1":
            return [Material("a")]
        else:
            return [Material("b")]

    monkeypatch.setattr(API, "get_calculations_by_search", mock_calculations_by_search)

    db = MaterialsDatabase(filepath=tmpdir, api=API(), log_mode="stream")

    db.fill_database("1", None, None)

    db.fill_database("2", None, None)