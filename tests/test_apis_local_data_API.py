import pytest

import os

import pandas as pd
import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.io import write

from simdatframe.apis.local_data_API import FileReaderASE, CSVPropertyReader, API
from simdatframe import Material

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

    ref_material = Material(mid = api.gen_mid("a", "geometry.in", ".", "properties.csv"),
                            atoms=atoms,
                            data=data["a"])

    mat = api.get_calculation("a", "geometry.in", ".", "properties.csv", property_file_id="a")

    assert ref_material == mat, "Did not read correct material from test data"

def test_API_get_calculations_by_search(atoms_test_data, csv_test_data):
    
    tmpdir, atoms = atoms_test_data
    _, data = csv_test_data

    api = API(root=tmpdir)

    ref_material_a = Material(mid = api.gen_mid(os.path.join(tmpdir, "a"), 
                                                "geometry.in", 
                                                ".", 
                                                "properties.csv"),
                            atoms=atoms,
                            data=data["a"])
    ref_material_b = Material(mid = api.gen_mid(os.path.join(tmpdir, "b"), 
                                                "geometry.in", 
                                                ".", 
                                                "properties.csv"),
                            atoms=atoms,
                            data=data["b"])

    read_data = api.get_calculations_by_search(tmpdir, "geometry.in", ".", "properties.csv")

    assert read_data[0] == ref_material_a, "did not read material a"
    assert read_data[1] == ref_material_b, "did not read material b"
