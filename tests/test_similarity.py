import pytest
from simdatframe.data_framework import MaterialsDatabase
import os

@pytest.mark.skip()
def test_simat_clearing():
    if os.path.exists('test_simat_clearing.db'):
        os.remove('test_simat_clearing.db')
        os.remove('test_simat_clearing_errors.log')
        os.remove('test_simat_clearing_network.log')
        os.remove('test_simat_clearing_perf.log')

    small_json = {"search_by":{"element":"C,Na,O","exclusive":"1","page":1,"per_page":10},"has_dos":"Yes"}
    print('\n')
    db = MaterialsDatabase(filename = 'test_simat_clearing.db', path_to_api_key = '..', db_path = '.', silent_logging = True)
    db.fill_database(small_json)

    db.add_fingerprint("DOS")

    full_matrix = db.get_similarity_matrix("DOS")

    db.atoms_db.update(4, DOS = 'None')

    smaller_matrix = db.get_similarity_matrix("DOS")

    matrix1, matrix2, new_mids = full_matrix.get_matching_matrices(smaller_matrix)

    for lines in zip(matrix1, matrix2):
        assert lines[0].all() == lines[1].all()
