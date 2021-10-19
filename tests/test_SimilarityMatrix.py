import pytest
from simdatframe.data_framework import MaterialsDatabase
from simdatframe.similarity import SimilarityMatrix, OverlapSimilarityMatrix, BatchedSimilarityMatrix
from simdatframe._test_data import test_data_path
import os, shutil
import numpy as np
from copy import deepcopy
from random import shuffle


@pytest.fixture
def database(tmp_path):
    shutil.copy(os.path.join(test_data_path, 'similarity_matrix_class_test.db'), tmp_path)
    db = MaterialsDatabase(filename = 'similarity_matrix_class_test.db', db_path = str(tmp_path))
    return db

@pytest.fixture
def dos_simat(database):
    return database.get_similarity_matrix("DOS", dtype = np.float64)

@pytest.fixture
def test_fingerprint(database):
    return database.get_fingerprint("DOS", mid = database[0].mid)

@pytest.fixture
def all_dos_fingerprints(database):
    return database.get_fingerprints("DOS")

def test_similarity_matrix(dos_simat, test_fingerprint, tmp_path):
    assert (np.array([np.isclose(test, true) for test, true in zip(dos_simat[3], np.array([0.36703822, 0.12966073, 0.18209813, 1.        , 0.4494311 ,
       0.29072238, 0.31711481, 0.36916951, 0.41760391, 0.32205882,
       0.39399191, 0.2646354 , 0.24575835, 0.42183623, 0.24537219,
       0.35934959, 0.45540309, 0.39454691, 0.12593435]))])).all()
    assert (dos_simat.get_symmetric_matrix()[3] == dos_simat[3]).all()
    # test __next__
    for idx, row in enumerate(dos_simat):
        if idx == 3:
            assert all(dos_simat.matrix[3] == row), "Problem in __next__"
    leave_out_mids = [dos_simat.mids[idx] for idx in [0,10,-1]]
    shortened_dos_simat = dos_simat.get_cleared_matrix(leave_out_mids)
    assert (shortened_dos_simat.get_symmetric_matrix() == np.array([[entry for mid2, entry in zip(dos_simat.mids, row) if not mid2 in leave_out_mids] for mid1, row in zip(dos_simat.mids, dos_simat.get_symmetric_matrix()) if not mid1 in leave_out_mids])).all()
    assert shortened_dos_simat == dos_simat.get_sub_matrix([mid for mid in dos_simat.mids if not mid in leave_out_mids])
    print('\nFunction "get_data_frame()" implicitly tested with "get_sub_matrix()"')
    overlap_row_mids = dos_simat.mids[15:]
    overlap_column_mids = dos_simat.mids[:15]
    overlap_dos_simat = dos_simat.get_overlap_matrix(overlap_column_mids, overlap_row_mids)
    assert len(overlap_dos_simat) == 4
    assert (overlap_dos_simat.get_full_matrix() == np.array([[entry for mid2, entry in zip(dos_simat.mids, row) if mid2 in overlap_column_mids] for mid1, row in zip(dos_simat.mids, dos_simat.get_symmetric_matrix()) if mid1 in overlap_row_mids])).all()
    assert dos_simat.lookup_similarity(test_fingerprint, test_fingerprint) == 1.0
    print('Function "get_entry()" implicitly tested with "lookup_similarity()"')
    copied_dos_simat = dos_simat.get_sub_matrix(dos_simat.mids)
    copied_dos_simat.align(shortened_dos_simat)
    shuffled_mids = deepcopy(dos_simat.mids)
    shuffle(shuffled_mids)
    shuffled_matrix = dos_simat.get_sub_matrix(shuffled_mids)
    assert shuffled_matrix == dos_simat, "Identical matrices with shuffled mids need to equal"
    assert shortened_dos_simat == copied_dos_simat
    assert (1 - dos_simat.get_symmetric_matrix() == dos_simat.get_complement().get_symmetric_matrix()).all()
    print('Function "get_symmetric_matrix()" implicitly tested during all tests.')
    dos_simat.save(data_path = str(tmp_path))
    assert SimilarityMatrix.load(data_path=str(tmp_path)) == dos_simat, "Loading or saving failed."
    assert SimilarityMatrix.load(data_path=str(tmp_path)).fp_type == "DOS", "Did not load correct fp_type."
    print("Skipping some functions, to be added later.")

def test_overlap_similarity_matrix(dos_simat, all_dos_fingerprints):
    overlap_row_mids = dos_simat.mids[15:]
    overlap_column_mids = dos_simat.mids[:15]
    generated_overlap_matrix = dos_simat.get_overlap_matrix(overlap_column_mids, overlap_row_mids)
    new_overlap_matrix = OverlapSimilarityMatrix().calculate(all_dos_fingerprints[15:], all_dos_fingerprints[:15])
    assert new_overlap_matrix == generated_overlap_matrix
    assert (new_overlap_matrix[3] == new_overlap_matrix.matrix[3]).all()
    full_entries = []
    for row in new_overlap_matrix:
        for entry in row:
            full_entries.append(entry)
    assert sorted(np.array(full_entries)) == sorted(new_overlap_matrix.get_entries())
    shuffled_row_mids = deepcopy(overlap_row_mids)
    shuffled_column_mids = deepcopy(overlap_column_mids)
    shuffle(shuffled_row_mids)
    shuffle(shuffled_column_mids)
    assert new_overlap_matrix == new_overlap_matrix.get_sub_matrix(shuffled_row_mids, shuffled_column_mids)
    assert new_overlap_matrix + generated_overlap_matrix == new_overlap_matrix * 2

@pytest.mark.xfail()
def test_batched_similarity_matrix(tmp_path, dos_simat, all_dos_fingerprints):
    serial_matrix = dos_simat.get_symmetric_matrix()
    batched_matrix = []
    bsm = BatchedSimilarityMatrix().calculate(all_dos_fingerprints, folder_name = 'test_matrix', batch_size = 5, data_path=str(tmp_path))
    loaded_bsm = SimilarityMatrix().load(data_path = str(tmp_path), batched = True, batch_size = 5, batch_folder_name = 'test_matrix')
    assert (bsm[3] == loaded_bsm[3]).all()
    for mid in bsm.mids:
        batched_matrix.append(bsm.get_row(mid))
    assert np.allclose(serial_matrix, batched_matrix)
