import pytest
from simdatframe.data_framework import MaterialsDatabase
from simdatframe.similarity import SimilarityMatrix, OverlapSimilarityMatrix, BatchedSimilarityMatrix, MemoryMappedSimilarityMatrix
from simdatframe._test_data import test_data_path
import os
import numpy as np
from copy import deepcopy
from random import shuffle

#read db and fingerprints and so on.
db = MaterialsDatabase(filename = 'similarity_matrix_class_test.db', db_path = test_data_path, path_to_api_key='..')
print('\nSimilarity matrix test:')
print('\nRunning in directory:',os.getcwd())
print('\nLoaded a database with length: ', len(db))
dos_simat = db.get_similarity_matrix('DOS')
soap_simat = db.get_similarity_matrix('SOAP')
test_fingerprint = db.get_fingerprint("DOS", mid = db[0].mid)
all_dos_fingerprints = db.get_fingerprints("DOS")

def test_similarity_matrix():
    assert (np.array([np.isclose(test, true) for test, true in zip(dos_simat[3], np.array([0.36703822, 0.12966073, 0.18209813, 1.        , 0.4494311 ,
       0.29072238, 0.31711481, 0.36916951, 0.41760391, 0.32205882,
       0.39399191, 0.2646354 , 0.24575835, 0.42183623, 0.24537219,
       0.35934959, 0.45540309, 0.39454691, 0.12593435]))])).all()
    assert (dos_simat.get_symmetric_matrix()[3] == dos_simat[3]).all()
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
    assert shortened_dos_simat == copied_dos_simat
    assert (1 - dos_simat.get_symmetric_matrix() == dos_simat.get_complement().get_symmetric_matrix()).all()
    print('Function "get_symmetric_matrix()" implicitly tested during all tests.')
    print("Skipping some functions, to be added later.")

def test_overlap_similarity_matrix():
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

def test_batched_similarity_matrix():
    serial_matrix = dos_simat.get_symmetric_matrix()
    batched_matrix = []
    bsm = BatchedSimilarityMatrix().calculate(all_dos_fingerprints, folder_name = 'test_matrix', batch_size = 5)
    loaded_bsm = SimilarityMatrix().load(data_path = '.', batched = True, batch_size = 5, batch_folder_name = 'test_matrix')
    assert (bsm[3] == loaded_bsm[3]).all()
    for mid in bsm.mids:
        batched_matrix.append(bsm.get_row(mid))
    assert np.allclose(serial_matrix, batched_matrix)

def test_memory_mapped_similarity_matrix():
    mmsm = MemoryMappedSimilarityMatrix().calculate(all_dos_fingerprints, mids = [fp.mid for fp in all_dos_fingerprints], mapped_filename = 'mapped_test_matrix.npy', mids_filename = 'mapped_test_matrix_mids.npy')
    loaded_mmsm = SimilarityMatrix().load(matrix_filename = 'mapped_test_matrix.npy', mids_filename = 'mapped_test_matrix_mids.npy', memory_mapped = True)
    assert (mmsm[3] == loaded_mmsm[3]).all()
