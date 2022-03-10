from _pytest.assertion import pytest_sessionfinish
import pytest
from simdatframe import MaterialsDatabase
from simdatframe.fingerprint import Fingerprint
from simdatframe.similarity import SimilarityMatrix, OverlapSimilarityMatrix, BatchedSimilarityMatrix
from simdatframe._test_data import test_data_path
import os, shutil
import numpy as np
from copy import deepcopy
from random import random, shuffle


@pytest.fixture
def database(tmp_path):
    shutil.copy(os.path.join(test_data_path, 'similarity_matrix_class_test.db'), tmp_path)
    db = MaterialsDatabase(filename = 'similarity_matrix_class_test.db', rootpath = str(tmp_path), filepath=".")
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

@pytest.fixture()
def simat():
    simat = SimilarityMatrix()
    triangular_matrix = [[1,2,3],[4,5],[6]]
    mids = [str(idx) for idx in range(3)]
    simat.set_matrix(triangular_matrix)
    simat.set_mids(mids)
    return simat

class MockFingerpint():

    def __init__(self) -> None:
        self.fp_type = "Mock"
        self.name = "mock"
        self.mid = str(np.round(random(), 5))

    def calculate(self, *args, **kwargs):
        self.data =  random()
        return self

    def get_similarities(self, fps):
        value = self.data
        return [1/ (abs(value - other.data) + 1) for other in fps]

    def get_similarity(self, other):
        return 1/ (abs(self.data - other.data) + 1)

def test_SimilarityMatrix_metadata():

    fps = [MockFingerpint().calculate() for _ in range(10)]

    simat = SimilarityMatrix().calculate(fps)

    assert simat.fp_type == "Mock", "did not copy fp_type"
    assert simat.fp_name == "mock", "did not copy fp_name"

    metadata = simat.get_metadata()

    assert metadata["fp_type"] == "Mock", "did not copy fp_type to metadata"
    assert metadata["fp_name"] == "mock", "did not copy fp_name to metadata"

    simat.set_metadata({"fp_type" : "a", "fp_name" : "b"})

    assert simat.fp_type == "a", "did not set fp_type to metadata"
    assert simat.fp_name == "b", "did not set fp_name to metadata"

def test_SimilarityMatrix_set_matrix():

    simat = SimilarityMatrix()

    triangular_matrix = [[1,2,3],[4,5],[6]]

    simat.set_matrix(triangular_matrix)

    assert (simat.matrix == [[1,2,3], [2,4,5], [3,5,6]]).all(), "Did not set triangular matrix correctly"

    square_matrix = [[1,2,3], [4,5,6], [7,8,9]]

    simat.set_matrix(square_matrix)

    assert (simat.matrix == square_matrix).all(), "Did not set square matrix correctly"

def test_SimilarityMatrix_set_mids():

    simat = SimilarityMatrix()

    triangular_matrix = [[1,2,3],[4,5],[6]]

    simat.set_matrix(triangular_matrix)

    mids = [str(idx) for idx in range(3)]

    simat.set_mids(mids)

    assert all(mids == simat.mids), "Did not set mids correctly"

def test_SimilarityMatrix_set_dataframe():

    simat = SimilarityMatrix()

    square_matrix = [[1,2,3], [4,5,6], [7,8,9]]

    from pandas import DataFrame

    frame = DataFrame(square_matrix, index=["a", "b", "c"], columns=["a", "b", "c"])

    simat.set_dataframe(frame)

    assert all(simat.dataframe == frame), "Did not set dataframe correctly"

    assert all(simat.mids == ["a", "b", "c"]), "Mids were not set corretly by dataframe"

    assert (simat.matrix == square_matrix).all(), "Matrix values were not set corretly by dataframe"

def test_SimilarityMatrix_calculate():

    fps = [MockFingerpint().calculate() for _ in range(10)]

    simat = SimilarityMatrix().calculate(fps)

    assert np.allclose(simat.matrix, [fp.get_similarities(fps) for fp in fps]), "Calculated similarities are wrong"

    simat = SimilarityMatrix().calculate(fps, symmetric=False)

    assert np.allclose(simat.matrix, [fp.get_similarities(fps) for fp in fps]), "Calculated similarities (not symmetric) are wrong"

    simat = SimilarityMatrix().calculate(fps, multiprocess=False)

    assert np.allclose(simat.matrix, [fp.get_similarities(fps) for fp in fps]), "Calculated similarities (serial) are wrong"

    simat = SimilarityMatrix().calculate(fps, multiprocess=False, symmetric=False)

    assert np.allclose(simat.matrix, [fp.get_similarities(fps) for fp in fps]), "Calculated similarities (serial, not symmetric) are wrong"

def test_SimilarityMatrix_get_sub_matrix(simat):

    sub_matrix = simat.get_sub_matrix(["2", "0"])

    assert (sub_matrix.matrix == [[6,3],[3,1]]).all(), "Sub-matrix values are wrong"

    assert all(sub_matrix.mids == ["2", "0"]), "Wrong mids for sub-matrix"

    simat.get_sub_matrix(["2", "0"], copy=False)

    assert simat == sub_matrix, "Inplace sub-matrix failed"

def test_SimilarityMatrix_get_overlap_matrix(simat):

    # matrix shape:
    #
    #     | "0" | "1" | "2" 
    # --------------------
    # "0" |  1  |  2  |  3
    # "1" |  2  |  4  |  5
    # "2" |  3  |  5  |  6

    overlap_matrix = simat.get_overlap_matrix(["2"], ["1", "0"])

    assert all(overlap_matrix.row_mids == ["1", "0"]), "Wrong row mids for overlap matrix"
    assert all(overlap_matrix.column_mids == ["2"]), "Wrong row mids for overlap matrix"

    assert (overlap_matrix.matrix == [[5], [3]]).all(), "Wrong values for overlap matrix"


def test_SimilarityMatrix_lookup_similarity():

    fps = [MockFingerpint().calculate() for _ in range(10)]

    simat = SimilarityMatrix().calculate(fps)

    fp1 = Fingerprint(similarity_function=simat.lookup_similarity)
    fp1.set_mid(fps[0].mid)

    fp2 = Fingerprint()
    fp2.set_mid(fps[1].mid)

    assert fp1.get_similarity(fp1) == 1, "Wrong self-similarity for lookup similarity"

    assert fp1.get_similarity(fp2) == fps[0].get_similarity(fps[1]), "Wrong similarity for lookup similarity"

def test_SimilarityMatrix_train_test_split(simat):

    train_mids = ["2", "1"]
    test_mids = ["0"]

    train_matrix, test_matrix = simat.train_test_split(train_mids, test_mids)

    assert (train_matrix.matrix == [[6,5], [5,4]]).all(), "Wrong values for train matrix"
    assert (test_matrix.matrix == [[3, 2]]).all(), "Wrong values for test matrix"

    assert all(train_matrix.mids == train_mids), "Wrong mids for train matrix"
    assert all(test_matrix.column_mids == train_mids), "Wrong column mids for test matrix"
    assert all(test_matrix.row_mids == test_mids), "Wrong row mids for test matrix"

def test_SimilarityMatrix_align():

    mids1 = ["a", "b", "c", "d", "e"]
    matrix1 = np.array(range(1,6)) * np.eye(5)
    mids2 = ["k", "b", "c", "j", "e"]
    matrix2 = np.array(range(6,11)) * np.eye(5)
    mids3 = ["b", "c"]
    matrix3 = np.array([11,12]) * np.eye(2)

    simat1 = SimilarityMatrix(matrix = matrix1, mids = mids1)
    simat2 = SimilarityMatrix(matrix = matrix2, mids = mids2)

    simat1.align(simat2)

    assert all(simat1.mids == ["b", "c", "e"]), "Did not align mids matrix1"
    assert all(simat2.mids == ["b", "c", "e"]), "Did not align mids matrix2"

    assert (simat1.matrix == [[2,0,0], [0,3,0], [0,0,5]]).all(), "Did not align values matrix1"
    assert (simat2.matrix == [[7,0,0], [0,8,0], [0,0,10]]).all(), "Did not align values matrix2"

    simat1 = SimilarityMatrix(matrix = matrix1, mids = mids1)
    simat2 = SimilarityMatrix(matrix = matrix2, mids = mids2)
    simat3 = SimilarityMatrix(matrix = matrix3, mids = mids3)

    simat1.align([simat2, simat3])

    assert all(simat1.mids == ["b", "c"]), "Did not align mids matrix1"
    assert all(simat2.mids == ["b", "c"]), "Did not align mids matrix2"
    assert all(simat3.mids == ["b", "c"]), "Did not align mids matrix3"

    assert (simat1.matrix == [[2,0], [0,3]]).all(), "Did not align values matrix1"
    assert (simat2.matrix == [[7,0], [0,8]]).all(), "Did not align values matrix2"
    assert (simat3.matrix == [[11,0], [0,12]]).all(), "Did not align values matrix3"

def test_SimilarityMatrix_get_entry(simat):

    assert simat.get_entry("0", "2") == 3, "Did not get correct value"

def test_SimilarityMatrix_get_row(simat):

    assert all(simat.get_row("1") == [2,4,5]), "Got wrong row values"

def test_SimilarityMatrix_get_unique_entries(simat):

    assert (simat.get_unique_entries() == list(range(1,7))).all(), "Wrong unique entries"

def test_SimilarityMatrix_get_k_most_similar(simat):

    most_similar = simat.get_k_most_similar("2")

    assert most_similar == {"2" : {"1" : 5.0, "0" : 3.0}}, "Wrong list of most similar materials"

def test_SimilarityMatrix_save_load(simat, tmpdir):

    simat.save(filepath = str(tmpdir))

    matrix = SimilarityMatrix.load(filepath=str(tmpdir))

    assert simat == matrix, "Save/load did not work properly"

def test_SimilarityMatrix_get_matching_matrices():

    mids1 = ["a", "b", "c", "d", "e"]
    matrix1 = np.array(range(1,6)) * np.eye(5)
    mids2 = ["k", "b", "c", "j", "e"]
    matrix2 = np.array(range(6,11)) * np.eye(5)

    simat1 = SimilarityMatrix(matrix = matrix1, mids = mids1)
    simat2 = SimilarityMatrix(matrix = matrix2, mids = mids2)

    simat1a, simat2a = simat1.get_matching_matrices(simat2)

    assert all(simat1a.mids == ["b", "c", "e"]), "Did not match mids matrix1"
    assert all(simat2a.mids == ["b", "c", "e"]), "Did not match mids matrix2"

    assert (simat1a.matrix == [[2,0,0], [0,3,0], [0,0,5]]).all(), "Did not match values matrix1"
    assert (simat2a.matrix == [[7,0,0], [0,8,0], [0,0,10]]).all(), "Did not match values matrix2"

def test_SimilarityMatrix_get_cleared_matrix(simat):

    cleared = simat.get_cleared_matrix(["1"])

    assert all(cleared.mids == ["0", "2"]), "Did not remove mid"

    assert (cleared.matrix == [[1,3], [3,6]]).all(), "Did not remove entries"

def test_OverlapSimilarityMatrix_set_mids():

    matrix = OverlapSimilarityMatrix([[1,0,0],[0,0,1]])

    matrix.set_mids(["a","b"], ["c", "d", "e"])

    rmids, cmids = matrix.mids

    assert all(rmids == ["a","b"]), "Did not set row mids correctly"
    assert all(cmids == ["c", "d", "e"]), "Did not set row mids correctly"

def test_OverlapSimilarityMatrix_calculate():

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    assert matrix.fp_type == "Mock", "Did not set fp_type"
    assert matrix.fp_name == "mock", "Did not set fp_name"

    mat = [fp.get_similarities(fps[:10]) for fp in fps[10:]]

    assert (mat == matrix.matrix).all(), "Did not calculate correct matrix"

    rmids, cmids = matrix.mids

    assert all(cmids == [fp.mid for fp in fps[:10]]), "Did not set correct column mids"
    assert all(rmids == [fp.mid for fp in fps[10:]]), "Did not set correct row mids"

def test_OverlapSimilarityMatrix_get_entries():

    matrix = OverlapSimilarityMatrix([[1,2,3], [4,5,6], [7,8,9]])

    assert all(matrix.get_entries() == list(range(1,10))), "Did not return correct entries"

def test_OverlapSimilarityMatrix_get_row():

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    row = matrix.get_row(0)

    assert all(row == fps[10].get_similarities(fps[:10])), "Did not return correct row by index"

    row = matrix.get_row(fps[11].mid)

    assert all(row == fps[11].get_similarities(fps[:10])), "Did not return correct row by mid"


def test_OverlapSimilarityMatrix_get_column():

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    column = matrix.get_column(0)

    assert all(column == fps[0].get_similarities(fps[10:])), "Did not return correct column by index"

    column = matrix.get_column(fps[1].mid)

    assert all(column == fps[1].get_similarities(fps[10:])), "Did not return correct column by mid"

def test_OverlapSimilarityMatrix_dataframe():

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    df = matrix.dataframe

    from pandas import DataFrame

    df2 = DataFrame(matrix.matrix, index = [fp.mid for fp in fps[10:]], columns=[fp.mid for fp in fps[:10]])

    assert all((df == df2).all()), "Did not return correct dataframe"

def test_OverlapSimilarityMatrix_get_sub_matrix():

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    sub_row_mids = [fp.mid for fp in fps[12:14]]
    sub_column_mids = [fp.mid for fp in fps[2:8]]

    sub_matrix = matrix.get_sub_matrix(sub_row_mids, sub_column_mids)

    sub_matrix_values = [fp.get_similarities(fps[2:8]) for fp in fps[12:14]]

    assert (sub_matrix.matrix == sub_matrix_values).all(), "Wrong values for sub matrix"

    assert all(sub_matrix.column_mids == sub_column_mids), "Wrong column mids for sub matrix"

    assert all(sub_matrix.row_mids == sub_row_mids), "Wrong row mids for sub matrix"

    matrix.get_sub_matrix(sub_row_mids, sub_column_mids, copy = False)

    assert matrix == sub_matrix, "Not copying does not return correct sub matrix"

def test_OverlapSimilarityMatrix_save_load(tmpdir):

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    matrix.save(filepath = str(tmpdir))

    loaded = SimilarityMatrix.load(filename="overlap_similarity_matrix.npy", filepath=str(tmpdir))

    assert matrix == loaded, "Did not save/load correct matrix data"

def test_OverlapSimilarityMatrix_get_entry():

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    row_mid = fps[13].mid
    column_mid = fps[2].mid

    sim = fps[2].get_similarity(fps[13])

    assert matrix.get_entry(row_mid, column_mid) == sim, "Returned wrong similarity value"

def test_OverlapSimilarityMatrix_transpose():

    fps = [MockFingerpint().calculate() for _ in range(15)]

    matrix = OverlapSimilarityMatrix().calculate(fps[:10], fps[10:])

    row_mids, column_mids = matrix.mids

    matrix.transpose()

    assert all(matrix.row_mids == column_mids), "Did not transpose row mids"
    assert all(matrix.column_mids == row_mids), "Did not transpose column mids"

    assert (matrix.matrix == [fp.get_similarities(fps[10:]) for fp in fps[:10]]).all(), "Does not contain correct transposed similarities"

def test_OverlapSimilarityMatrix_align():

    r_mids1 = ["a", "b", "c"]
    c_mids1 = ["e", "f"]

    r_mids2 = ["g", "b", "c"]
    c_mids2 = ["e", "k"]

    matrix1 = [
        [1, 0],
        [0, 1],
        [1, 1]
    ]

    matrix2 = 2 * np.array(matrix1)

    osm1 = OverlapSimilarityMatrix(matrix1, r_mids1, c_mids1)
    osm2 = OverlapSimilarityMatrix(matrix2, r_mids2, c_mids2)

    osm1.align(osm2)

    assert (osm1.matrix == [[0], [1]]).all(), "Did not align matrix1 values correctly"
    assert (osm2.matrix == [[0], [2]]).all(), "Did not align matrix2 values correctly"

    assert all(osm1.row_mids == ["b", "c"]), "Did not align row mids matrix1"
    assert all(osm2.row_mids == ["b", "c"]), "Did not align row mids matrix2"
    assert all(osm1.column_mids == ["e"]), "Did not align column mids matrix1"
    assert all(osm2.column_mids == ["e"]), "Did not align column mids matrix2"

    r_mids1 = ["a", "b", "c"]
    c_mids1 = ["e", "f"]

    r_mids2 = ["g", "b", "c"]
    c_mids2 = ["e", "k"]

    r_mids3 = ["g", "j", "c"]
    c_mids3 = ["e", "k"]

    matrix1 = [
        [1, 0],
        [0, 1],
        [1, 1]
    ]

    matrix2 = 2 * np.array(matrix1)
    matrix3 = 3 * np.array(matrix1)

    osm1 = OverlapSimilarityMatrix(matrix1, r_mids1, c_mids1)
    osm2 = OverlapSimilarityMatrix(matrix2, r_mids2, c_mids2)
    osm3 = OverlapSimilarityMatrix(matrix3, r_mids3, c_mids3)

    osm1.align([osm2, osm3])

    assert (osm1.matrix == [[1]]).all(), "Did not align matrix1 values correctly"
    assert (osm2.matrix == [[2]]).all(), "Did not align matrix2 values correctly"
    assert (osm3.matrix == [[3]]).all(), "Did not align matrix3 values correctly"

    assert all(osm1.row_mids == ["c"]), "Did not align row mids matrix1"
    assert all(osm2.row_mids == ["c"]), "Did not align row mids matrix2"
    assert all(osm3.row_mids == ["c"]), "Did not align row mids matrix3"
    assert all(osm1.column_mids == ["e"]), "Did not align column mids matrix1"
    assert all(osm2.column_mids == ["e"]), "Did not align column mids matrix2"
    assert all(osm3.column_mids == ["e"]), "Did not align column mids matrix3"

@pytest.mark.skip()
def test_batched_similarity_matrix(tmp_path, dos_simat, all_dos_fingerprints):
    serial_matrix = dos_simat.get_symmetric_matrix()
    batched_matrix = []
    bsm = BatchedSimilarityMatrix().calculate(all_dos_fingerprints, folder_name = 'test_matrix', batch_size = 5, data_path=str(tmp_path))
    loaded_bsm = SimilarityMatrix().load(data_path = str(tmp_path), batched = True, batch_size = 5, batch_folder_name = 'test_matrix')
    assert (bsm[3] == loaded_bsm[3]).all()
    for mid in bsm.mids:
        batched_matrix.append(bsm.get_row(mid))
    assert np.allclose(serial_matrix, batched_matrix)
