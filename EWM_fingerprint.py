from dscribe.descriptors import EwaldMatrix
from scipy.sparse import coo_matrix
import numpy as np
import json

from fingerprint import Fingerprint

ewmatrix = EwaldMatrix(118)

class EWMFingerprint(Fingerprint):

    def __init__(self, db_row = None):
        self._init_from_db_row(db_row)

    def calculate(self, db_row):
        atoms = db_row.toatoms()
        self.matrix = coo_matrix(ewmatrix.create(atoms))

    def reconstruct(self, db_row):
        data = self._data_from_db_row(db_row)
        data = json.loads(data)
        self.matrix = coo_matrix((data[1], (data[2], data[3])), shape = data[0])

    def get_data(self):
        data = json.dumps([self.matrix.shape, self.matrix.data.tolist(), self.matrix.row.tolist(), self.matrix.col.tolist()])
        return data


def EWM_similarity(fingerprint1, fingerprint2):
    arr1 = fingerprint1.matrix.toarray()[0]
    arr2 = fingerprint2.matrix.toarray()[0]
    s = (1 + np.dot(arr1,arr2)/ (np.linalg.norm(arr1) * np.linalg.norm(arr2))) / 2 #shift to have 0 < s < 1
    return s
