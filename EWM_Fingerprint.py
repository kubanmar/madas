from dscribe.descriptors import EwaldMatrix
from scipy.sparse import coo_matrix
import numpy as np
import json

ewmatrix = EwaldMatrix(118)

class EWMFingerprint():
    def __init__(self, atoms, data = None):
        if atoms == None:
            data = json.loads(data)
            self.matrix = coo_matrix((data[1], (data[2], data[3])), shape = data[0])
        else:
            self.matrix = coo_matrix(ewmatrix.create(atoms))

    def get_data(self):
        data = json.dumps([self.matrix.shape, self.matrix.data.tolist(), self.matrix.row.tolist(), self.matrix.col.tolist()])
        return data


def get_EWM_sim(ewm1, ewm2):
    arr1 = ewm1.matrix.toarray()[0]
    arr2 = ewm2.matrix.toarray()[0]
    s = (1 + np.dot(arr1,arr2)/ (np.linalg.norm(arr1) * np.linalg.norm(arr2))) / 2 #shift to have 0 < s < 1
    return s
