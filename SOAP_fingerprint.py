import soaplite
from dscribe.descriptors import SOAP
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import numpy as np
import json

from fingerprint import Fingerprint
from dscribe.kernels import AverageKernel


atomic_numbers = [x for x in range(1,119)]
rcut = 6.0
nmax = 8
lmax = 6
soap = SOAP(
    species=atomic_numbers,
    periodic=True,
    average = True,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse = True
)

kernel = AverageKernel(metric="linear")

class SOAPFingerprint(Fingerprint):
    def __init__(self, db_row = None):
        self._init_from_db_row(db_row)

    def calculate(self, db_row):
        atoms = db_row.toatoms()
        self.matrix = soap.create(atoms)

    def reconstruct(self, db_row):
        data = self._data_from_db_row(db_row)
        data = json.loads(data)
        self.matrix = coo_matrix((data[1], (data[2], data[3])), shape = data[0])

    def get_data(self):
        data = json.dumps([self.matrix.shape, self.matrix.data.tolist(), self.matrix.row.tolist(), self.matrix.col.tolist()])
        return data

    @staticmethod
    def _calculate_similarity_matrix(soap_fingerprints):
        from similarity import SimilarityMatrix
        soap_sim_mat = kernel.create([fp.matrix.toarray() for fp in soap_fingerprints])
        soap_simat = SimilarityMatrix()
        soap_simat.matrix = soap_simat.triangular_from_square_matrix(soap_sim_mat)
        soap_simat.mids = [fp.mid for fp in soap_fingerprints]
        return soap_simat

def SOAP_similarity(fingerprint1, fingerprint2):
    arr1 = fingerprint1.matrix.toarray()
    arr2 = fingerprint2.matrix.toarray()
    sims = kernel.create([arr1, arr2])
    return round(sims[0][1],15)#(np.sqrt(2.0) - distance[0][1])/np.sqrt(2.0) #we need similarity, thus max(distance) - distance; assuming max(distance) = sqrt(2) here.
