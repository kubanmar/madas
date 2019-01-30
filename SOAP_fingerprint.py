import soaplite
from dscribe.descriptors import SOAP
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import numpy as np
import json

atomic_numbers = [x for x in range(1,119)]
rcut = 6.0
nmax = 8
lmax = 6
soap = SOAP(
    atomic_numbers=atomic_numbers,
    periodic=True,
    average = True,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse = True
)


class SOAPfingerprint():
    """
    def __init__(self, atoms_object):
        n_max = 5 #simply copied from soaplite
        l_max = 5
        r_cut = 10.0
        if atoms_object != None:
            self.atoms = atoms_object
            my_alphas, my_betas = soaplite.getBasisFunc(r_cut, n_max)
            self.x = soaplite.get_periodic_soap_structure(atoms_object, my_alphas, my_betas)

    def get_data(self):
        data = {'densities':[]}
        data['positions'] = self.atoms.get_positions().tolist()
        for position in self.x:
            density = [value for value in position]
            data['densities'].append(density)
        return data
    """
    def __init__(self, atoms_object, data = None):
        if atoms_object == None:
            data = json.loads(data)
            self.matrix = coo_matrix((data[1], (data[2], data[3])), shape = data[0])
        else:
            self.matrix = soap.create(atoms_object)

    def get_data(self):
        data = json.dumps([self.matrix.shape, self.matrix.data.tolist(), self.matrix.row.tolist(), self.matrix.col.tolist()])
        return data

class ClusterSOAPFingerprint():

    def __init__(self, atoms_object):
        pass

def get_SOAP_sim(soap1, soap2):
    arr1 = soap1.matrix.toarray()
    arr2 = soap2.matrix.toarray()
    molecules = np.vstack([arr1, arr2])
    distance = squareform(pdist(molecules))
    return (2.0 - distance[0][1])/2.0 #we need similarity, thus max(distance) - distance; assuming max(distance) = 2 here.
