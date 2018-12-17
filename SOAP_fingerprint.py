import soaplite
from dscribe.descriptors import SOAP
from scipy.spatial.distance import pdist, squareform
import numpy as np

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
    sparse = False
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
            self.data = data
        else:
            self.data = soap.create(atoms_object).tolist()

    def get_data(self):
        return self.data

class ClusterSOAPFingerprint():

    def __init__(self, atoms_object):
        pass

def get_SOAP_sim(soap1, soap2):
    molecules = np.vstack([soap1, soap2])
    distance = squareform(pdist(molecules))
    return distance[0][1]
