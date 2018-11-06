import soaplite

class SOAPfingerprint():

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

class ClusterSOAPFingerprint():

    def __init__(self, atoms_object):
        pass
