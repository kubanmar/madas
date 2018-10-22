from ase.spacegroup import get_spacegroup, Spacegroup
from numpy import ndarray

class SYMFingerprint():

    def __init__(self, atoms_object, symop = None, sg = None):
        if atoms_object != None:
            self.sg = get_spacegroup(atoms_object, method = 'spglib')
            self.symop = self.sg.get_symop()
        else:
            self.sg = Spacegroup(sg)
            self.symop = symop


    def get_data(self):
        data = {'symop':[]}
        data['sg'] = self.sg.no
        for item in self.symop: #DEBUG
            symop = []
            for subitem in item:
                symop.append(subitem.tolist())
            data['symop'].append(symop)
        return data

def get_SYM_sim(symop_mat_1, symop_mat_2):
    #Pseudocode:: for item in fingerprint do check if symop in fingerprint2, then do some kind of tanimoto with the total number entries in each thing.
    if len(symop_mat_1) < len(symop_mat_2):
        symop1 = symop_mat_1
        symop2 = symop_mat_2
    else:
        symop1 = symop_mat_2
        symop2 = symop_mat_1
    a = len(symop1)
    b = len(symop2)
    c = 0
    for item in symop1:
        if item in symop2:
            c += 1
    return c / (a+b-c)
