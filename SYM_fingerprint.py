from ase.spacegroup import get_spacegroup, Spacegroup
from numpy import ndarray

from fingerprint import Fingerprint

class SYMFingerprint(Fingerprint):

    def __init__(self, db_row = None):
        self._init_from_db_row(db_row)

    def calculate(self, db_row):
        atoms = db_row.toatoms()
        self.sg = get_spacegroup(atoms)
        symop = self.sg.get_symop()
        self.symop = []
        for item in symop:
            op = []
            for subitem in item:
                op.append(subitem.tolist())
            self.symop.append(op)


    def reconstruct(self, db_row):
        data = self._data_from_db_row(db_row)
        self.sg = Spacegroup(data['sg'])
        self.symop = data['symop']

    def get_data(self):
        data = {'symop':[]}
        data['sg'] = self.sg.no
        data['symop'] = self.symop
        return data

def SYM_similarity(fingerprint1, fingerprint2):
    symop_mat_1 = fingerprint1.symop
    symop_mat_2 = fingerprint2.symop
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
    try:
        tc = c / (a+b-c)
    except ZeroDivisionError:
        tc = 0
    return tc
