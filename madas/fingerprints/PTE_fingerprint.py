from madas import Fingerprint

from ase import Atoms
import numpy as np

def PTE_similarity(fingerprint1, fingerprint2):
    """
    Calculate similarity between PTE fingerprints.

    The similarity is calculated as:

    *similarity = 1 / ( 1 + |PTE1 - PTE2|)*
    """
    diff = abs(fingerprint1.data["averaged_PTE_column"] - fingerprint2.data["averaged_PTE_column"])
    return 1 / (1 + diff)

class PTEFingerprint(Fingerprint):
    """
    A fingerprint comparing the column of the periodic table of elements for all species of the material.
    """
    def __init__(self, 
                 name = "PTE", 
                 similarity_function=PTE_similarity, 
                 pass_on_exceptions=True) -> None:
        self.set_fp_type("PTE")
        self.set_name(name)
        self.set_pass_on_exceptions(pass_on_exceptions)
        self.set_similarity_function(similarity_function)

    def from_material(self, material):
        """
        Calculate PTEFingerprint from a `MADAS` `Material` object.
        """
        self.set_mid(material)
        return self.calculate(material.atoms)
    
    def calculate(self, atoms: Atoms) -> object | None:
        """
        Calculate average periodic table of elements column number.
        """
        column_numbers = []
        for number in atoms.get_atomic_numbers():
                column_numbers.append(COLUMN_PER_ATOM[number])
        self.set_data("averaged_PTE_column", np.mean(column_numbers))
        return self


MAIN_GROUP_DATA = {
    1 : [1,3,11,19,37, 55, 87],
    2 : [2,4,12,20,38,56,88],
    3 : [21,39],
    4 : [22,40,72,104],
    5 : [23,41,73,105],
    6 : [24,42,74,106],
    7 : [25,43,75,107],
    8 : [26,44,76,108],
    9 : [27,45,77,109],
    10 : [28, 46, 78, 110],
    11 : [29, 47, 79, 111],
    12 : [30, 48, 80, 112],
    13 : [5, 13, 31, 49, 81, 113],
    14 : [6, 14, 32, 50, 82, 114],
    15 : [7, 15, 33, 51, 83, 115],
    16 : [8, 16, 34, 52, 84, 116],
    17 : [9, 17, 35, 53, 85, 117],
    18 : [10, 18, 36, 54, 86, 118]
}

MAIN_GROUP_DATA[3].extend(range(57,72)) # Lanthanides
MAIN_GROUP_DATA[3].extend(range(89,104)) # Actinides

COLUMN_PER_ATOM = {}
for group, atomic_numbers in MAIN_GROUP_DATA.items():
    for number in atomic_numbers:
        COLUMN_PER_ATOM[number] = group