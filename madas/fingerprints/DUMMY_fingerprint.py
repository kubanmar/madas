import numpy as np

from madas.fingerprint import Fingerprint
from madas.utils import safe_log

def DUMMY_similarity(fingerprint1: Fingerprint, 
                     fingerprint2: Fingerprint,
                     alpha = 1.0) -> np.float64:
    """
    Similarity between cartesian vectors given as DUMMYFingerprints. Calculates:

    *similarity = alpha / ( alpha + |vector1 - vector2|_2)*

    **Arguments:**

    fingerprint1, fingerprint2: *madas.fingerprints.DUMMY_fingerprint.DUMMYFingerprint*
        DUMMY fingerprints to calculate similarity.

    **Keyword arguments:**

    alpha: `float`
        Scaling factor for similarity 

        default: `1.0`

    **Returns:**
    
    similarity: *numpy.float64*
        Similarity between both fingerprints
    """
    return alpha / (alpha +np.linalg.norm(np.array(fingerprint1['data']) - np.array(fingerprint2['data'])))

def dummy_target(x):
    return sum(np.array(x)**2)

class DUMMYFingerprint(Fingerprint):
    """
    A dummy fingerprint of cartesian vectors to demonstrate the usage of learning methods.
    """

    def __init__(self, name = "DUMMY", similarity_function = DUMMY_similarity, target_function = dummy_target):
        self.set_name(name)
        self.set_fp_type("DUMMY")
        self.set_similarity_function(similarity_function)
        self.target_function = target_function

    def calculate(self, list_: list):
        """
        Generate DUMMYFingerprints from a vector.
        """
        self.set_data("data", list(list_))
        self.set_name("DUMMY_from_list")
        return self

    def from_material(self, material, property_path="", value=None):
        """
        Calculate fingerprint by setting its data property to "Material.data".
        """
        self.set_mid(material)
        if value is not None:
            self.calculate(value)
            return self
        try:
            self.set_data("data", list(material.get_data_by_path(property_path)))
        except Exception as ex_:
            if self.pass_on_exceptions:
                safe_log(f"Could not retrieve property {property_path} for material with mid {self.mid}, because of {str(ex_)}", None)
                return None 
            else:
                raise ex_
        return self

    @property
    def y(self):
        return self.target_function(self["data"])