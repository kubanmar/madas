from typing import Callable, List

from numpy import ndarray

from madas import Material
from madas.fingerprint import Fingerprint

from nomad_dos_fingerprints import DOSFingerprint as NMDDOSFingerprint, Grid


def _initialize_fingerprint(fingerprint):
    fingerprint.fingerprint = DOSFingerprint.from_dict(fingerprint.data['fingerprint'])
    return fingerprint

def DOS_similarity(fingerprint1, fingerprint2):
    """
    Compute Tanimoto coefficient (Tc) between two DOS fingerprints.
    """
    # fingerprint was deserialized and not initialized
    if not hasattr(fingerprint1, "fingerprint"):
        fingerprint1 = _initialize_fingerprint(fingerprint1)
    if not hasattr(fingerprint2, "fingerprint"):
        fingerprint2 = _initialize_fingerprint(fingerprint2)
    
    return fingerprint1.fingerprint.get_similarity(fingerprint2.fingerprint)


class DOSFingerprint(Fingerprint):
    """
    A DOS fingerprint that uses the NOMAD DOS fingerprint package.
    """
    def __init__(self, 
                 name: str = "DOS", 
                 grid_id: str = None,
                 pass_on_exceptions: bool = False,
                 similarity_function = DOS_similarity):
        self.set_fp_type('DOS')
        self.set_name(name)
        self.set_pass_on_exceptions(pass_on_exceptions)
        self.set_similarity_function(similarity_function)
        self.grid_id = grid_id

    def calculate(self, 
                  energy: List[float] | ndarray,
                  dos: List[float] | ndarray,
                  convert_data: Callable | None = None, **kwargs):
        """
        Calculate the fingerprint. Possible kwargs (and defaults) are:
            grid_id = 'dg_cut:56:-2:7:(-10, 5):56'
            unit_cell_volume = 1
            n_atoms = 1
        """
        self.fingerprint = NMDDOSFingerprint().calculate(energy, dos, grid_id = self.grid_id, convert_data = convert_data, **kwargs)
        self.set_data("fingerprint", self.fingerprint.to_dict())
        return self

    def from_material(self, 
                      material: Material, 
                      energy_path: str = "electronic_dos_energies", 
                      dos_path: str = "electronic_dos_values",
                      convert_data : Callable | None = None, **kwargs):
        self.set_mid(material)
        energy = material.get_data_by_path(energy_path)
        dos = material.get_data_by_path(dos_path)
        return self.calculate(energy, dos, convert_data=convert_data, **kwargs)

    def get_grid(self):
        return Grid.create(grid_id = self.fingerprint.grid_id)

    def from_data(self, data):
        for key, value in data.items():
            self.set_data(key, value)
        self.fingerprint = NMDDOSFingerprint.from_dict(data['fingerprint'])
        return self

    @classmethod
    def from_nomad_dos_fingerprint(cls, 
                                   fingerprint: NMDDOSFingerprint, 
                                   name: str = "DOS", 
                                   pass_on_exceptions: bool = False,
                                   similarity_function: Callable = DOS_similarity) -> object:
        """
        Create MADAS DOSFingerprint from NOMAD DOS fingerprint object.

        **Arguments:**

        fingerprint: `nomad_dos_fingerprints.DOSFingerprint`
            `nomad_dos_fingerprints.DOSFingerprint` object to convert to `madas.fingerprints.DOSFingerprint`

        **Keyword arguments:**

        name: `str`
            Name of the fingerprint

            default: `"DOS"`

        pass_on_exceptions: `bool`
            Set if fingerprint should pass and return None if an exception occures during calculation of similarities, or raise an Exception.

            default: `False`

        similarity_function: `Callable`
            Similarity function to be used by the fingerprints.

            default: `madas.fingerprints.DOS_fingerprint.DOS_similarity`

        **Returns:**

        self: `madas.fingerprints.DOS_fingerprint.DOSFingerprint`
        """
        self = cls(name=name, pass_on_exceptions=pass_on_exceptions, similarity_function=similarity_function)
        self.grid_id = fingerprint.grid_id
        self.set_data("fingerprint", fingerprint.to_dict())
        self.fingerprint = fingerprint
        return self


    @staticmethod
    def get_default_grid() -> Grid:
        return Grid.create()

    @property
    def indices(self):
        return self.fingerprint.indices
    
    @property
    def bins(self):
        return self.fingerprint.bins

    @property
    def filling_factor(self):
        return self.fingerprint.filling_factor

    @property
    def overflow(self):
        return self.fingerprint.overflow
