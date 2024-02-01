from typing import Callable, List

from numpy import ndarray

from madas import Material
from madas.fingerprint import Fingerprint


def _initialize_fingerprint(fingerprint):
    version = fingerprint.data["version"]
    if version == 1:
        from nomad_dos_fingerprints.v1 import DOSFingerprint
    elif version == 2:
        from nomad_dos_fingerprints.v2 import DOSFingerprint
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
                  version: int  = 2,
                  convert_data: Callable | None =None, **kwargs):
        """
        Calculate the fingerprint. Possible kwargs (and defaults) are:
            grid_id = 'dg_cut:56:-2:7:(-10, 5):56'
            unit_cell_volume = 1
            n_atoms = 1
        """
        if version == 1:
            from nomad_dos_fingerprints.v1 import DOSFingerprint
        elif version == 2:
            from nomad_dos_fingerprints.v2 import DOSFingerprint
        self.set_data("version", 2)
        self.fingerprint = DOSFingerprint().calculate(energy, dos, grid_id = self.grid_id, convert_data = convert_data, **kwargs)
        self.set_data("fingerprint", self.fingerprint.to_dict())
        return self

    def from_material(self, 
                      material: Material, 
                      energy_path: str = "electronic_dos_energies", 
                      dos_path: str = "electronic_dos_values",
                      version: int = 2,
                      convert_data : Callable | None = None, **kwargs):
        self.set_mid(material)
        energy = material.get_data_by_path(energy_path)
        dos = material.get_data_by_path(dos_path)
        return self.calculate(energy, dos, version=version, convert_data=convert_data, **kwargs)

    def get_grid(self):
        if self.data["version"] == 1:
            from nomad_dos_fingerprints.v1 import Grid
        elif self.data["version"] == 2:
            from nomad_dos_fingerprints.v2 import Grid
        return Grid.create(grid_id = self.fingerprint.grid_id)

    def from_data(self, data):
        for key, value in data.items():
            self.set_data(key, value)
        version = data["version"]
        if version == 1:
            from nomad_dos_fingerprints.v1 import DOSFingerprint
        elif version == 2:
            from nomad_dos_fingerprints.v2 import DOSFingerprint
        self.fingerprint = DOSFingerprint.from_dict(data['fingerprint'])
        return self

    @property
    def indices(self):
        return self.fingerprint.indices
    
    @property
    def bins(self):
        if self.data["version"] == 1:
            return bytes.fromhex(self.fingerprint.bins)
        else:
            return self.fingerprint.bins

    @property
    def filling_factor(self):
        return self.fingerprint.filling_factor

    @property
    def overflow(self):
        return self.fingerprint.overflow
