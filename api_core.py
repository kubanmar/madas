from ase import Atoms

class Material():
    """
    A materials base class. Contains the atomic positions as an ASE Atoms object and additional properties as a dictionary.
    Kwargs:
        * atoms: ase.Atoms(); default: None; Atoms object describing the unit cell
        * data: dict; default: None; Additional properties of the material
    """

    def __init__(self, mid, atoms = None, data = None):
        self.mid = mid
        self.set_atoms(atoms)
        self.set_data(data)

    def set_atoms(self, atoms):
        self.atoms = atoms

    def set_data(self, data):
        self.data = data

class APIClass():
    """
    Base class for APIs to different data sources.
    """

    def __init__(self, logger = None):
        self.set_logger(logger)

    def get_calculation(self, *args, **kwargs):
        raise NotImplementedError('The option to get individual calculations is not implemented in this API.')
        return Material()

    def get_calculations_by_search(self, *args, **kwargs):
        raise NotImplementedError('The option to get sets of calculations is not implemented in this API.')
        return [Material()]

    def get_property(self, *args, **kwargs):
        raise NotImplementedError('The option to get individual properties is not implemented in this API.')
        return {'property':None}

    def gen_mid(self):
        raise NotImplementedError('The option to generate material ids is not implemented in this API.')
        return '<unique>:<identifyer>'

    def set_logger(self, logger):
        self.log = logger

    def _report_error(self, error_message):
        if self.log != None:
            self.log.error(error_message)
        else:
            print(error_message)
