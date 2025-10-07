from ase import Atoms
import numpy as np

class Material():
    """
    A materials base class. Contains the atomic positions as an ASE Atoms object and additional properties as a dictionary.

    **Arguments:**

    mid: *str*
        A (unique) identifier that is used as the main identifier in a database.
    
    **Keyword arguments:**

    atoms: *ase.Atoms()*
        Atoms object describing the material

        default: *None*

    data: *dict*
        Additional data of the material
        
        default: *None*

    properties: *dict*
        Additional properies of the material

        default: *None*
    """

    def __init__(self, mid, atoms: Atoms | None = None, data = None, properties = None):
        self.mid = str(mid)
        self.set_atoms(atoms)
        self.set_data(data)
        self.set_properties(properties)

    @property
    def properties(self):
        return self._properties

    @property
    def data(self):
        return self._data

    def set_atoms(self, atoms):
        self.atoms = atoms

    def set_data(self, data):
        if data is None:
            data = {}
        self._data = data

    def set_properties(self, properties):
        if properties is None:
            properties = {}
        self._properties = properties

    def get_property_by_path(self, property_path: str):
        property_path_ = property_path.split("/")
        property_root = property_path_.pop(0)
        if property_root in self.properties.keys():
            prop = self.properties[property_root]
        else:
            raise KeyError(f"No property {property_path} for material {self.mid}")
        while len(property_path_) > 0:
            key = property_path_.pop(0)
            try:
                prop = prop[key]
            except KeyError:
                raise KeyError(f"No property {property_path} for material {self.mid}")
        return prop

    def get_data_by_path(self, data_path: str):
        data_path_ = data_path.split("/")
        data_root = data_path_.pop(0)
        if data_root in self.data.keys():
            prop = self.data[data_root]
        else:
            raise KeyError(f"No data {data_path} for material {self.mid}")
        while len(data_path_) > 0:
            key = data_path_.pop(0)
            try:
                prop = prop[key]
            except KeyError:
                raise KeyError(f"No data {data_path} for material {self.mid}")
            except TypeError as t_error:
                try:
                    prop = prop[int(key)]
                except Exception:                
                    print(f"Can't resolve: {key} of type {type(key)}. Remaining path: {data_path_}.")
                    raise t_error
        return prop

    def to_dict(self) -> dict:
        atoms_dict = self.atoms.todict()
        for key in atoms_dict.keys():
            if isinstance(atoms_dict[key], np.ndarray):
                atoms_dict[key]=atoms_dict[key].tolist()
        return {
            "mid" : self.mid,
            "atoms" : atoms_dict,
            "properties" : self.properties,
            "data" : self.data
        }

    @classmethod
    def from_dict(cls, data: dict) -> object:
        self = cls(data["mid"], 
                   atoms = Atoms.fromdict(data["atoms"]), 
                   properties = data["properties"],
                   data = data["data"])
        return self

    def __eq__(self, other: object) -> bool:
        if not self.mid == other.mid:
            return False
        if not self.atoms == other.atoms:
            return False
        if not self.data == other.data:
            return False
        if not self.properties == other.properties:
            return False
        return True

    def __repr__(self) -> str:
        return f"Material(mid = {self.mid}, data = {set(self.data.keys())}, properties = {set(self.properties.keys())})"

    def __hash__(self) -> int:
        return hash(f"{self.mid}")