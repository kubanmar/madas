from typing import List


from madas.backend.backend_core import Backend
from madas.material import Material
from madas.utils import safe_log, JSONNumpyEncoder

class DictBackend(Backend):
    """
    Storage of materials data in a dictionary in memory.

    Inteded use are development and testing, however, data can be stored by `json.dump` ing the dict.
    """

    def __init__(self, 
                 filename=None, 
                 filepath=None, 
                 rootpath=None, 
                 make_dirs=True, 
                 key_name="mid", 
                 log=None):
        self.key_name = key_name
        self._metadata = {}
        self._dict = {}
        self.log = log
        #  kept for compatibility, but ignored in the remainder
        self.filename = filename 
        self.filepath = filepath
        self.rootpath = rootpath

    @property
    def abs_path(self):
        """
        Included for compatibility, returns `None`.
        """
        return None

    def get_single(self, mid: str) -> Material:
        """
        Get a single entry from the database.
        """
        return Material.from_dict(self._dict[mid])

    def get_many(self, mids: List[str]) -> List[Material]:
        """
        Get several entry from the database.
        """
        return [self.get_single(mid) for mid in mids]

    def get_by_id(self, db_id: int) -> Material:
        """
        Return a single entry from an (integer valued) database id.
        """
        mid = list(self._dict.keys())[db_id]
        return self.get_single(mid)

    def add_single(self, material: Material) -> None:
        """
        Add data to the database.
        """
        self._dict.update(**{material.mid : material.to_dict()})
        safe_log(self._log_write_message(material.mid), logger=self.log, level = "info")
        
    def add_many(self, materials: List[Material]) -> None:
        """
        Add data to the database.
        """
        new_data = {material.mid : material.to_dict() for material in materials}
        self._dict.update(**new_data)

    def update_single(self, mid: str, **kwargs) -> None:
        """
        Update a single entry in the database.

        The `properties` attribute of entry `mid` will be updated with `**kwargs`.
        """
        self._dict[mid]["properties"].update(**kwargs)
    
    def update_many(self, mids: List[str], kwargs_list: List[dict] = []):
        """
        Update several entries in the database.

        The `properties` attributes of entries `mids` will be updated with entries of `kwargs_list`.
        """
        for mid, props in zip(mids, kwargs_list):
            self._dict[mid]["properties"].update(**props)

    def update_metadata(self, **kwargs) -> None:
        """
        Updata database metadata.
        """
        self._metadata.update(**kwargs)

    def has_entry(self, entry_id: str) -> bool:
        """
        Check if an entry with the given id is present in the database.
        """
        if entry_id in self._dict.keys():
            return True
        return False

    def get_length(self) -> int:
        """
        Return the length of the database, i.e. the total number of entries.
        """
        return len(self._dict)

    def to_json(self) -> str:
        """
        `json.dumps` the content of the database.
        """
        import json
        print(self._dict)
        return json.dumps({"data": self._dict, "metadata" : self.metadata}, cls=JSONNumpyEncoder)

    @staticmethod
    def from_json(json_data) -> object:
        """
        Restore data from string. 
        Inverse option to `DictBackend().to_json()`.
        """
        import json
        data = json.loads(json_data)
        self = DictBackend()
        self._dict = data["data"]
        self._metadata = data["metadata"]
        return self