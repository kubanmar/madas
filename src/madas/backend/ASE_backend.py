from typing import List
from .backend_core import Backend
from madas.data_framework import Material
from madas.utils import safe_log
from ase.db import connect

class ASEBackend(Backend):
    """
    Database backend that connects to a ASEAtomsDatabase object.
    """
    
    def __init__(self, 
                 filename = "materials_database.db", 
                 filepath = "data", 
                 make_dirs = True,
                 key_name = "mid", 
                 log = None) -> None:
        super().__init__(filename, filepath, make_dirs, key_name, log)
        self._db = connect(self.abs_path)
        self._metadata = self._db.metadata
        self._write_buffer = {}

    def add_single(self, material: Material) -> None:
        """
        Add a single material to the database.

        **Arguments:**

        material: *madas.Material*
            Material object to be added to the database

        **Return:**

        `None`
        """
        self._db.write(material.atoms, **{self.key_name : material.mid}, data = material.data, **material.properties)
        safe_log(self._log_write_message(material.mid), logger=self.log, level = "info")

    def add_many(self, materials: List[Material]) -> None:
        """
        Add many materials to the database.

        **Arguments:**

        materials: *List[madas.Material]*
            List of material objects to be added to the database

        **Return:**

        `None`
        """
        with self._db as db:
            for material in materials:
                db.write(material.atoms, **{self.key_name : material.mid}, data = material.data, **material.properties)
                safe_log(self._log_write_message(material.mid), logger=self.log, level = "info")
            
    def get_single(self, mid: str = None, **kwargs) -> Material:
        """
        Retrieve a single entry from the database.

        **Keyword arguments**

        mid: *str*
            default: `None`

            Material identifier used in the database and madas.Material objects

        Additional kwargs are used to retrieve entries from the database, e.g. if a single Material object has a specific property value. 
        
        **Returns:**

        material: *madas.Material*
            Material object of the selected database entry

        **Raises:**

        *ValueError*
            If neither a mid nor a keyword argument is given, this error is raised.

        *AssertionError*
            If more than one row matches the keyword arguments, the ASE AtomsDatabase raises this error.

        *KeyError*
            If no material has the specified mid or property values, this error is raised.
        """
        self._check_select_arguments(mid, **kwargs)
        row = self._get_single_row(mid, **kwargs)
        properties = row.key_value_pairs
        mid = properties.pop(self.key_name)
        material = Material(mid, row.toatoms(), data = row.data, properties = properties)
        return material

    def get_many(self, mids: List[str] = None, **kwargs) -> List[Material]:
        """
        Retrieve many entries from the database.

        **Keyword arguments**

        mids: *List[str]*
            default: `None`

            List of material identifiers used in the database and madas.Material objects

        Additional kwargs are used to retrieve entries from the database, e.g. if several Material objects have a specific property value. 
        
        **Returns:**

        materials: *List[madas.Material]*
            List of Material objects of the selected database entries

        **Raises:**

        *ValueError*
            If neither a mid nor a keyword argument is given, this error is raised.

        *KeyError*
            If no material has the specified mid or property values, this error is raised.
        """
        self._check_select_arguments(mids, **kwargs)
        materials = []
        if mids is not  None:
            for mid in mids:
                row = self._db.get(**{self.key_name : mid})
                properties = row.key_value_pairs
                properties.pop(self.key_name)
                materials.append(Material(mid, row.toatoms(), data = row.data, properties = properties))
        else:
            rows = self._db.select(**kwargs)
            for row in rows:
                properties = row.key_value_pairs
                mid = properties.pop(self.key_name)
                materials.append(Material(mid, row.toatoms(), data = row.data, properties = properties))
        return materials

    def get_by_id(self, db_id):
        """
        Get a single entry by the integer valued database id, i.e. the count of when the material was added.

        WARNING: Different than the ase.AtomsDatabase, this index starts with entry 0. 

        **Arguments**

        db_id: *int*
            Index of material in database

        **Returns**

        material: *madas.Material*
            Material object associated with the database entry

        *KeyError*
            If no material has the specified index, this error is raised.
        """
        row = self._db.get(db_id+1)
        properties = row.key_value_pairs
        mid = properties.pop(self.key_name)
        return Material(mid, row.toatoms(), row.data, properties)

    def update_single(self, mid: str = None, **kwargs) -> None:
        """
        Update a single entry from the database.

        **Keyword arguments**

        mid: *str*
            default: `None`

            Material identifier used in the database and madas.Material objects

        Additional kwargs are used to identify entries from the database, e.g. if a single Material object has a specific property value. 
        
        **Returns:**

        `None`

        **Raises:**

        *ValueError*
            If neither a mid nor a keyword argument is given, this error is raised.

        *AssertionError*
            If more than one row matches the keyword arguments, the ASE AtomsDatabase raises this error.

        *KeyError*
            If no material has the specified mid or property values, this error is raised.
        """
        self._check_select_arguments(mid, **kwargs)
        row = self._get_single_row(mid, **kwargs)
        _id = row.id
        self._db.update(_id, **kwargs)

    def update_many(self, mids: List[str] = None, kwargs_list: List[dict] = []) -> None:
        """
        Update a set of entries from the database.

        **Keyword arguments**

        mids: *List[str]*
            default: `None`

            List of material identifiers used in the database and madas.Material objects
        
        kwargs_list: `List[dict]`
            default: `[]`

            List of dictionaries that contain the data that should be written as properties to the database.

        **Returns:**

        `None`

        **Raises:**

        *ValueError*
            If neither a mid nor a keyword argument is given, this error is raised.

        *AssertionError*
            If more than one row matches the keyword arguments, the ASE AtomsDatabase raises this error.

        *KeyError*
            If no material has the specified mid or property values, this error is raised.
        """
        buffer_size = 1000
        ids = []
        for mid in mids:
            self._check_select_arguments(mid)
            row = self._get_single_row(mid)
            ids.append(row.id)
        update_buffer = []
        for update_single in zip(ids, kwargs_list):
            if len(update_buffer) >= buffer_size:
                self._update_buffer(update_buffer)
                update_buffer = []
            update_buffer.append(update_single)
        if len(update_buffer) > 0:
            self._update_buffer(update_buffer)

    def _update_buffer(self, buffer):
        with self._db:
            for _id, kwargs in buffer:
                self._db.update(_id, **kwargs)


    def update_metadata(self, **kwargs) -> None:
        """
        Update database metadata.
        
        Keyword arguments are directly written to database metadata.

        **Returns:**

        `None`
        """
        metadata = self.metadata
        metadata.update(**kwargs)
        self._db.metadata = metadata
        self._metadata = metadata

    def has_entry(self, entry_id: str) -> bool:
        """
        Test if an entry with given id exists in the database.

        **Arguments:**

        entry_id: *str*
            Material identifier of the expected entry.

        **Returns:**

        has_entry: *bool*
            True if an entry with this key exists, False else
        """
        try:
            self._db.get(**{self.key_name : entry_id})
            return True
        except KeyError:
            return False        

    def get_length(self):
        """
        Return the length of the database, i.e. the number of entries.
        """
        return self._db.count()

    def _get_single_row(self, mid: str, **kwargs) -> object:
        if mid is not None:
            row = self._db.get(**{self.key_name : mid})
        else:
            row = self._db.get(**kwargs)
            mid = row.__getattribute__(self.key_name)
        return row
