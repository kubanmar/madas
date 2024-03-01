import logging
import os


class Backend():
    """
    A database backend wrapper to unify storage of materials data. 
    """

    def __init__(self, 
                 filename = "materials_database.db", 
                 filepath = "data", 
                 make_dirs = True,
                 key_name = "mid",
                 log = None):
        self.filename = filename
        self.filepath = filepath
        if not os.path.exists(filepath):
            if not make_dirs:
                raise FileNotFoundError("Path for storing database does not exist. To create it, set `make_dirs = True`.")
            else:
                os.makedirs(filepath)
        self._metadata = {}
        self.key_name = key_name
        self.set_logger(log)

    # properties

    @property
    def metadata(self):
        """
        Metadata property, returns metadata attached to the backend.
        """
        return self._metadata

    @property
    def abs_path(self):
        """
        Absolute path property, contains the absolute path of the backend file.
        """
        return os.path.abspath(os.path.join(self.filepath, self.filename))

    @property
    def log(self):
        """
        Logger property, returns the log.
        """
        return self._log

    # methods

    def get_single(self, mid = None, **kwargs):
        """
        Get a single entry from the database.
        """
        raise NotImplementedError(f"Function 'get_single' is not implemented for object of class {self.__class__}")

    def get_many(self, mids = None, **kwargs):
        """
        Get a single entry from the database.
        """
        raise NotImplementedError(f"Function 'get_many' is not implemented for object of class {self.__class__}")

    def get_by_id(self, db_id):
        """
        Return a single entry from an (integer valued) database id.
        """
        raise NotImplementedError(f"Function 'get_by_id' is not implemented for object of class {self.__class__}")

    def add_single(self, *args, **kwargs):
        """
        Add data to the database.
        """
        raise NotImplementedError(f"Function 'add_single' is not implemented for object of class {self.__class__}")
        
    def add_many(self, *args, **kwargs):
        """
        Add data to the database.
        """
        raise NotImplementedError(f"Function 'add_many' is not implemented for object of class {self.__class__}")

    def update_single(self, *args, **kwargs):
        """
        Update a single entry in the database.
        """
        raise NotImplementedError(f"Function 'update_single' is not implemented for object of class {self.__class__}")
    
    def update_many(self, *args, **kwargs):
        """
        Update several entries in the database.
        """
        raise NotImplementedError(f"Function 'update_many' is not implemented for object of class {self.__class__}")

    def update_metadata(self, *args, **kwargs):
        """
        Updata database metadata.
        """
        raise NotImplementedError(f"Function 'update_metadata' is not implemented for object of class {self.__class__}")

    def has_entry(self, entry_id):
        """
        Check if an entry with the given id is present in the database.
        """
        raise NotImplementedError(f"Function 'has_entry' is not implemented for object of class {self.__class__}")

    def get_length(self):
        """
        Return the length of the database, i.e. the total number of entries.
        """
        raise NotImplementedError(f"Function 'get_length' is not implemented for object of class {self.__class__}")

    def set_logger(self, logger: logging.Logger) -> None:
        """
        Set logger.
        """
        self._log = logger

    # private methods

    @staticmethod
    def _log_write_message(mid):
        return f"Wrote material with id {mid}."

    def _check_select_arguments(self, mid_s, **kwargs):
        if mid_s is None and len(kwargs) == 0:
            raise ValueError("Neither material ids nor properties to select materials given.")