from abc import ABC, abstractmethod
from typing import Any, List
from .api_core import APIClass
from ..data_framework import Material 
from ..utils import safe_log, tqdm
import ase
import os
import csv
import hashlib
import json

class FileReader(ABC):
    """
    Base class for file reader objects.

    **Keyword arguments:**

    file_path: `str`
        Path to file to be read.

    **Methods:**
    """
    def __init__(self, file_path: str = '.'):
        self.set_file_path(file_path)

    def set_file_path(self, file_path: str) -> None:
        """
        Set the file path.

        **Arguments:**

        file_path: `str`
            File path of file to be read.
        """
        self.file_path = file_path

    @abstractmethod
    def read(file_name, file_path: str = None) -> Any:
        return

class FileReaderASE(FileReader):
    """
    Wrapper class around the ASE IO module.

    Loads geometry from a DFT input file to an ASE Atoms object.

    **Keyword arguments:**

    file_path: `str`
        Path to file to be read.

    **Methods:**
    """

    def read(self, 
             file_name: str, 
             file_path: str = None, 
             **kwargs) -> ase.Atoms:
        """
        Read geometry from DFT input file.

        **Arguments:**

        file_name: `str`
            Name of file to be read. Format is guessed from extension.

        **Keyword arguments:**

        file_path: `str`
            Path of file to be read. Sets the `file_path` attribute.

            default: `None` --> use `self.file_path`

        Additional keyword arguments are forwarded to `ase.io.read`.

        **Returns:**

        atoms: `ase.Atoms`
            Atoms object with geometry read from file.
        """
        if file_path is not None:
            self.set_file_path(file_path)
        atoms = ase.io.read(os.path.join(self.file_path, file_name), **kwargs)
        return atoms

class CSVPropertyReader(FileReader):
    """
    Read material properties from a CSV file and return as dictionary.

    Assumes that the first column of CSV file contains a key. Example content:


    .. code-block:: 

        id,property1,property2   
        0, 1.4, 2.3
        1, 2, 2.5

    Calling `read()` returns:
 
    .. code-block:: python

       {"0" : {"property1" : 1.4, "property2" : 2.3},
       "1" : {"property1" : 2, "property2" : 2.5}}

    **Keyword arguments:**

    file_path: `str`
        Path to file to be read.

    **Methods:**
    """

    def read(self, 
             file_name_csv: str, 
             file_path_csv: str = None, 
             **kwargs) -> dict:
        """
        Read CSV data from file to dictionary.

        **Arguments:**

        file_name_csv: `str`
            Name of CSV file to be read.

        **Keyword arguments:**

        file_path_csv: `str`
            File path to file to be read.

            default: `None` --> use `self.file_path`

        Additional keyword arguments are ignored.

        **Returns:**

        data_dict: `dict`
            Dictionary of properties. First column of csv data is used as dictionary keys.
        """
        if file_path_csv is not None:
            self.set_file_path(file_path_csv)
        with open(os.path.join(self.file_path, file_name_csv),'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            data = [x for x in reader]
            header = data[0]
            data = data[1:]
        data_dict = {}
        for entry in data:
            data_dict[entry[0]] = {}
            for key, value in zip(header[1:],entry[1:]):
                data_dict[entry[0]][key] = json.loads(value)
        return data_dict

class API(APIClass):
    """
    API class to add materials from a local folder structure.

    **Requires the following data structure:**

    * root_directory
        * <properties.file>
        * directory[0]
            * <input.file>
        * directory[1]
            * <input.file>
        * ...

    where the directory names correspond to the entries of the first column of the CSV file.

    The name of individual files can be chosen on demand.

    **Keyword arguments**

    root: `str`
        Root directory.

        default: '.'

    file_reader: `FileReader`
        File reader object for reading input files.

        default: `FileReaderASE()`

    property_reader: `FileReader`
        File reader object for reading properties file.

        default: `CSVPropertyReader()`

    logger: `logging.Logger`
        Logger to write error and info messages.

    **Methods:**
    """

    def __init__(self, 
                 root = '.', 
                 file_reader = FileReaderASE(), 
                 property_reader = CSVPropertyReader(), 
                 logger = None):
        self.root = root
        self.log = logger
        self.file_reader = FileReaderASE(file_path = root) if file_reader is None else file_reader
        self.property_reader = CSVPropertyReader(file_path = root) if property_reader is None else property_reader

    def get_calculation(self, 
                        file_path: str, 
                        file_name: str, 
                        property_file_path: str, 
                        property_file_name: str, 
                        property_file_id: str, 
                        file_reader_kwargs: dict = {}, 
                        property_reader_kwargs: dict = {}) -> Material:
        """
        Get calculation from a local file.

        **Arguments:**

        file_path: `str`
            Path to the local file of the structure. Passed to `FileReader.read()` method of file reader. 

        file_name: `str`
            Name of the local file of the structure. Passed to `FileReader.read()` method of file reader.

        property_file_path: `str`
            Path of the file containing the properties of the calculation. 
            Passed to `FileReader.read()` method of property reader.

        property_file_name: `str`
            Name of the file containing the properties of the calculation.
            Passed to `FileReader.read()` method of property reader.

        property_file_id: `str`
            Id of the structure in the property file.
            If no string is provided, it will be converted to string.

        **Keyword arguments:**

        file_reader_kwargs: `dict`
            Additional keyword arguments passed to the file reader object.

            default: `{}`

        property_reader_kwargs: `dict`
            Additional keyword arguments passed to the property reader object.

            default: `{}`

        **Returns:**

        material: `Material`
            Material object with set material id, atomic structure and `data` properties.
        """
        atoms = self.file_reader.read(file_name, file_path = os.path.join(self.root, file_path), **file_reader_kwargs)
        properties = self.property_reader.read(property_file_name, os.path.join(self.root, property_file_path), **property_reader_kwargs)[str(property_file_id)]
        mid = self._gen_mid(file_path, file_name, property_file_path, property_file_name)
        return Material(mid, atoms = atoms, data = properties)

    def get_calculations_by_search(self, 
                                   folder_path: str, 
                                   file_name: str, 
                                   property_file_path: str, 
                                   property_file_name: str, 
                                   file_reader_kwargs: str = {},
                                   show_progress: bool = True, 
                                   property_reader_kwargs: str = {}) -> List[Material]:
        """
        Get calculations from a local file-structure.

        Assumes that every struture is stored in a separate directory, which has a name corresponding to the id extracted from the properties file.

        **Arguments:**

        file_path: `str`
            Path to the local file of the structure. Passed to `FileReader.read()` method of file reader. 

        file_name: `str`
            Name of the local file of the structure. Passed to `FileReader.read()` method of file reader.

        property_file_path: `str`
            Path of the file containing the properties of the calculation. 
            Passed to `FileReader.read()` method of property reader.

        property_file_name: `str`
            Name of the file containing the properties of the calculation.
            Passed to `FileReader.read()` method of property reader.

        **Keyword arguments:**

        file_reader_kwargs: `dict`
            Additional keyword arguments passed to the file reader object.

            default: `{}`

        property_reader_kwargs: `dict`
            Additional keyword arguments passed to the property reader object.

            default: `{}`

        **Returns:**

        materials: `List[Material]`
            List of `Material` objects with set material id, atomic structure and `data` properties.
        """
        calculations = []
        folders = os.listdir(os.path.join(self.root, folder_path))
        for folder in tqdm(folders, disable=not show_progress):
            if not os.path.isdir(os.path.join(self.root, folder_path, folder)):
                safe_log(f"Skipping: {folder} - not a directory", logger=self.log, level="info")
                continue
            try:
                calculations.append(self.get_calculation(os.path.join(folder_path,folder), 
                                                         file_name, 
                                                         property_file_path, 
                                                         property_file_name, 
                                                         property_file_id=folder, 
                                                         file_reader_kwargs=file_reader_kwargs,
                                                         property_reader_kwargs = property_reader_kwargs))
            except Exception as exc:
                error_message = 'Could not read file in folder:' + os.path.join(folder_path,folder) + '! Reason: ' + str(exc)
                safe_log(error_message, logger=self.log)
        return calculations

    def get_property(self, *args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError('Not implemented.')

    def _gen_mid(self, *args) -> str:
        """
        Generate mid from arguments by hashing over their `str` values.
        """
        string = ':'.join([str(arg) for arg in args])
        return '!' + hashlib.md5(bytes(string,'utf8')).hexdigest()
