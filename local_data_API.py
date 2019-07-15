from api_core import APIClass, Material
import ase, os, csv, hashlib

class FileReader():

    def __init__(self, file_path = '.'):
        self.set_file_path(file_path)

    def set_file_path(self, file_path):
        self.file_path = file_path

    def read(self, file_name, file_path = None, **kwargs):
        if file_path != None:
            self.set_file_path(file_path)
        atoms = ase.io.read(os.path.join(self.file_path, file_name), **kwargs)
        return atoms

class CSVPropertyReader():

    def __init__(self, file_path = '.'):
        self.set_file_path(file_path)

    def set_file_path(self, file_path):
        self.file_path_csv = file_path

    def read(self, file_name_csv, file_path_csv = None, **kwargs):
        if file_path_csv != None:
            self.set_file_path(file_path_csv)
        with open(os.path.join(self.file_path_csv, file_name_csv),'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            data = [x for x in reader]
            header = data[0]
            data = data[1:]
        data_dict = {}
        for entry in data:
            data_dict[entry[0]] = {}
            for key, value in zip(header[1:],entry[1:]):
                data_dict[entry[0]][key] = value
        return data_dict

class API(APIClass):
    """
    API class to add materials from a local folder structure.
    """

    def __init__(self, root = '.', file_reader = FileReader(), property_reader = CSVPropertyReader(), logger = None):
        self.root = root
        self.log = logger
        self.file_reader = FileReader(file_path = root) if file_reader == None else file_reader
        self.property_reader = CSVPropertyReader(file_path = root) if property_reader == None else property_reader

    def get_calculation(self, file_path, file_name, property_file_path, property_file_name, property_file_id = None, file_reader_kwargs = {}, property_reader_kwargs = {}):
        """
        Get calculation from a local file.
        Args:
            * file_path: string; path to the local file of the structure
            * file_name: string; name of the local file of the structure
            * property_file_path: string; path of the file containing the properties of the calculation
            * property_file_name: string; name of the file containing the properties of the calculation
        Kwargs:
            * property_file_id: string, int; id of the structure in case the property file contains information about several materials
            * file_reader_kwargs: dict; kwargs to be passed to the file reader object
            * property_reader_kwargs: dict; kwargs to be passed to the property reader object
        """
        atoms = self.file_reader.read(file_name, file_path = os.path.join(self.root, file_path), **file_reader_kwargs)
        properties = self.property_reader.read(property_file_name, os.path.join(self.root, property_file_path), **property_reader_kwargs)[str(property_file_id)]
        mid = self.gen_mid(file_path, file_name, property_file_path, property_file_name)
        return Material(mid, atoms = atoms, data = properties)

    def get_calculations_by_search(self, folder_path, file_name, property_file_path, property_file_name, file_reader_kwargs = {'format':'aims'}, property_reader_kwargs = {}):
        """
        Get calculations from a local file-structure.
        Args:
            * folder_path: string; path to the local folder of the structures
            * file_name: string; name of the local files of the structures
            * property_file_path: string; path of the file containing the properties of the calculation
            * property_file_name: string; name of the file containing the properties of the calculation
        Kwargs:
            * file_reader_kwargs: dict; kwargs to be passed to the file reader object
            * property_reader_kwargs: dict; kwargs to be passed to the property reader object
        """
        calculations = []
        folders = os.listdir(folder_path)
        for folder in folders:
            try:
                calculations.append(self.get_calculation(os.path.join(folder_path,folder), file_name, property_file_path, property_file_name, property_file_id=folder, file_reader_kwargs=file_reader_kwargs, property_reader_kwargs = property_reader_kwargs))
            except Exception as exc:
                error_message = 'Could not read file in folder:' + os.path.join(folder_path,folder) + '! Reason: ' + str(exc)
                self._report_error(error_message)
        return calculations

    def get_property(self, file_path):
        return {'property':None}

    def gen_mid(self, *args):
        string = ':'.join([str(arg) for arg in args])
        return '!' + hashlib.md5(bytes(string,'utf8')).hexdigest()#hash(string).to_bytes(int(len(string)/8), 'little', signed = True).hex()
