import logging, multiprocessing
import importlib
import json

from utils import report_error

def import_fingerprint_module(fp_type, file_suffix = '_fingerprint', class_suffix = 'Fingerprint', similarity_measure_suffix = '_similarity'):
    """
    Function to import specific fingerprint classes and similarity measures from different files.
    """
    module_name = 'fingerprints.' + fp_type + file_suffix
    module = importlib.import_module(module_name)
    class_name = fp_type + class_suffix
    similarity_measure_name = fp_type + similarity_measure_suffix
    fingerprint_class = getattr(module, class_name)
    similarity_function = getattr(module, similarity_measure_name)
    return fingerprint_class, similarity_function

class Fingerprint():
    """
    Base class for all fingerprints.
    kwargs:
        * db_row: AtomsRow object of ASE Database
        * log: logging.Logger object
        * fp_type: string; Type of fingerprint, as given by Python module
        * importfunction: function; used to import different fingerprint types individually
    """

    def __init__(self, fp_type = None, name = None, db_row = None, logger = None, importfunction = import_fingerprint_module, **kwargs):
        self.log = logger
        self.set_import_function(importfunction)
        self.db_row = db_row
        if hasattr(self, 'db_row'):
            if hasattr(self.db_row,'mid'):
                self.mid = self.db_row.mid
        self.fp_type = fp_type
        self.name = name if name != None else fp_type
        if self.fp_type != None:
            fingerprint_class, similarity_function = self.importfunction(fp_type)
            self.specify(fingerprint_class, **kwargs)
            self.set_similarity_function(similarity_function)

    def get_similarity(self, fingerprint):
        """
        Calculate similarity to another fingerprint. Will raise TypeError if different fingerpint types are used.
        Args:
            * fingerpint: Fingerpint() object; fingerprint to calculate similarity
        """
        if not self.fp_type == fingerprint.fp_type:
            report_error(self.log, 'Can not calculate similarity for fingerprints of different types. (%s and %s)' %(self.fp_type, fingerprint.fp_type))
            raise TypeError("Similarty of differing types of fingerprints can not be calculated.")
        try:
            similarity = self.similarity_function(self, fingerprint)
            return similarity
        except Exception as err:
            mid1 = 'unknown' if not hasattr(self, 'mid') else self.mid
            mid2 = 'unknown' if not hasattr(fingerprint, 'mid') else fingerprint.mid
            error_message = 'Could not calculate similarity for materials: ' + mid1 + ' and ' + mid2 + ' because of error: {0}'.format(err)
            report_error(self.log, error_message)
            return None

    def get_similarities(self, fingerprint_list):
        """
        Calculate similarities to a list of fingerprints. Will raise TypeError if different fingerpint types are used.
        Args:
            * fingerpint_list: list of Fingerpint() objects; fingerprints to calculate similarity
        """
        similarities = [self.get_similarity(fp) for fp in fingerprint_list]
        return similarities

    def set_similarity_function(self, similarity_function):
        """
        Set the function for calculating similarity.
        Args:
            * similarity_function: Python function; a function that takes two fingerprint objects as input and calculates the similarity between them.
        """
        self.similarity_function = similarity_function

    def set_import_function(self, import_function):
        """
        Set the function used to import specialized fingerprint objects.
        Args:
            * import_function: Python function; a function that returns a specialized Fingerprint() class and a similarity function.
        """
        self.importfunction = import_function

    def specify(self, fingerprint_class, **kwargs):
        """
        Change self to class given by ``fingerprint_class``.
        Args:
            * fingerprint_class: Python class; specialized Fingerprint() class
        Keyword arguments are passed to the __init__ function of the specialized fingerprint.
        """
        self.__class__ = fingerprint_class
        self.__init__(**kwargs)

    def calculate(self, db_row):
        """
        To be defined in child class. Calculate the fingerprint using AtomsRow object.
        """
        pass

    def reconstruct(self, db_row):
        """
        To be defined in child class. Reconstruct fingerprint using the data from AtomsRow object.
        """
        pass

    def get_data(self):
        """
        To be defined in child class. Return data to be written to database.
        """
        pass

    def get_data_json(self):
        """
        Return fingerprint data in as a json-coded string.
        """
        json_data = json.dumps(self.get_data())
        return json_data

    def _init_from_db_row(self, db_row, **kwargs):
        if not hasattr(self, 'db_row'):
            self.db_row = db_row
        if self.db_row != None:
            if not hasattr(self.db_row, self.name):
                self.calculate(self.db_row, **kwargs)
            else:
                try:
                    self.reconstruct(self.db_row)
                except TypeError:
                    error_message = 'Failed to reconstruct fingerprint for material ' + db_row.mid + '. Calculating fingerprint instead.'
                    report_error(self.log, error_message)
                    self.calculate(self.db_row, **kwargs)

    def _data_from_db_row(self, db_row):
        try:
            data = json.loads(db_row[self.name])
        except KeyError:
            report_error(self.log, "Error in reconstructing Fingerprint: AtomsRow does not have attribute of name: " + self.name)
        if hasattr(db_row, 'mid'):
            self.mid = db_row.mid
        elif 'mid' in data.keys():
            self.mid = data['mid']
        return data

class DBRowWrapper(dict):
    """
    Wrapper class to imitate the bahavior of an ASE AtomsRow object, if no ASE database is used.
    Arguments:
    data: dict; key-value pairs of data
    """
    def __init__(self, data):
        self.__dict__.update(**data)
        self['data'] = {}
        self['data'].update(**data)
