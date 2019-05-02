import logging, multiprocessing
import importlib
import json

from utils import report_error

def import_fingerprint_module(self, fp_type, file_suffix = '_fingerprint', class_suffix = 'Fingerprint', similarity_measure_suffix = '_similarity'):
    """
    Function to import specific fingerprint classes and similarity measures from different files.
    """
    module_name = fp_type + file_suffix
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
    db_row: AtomsRow object of ASE Database
    log: multiprocessing.logger object
    fp_type: string; Type of fingerprint, as defined in FingerprintParser
    importfunction: function, used to import different fingerprint types individually
    """

    def __init__(self, fp_type = None, name = None, importfunction = import_fingerprint_module, **kwargs):
        self.log = None
        self.importfunction = importfunction
        self.__dict__.update(kwargs) # Initialize all kwargs as attributes. Thus there is maximal flexibility.
        if hasattr(self, 'db_row'):
            if hasattr(self.db_row,'mid'):
                self.mid = self.db_row.mid
        self.fp_type = fp_type
        self.name = name if name != None else fp_type
        self.data = None if not 'data' in kwargs.keys() else kwargs['data']

    def get_similarity(self, fingerprint):
        try:
            similarity = self.similarity_function(self, fingerprint)
        except:
            mid1 = 'unknown' if not hasattr(self, mid) else self.mid
            mid2 = 'unknown' if not hasattr(fingerprint, mid) else fingerprint.mid
            error_message = 'Could not calculate similarity for materials: ' + mid1 + ' and ' + mid2
            report_error(self.log, error_message)
        return similarity

    def set_similarity_function(self, similarity_function):
        self.similarity_function = similarity_function

    def specify(self, fingerprint_class, **kwargs):
        """
        Change self to class given by ``fingerprint_class``.
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
