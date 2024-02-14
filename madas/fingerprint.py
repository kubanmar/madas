import importlib
import json
from typing import Callable, List
from typing import Any
from copy import deepcopy
from functools import partial

from madas import Material

from .utils import report_error

def import_builtin_module(fp_type: str, 
                          file_suffix: str = '_fingerprint', 
                          class_suffix: str = 'Fingerprint', 
                          similarity_measure_suffix: str = '_similarity'):
    """
    Function to import specific fingerprint classes and similarity measures from different files.
    """
    module_name = 'madas.fingerprints.' + fp_type + file_suffix
    module = importlib.import_module(module_name)
    class_name = fp_type + class_suffix
    similarity_measure_name = fp_type + similarity_measure_suffix
    fingerprint_class = getattr(module, class_name)
    similarity_function = getattr(module, similarity_measure_name)
    return fingerprint_class, similarity_function

class Fingerprint():
    """
    Generic description of a fingerprint.

    Intended use case is subclassing to integrate with the `madas` framework.

    Arbitrary data can be passed to the Fingerprint object by using `Fingerprint().set_data(<key>: str, <data>: Any)`. It can be retrieved either through the property `Fingerprint().data` as a dictionary, or directly via `Fingerprint()[<key>]`.

    **Parameters:**

    fp_type: *str* or *type*

    Keyword arguments are passed to the `__init__` of the specific fingerprint.

    **Methods:**
    """

    def __init__(self, fp_type = None, 
                       name = None,
                       similarity_function = None,
                       pass_on_exceptions = False,
                       importfunction = import_builtin_module, 
                       **kwargs) -> None:
        if isinstance(fp_type, str) or fp_type is None:
            fingerprint_class = None
        elif isinstance(fp_type, type):
            fingerprint_class = fp_type
            fp_type = fp_type.__name__
        else:
            raise TypeError("Attibute fp_type must be either a string or a Fingerprint child type")
        name = self._name_from_type(name, fp_type)
        self._safe_set_attribute("similarity_function", similarity_function)
        self._safe_set_attribute("_fp_type", fp_type)
        self._safe_set_attribute("_name", name)
        self._safe_set_attribute("_pass_on_exceptions", pass_on_exceptions)
        self._safe_set_attribute("_data", {})
        self._safe_set_attribute("_mid", "unknown")
        if self.fp_type is not None and type(self) == Fingerprint:
            if fingerprint_class is None:
                self.set_import_function(importfunction)
                fingerprint_class, similarity_function = self.importfunction(fp_type)
            self.specify(fingerprint_class, **kwargs)
            if self.similarity_function is None:
                self.set_similarity_function(similarity_function)

    @property
    def data(self) -> dict:
        """
        Get data dictionary of the Fingerprint.
        """
        try:
            return self._data
        except AttributeError:
            return {}

    @property
    def mid(self):
        try:
            return self._mid
        except AttributeError:
            return "unknown"

    @property
    def pass_on_exceptions(self):
        try:
            return self._pass_on_exceptions
        except AttributeError:
            return True

    @property
    def fp_type(self):
        try:
            return self._fp_type
        except AttributeError:
            return self.__class__.__name__

    @property 
    def name(self):
        try:
            return self._name
        except AttributeError:
            return self.fp_type

    def set_pass_on_exceptions(self, behavior: bool) -> None:
        """
        Set if fingerprint should pass and return None if an exception occures during calculation of similarities, or raise an Exception.

        **Arguments:**

        behavior: *bool*
            Set behavior.

            True -> Return *None* if exceptions are raised during calculation of similarities.

            False -> Raise exceptions.

        """
        self._pass_on_exceptions = behavior

    def set_data(self, key: str, data: Any) -> None:
        """
        Set data entries of the Fingerprint.

        **Arguments:**
        
        key: *str*
            Name of the property that is meant to be stored
        
        data: *Any*
            Properties to be stored as fingerprint data 

        """
        try:
            self._data[key] = data
        except AttributeError:
            self._data = {}
            self._data[key] = data

    def set_mid(self, mid: str | Material) -> None:
        """
        Set material id of Fingerprint.

        **Arguments:**

        mid: *str* or *Material*
            Material id to be set.

            If a Material is given, the mid will be extracted from there.

        **Returns:**

        None
        """
        if isinstance(mid, Material):
            mid = mid.mid
        self._mid = mid

    def set_fp_type(self, fp_type: str):
        if not isinstance(fp_type, str):
            raise TypeError("This function only sets the string describing the fingerprint. To specify the type of fingerprint, use the function specify().")
        self._fp_type = fp_type

    def set_name(self, name: str):
        self._name = name

    def get_similarity(self, other: object) -> float:
        """
        Calculate similarity to another fingerprint. 
        
        **Arguments:**
    
        other: *Fingerpint*
            Other fingerprint to calculate similarity

        **Returns:**

        similarity: *float*
            Similarity between both fingerprints

        **Raises:**

        TypeError: Different fingerprint types are used.
        """
        if not self._check_type_compatibility(other):
            return 0
        try:
            similarity = self.similarity_function(self, other)
            return similarity
        except Exception as exception:
            error_message = f'Could not calculate similarity for materials: {self.mid} and {other.mid} because of error: {str(exception)}'
            report_error(None, error_message)
            if self.pass_on_exceptions:
                report_error(None, "Setting similarity to 0.")
                return 0
            else:
                raise exception

    def get_similarities(self, fingerprint_list: list) -> List[float]:
        """
        Calculate similarity to all other fingerprints in *fingerprint_list*. 
        
        **Arguments:**
    
        fingerprint_list: *List[Fingerpint]*
            Other fingerprints to calculate similarity

        **Returns:**

        similarities: *List[float]*
            Similarities of self to all other fingerprints

        **Raises:**

        TypeError: Different fingerprint types are used.
        """
        similarities = [self.get_similarity(fp) for fp in fingerprint_list]
        return similarities

    def set_similarity_function(self, similarity_function: Callable, **kwargs) -> object:
        """
        Set the function for calculating similarity and return self.

        **Arguments:**
        
        similarity_function: *Callable* 
            Function that takes two fingerprint objects as input and calculates the similarity between them

        Keyword arguments are passed to the function.
        """
        if len(kwargs) != 0:
            self.similarity_function = partial(similarity_function, **kwargs)
        else:
           self.similarity_function = similarity_function
        return self

    def set_import_function(self, import_function: Callable) -> None:
        """
        Set the function used to import specialized fingerprint objects.
        
        **Arguments:**

        import_function: *Callable*; 
            A function that returns a specialized Fingerprint class and a similarity function.
        """
        self.importfunction = import_function

    def specify(self, fingerprint_class, **kwargs):
        """
        Change self to class given by ``fingerprint_class``.
        
        **Arguments:**

        fingerprint_class: *type*
            Subclass of *Fingerprint*

        Keyword arguments are passed to the __init__ function of the specialized fingerprint.
        """
        self.__class__ = fingerprint_class
        self.__init__(**kwargs)

    def calculate(self, mid: str, *args, **kwargs) -> object | None:
        """
        To be defined in child class. Calculate the fingerprint from values.
        
        **Expected return value:**
        
        self: *Fingerprint*
            It is expected that the object returns itself after calling *calculate*.
        """
        self.set_mid(mid)
        return self
    
    def from_material(self, material: Material, *args, **kwargs):
        """
        To be defined in child class. Calculate the fingerprint from a Material object.
        
        **Expected return value:**
        
        self: *Fingerprint*
            It is expected that the object returns itself after calling *calculate*.
        """
        self.set_mid(material)
        return self

    def get_data_json(self) -> str:
        """
        Return fingerprint data in as a json-coded string.
        """
        json_data = json.dumps(self.data)
        return json_data
    
    def copy(self) -> object:
        """
        Return a deep copy of the fingerprint.
        """
        return deepcopy(self)

    def serialize(self) -> str:
        """
        Write fingerprint data and metadata to dictionary or string.

        **Returns:**

        data: *str*
            json dump of fingerprint
        """
        data = {
            "mid" : self.mid, 
            "name" : self.name,
            "fp_type" : self.fp_type,
            "data" : self.get_data_json(),
            "pass_on_exceptions" : self.pass_on_exceptions 
            }
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data: str) -> object:
        """
        Read a fingerprint object from string data.
        This returns a *Fingerprint* object, _not_ a subclass of it. 
        The similarity function needs to be set separately, and calculate() won't work.

        **Arguments:**
        
        data: *str*
            Sting data of fingerprint (json serialized), obtained by Fingerprint().serialize().

        **Returns:**

        fingerprint: *Fingerprint*
            Fingerprint object with all fields set. 
        """
        data = json.loads(data)
        self = cls()
        self.set_mid(data["mid"])
        self.set_name(data["name"])
        self.set_fp_type(data["fp_type"])
        self.set_pass_on_exceptions(data["pass_on_exceptions"])
        self.from_data(json.loads(data["data"]))
        return self

    @staticmethod
    def _name_from_type(name, fp_type):
        if name is None:
            if isinstance(fp_type, type):
                name = fp_type.__name__
            else:
                name = str(fp_type)
        return name

    def from_data(self, data: dict): #TODO test, and make sure this is what you want
        self._data = data
        return self

    def _check_type_compatibility(self, other):
        if not isinstance(other, Fingerprint):
            error_message = f"Object of type {type(other)} is not a Fingerprint(). Can not calculate similarity"
            if self.pass_on_exceptions:
                report_error(None, error_message)
                return False
            else:
                raise TypeError(error_message)
        if not self.fp_type == other.fp_type:
            error_message = f"Can not calculate similarity for fingerprints of different types. self: {self.fp_type}, other: {other.fp_type}"
            if self.pass_on_exceptions:
                report_error(None, error_message)
                return False
            else:
                raise TypeError(error_message)
        return True

    def _safe_set_attribute(self, attibute_name: str, default: Any):
        if not hasattr(self, attibute_name):
            self.__setattr__(attibute_name, default)

    def __getitem__(self, key):
        return self.data[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fingerprint):
            return False
        if self.fp_type == other.fp_type:
            if self.mid == other.mid:
                if self.name == other.name:
                    if self.get_data_json() == other.get_data_json():
                        return True
        return False

    def __repr__(self) -> str:
        return f"Fingerprint({self.fp_type}, mid = {self.mid}, name = {self.name})"
