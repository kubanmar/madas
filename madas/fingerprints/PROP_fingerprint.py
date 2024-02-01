from typing import Callable 
from typing import Any

from madas import Material
from madas.fingerprint import Fingerprint
from madas.utils import report_error, resolve_nested_dict

def PROP_similarity(fingerprint1: Fingerprint, 
                    fingerprint2: Fingerprint, 
                    delta: float = 1, 
                    scaling_function: Callable = lambda x: x) -> float:
    """
    Calculate similarity between PROPFingerprints.

    Calculates *S(a, b) = delta / ( scaling_function(|a - b|) + delta )*

    where `S` is the similarity, `a` and `b` are scalar material properties encoded in `PROPFingerprint` objects, `delta` is an arbitrary scalar value to rescale the similarity, and `scaling_function` is any function f(x) -> y with real numbers x,y >= 0.

    **Arguments:**

    fingerprint1, fingerprint2: `Fingerprint`
        PROPFingerprints of to calculate similarity

    **Keyword arguments:**

    delta: `float`
        Scaling factor that allows to define how the absolute differences in the scalar property affect the similarity.

    scaling_function: `Callable`
        Function to dynamically rescale the absolute differences in properties.

    **Returns:**

    similarity: `float`
        Similarity between input fingerprints.
    """
    if not fingerprint1.fp_type == fingerprint2.fp_type == "PROP":
        raise TypeError("PROP similarity can only be used for PROP fingerprints.")
    property_name = fingerprint1.data["property_path"]
    if not fingerprint2.data["property_path"] == property_name:
        raise ValueError("Trying to calculate similarity of different properties, which is impossible.")
    value1 = fingerprint1.data["value"]
    value2 = fingerprint2.data["value"]
    distance = abs(value1 - value2)
    similarity = delta / (scaling_function(distance) + delta)
    return similarity

class PROPFingerprint(Fingerprint):
    """
    A fingerprint object to represent a user defined property.

    Provide the path to a property to compute 
    """

    def __init__(self, 
                 name = "PROP", 
                 property_path: str | Callable = 'atomic_density',
                 similarity_function = PROP_similarity, 
                 from_data: bool = True):
        self._from_data = from_data
        if callable(property_path):
            self.set_data("property_path", f"function_{property_path.__name__}")
            self._compute_function = property_path
        else:      
            self.set_data("property_path", property_path)
            self._compute_function = None
        self.set_fp_type("PROP")
        self.set_name(name)
        self.set_similarity_function(similarity_function)

    def from_material(self, material: Material) -> object:
        """
        Set fingerprint data to property_path given during initialization. 

        Defines the following data in `Fingerprint.data`:

        *value* : Poperty value, extracted from Material.data, specified by *property_path*.

        **Returns:**

        self: *PROPFingerprint*

        **Raises:**

        KeyError: if ``self.pass_on_exceptions == False`` and the material has not data entry with `property_path`.
        """
        self.set_mid(material)
        if self._compute_function is not None:
            self.set_data("value", self._compute_function(material))
            return self
        try:
            prop = material.get_data_by_path(self.data["property_path"])
        except KeyError as e:
            message = f'No property of type {self.data["property_path"]} for material {material.mid}.'
            if self.pass_on_exceptions:
                report_error(None, message)
                return None
            else:
                report_error(None, message)
                raise e
        self.set_data("value", prop)
        return self

    def calculate(self, data: Any = None) -> object:
        """
        Added for compatibility with the API.
        """
        if self._compute_function is not None:
            self.set_data("value", self._compute_function(data))
            return self
        try:
            prop = resolve_nested_dict(data, self.data['property_path'], fail_on_key_error=True)
        except KeyError as key_error:
            message = f'No property of type {self.data["property_path"]}.'
            if self.pass_on_exceptions:
                report_error(None, message)
                return None
            else:
                report_error(None, message)
                raise key_error
        self.set_data("value", prop)
        return self
