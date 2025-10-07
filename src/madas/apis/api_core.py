import logging
import sys
import traceback
from typing import Callable, List
from typing import Any
from hashlib import md5

from madas.material import Material
from madas.utils import safe_log

class APIClass():
    """
    Base class for APIs to different data sources.
    """

    def __init__(self, 
                 logger = None):
        self.set_logger(logger)

    def get_calculation(self, *args, **kwargs) -> Material:
        """
        Get a single material from the external data source.

        **Expected input:**

        ids: *Any*
            Any piece of data that allows the external resource to identify which data shall be returned.

        **Expected to return:**

        material: *madas.Material*
            Material data in a Material object.

        **Raises:**

        NotImplementedError: This function is not implemented here.
        """
        raise NotImplementedError('The option to get individual calculations is not implemented in this API.')
        return Material()

    def get_calculations_by_search(self, *args, **kwargs) -> List[Material]:
        """
        Get a list of materials from the external data source.

        **Expected input:**

        search_query: *Any*
            Any piece of data that allows the external resource to identify which data shall be returned.

        **Expected to return:**

        material: *List[madas.Material]*
            List of Material objects, containing the data from external source.

        **Raises:**

        NotImplementedError: This function is not implemented here.
        """
        raise NotImplementedError('The option to get sets of calculations is not implemented in this API.')
        return [Material()]

    def get_property(self, **kwargs) -> Any:
        """
        Get a property from the external source.

        **Expected input:**

        kwargs: *Any*
            Any piece of data that allows the external resource to identify which property for which data shall be returned.

        **Expected to return:**

        property: *Any*
            A single property, e.g. a string or float, that can be stored to the database.

        **Raises:**

        NotImplementedError: This function is not implemented here.
        """
        raise NotImplementedError('The option to get individual properties is not implemented in this API.')
        return {'property':None}
    
    def hash_query(self, *args, **kwargs) -> str:
        """
        Hash a query that is passed to the API. This is used to avoid repeatedly querying the same data.
        """
        query_string = ""
        for arg in sorted(map(str, args)):
            query_string+=arg
        for key, val in sorted(kwargs.items(), key = lambda x: x[0]):
            query_string+=f"{key}{val}"
        return md5(query_string.encode()).hexdigest()

    def _gen_mid(self, *args) -> str:
        """
        Generate the unique material identifier from arguments. Mids are used to identify the material both in the external database and the *MaterialsDatabase*.

        **Expected input:**

        args: *Any*
            Any piece of data that allows to generate a unique id.

        **Expected to return:**

        mid: *str*
            Unique identifier of the material.

        **Raises:**

        NotImplementedError: This function is not implemented here.        
        """
        raise NotImplementedError('The option to generate material ids is not implemented in this API.')
        return '<unique>:<identifyer>'

    def set_logger(self, logger: logging.Logger) -> None:
        """
        Set logger of the API. 
        The log is used to improve analysis of errors during collection of materials and increase robustness due to failures of external resources.

        **Arguments:**

        logger: *logging.Logger* or *None*
            Logger object.
            If *logger = None*, errors will be written to STDERR.

        **Returns:**

        None
        """
        self.log = logger

    def _report_error(self, error_message, level = "error"):
        """
        Write error to logging.log (if exists), or stderr.
        """
        safe_log(error_message, logger = self.log, level=level)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

class APIError(Exception):
    """
    A generic exception to specify errors that are raised in an *APIClass*.

    **Usage:**

        message = "Something went wrong!"

        raise APIError(message)
    """

    def __init__(self, message: str):
        super().__init__(message)

class DatabaseEntryDoesNotExistError(Exception):
    """
    A generic exception to be raised if a specific entry does not exist in an external database.

    **Usage:**

        <code that can not retieve data>

        raise DatabaseEntryDoesNotExistError("Can not retrieve entry!")
    """
    def __init__(self, message):
        super().__init__(message)

def api_call(call, 
             retries: int = 3, 
             report_function: Callable = print, 
             report_function_parameters: dict = {"file" : sys.stderr}) -> Callable:
    """
    Decorator function to realize 10 tries for an API call before raising an exception.
    For each failed function call, the decorator will pass the exception name and the number of trials to *report_function*.
    If the number of trials is exceeded, the decorator will pass the name of the failing function, the arguments and keyword arguments to *report_function*.

    **Arguments:**
    
    call: *Callable*
        Function that tries to make an API call to an external database.

    **Keyword arguments:**

    retries: `int`
        Number of retries before passing on the exceptions

        Default: `3`

    report_function: *Callable*
        Function to report error.

        Default: print

    report_function_parameters: `dict`
        Keyword arguments passed to report function.
        
        default: ``{"file" : sys.stderr}`` 

    **Raises:**

    APIError: could not retrieve data

    """
    def call_api(*args, **kwargs):
        trials = 0
        success = False
        while trials < retries and not success:
            try:
                answer = call(*args, **kwargs)
                success = True
            except Exception as error:
                error_message = f'Failed connection to server because of {type(error).__name__}: {str(error)}\nThis was attempt number {str(trials)}.\ntraceback: {traceback.format_exc()}'
                report_function(error_message, **report_function_parameters)
                trials += 1
        if not success:
            error_message = f'Function {call.__name__}: Could not connect to external API! Using args: {args} and kwargs: {kwargs}.'
            report_function(error_message, **report_function_parameters)
            raise APIError(error_message)
        return answer

    return call_api
