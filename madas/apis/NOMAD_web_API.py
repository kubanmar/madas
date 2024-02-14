from typing import Any, List, Callable
from functools import partial
from itertools import islice
from copy import deepcopy
from multiprocessing.pool import ThreadPool
import traceback

import requests
from ase import Atoms
import numpy as np

from madas.apis.api_core import APIClass, APIError
from madas import Material
from madas.utils import tqdm

def get_atoms(NOMAD_respose: dict) -> Atoms:
    """
    Creates an ASE Atoms object from a NOMAD API response.
    """
    structure_data = NOMAD_respose["archive"]["results"]['properties']["structures"]["structure_original"]
    pbc = NOMAD_respose["archive"]["run"][0]["system"][0]["atoms"]["periodic"]
    at = Atoms(positions=np.array(structure_data['cartesian_site_positions']) * 1e10, 
               cell=np.array(structure_data["lattice_vectors"]) * 1e10, 
               symbols=structure_data["species_at_sites"], 
               pbc=pbc)
    return at

def get_archive(NOMAD_respose: dict) -> dict:
    """
    Return the Archive entry of a NOMAD entry from the API response. 
    """
    return NOMAD_respose['archive']

DEFAULT_PROCESSING = {
    "atoms" : get_atoms,
    "archive" : get_archive
}

DEFAULT_BASE_URL = "https://nomad-lab.eu/prod/v1/api/v1"

class API(APIClass):
    """
    API connection wrapper for the NOMAD project.

    Allows to download and process data using the NOMAD web API.

    **Keyword arguments**

    processing: `dict[str:Callable]`
        Dictionary that provides functions to process the data that is downloaded from NOMAD.
        By default, the full NOMAD Archive is stored. This is a considerable overhead if 
        not all data is required. In that case, it is suggested to limit the amout of data that is 
        requested from NOMAD (see, e.g., the docstring of `get_calculation`) and to provide a custom 
        function to process the data, such that only the necessary meta-data is kept.

        default: 
        
        .. code-block:: python

            DEFAULT_PROCESSING = {
                "atoms" : madas.apis.NOMAD_web_API.get_atoms,
                "archive" : madas.apis.NOMAD_web_API.get_archive
            }

    logger: `logging.Logger` | `None`
        Logger to write error and info messages.

    **Methods:**         
    """

    def __init__(self, processing=DEFAULT_PROCESSING, logger=None):
        self.set_logger(logger)
        self.set_processing(processing)
        self._failed_download = set()

    @property
    def processing(self):
        """
        Dictionary defining how data from NOMAD will be processed.

        Keys indicate the name of the property in MADAS.

        Functions are used to read values from the NOMAD archives.
        """
        return deepcopy(self._processing)

    @property
    def failed_download(self):
        """
        List of entry_id of calculations that could not be downloaded.

        Use `API().retry()` to try to retrieve them.
        """
        return list(self._failed_download)

    def get_calculation(self, 
                        entry_id: str, 
                        required: dict = {"required" : "*"}, 
                        return_raw: bool = False,
                        fail_quietly: bool = False) -> Material | dict:
        """
        Download data for a single calculation from NOMAD.

        **Arguments:**

        entry_id: `str`
            Unique `entry_id` as it is used by NOMAD

        **Keyword arguments:**
        
        required: `dict`
            Dictionary containing the information which sections of the NONAD Archive are downloaded.
            This argument is passed directly to the NOMAD API and _must_ follow its definition given
            in the NOMAD documentation. Downloads the full Archive by default.

            default: ``{'required' : '*'}``

        return_raw: `bool`
            Return the response from the NOMAD API withour processing. This is helpful for debugging
            and for development of own `processing` functions.

            NOTE: The subsequent parts of the code use only the 'data' part of the response.

            default: `False`

        fail_quietly: `bool`
            Instead of raising an Exception, return a dictionary with the `entry_id`, error message and traceback.

            default: `False`

        **Returns:**

        material: `madas.material.Material`
            `Material` object with parsed properties

        or

        error_dict: `dict`
            Only if `fail_quietly==True`, dictionary describing the error that occured.

            Format: ``{error_message: str, entry_id: entry_id, traceback: traceback}`` 
        """
        url = self._URL_from_entry_id(entry_id)
        try:
            resp = requests.post(url, json=required).json()
            if return_raw:
                return resp
            resp = resp["data"]
            atoms = self.processing["atoms"](resp) if "atoms" in self.processing.keys() else None
            data = {key:func(resp) for key, func in self.processing.items() if not key == "atoms"}
        except Exception as e:
            if fail_quietly:
                return {"error_message" : str(e), "entry_id":entry_id, "traceback" : traceback.format_exc()}
            raise e        
        return Material(entry_id, atoms=atoms, data=data)
    
    def get_calculations_by_search(self, 
                                   query: dict, 
                                   required: dict = {"required" : "*"},
                                   n_threads: int = 5,
                                   max_entries: int | None = None,
                                   ignore_entries: list | None = None) -> List[Material]:
        """
        Download several calculations from NOMAD, defined by a query. 
        Uses multithreading to maximize download speed.
        If entries can not be downloaded, a message will be written to the log.

        **Arguments:**

        query: `dict`
            Query submitted to the NOMAD API. The format must comply with the NOMAD standard.

        **Keyword arguments:**
        
        required: `dict`
            Dictionary containing the information which sections of the NONAD Archive are downloaded.
            This argument is passed directly to the NOMAD API and _must_ follow its definition given
            in the NOMAD documentation. Downloads the full Archive by default.

            default: ``{'required' : '*'}``

        n_threads: `int`
            Number of threads to start. A too high number may result in unexpected behaviour and unnecessary overhead.

            Set to any number smaller than 1 (one) to disable threading. 
            
            default: `100`


        max_entries: `int` or `None`
            Maximal number of entries to retrieve. Set to `None` to retrieve all data.

            default: `None`

        ignore_entries: `List[str]` or `None`
            Do not download entries with the any id from the given list.

            default: `None` 

        **Returns:**

        materials_list: `List[madas.material.Material]`
            List of Materials for the given query. 
                    
        """
        ids = self._query_for_entries(query, max_entries=max_entries)
        self._report_error(f"Found {len(ids)} entries", level="info")
        if max_entries is not None:
            self._report_error("Possibly not all entries discovered due to max_entries limit", level="info")        
            self._report_error(f"Downloading {int(max_entries)} entries", level="info")        
            ids = set(islice(ids, int(max_entries)))
        if ignore_entries is not None:
            for id in ignore_entries:
                ids.discard(id)
            self._report_error(f"Download data for {len(ids)} entries", level="info")
        query_function = partial(self.get_calculation, required=required, fail_quietly=True)
        if n_threads < 1:
            materials = [query_function(id_) for id_ in ids]
        else:
            with ThreadPool(int(n_threads)) as pool:
                materials = pool.map(query_function, ids)
        self._report_error("Finished download.", level="info")
        errornous_entries = []
        for idx, mat in enumerate(materials):
            if not isinstance(mat, Material):
                message = f"Could not get calculation {mat['entry_id']}, because of error {mat['error_message']} with traceback {mat['traceback']}."
                self._report_error(message)
                errornous_entries.append(idx)
        for idx in sorted(errornous_entries, reverse=True):
            failed_download = materials.pop(idx)
            self._failed_download.add(failed_download['entry_id'])
        return materials
        
    def get_property(self,
                     processing_function: Callable,
                     entry_id: str, 
                     required: dict = {"required" : "*"}) -> Any:
        """
        Receive a property from NOMAD.

        **Arguments:**

        processing_function: `Callable`
            Function to extract data from NOMAD Archive.
            For more details please refer to the documentation of the class.

        entry_id: `str`
            NOMAD entry id of the calculation that shall be downloaded.

        **Keyword arguments:**

        required: `dict`
            Dictionary containing the information which sections of the NONAD Archive are downloaded.
            This argument is passed directly to the NOMAD API and _must_ follow its definition given
            in the NOMAD documentation. Downloads the full Archive by default.

            default: ``{'required' : '*'}``

        **Returns:**

        property: `Any`
            The desired property, parsed from the NOMAD Archive.
        """
        resp = requests.post(self._URL_from_entry_id(entry_id), json=required).json()
        return processing_function(resp["data"])

    def set_processing(self, processing: dict) -> None:
        """
        Set `processing` property of the API.

        See documentation of the class for more information.
        """
        self._processing = processing

    def retry(self, required: dict = {"required" : "*"}, use_progress_bar: bool= False) -> List[Material]:
        """
        Retry previously failed downloads.

        **Keyword arguments:**

        required: `dict`
            Dictionary containing the information which sections of the NONAD Archive are downloaded.
            This argument is passed directly to the NOMAD API and _must_ follow its definition given
            in the NOMAD documentation. Downloads the full Archive by default.

            default: ``{'required' : '*'}``

        use_progress_bar: `bool`
            Show progress bar when downloading data.

        **Returns:**

        materials: `List[madas.Material]`
            Materials that could successfully be downloaded with this attempt.
        """
        if len(self.failed_download) == 0:
            self._report_error("Retry list is empty.")
            return []
        self._report_error(f"Retrying {len(self.failed_download)} entries.", level="info")
        materials = []
        for mid in tqdm(self.failed_download, disable=not use_progress_bar):
            new_mat = self.get_calculation(mid, fail_quietly=True, required=required)
            if not isinstance(new_mat, Material):
                self._report_error(f"Retry failed for material with entry_id {mid}. Error message: {new_mat['error_message']}")
                continue
            self._failed_download.discard(mid)
            materials.append(new_mat)
        return materials

    def _URL_from_entry_id(self, entry_id: str):
        return f"{DEFAULT_BASE_URL}/entries/{entry_id}/archive/query"


    def _gen_mid(self, query):
        if isinstance(query, str):
            return query
        if "entry_id:any" in query.keys():
            return query["entry_id:any"][0]
        else:
            raise ValueError("Could not retrieve entry id from query. Is this query correct?")           
            
    def _query_for_entries(self, query: dict, max_entries: int | None = None) -> List[str]:
        entries = set()
        payload = {"query" : query, "required" : {"include": ["entry_id"]}}
        resp = requests.post(f"{DEFAULT_BASE_URL}/entries/query", json=payload).json()
        entries.update(self._ids_from_response(resp))
        total_entries = self._get_max_entries(resp)
        if max_entries is not None:
            total_entries = min([total_entries, max_entries])
        while len(entries) < total_entries:
            last_len = len(entries)
            payload = self._update_pagination(payload, page_after=self._get_page_after(resp)) 
            resp = requests.post(f"{DEFAULT_BASE_URL}/entries/query", json=payload).json()
            entries.update(self._ids_from_response(resp))
            if not len(entries) > last_len:
                raise APIError(f"Did not add any entry ids at {len(entries)} of {total_entries} entries")
        return entries
            
    def _ids_from_response(self, response: dict) -> List[str]:
        return [entry["entry_id"] for entry in response["data"]]

    def _update_pagination(self, json_data: dict, page_after=None) -> dict:
        if page_after is not None:
            if "pagination" not in json_data.keys():
                json_data["pagination"] = {"page_after_value" : page_after}
            else:
                json_data["pagination"]["page_after_value"] = page_after
        return json_data

    def _get_page_after(self, response: dict) -> str:
        return response["pagination"]["next_page_after_value"]
    
    def _get_max_entries(self, response: dict) -> int:
        return response["pagination"]["total"]