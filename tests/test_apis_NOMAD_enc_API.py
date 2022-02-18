import pytest
import requests

from simdatframe.apis.NOMAD_enc_API import API
from bravado.client import SwaggerClient

@pytest.fixture
def test_response():
    return {'test' : 'answer'}

@pytest.fixture
def nomad_test_url():
    return "test/url"

@pytest.fixture
def test_API(monkeypatch, nomad_test_url):

    def mock_from_url(url, *args, **kwargs):
        return MockSwaggerClient(url = url)

    monkeypatch.setattr(SwaggerClient, "from_url", mock_from_url)

    def mock_selector(list_ : list) -> dict:
        return list_[0]

    test_api = API(nomad_url=nomad_test_url, logger = MockLogger(0), representative_selector = mock_selector)
    return test_api

@pytest.fixture
def get_material_calculations_response():
    return {
                "total_results": 0,
                "results": [
                    {
                    "calc_id": "string",
                    "upload_id": "string",
                    "code_name": "string",
                    "code_version": "string",
                    "functional_type": "string",
                    "basis_set_type": "string",
                    "core_electron_treatment": "string",
                    "run_type": "string",
                    "has_dos": True,
                    "has_band_structure": True,
                    "has_thermal_properties": True
                    }
                ],
                "representatives": {
                    "idealized_structure": "string",
                    "electronic_band_structure": "string",
                    "electronic_dos": "string",
                    "thermodynamical_properties": "string"
                }
            }

@pytest.fixture
def search_materials_response():
    return {
                "search_by": {
                    "exclusive": False,
                    "formula": "TiO2",
                    "elements": [
                    "Ti",
                    "O"
                    ],
                    "page": 1,
                    "per_page": 10,
                    "restricted": False
                },
                "query": "string",
                "material_type": [
                    "bulk"
                ],
                "material_name": [
                    "string"
                ],
                "structure_type": [
                    "string"
                ],
                "space_group_number": [
                    0
                ],
                "crystal_system": [
                    "triclinic"
                ],
                "band_gap": {
                    "max": 0,
                    "min": 0
                },
                "has_band_structure": True,
                "has_dos": True,
                "has_thermal_properties": True,
                "functional_type": [
                    "GGA"
                ],
                "basis_set": [
                    "numeric AOs"
                ],
                "code_name": [
                    "ABINIT"
                ]
            }

@pytest.fixture
def get_material_response():
    return {
                "material_id": "string",
                "formula": "string",
                "formula_reduced": "string",
                "material_type": "string",
                "n_calculations": 0,
                "has_free_wyckoff_parameters": True,
                "strukturbericht_designation": "string",
                "material_name": "string",
                "bravais_lattice": "string",
                "crystal_system": "string",
                "point_group": "string",
                "space_group_number": 0,
                "space_group_international_short_symbol": "string",
                "structure_prototype": "string",
                "structure_type": "string",
                "similarity": [
                    {
                    "material_id": "string",
                    "value": 0,
                    "formula": "string",
                    "space_group_number": 0
                    }
                ]
            }

@pytest.fixture
def mock_get_archive_calc_response():
    return {
        "section_metadata" : {
            "encyclopedia" : {
                "material" : {
                    "idealized_structure" : {
                        'atom_labels': ['C'],
                        'atom_positions': [[0, 0, 0]],
                        'lattice_vectors': [[1.7760600000000004e-10, 0, 0],
                        [0, 1.7760600000000004e-10, 0],
                        [0, 0, 1.7760600000000004e-10]],
                        'lattice_vectors_primitive': [[1.7760600000000004e-10, 0, 0],
                        [0, 1.7760600000000004e-10, 0],
                        [0, 0, 1.7760600000000004e-10]],
                        'periodicity': [True, True, True],
                        'number_of_atoms': 1,
                        'cell_volume': 5.60238434686102e-30,
                        'wyckoff_sets': [{'wyckoff_letter': 'a', 'indices': [0], 'element': 'C'}],
                        'lattice_parameters': {'a': 1.7760600000000004e-10,
                        'b': 1.7760600000000004e-10,
                        'c': 1.7760600000000004e-10,
                        'alpha': 1.5707963267948966,
                        'beta': 1.5707963267948966,
                        'gamma': 1.5707963267948966}
                    }
                },
                "properties" : {
                    "energies" : {
                        "energy_total" : -6.978074033123606e-19
                    }
                }
            }
        }
    }

@pytest.fixture
def get_calculation_response():
    return {
        "lattice_parameters": {
            "a": 0,
            "b": 0,
            "c": 0,
            "alpha": 0,
            "beta": 0,
            "gamma": 0
        },
        "energies": {
            "energy_total": 0,
            "energy_total_T0": 0,
            "energy_free": 0
        },
        "mass_density": 0,
        "atomic_density": 0,
        "cell_volume": 0,
        "wyckoff_sets": {
            "wyckoff_letter": "string",
            "indices": [
            0
            ],
            "element": "string",
            "variables": {
            "x": 0,
            "y": 0,
            "z": 0
            }
        },
        "idealized_structure": {
            "atom_labels": [
            "string"
            ],
            "atom_positions": [
            [
                0
            ]
            ],
            "lattice_vectors": [
            [
                0
            ]
            ],
            "lattice_vectors_primitive": [
            [
                0
            ]
            ],
            "lattice_parameters": {
            "a": 0,
            "b": 0,
            "c": 0,
            "alpha": 0,
            "beta": 0,
            "gamma": 0
            },
            "periodicity": [
            True
            ],
            "number_of_atoms": 0,
            "cell_volume": 0,
            "wyckoff_sets": [
            {
                "wyckoff_letter": "string",
                "indices": [
                0
                ],
                "element": "string",
                "variables": {
                "x": 0,
                "y": 0,
                "z": 0
                }
            }
            ]
        },
        "band_gap": 0,
        "electronic_band_structure": {
            "reciprocal_cell": [
            [
                0
            ]
            ],
            "brillouin_zone": {},
            "section_k_band_segment": {},
            "section_band_gap": {}
        },
        "electronic_dos": {
            "dos_energies": [
            0
            ],
            "dos_values": [
            [
                0
            ]
            ]
        },
        "phonon_band_structure": {},
        "phonon_dos": {},
        "thermodynamical_properties": {}
    }

class MockSwaggerClient():
    """
    Mock class to test the responses of the NOMAD API without actually calling it.
    """
    
    def __init__(self, url) -> None:
        self._url = url

    @property
    def encyclopedia(self):
        return MockSwaggerSection()

    @property
    def url(self):
        return self._url

class MockSwaggerResponse():

    response_data = {}

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self):
        response_data = self.response_data
        response_data.update(**self.payload)
        return response_data

class MockSwaggerRequest():

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    @property
    def incoming_response(self):
        return MockSwaggerResponse(self.payload)

class MockSwaggerResponseData():

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def response(self):
        return MockSwaggerRequest(payload=self.payload)

class MockSwaggerSection():

    def search_materials(self, payload = {}):
        return MockSwaggerResponseData(payload)

class MockLogger():

    def __init__(self, id = 0) -> None:
        self.logs = []
        self.id = id

    def info(self, message):
        self.logs.append(message)

class MockRequestsRespose():

    response_data = {}

    def __init__(self, url : str) -> None:
        self.url = url

    def json(self):
        resp = self.response_data
        resp.update(url = self.url)
        return resp

def test_init(nomad_test_url, test_API):

    assert test_API.base_url == nomad_test_url, "wrong url set in initialization"

    assert test_API.client.url == nomad_test_url + "/swagger.json", "call to `SwaggerClient.from_url()` failed"

    assert test_API.log.id == 0, "wrong logger is set during initialization"
    
def test_api_calls(test_API, test_response, monkeypatch):

    # _get_materials

    monkeypatch.setattr(MockSwaggerResponse, 'response_data', test_response)

    response = test_API._get_materials({'Hello' : 'world'})

    assert response['test'] == 'answer', "Did not recieve correct answer from MockAPI"

    assert response['Hello'] == 'world', "API call was made with wrong payload"

    # _get_material

    def mock_requests_get(url):
        return MockRequestsRespose(url)

    monkeypatch.setattr(requests, 'get', mock_requests_get)
    monkeypatch.setattr(MockRequestsRespose, 'response_data', test_response)

    response = test_API._get_material('a')

    assert response['url'] == 'http://nomad-lab.eu/prod/rae/api/encyclopedia/materials/a', "setting wrong url in _get_material"

    assert response['test'] == 'answer', "_get_material made wrong request"

@pytest.mark.skip()
def test_get_calculation(test_API, 
                         monkeypatch, 
                         mock_get_archive_calc_response,
                         get_material_response, ):

    """
    This test is not implemented for two reasons:
        1. It is hard to test the functions not implementation specific, and the implementation may change in the (near) future.
        2. The API may be adopted to use the NOMAD package instead. Then, the whole structure changes.
    """    


    def mock_get_material(*args, **kwargs):
        return get_material_response

    monkeypatch.setattr(test_API, '_get_material', mock_get_material)
    monkeypatch.setattr(test_API, '_get_archive_data', mock_get_archive_calc_response)
    #monkeypatch.setattr(test_API, '_get_material_properties', mock_get_material_properties)
    #monkeypatch.setattr(test_API, '_get_calculation_list', mock_get_calculation_list)

    # test without providing calculation data
    test_material = test_API.get_calculation("a", "b")

