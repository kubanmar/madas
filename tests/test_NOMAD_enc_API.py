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
def test_get_calculation(test_API, monkeypatch):

    """
    This test is too much tied to the actual implementation. TODO: add actual data and mock out the responses with this NOMAD data.
    
    """

    # test without providing calculation data
    test_material = test_API.get_calculation("a", "b")

    def mock_get_material():
        return None

    def mock_get_archive_data():
        return None

    def mock_get_material_properties():
        return None

    def mock_get_calculation_list():
        return None

    monkeypatch.setattr(test_API, '_get_material', mock_get_material)
    monkeypatch.setattr(test_API, '_get_archive_data', mock_get_archive_data)
    monkeypatch.setattr(test_API, '_get_material_properties', mock_get_material_properties)
    monkeypatch.setattr(test_API, '_get_calculation_list', mock_get_calculation_list)
