import pytest

from simdatframe.apis.NOMAD_enc_API import API
from bravado.client import SwaggerClient

@pytest.fixture
def nomad_test_url():
    return "test/url"

class MockSwaggerClient():
    """
    Mock class to test the responses of the NOMAD API without actually calling it.
    """
    
    def __init__(self, url) -> None:
        self._url = url

    @property
    def url(self):
        return self._url

class MockLogger():

    def __init__(self, id = 0) -> None:
        self.logs = []
        self.id = id

    def info(self, message):
        self.logs.append(message)

def mock_selector(list_ : list) -> dict:
    return list_[0]

def test_init(monkeypatch, nomad_test_url):

    def mock_from_url(url, *args, **kwargs):
        return MockSwaggerClient(url = url)

    monkeypatch.setattr(SwaggerClient, "from_url", mock_from_url)

    test_API = API(nomad_url=nomad_test_url, logger = MockLogger(0), representative_selector = mock_selector)

    assert test_API.base_url == nomad_test_url, "wrong url set in initialization"

    assert test_API.client.url == nomad_test_url + "/swagger.json", "call to `SwaggerClient.from_url()` failed"

    assert test_API.log.id == 0, "wrong logger is set during initialization"
    