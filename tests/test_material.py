import pytest

from simdatframe import Material

@pytest.fixture()
def material():
    return Material("a")

def test_init(material):
    assert material.mid == "a", "Did not set mid correctly"
    assert material.atoms == None, "Did not set atoms correctly"
    assert material.data == {}, "Did not set data correctly"
    assert material.properties == {}, "Did not set properties correctly"

def test_get_property_by_path(material):

    with pytest.raises(KeyError):
        material.get_property_by_path("test/data")

    material.set_properties({"test" : {"data" : 1}})

    assert material.get_property_by_path("test/data") == 1, "Did not retrieve property by path"

def test_get_data_by_path(material):

    with pytest.raises(KeyError):
        material.get_data_by_path("test/data")

    material.set_data({"test" : {"data" : 1}})

    assert material.get_data_by_path("test/data") == 1, "Did not retrieve data by path"