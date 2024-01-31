from simdatframe import Fingerprint, Material

import pytest

@pytest.fixture()
def material():
    return Material("a")

def test_from_material(material):

    prop = {"test" : {"path": 1}}
    material.set_data(prop)

    fp = Fingerprint("PROP", property_path = "test/path").from_material(material)

    assert fp.get_similarity(fp) == 1, "Same fingerprint is not identical"

    prop = {"test" : {"path": 2}}
    material.set_data(prop)

    fp2 = Fingerprint("PROP", property_path = "test/path").from_material(material)

    assert fp.get_similarity(fp2) == 0.5, "Wrong similarity between prop fingerprints"

def test_similarity(material):

    prop = {"test" : {"path": 1}}
    material.set_data(prop)

    fp = Fingerprint("PROP", property_path = "test/path", pass_on_exceptions=False).from_material(material)

    prop = {"test1" : {"path": 2}}
    material.set_data(prop)

    fp2 = Fingerprint("PROP", property_path = "test1/path", pass_on_exceptions=False).from_material(material)

    print(fp.data["property_path"])
    print(fp2.data["property_path"])

    with pytest.raises(ValueError):
        fp.get_similarity(fp2)