from simdatframe import Fingerprint, Material
import pytest

@pytest.fixture
def fingerprint():
    return Fingerprint()

@pytest.fixture()
def test_material():
    return Material("a:b", atoms=None, data = {"test" : "data"})

def test_imports():
    
    from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint, DUMMY_similarity
    from simdatframe.fingerprint import import_builtin_module
    
    fingerprint_class, similarity_function = import_builtin_module("DUMMY")

    assert fingerprint_class == DUMMYFingerprint, "Error importing fingerprint class"
    assert similarity_function == DUMMY_similarity, "Error importing similarity function"

def test_specify(fingerprint):

    from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint

    def mock_target_function(x: list):
        return sum(x)

    fingerprint.specify(DUMMYFingerprint, target_function = mock_target_function)

    assert isinstance(fingerprint, DUMMYFingerprint), "Fingerprint class was not updated during `specify`"

    fingerprint.set_data("data", [1,2,3])

    assert fingerprint.y == 6, "Target function was not passed as kwarg to fingerprint"

def test_specify_on_init():

    from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint

    def mock_target_function(x: list):
        return sum(x)

    fingerprint = Fingerprint("DUMMY", target_function = mock_target_function)

    assert isinstance(fingerprint, DUMMYFingerprint), "Specify was not called upon initialization."

    fingerprint.set_data("data", [1,2,3])

    assert fingerprint.y == 6, "Target function was not passed as kwarg to fingerprint"

    class NewFingerprint(Fingerprint):

        def __init__(self, a = 1, b = 1):
            self.a = a
            self.b = b

        def calculate(self, *args):
            self.set_data("result", self.a + self.b)
            return self

    fingerprint = Fingerprint(NewFingerprint, a=2, b=3)

    assert fingerprint.calculate().data["result"] == 5, "Did not initialize fingerprint from type object"
    assert fingerprint.fp_type == "NewFingerprint", "Did not initialize fingerprint from type object"

def test_subclassing(test_material):

    class SubFingerprint(Fingerprint):

        def __init__(self, a = 1, b = 1):
            self.a = a
            self.b = b

        def calculate(self, material):
            self.set_mid(material)
            self.set_data("result", self.a + self.b)
            return self

    sub_fp = SubFingerprint().calculate(test_material)

    assert sub_fp.mid == test_material.mid, "Did not set mid from material in subclass"


def test_set_data(fingerprint):

    fingerprint.set_data("test", "data")

    assert fingerprint.data["test"] == "data", "Data was not set correctly."

    assert fingerprint["test"] == "data", "Data could not be retrieved via __getitem__"

    fingerprint.set_data("list", [1,2,3])

    assert fingerprint["list"] == [1,2,3], "Failed to set two different data entries"

def test_set_similarity(fingerprint):

    def dummy_similarity(*args):
        return "Passed"

    fingerprint.set_similarity_function(dummy_similarity)

    assert fingerprint.get_similarity(Fingerprint()) == "Passed", "Unable to set similarity function"

def test_get_similarity(fingerprint):

    fp2 = Fingerprint()

    def mock_similarity_function(*args):
        return True

    fingerprint.set_similarity_function(mock_similarity_function)

    assert all(fingerprint.get_similarities([fp2, fp2])), "Could not calculate similarities for several fingerprints"

    fingerprint.set_pass_on_exceptions(False)

    fp2.set_fp_type("Whatever")

    with pytest.raises(TypeError):
        # This should fail because the fingerprint types are not the same
        fingerprint.get_similarity(fp2)

    with pytest.raises(TypeError):
        # This should fail because the other is no fingerprint
        fingerprint.get_similarity(0)

    fingerprint.set_pass_on_exceptions(True)

    assert fingerprint.get_similarity(fp2) == None, "Passing on exceptions, invalid fingerprints should return None as similarity"

    assert fingerprint.get_similarity(0) == None, "Passing on exceptions, invalid fingerprints should return None as similarity"

    def mock_failing_similarity_function(*args):
        raise AssertionError("I am failing.")

    fingerprint.set_similarity_function(mock_failing_similarity_function)

    fp2 = Fingerprint()

    assert fingerprint.get_similarity(fp2) == None, "Passing on exceptions, failing similarity functions should return None as similarity"

    fingerprint.set_pass_on_exceptions(False)

    with pytest.raises(AssertionError):
        # This should fail because the similarity function raises an exception
        fingerprint.get_similarity(fp2)
