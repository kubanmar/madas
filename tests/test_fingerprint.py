from simdatframe.fingerprint import Fingerprint, DBRowWrapper
import pytest

@pytest.fixture
def fingerprint():
    return Fingerprint()

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

    fingerprint = Fingerprint("DUMMY")

    assert isinstance(fingerprint, DUMMYFingerprint), "Specify was not called upon initialization."

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

    fingerprint.pass_on_exceptions = False

    fp2.fp_type = "Whatever"

    with pytest.raises(TypeError):
        # This should fail because the fingerprint types are not the same
        fingerprint.get_similarity(fp2)

    with pytest.raises(TypeError):
        # This should fail because the other is no fingerprint
        fingerprint.get_similarity(0)

    fingerprint.pass_on_exceptions = True

    assert fingerprint.get_similarity(fp2) == None, "Passing on exceptions, invalid fingerprints should return None as similarity"

    assert fingerprint.get_similarity(0) == None, "Passing on exceptions, invalid fingerprints should return None as similarity"

    def mock_failing_similarity_function(*args):
        raise AssertionError("I am failing.")

    fingerprint.set_similarity_function(mock_failing_similarity_function)

    fp2 = Fingerprint()

    assert fingerprint.get_similarity(fp2) == None, "Passing on exceptions, failing similarity functions should return None as similarity"

    fingerprint.pass_on_exceptions = False

    with pytest.raises(AssertionError):
        # This should fail because the similarity function raises an exception
        fingerprint.get_similarity(fp2)
