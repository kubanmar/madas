from simdatframe.fingerprint import Fingerprint, DBRowWrapper
import pytest


#api = API(key_path = "/users/stud/kuban/PhD/similarity/similarity_data_framework")

#material1 = api.get_calculation(1000885, 2913391)
#material2 = api.get_calculation(1000903, 2908312)
"""
@pytest.mark.skip()
def test_fingerprint():

    row1 = DBRowWrapper(mid = material1.mid, atoms = material1.atoms, data = material1.data)
    row2 = DBRowWrapper(mid = material2.mid, atoms = material2.atoms, data = material2.data)


    fingerprint1 = Fingerprint(fp_type = "DOS", db_row = row1)
    fingerprint2 = Fingerprint(fp_type = "DOS", db_row = row2, name = "MARTIN", stepsize = 0.001)

    row1["DOS"] = fingerprint1.get_data_json()
    row2["MARTIN"] = fingerprint2.get_data_json()

    fingerprint1_cp = Fingerprint(fp_type = "DOS", db_row = row1)
    fingerprint2_cp = Fingerprint(fp_type = "DOS", db_row = row2)

    assert fingerprint1.get_similarity(fingerprint1_cp) == 1
    assert fingerprint2.get_similarity(fingerprint2_cp) == 1
    assert fingerprint2.stepsize == 0.001
"""

@pytest.fixture
def fingerprint():
    return Fingerprint()

def test_imports(fingerprint):
    
    from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint, DUMMY_similarity
    
    fingerprint_class, similarity_function = fingerprint.importfunction("DUMMY")

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
