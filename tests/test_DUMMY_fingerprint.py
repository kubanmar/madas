from simdatframe.fingerprint import Fingerprint
from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint

def test_dummy_fingerprint():
    data = [1,2,3,4]
    fp = DUMMYFingerprint.from_list(data, pass_on_exceptions = False)
    assert fp.fp_id == "1:2:3:4", "Failed setting fingerprint id"
    assert not fp.pass_on_exceptions, "kwargs are not passed to Fingerprint"
    assert fp.get_similarity(fp) == 1, "Wrong similarity"
    assert fp.y == 30, "wrong target value"
