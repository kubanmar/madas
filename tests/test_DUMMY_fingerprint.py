from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint

def test_dummy_fingerprint():
    data = [1,2,3,4]
    fp = DUMMYFingerprint(data)
    assert fp.get_similarity(fp) == 1, "Wrong similarity"
    assert fp.y == 30, "wrong target value"

