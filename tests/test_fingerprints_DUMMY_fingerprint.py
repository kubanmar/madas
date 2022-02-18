from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint

def test_dummy_fingerprint():
    data = [1,2,3,4]
    fp = DUMMYFingerprint.from_list(data)
    fp.set_pass_on_exceptions(True)
    assert fp.mid == "1:2:3:4", "Failed setting fingerprint id"
    assert fp.get_similarity(fp) == 1, "Wrong similarity"
    assert fp.y == 30, "wrong target value"
