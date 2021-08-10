from simdatframe.fingerprint import Fingerprint, DBRowWrapper
from simdatframe.apis.NOMAD_enc_API import API
import pytest


#api = API(key_path = "/users/stud/kuban/PhD/similarity/similarity_data_framework")

#material1 = api.get_calculation(1000885, 2913391)
#material2 = api.get_calculation(1000903, 2908312)

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
