import pytest
import numpy as np

from simdatframe import Fingerprint, SimilarityMatrix

from simdatframe.analysis import MetricSpaceTest

@pytest.fixture
def fingerprints():
    points = np.arange(0,1.01,0.01), np.arange(0,1.01,0.01)
    func = lambda x, y: np.exp(-0.5 * (x**2 + y**2) / 0.25)
    fps = [Fingerprint("DUMMY", target_function = func).from_list(list(point)) for point in zip(*points)]
    return fps

def unity(*args):
    return 1

def non_symmetric(x,y):
    if x["data"][0] <= y["data"][0]:
        return 0.5
    else:
        return 1

def test_init(fingerprints):

    simat = SimilarityMatrix().calculate(fingerprints)

    mst = MetricSpaceTest(fingerprints)

    assert mst.similarity_matrix == simat, "Did not calculate correct similarity matrix"

def test_call(fingerprints):

    mst = MetricSpaceTest(fingerprints)

    assert mst() == [True, True, True, True], "Test failed for metric space"

def test_uniqueness(fingerprints):

    for fp in fingerprints:
        fp.set_similarity_function(unity)

    mst = MetricSpaceTest(fingerprints)

    assert not mst(only=["uniqueness"])[0], "Uniqueness passed for non-unique space"

def test_symmetry(fingerprints):

    for fp in fingerprints:
        fp.set_similarity_function(non_symmetric)

    mst = MetricSpaceTest(fingerprints)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(mst.similarity_matrix)
    plt.colorbar()
    plt.show()

    assert not mst(only=["symmetry"])[0], "Symmetry passed for non-symmetric similarity measure"