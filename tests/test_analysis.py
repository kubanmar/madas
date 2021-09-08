import pytest
import numpy as np

from simdatframe import Fingerprint
from simdatframe.fingerprint import DBRowWrapper

from simdatframe.analysis import MetricSpaceTest

@pytest.fixture
def fingerprints():
    points = np.meshgrid(np.arange(0,1.01,0.01), np.arange(0,1.01,0.01))
    func = lambda x, y: np.exp(-0.5 * (x**2 + y**2) / 0.25)
    fps = [Fingerprint("DUMMY", db_row=DBRowWrapper({"x" : x, "y" : y, "z" : func(x,y)})) for x, y in zip(points[0][0], points[1][0])]
    return fps

@pytest.mark.skip()
def test_identity(fingerprints):

    mst = MetricSpaceTest(fingerprints)

    