import pytest

from simdatframe import SimilarityMatrix, Fingerprint
from simdatframe.machine_learning import SimilarityMatrixRegression

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def target_function(x):
    return np.exp(-1 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.05)

@pytest.fixture
def fingerprints():
    from itertools import product
    fps = [Fingerprint("DUMMY").from_list(arr, target_function = target_function) for arr in product(np.linspace(0,1,num = 20), np.linspace(0,1,num = 20))]    
    return fps

@pytest.fixture
def similarity_matrix(fingerprints):
    return SimilarityMatrix().calculate(fingerprints)

def test_SimilarityMatrixRegression(similarity_matrix, fingerprints):

    train_set_fps, test_set_fps = train_test_split(fingerprints, random_state=0)

    train_mids  = [fp.mid for fp in train_set_fps]
    test_mids  = [fp.mid for fp in test_set_fps]
    train_y = [fp.y for fp in train_set_fps]
    test_y = [fp.y for fp in test_set_fps]

    train_simat, test_simat = similarity_matrix.train_test_split(train_mids, test_mids)

    smr = SimilarityMatrixRegression(train_simat, test_simat)

    smr.fitCV(train_y, regressor_parameters = {'alpha': np.logspace(-1,-12, num = 5)}, show_progress=True, random_seed=0)

    preds = smr.predict()
    preds_train = smr.predict_fit()

    assert smr.evaluate_test(test_y) <= 1e-4, "Failed prediction"

    assert np.allclose(preds, test_y, atol=5e-2)

    assert mean_squared_error(preds_train, train_y) <= 1e-24, "Failed fit"
