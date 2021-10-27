import pytest

from simdatframe.machine_learning import MatrixMultiKernelLearning
from simdatframe.fingerprints.DUMMY_fingerprint import DUMMYFingerprint
from simdatframe import SimilarityMatrix, fingerprint

import numpy as np
from sklearn.model_selection import train_test_split

@pytest.fixture
def fingerprints():
    x,y = np.meshgrid(np.round(np.linspace(-1,1, 21), 13), np.round(np.linspace(-1,1, 21), 13))
    fps = [DUMMYFingerprint.from_list([x_i, y_i]) for x_i, y_i in zip(np.concatenate(x),np.concatenate(y))]
    return fps

#@pytest.fixture
#def target_values(fingerprints):
#    return [fp.y for fp in fingerprints]

@pytest.fixture
def similarity_matrix(fingerprints):
    return SimilarityMatrix().calculate(fingerprints)

@pytest.fixture
def predictions():
    return np.array([0.85212033, 0.65186677, 0.85083564, 1.17246202, 1.45456046,
       1.3039208 , 1.04890647, 1.81332328, 0.02171413, 0.37266654,
       0.74205951, 0.80209557, 0.52306346, 0.53255931, 0.40254224,
       1.81366104, 0.26188152, 1.81407074, 1.0093486 , 0.82097386,
       0.5821174 , 1.17228633, 0.65259416, 0.82037779, 0.45255129,
       0.72201384, 0.0529696 , 1.62802466, 0.6135294 , 0.50261377,
       0.98199621, 0.09353276, 0.7419117 , 0.40334923, 0.25180607,
       1.1029956 , 0.05416612, 0.10189236, 0.64247004, 0.20189167,
       0.10468879, 0.34183855, 0.85207326, 1.36555685, 0.50186728,
       0.80195149, 0.58211389, 0.68175177, 0.09308015, 0.5318232 ,
       0.10266798, 1.18734279, 0.45286437, 0.26419844, 0.13393568,
       1.25691   , 0.50293176, 1.28308047, 0.10360016, 0.52272765,
       0.80196708, 1.06166992, 0.8919839 , 0.32190673, 1.13408093,
       1.00208927, 0.18271975, 0.41251757, 0.68192247, 1.96669111,
       0.52192239, 1.13299863, 0.37337646, 0.65198138, 1.167532  ,
       1.13210155, 1.49688741, 0.13348902, 0.74232435, 0.5821402 ,
       1.0486623 , 0.40219801, 0.68260358, 0.29265626, 1.28386826,
       0.49193344, 0.20359403, 0.73051945, 0.64159534, 0.50215567,
       1.06842983, 1.13169671, 0.53194736, 0.18307647, 1.37571816,
       0.73183673, 0.25202108, 0.29304008, 0.08200937, 1.0586109 ,
       0.82069902, 1.27546396, 1.49379033, 0.02278131, 0.97122604,
       1.11564608, 0.25288028, 0.53314006, 1.62889531, 0.74181255,
       1.36800575])

@pytest.mark.skip
def test_linear_comb_sim_mat():
    pass

def test_MatrixMultiKernelLearning(fingerprints, similarity_matrix, predictions):
    mids, targets = np.transpose(np.array([[fp.mid, fp.y] for fp in fingerprints], dtype = object))
    np.random.seed(0)
    mids_train, mids_test, target_train, target_test = train_test_split(mids, targets)
    train_matrix, test_matrix = similarity_matrix.train_test_split(mids_train, mids_test)
    mmkl = MatrixMultiKernelLearning(kernel_matrices=[train_matrix], 
                                 prediction_matrices=[test_matrix], 
                                 regressor_params={'alpha': 1e-12, 'fit_intercept': True, 'normalize': False})
    mmkl.fit(target_train)
    preds = mmkl.predict()
    assert np.allclose(preds, predictions), "Predictions deviate too much from reference."