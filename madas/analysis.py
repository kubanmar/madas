import warnings
from typing import List
from itertools import combinations

from scipy.special import binom
from nomad_dos_fingerprints import DOSFingerprint, Grid
import numpy as np

from .fingerprint import Fingerprint
from .similarity import SimilarityMatrix
from .utils import tqdm


class MetricSpaceTest():
    """
    Class for fingerprints to test if the fingerprints and their respective similarity measures span a metric space.

    **Usage:**

    ``mst = MetricSpaceTest(fingerprints: List[Fingerprint])``

    ``is_metric = all(mst())``
    """
    
    def __init__(self, 
                 fingerprints: List[Fingerprint] = None, 
                 similarity_matrix: SimilarityMatrix = None, 
                 show_progress: bool = True):
        assert not (similarity_matrix is None and fingerprints is None), "Either fingerprints or similarity matrix need to be provided."   
        if similarity_matrix is None:
            self.similarity_matrix = SimilarityMatrix().calculate(fingerprints, mids = [fp.mid for fp in fingerprints], symmetric=False)
        else:
            self.similarity_matrix = similarity_matrix
        self.fingerprints = fingerprints
        self.show_progress = show_progress
        
    def __call__(self, only = None):
        only = only if only is not None else ["identity", "uniqueness", "symmetry", "triangle inequality"]
        results = [self.call_single(test_name) for test_name in only]
        return results
            
    def call_single(self, test_name : str) -> bool:
        """
        Call a single test.

        **Arguments:**

        test_name: `str`
            Name of the test. Choose from: 'uniqueness', 'identity', 'symmetry', 'triangle inequality'

        **Returns:**

        result: `bool`
            Result of the test.

        **Raises:**

        KeyError: Name of the test is not recogized.
        """
        if test_name == "uniqueness":
            result = self.uniqueness(self.similarity_matrix, self.fingerprints, show_progress = self.show_progress)
        elif test_name == "identity":
            result = self.identity(self.similarity_matrix, show_progress = self.show_progress)
        elif test_name == "symmetry":        
            result = self.symmetry(self.similarity_matrix, show_progress = self.show_progress)
        elif test_name == "triangle inequality":
            result = self.triangle_inequality(self.similarity_matrix, show_progress = self.show_progress)
        else:
            raise KeyError("No test of this name is implemented. Choose from: 'uniqueness', 'identity', 'symmetry', 'triangle inequality'")
        return result
    
    @staticmethod
    def symmetry(similarity_matrix: SimilarityMatrix, show_progress = True):
        """
        Test symmetry property of a similarity matrix. The threshold for being equal is `1e-8`.

        Tests: S(i,j) == S(j,i) for fingerprints i,j and similarity S

        **Arguments:**

        similarity_matrix: `SimilarityMatrix`
            Similarity matrix to test.

        **Keyword arguments:**

        show_progress: `bool`
            Show progress of test.

            default: `True`

        **Returns:**

        result: `bool`
            Result of the test: `True`: matrix is symmetric, `False` else
        """
        if show_progress:
            print('Checking symmetry property!')
        matrix = similarity_matrix.matrix
        combs = binom(len(similarity_matrix), 2)
        symmetries = [abs(matrix[idx1][idx2] - matrix[idx2][idx1]) < 1e-8 for (idx1, idx2) in tqdm(combinations(range(len(similarity_matrix)), 2), ncols = 500, disable = not show_progress, total = combs)]
        return all(symmetries)

    
    @staticmethod
    def identity(similarity_matrix, show_progress = True):
        """
        Test identity property of a similarity matrix.

        Tests: S(i,i) == 1 for all fingerprints i and similarity S

        **Arguments:**

        similarity_matrix: `SimilarityMatrix`
            Similarity matrix to test.

        **Keyword arguments:**

        show_progress: `bool`
            Show progress of test.

            default: `True`

        **Returns:**

        result: `bool`
            Result of the test: `True`: identical fingerprints have S == 1, `False` else
        """
        if show_progress:
            print('Checking self-similarity property!')
        matrix = similarity_matrix.matrix
        identities = [matrix[idx][idx] == 1 for idx in tqdm(range(len(similarity_matrix)), ncols = 500, disable = not show_progress)]
        return all(identities)

    @staticmethod
    def uniqueness(similarity_matrix, fingerprints, show_progress = True):
        """
        Test uniqueness property of a similarity matrix.

        Tests: i != j -> S(i,j) != 1 for fingerprints i,j and similarity S

        **Arguments:**

        similarity_matrix: `SimilarityMatrix`
            Similarity matrix to test.

        fingerprints: `List[Fingerprint]`
            List of fingerprints to verfy if two fingerprints are equal.

        **Keyword arguments:**

        show_progress: `bool`
            Show progress of test.

            default: `True`

        **Returns:**

        result: `bool`
            Result of the test: `True`: fingerprints are unique, `False` else
        """
        if fingerprints is None:
            raise ValueError("This test requires to give fingerprints to the MetricSpaceTest.")
        if show_progress:
            print('Checking non-identic --> S < 1 property!')
        matrix = similarity_matrix.matrix
        combs = binom(len(similarity_matrix), 2)
        for (idx1, idx2) in tqdm(combinations(range(len(similarity_matrix)), 2), ncols = 500, disable = not show_progress, total = combs):
            if idx1==idx2:
                assert False
            if matrix[idx1][idx2] == 1:
                if not fingerprints[idx1] == fingerprints[idx2]:
                    return False
        return True

    @staticmethod
    def triangle_inequality(similarity_matrix, show_progress = True):
        """
        Test triangle inequality property of a similarity matrix. The threshold for being equal is `1e-8`.

        Tests: S(i,k) + S(j,k) <= S(j,i) + 1 for fingerprints i,j,k and similarity S

        **Arguments:**

        similarity_matrix: `SimilarityMatrix`
            Similarity matrix to test.

        **Keyword arguments:**

        show_progress: `bool`
            Show progress of test.

            default: `True`

        **Returns:**

        result: `bool`
            Result of the test: `True`: matrix obays triangle inequality, `False` else
        """
        if show_progress:
            print("Checking triangle inequality!")
        matrix = similarity_matrix.matrix
        combs = binom(len(similarity_matrix), 3)
        for (idx1, idx2, idx3) in tqdm(combinations(range(len(similarity_matrix)), 3), ncols = 500, disable = not show_progress, total = combs):
            truth = matrix[idx1][idx3] +  matrix[idx2][idx3] <=  matrix[idx1][idx2] + 1
            if not truth:
                return False
        return True

class StrideSpectrumComparison():
    """
    Compare two spectra by "striding" small spectum-fingerprint feature regions along the x-axis.
    
    **Usage**:
        For spectra 1 and 2 with x-values x1 and x2 and y-values y1 and y2:

        ssc = StrideSpectrumComparison()
        stride_list, similarities = ssc(x1, y1, x2, y2)


    **Returns:**
    
    stride_list: `list` of `float`
        List of strides that were used for calculating the similarities

    similarities:  `list` of `float`
        Similiarity of the spectra in a window centered at the respective stride
    """
        
    def __init__(self,
                 grid_id: str = 'nonuniform:-2:0.05:1.05:0.3:8:0.5:-1:1:1024',
                 cutoff: List[float] = [-1,1],
                 eref_stepsize: float = 0.1,
                 show_progress: bool = True,
                 **kwargs):
        self._grid = Grid.create(grid_id=grid_id)
        self.grid.cutoff_min, self.grid.cutoff_max = cutoff
        self.stepsize = eref_stepsize
        self.show_progress = show_progress
        self.kwargs = kwargs
    
    @property
    def grid(self):
        return self._grid
    
    def __call__(self, 
                 x1: list, 
                 y1: list, 
                 x2: list, 
                 y2: list) -> List[List[float]]:
        # determine max range of strides
        min_x = max([min(x1), min(x2)])
        max_x = min([max(x1), max(x2)])
        
        offsets = np.arange(min_x, max_x, self.stepsize)
        
        sims = []
        for offset in tqdm(offsets, disable=not self.show_progress):
            self.grid.e_ref = offset
            fp1 = DOSFingerprint(**self.kwargs).calculate(x1, y1, convert_data=None, grid_id = self.grid.get_grid_id())
            fp2 = DOSFingerprint(**self.kwargs).calculate(x2, y2, convert_data=None, grid_id = self.grid.get_grid_id())
            if any(map(lambda x: x.overflow>0, [fp1,fp2])):
                warnings.warn("Fingerprint overflows. Consider increasing Grid().n_pix.")
            if fp1.filling_factor == 0 and fp2.filling_factor == 0:
                sims.append(1)
            else:
                sims.append(fp1.get_similarity(fp2))
        return offsets.tolist(), sims
    