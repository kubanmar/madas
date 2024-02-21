from typing import List, Tuple, Callable
import numpy as np
import pandas as pd
import os
import json
import multiprocessing
from functools import partial

from .utils import report_error, BatchIterator
from .fingerprint import Fingerprint

class SimilarityMatrix():
    """
    A matrix, that stores all similarites between materials and the corresponding material identifier.    
    """

    def __init__(self, 
                 matrix: List[list] = [], 
                 mids: List[str] = None, 
                 dtype: type = np.float64):
        '''
        **Keyword arguments:**

        matrix: `np.ndarray` or `List[list]`; 
            Initialize matrix with precomputed similarities.
            
            default: `[]`

        mids: `List[str]`
            Material ids of precomputed similarities.
        
            default: `None` 
        '''
        self._dtype = dtype
        self._dataframe = None
        self.set_matrix(matrix)
        self.set_mids(mids)
        self.fp_type = None
        self.fp_name = None
        self._iter_index = 0

    @property
    def matrix(self) -> np.ndarray:
        """
        Matrix values.
        """
        return self._dataframe.values

    @property
    def mids(self) -> np.ndarray:
        """
        Material identifier, corresponding to rows and columns of the matrix.
        """
        return np.array(self._dataframe.index, dtype=str)

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return `pandas.DataFrame` object containing similarities as values and material ids as indices.
        """
        if not hasattr(self, "_dataframe"):
            raise AttributeError("Matrix is not calculated! Can not return data.")
        return self._dataframe.copy()

    def get_metadata(self) -> dict:
        """
        Get dictionary of fingerprint type and name.
        """
        return {"fp_type" : self.fp_type, "fp_name" : self.fp_name}

    def set_metadata(self, metadata: dict) -> None:
        """
        Set fingerprint type and name from dictionary. Ignores other keys than `fp_type` and `fp_name`.
        """
        if "fp_type" in metadata.keys():
            self.fp_type = metadata["fp_type"]
        if "fp_name" in metadata.keys():
            self.fp_name = metadata["fp_name"]

    def set_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the values of the similarity matrix.
        
        **Arguments:**

        matrix: `np.ndarray`
            array of values for the matrix

            Can be a square matrix or an upper triangular matrix.
        """

        if len(matrix) == 0:
            self._dataframe = pd.DataFrame()
            return
        if len(np.unique([len(row) for row in matrix])) != 1:
            matrix = self._square_from_triangular_matrix(matrix)
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        if not isinstance(self._dataframe, pd.DataFrame) or matrix.shape != self._dataframe.shape:
            self._dataframe = pd.DataFrame(data = matrix)
        else:
            self._dataframe.values[:] = np.array(matrix, dtype = self._dtype)

    def set_mids(self, mids: list) -> None:
        """
        Set the matrix row and column identifiers to given values.

        **Arguments:**

        mids: `List[str]`
            List of identifiers to be set
        """
        if isinstance(mids, np.ndarray):
            mids = mids.tolist()
        if mids is None or len(mids) == 0:
            return
        self._dataframe.rename(columns=dict(zip(self._dataframe.columns, mids)), 
                               index=dict(zip(self._dataframe.index, mids)), 
                               inplace=True)

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """
        Set the dataframe of the matrix.

        **Arguments:**

        dataframe: `pandas.DataFrame`
            Dataframe that should contain similarities as values and material identifier as indices.
        """
        self._dataframe = dataframe

    def calculate(self, 
                  fingerprints: List[Fingerprint], 
                  mids: List[str] = None, 
                  multiprocess:  int | None = -1, 
                  symmetric: bool = True):
        """
        Calculate similarity matrix.

        **Arguments:**

        fingerprints: `List[Fingerprint]`
            Fingerprints to calculate similarities.

        **Keyword arguments:**

        mids: `List[str]`
            Material ids of fingerprints.

            default: `None` -> mids are extracted from `fingerprints`

        multiprocess: `int`
            Calculate similarities on available processors.
            Set to `-1` to use all available processors.
            Set to `None` for serial execution
            Set to any positive integer to use as may processes.

            default: `-1`

        symmetric: `bool` 
            Reduce computation cost by calculating only unique half of symmetric matrix
            
            default: `True`

        **Returns:**

        self: `SimilarityMatrix`
            Populated similarity matrix.
        """
        matrix = []
        mids = mids if mids is not None else [fp.mid for fp in fingerprints]
        try:
            self.fp_type = fingerprints[0].fp_type
        except Exception as e:
            report_error(None, f"Unable to set Fingerprint type because of: {str(e)}")
            self.fp_type = "Unknown"
        try:
            self.fp_name = fingerprints[0].name
        except Exception as e:
            report_error(None, f"Unable to set Fingerprint name because of: {str(e)}")
            self.fp_name = "Unknown"
        if multiprocess is not None:
            if multiprocess <= 0:
                if multiprocess == -1:
                    n_processes = multiprocessing.cpu_count()
                else:
                    raise ValueError("Invalid input for `multiprocess`. Choose from positive integer, `-1` or `None`.")
            else:
                n_processes = multiprocess
            if symmetric:
                calc_function = self._get_similarities_list_index
            else:
                calc_function = partial(self._get_similarities_list_index, square_matrix = True)
            with multiprocessing.Pool(n_processes) as p:
                matrix = p.map(calc_function, [(idx, fingerprints) for idx in range(len(fingerprints))]) # TODO refactor, as unneccessary data is copied
                if not symmetric:
                    matrix = np.array(matrix)
        else:
            for idx, fp in enumerate(fingerprints):
                if symmetric:
                    matrix_row = []
                    for fp2 in fingerprints[idx:]:
                        matrix_row.append(fp.get_similarity(fp2))
                else:
                    matrix_row = fp.get_similarities(fingerprints)
                matrix.append(np.array(matrix_row))
        self.set_matrix(matrix)
        self.set_mids(mids)
        return self
    
    def get_sub_matrix(self, 
                       mid_list: List[str], 
                       copy: bool = True) -> object:
        """
        Get sub matrix of all elements in mid_list sorted by occurrence in mid list.

        **Arguments:**

        mid_list: `List[str]`
            List of (unique) mids of materials to include in sub matrix

        **Keyword arguments:**

        copy: `bool`
            Return a new similarity matrix. If set to False, apply changes to `self`.
            
            default: `True` 

        **Returns:**

        `SimilarityMatrix()` object of sub matrix if `copy == True`

        `self` restricted to, and sorted by, elements in `mid_list`
        """
        assert len(set(mid_list)) == len(mid_list), "mid mist contains no-unique entries"
        frame = self.dataframe
        frame = frame[mid_list]
        frame = frame.transpose()
        frame = frame[mid_list]
        new_matrix = frame.values
        if copy:
            simat = SimilarityMatrix(matrix = new_matrix, mids = mid_list, dtype = self._dtype)
            simat.fp_name = self.fp_name
            simat.fp_type = self.fp_type
            return simat
        else:
            self.set_matrix(new_matrix)
            self.set_mids(mid_list)
            return self

    def get_overlap_matrix(self, 
                           column_mids: List[str], 
                           row_mids: List[str]) -> object:
        """
        Get `OverlapSimilarityMatrix()` from matrix. This new matrix contains (mostly) off-diagonal elements of the original matrix.

        **Arguments:**

        row_mids: `List[str]` 
            mids associated with the rows of the new matrix

        column_mids: `List[str]`
            mids associated with the columns of the matrix

        **Returns:**

        overlap_matrix: `OverlapSimilarityMatrix`
            New matrix with rows and columns corresponding to materials with identifier `row_mids` and `column_mids`
        """
        frame = self.dataframe
        frame = frame[column_mids]
        frame = frame.transpose()
        frame = frame[row_mids]
        frame = frame.transpose()
        simat =  OverlapSimilarityMatrix(matrix = frame.values, row_mids = row_mids, column_mids = column_mids)
        simat.fp_type = self.fp_type
        simat.fp_name = self.fp_name
        return simat

    def lookup_similarity(self, 
                          fp1: Fingerprint, 
                          fp2: Fingerprint):
        """
        Return similarity between two fingerprints from the matrix.

        The expected usecase for this is to pass this function `set_similarity_function` of a `Fingerprint` object.

        fp1, fp2: `Fingerprint`
            Fingerprint objects to retrieve similarity

        **Returns:**

        similarity: `float`
            Similarity between materials with mids `fp1.mid` and `fp2.mid`

        **Raises:**

        KeyError: Similarity matrix does not contain an entry with these mids.
        """
        return self.get_entry(fp1.mid, fp2.mid)

    def train_test_split(self, 
                         train_mids: List[str], 
                         test_mids: List[str]) -> set:
        """
        Split similarity matrix into a (symmetric) train matrix and a (off-diagonal) test matrix.

        **Arguments:**

        train_mids: `List[str]`
            mids that identify materials of the training set

        test_mids: `List[str]` 
            mids that identify materials in the test set

        **Returns:**

        train_matrix, test_matrix: `Set[SimilarityMatrix, OverlapSimilarityMatrix]`
        """
        return self.get_sub_matrix(train_mids), self.get_overlap_matrix(train_mids, test_mids) 

    def align(self, matrices): #TODO TEST PROPERLY
        """
        Align the materials in this matrix and all provided matrices.

        **Arguments:**

        matrices: `SimilarityMatrix` or `List[SimilarityMatrix]`
            Matrix or list of matrices to align
        
        **Returns:**

        `None`

        WARNING! Entries in both matrices will be altered, i.e. unique entries in each matrix will be dropped.
        """
        if isinstance(matrices, (list, np.ndarray)):
            #shared_mids = [mid for mid in self.mids if np.array([mid in matrix.mids for matrix in matrices]).all()]
            shared_mids = set.intersection(*[set(simat.mids) for simat in matrices])
            shared_mids = list(shared_mids.intersection(set(self.mids)))
            shared_mids.sort()
            for matrix in matrices:
                matrix.get_sub_matrix(shared_mids, copy = False)
        else:
            shared_mids = [mid for mid in self.mids if mid in matrices.mids]
            matrices.get_sub_matrix(shared_mids, copy = False)
        self.get_sub_matrix(shared_mids, copy = False)

    def get_entry(self, 
                  mid1: str, 
                  mid2: str) -> np.float64:
        """
        Get any entry of the matrix.
        
        **Arguments:**

        mid1, mid2: `str`
            material ids of the requested material
        
        **Returns:**
        
        similarity: `float`
            Similarity between both materials

        **Raises:**

        KeyError: No entry for material with given mid.
        """
        return self._dataframe[mid1][mid2]

    def get_symmetric_matrix(self):
        """
        Deprecated! Use `SimilarityMatrix().matrix` property!
        Get square matrix form.
        
        **Returns:**

        square_matrix: `np.ndarray`
            matrix of similarities
        """
        report_error(None, "Deprecation warning: use SimilarityMatrix.matrix instead of get_symmetric matrix")
        return self.matrix

    def get_row(self, mid: str) -> np.ndarray:
        """
        Get a row of the matrix, by mid or index.

        **Arguments:**

        mid: `str` or `int`
            Material id oder matrix index of requested matrix row

        **Returns:**

        row: `np.ndarray`
            Similarities of material with given mid to all other materials in the matrix.

        **Raises:**

        KeyError: No entry with given mid.

        IndexError: Matrix index out of range.
        """
        if isinstance(mid, int):
            mid = self.mids[mid]
        return np.array(self._dataframe[mid])

    def get_entries(self):
        import warnings
        warnings.warn("Function is deprecated due to ambiguous name. Use `get_unique_entries`.", DeprecationWarning)
        return self.get_unique_entries()

    def get_unique_entries(self) -> np.ndarray:
        """
        Return all enries of the upper triangular matrix.

        **Returns:**

        entries: `np.ndarray`
            list of all unique entries of the matrix
        """
        entries = np.concatenate(self._triangular_from_square_matrix(self.matrix)) #this version should be a lot faster
        return np.array(entries, dtype = np.float64)

    def get_k_most_similar(self, 
                           ref_mid: str, 
                           k: int = 10, 
                           remove_self: bool = True) -> dict:
        """
        Get the k most similar materials and the respective similarities for a material.

        WARNING! Accurate results can only be obtained for symmetric similarity matrices.
        For asymmetric similarity measures, the assignment may be ambiguous.

        **Arguments:**

        ref_mid: `str`
            Material id of the reference material

        **Keyword arguments:**

        k: `int`
            Number of most similar materials to return

            default: 10

        remove_self: `bool`
            Remove the requested material from the results list

            default: `True`

        **Returns:**

        dict: {<1st_nearest_mid>:<similarity>, 2nd_nearest_mid>:<similiarty>, ...}
        """
        row = [(mid,entry) for mid, entry in zip(self.mids, self.get_row(ref_mid))]
        row.sort(reverse = True, key = lambda x: x[1])
        if remove_self:
            for idx, item in enumerate(row):
                if item[0] == ref_mid:
                    row.pop(idx)
                    break
        return {mid: entry for mid, entry in row[:k]}

    def save(self, 
             filename: str = 'similarity_matrix.npy', 
             filepath: str = '.', 
             overwrite: bool = False) -> None:
        """
        Save SimilarityMatrix to numpy binary file.

        **Keyword arguments:**

        filename: `str`
            Name of the file

            default: 'similarity_matrix.npy'

        filepath: `str`
            Relative path to created files
            
            default: '.' 

        overwrite: `bool`
            Overwrite matrix file if it exists.

            default: `False`
        """
        matrix_path = os.path.join(filepath, filename)
        if os.path.exists(matrix_path) and not overwrite:
            raise FileExistsError(f"Matrix at {matrix_path} exists. To overwrite set `overwrite = True`.")
        np.save(matrix_path, np.array([self.mids, self.get_metadata(), self.matrix], dtype = object))

    @staticmethod
    def load(filename: str = 'similarity_matrix.npy', filepath: str = '.') -> object:
        """
        Static method. Load SimilarityMatrix from file. 
        If the target file is created from an OverlapSimilarityMatrix object, return OverlapSimilarityMatrix object

        Warning: This methods loads a pickled file. Only load files of known origin.

        **Keyword arguments:**

        filename: `str` 
            Name of a saved similarity matrix file
            
            default: 'similarity_matrix.npy'

        filepath: `str`
            Relative path to SimilarityMatrix files
            
            default: '.'

        **Returns:**
        
        similarity_matrix: `SimilarityMatrix` or `OverlapSimilarityMatrix`

        **Raises:**

        IndexError: Wrong format of data in file. Can not load.
        """
        matrix_path = os.path.join(filepath, filename)
        matrix = np.load(matrix_path, allow_pickle = True)
        if len(matrix) == 3:
            mids, metadata, matrix = matrix
        elif len(matrix) == 2:
            mids, matrix = matrix
            metadata = {}
        else:
            raise IndexError("Can not load matrix: wrong format of data")
        if len(mids) == 2:
            self = OverlapSimilarityMatrix()
            row_mids = mids[0]
            column_mids = mids[1]
            self.set_matrix(matrix)
            self.set_mids(row_mids, column_mids)
        else:
            self = SimilarityMatrix()
            self.set_matrix(matrix)
            self.set_mids(mids)
        self.set_metadata(metadata)
        return self

    def get_matching_matrices(self, second_matrix):
        """
        Match matrices such that they contain the same materials in the same order.
        
        **Arguments:**

        second_matrix: `SimilarityMatrix`
            Matrix to match materials

        **Returns:**

            new_self, new_matrix: `tuple(SimilarityMatrix)`
                Matching similarity matrices
        """
        shared_mids = [mid for mid in self.mids if mid in second_matrix.mids]
        new_self = self.get_sub_matrix(shared_mids)
        new_matrix = second_matrix.get_sub_matrix(shared_mids)
        return new_self, new_matrix

    def get_cleared_matrix(self, leave_out_mids, copy = True):
        """
        Return a matrix where all materials with mids specified in `leave_out_mids` are removed from the matrix.

        **Arguments:**

        leave_out_mids: `List[str]`
            mids of materials to leave out of the matrix
        
        **Keyword arguments:**

        copy: `bool`
            Return copy of SimilarityMatrix(); apply changes to self, if False
            
            default: `True` 

        **Returns:**

        cleared_matrix: `SimilarityMatrix`
            Copy of similarity matrix if copy is True, `self` otherwise without `leave_out_mids`
        """
        new_mids = [mid for mid in self.mids if mid not in leave_out_mids]
        new_matrix = self.get_sub_matrix(new_mids, copy = copy)
        return new_matrix

    def _get_similarities_list_index(self, idx__list, square_matrix = False): # TODO refactor
        idx, fp_list = idx__list
        if square_matrix:
            sims = np.array(fp_list[idx].get_similarities(fp_list))
        else:
            sims = fp_list[idx].get_similarities(fp_list[idx:])
        return np.array(sims)

    def _triangular_from_square_matrix(self, matrix):
        """
        Get the upper triangular part of a square matrix.
        Args:
            * matrix: list of lists, np.ndarray; symmetric square matrix
        Returns:
            * triangular_matrix: np.ndarray; triangular form of input matrix
        """
        triangular_matrix = []
        for index, row in enumerate(matrix):
            triangular_matrix.append(np.array(row[index:], dtype = self._dtype))
        return np.array(triangular_matrix, dtype = object)

    def _square_from_triangular_matrix(self, matrix):
        symmetric_matrix = np.zeros((len(matrix[0]),len(matrix[0])))
        for idx, row in enumerate(matrix):
            for jdx, entry in enumerate(row):
                symmetric_matrix[idx][jdx + idx] = entry
                symmetric_matrix[jdx + idx][idx] = entry
        return np.array(symmetric_matrix, dtype = self._dtype)

    def __len__(self):
        return len(self._dataframe)

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= len(self):
            self._iter_index = 0
            raise StopIteration
        else:
            self._iter_index += 1
            return self.get_row(self._iter_index - 1)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self.get_row(key)
            except KeyError:
                raise KeyError("No entry with mid = " + key + '.')
        elif isinstance(key, int):
            try:
                return self._dataframe.values[key]
            except KeyError:
                raise KeyError('No entry with id = ' + str(key) + '.')
        else:
            raise KeyError('Key can not be interpreted as matrix key.')

    def __add__(self, simat):
        new_matrix = SimilarityMatrix()
        new_matrix.set_dataframe(self.dataframe + simat.dataframe)
        return new_matrix

    def __sub__(self, simat):
        new_matrix = SimilarityMatrix()
        new_matrix.set_dataframe(self.dataframe - simat.dataframe)
        return new_matrix

    def __mul__(self, simat):
        new_matrix = SimilarityMatrix()
        if isinstance(simat, SimilarityMatrix):
            new_matrix.set_dataframe(self.dataframe * simat.dataframe)
        else:
            new_matrix.set_dataframe(self.dataframe * simat)
        return new_matrix

    def __eq__(self, simat):
        try:
            return ((self._dataframe == simat._dataframe).all()).all()
        except ValueError:
            sorted_self = self.get_sub_matrix(simat.mids)
            return ((sorted_self._dataframe == simat._dataframe).all()).all()
            
    def __repr__(self) -> str:
        return f"SimilarityMatrix({self.fp_type}, {self.fp_name}, {len(self)})"

class OverlapSimilarityMatrix(SimilarityMatrix):
    """
    A SimilarityMatrix that is used to store similarities between different sets of fingerprints.
    """

    def __init__(self, 
                 matrix: List[list] = [], 
                 row_mids: list = [], 
                 column_mids: list = [], 
                 dtype = np.float64):
        """
        **Keyword arguments:**

        matrix: `List[list]`
            Precomputed matrix values

            default: `[]`

        row_mids: `List[str]`
            Material ids corresponding to the rows of the matrix

            default: `[]`

        column_mids: `List[str]`
            Material ids corresponding to the columns of the matrix

            default: `[]`

        dtype: `type`
            Data type for storing matrix values

            default: numpy.float64``
        """
        self._dtype = dtype
        self._dataframe = None
        self.set_matrix(matrix)
        self.set_mids(row_mids, column_mids)
        self.fp_type = None
        self.fp_name = None
        self._iter_index = 0

    @property
    def mids(self) -> Tuple[List[str], List[str]]:
        """
        Get mids corresponding to matrix row and columns.
        """
        return self.row_mids, self.column_mids

    @property
    def row_mids(self) -> np.ndarray:
        """
        Get mids corresponding to matrix rows.
        """
        return np.array(self._dataframe.index, dtype = str)

    @property
    def column_mids(self) -> np.ndarray:
        """
        Get mids corresponding to matrix columns.
        """
        return np.array(self._dataframe.columns, dtype = str)

    def set_mids(self, 
                 row_mids: List[str], 
                 column_mids: List[str]) -> None:
        """
        Set the mids for rows and columns of the matrix.
        """
        if isinstance(row_mids, np.ndarray):
            row_mids = row_mids.tolist()
        if isinstance(column_mids, np.ndarray):
            column_mids = column_mids.tolist()
        if column_mids is None or len(column_mids) == 0:
            return
        if row_mids is None or len(row_mids) == 0:
            return        
        self._dataframe.rename(columns=dict(zip(self._dataframe.columns, column_mids)), 
                        index=dict(zip(self._dataframe.index, row_mids)), 
                        inplace=True)

    def set_matrix(self, matrix: np.ndarray) -> None:
        """
        Set matrix values

        If the values do not fit the shape of the original matrix, values and mids will be overwritten.
        
        **Arguments:**

        matrix: `np.ndarray`
            Matrix to be set
        """
        if self._dataframe is None:
            self._dataframe = pd.DataFrame(matrix)
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=self._dtype)
        if not isinstance(self._dataframe, pd.DataFrame) or matrix.shape != self._dataframe.shape:
            self._dataframe = pd.DataFrame(data = matrix)
        else:
            self._dataframe.values[:] = np.array(matrix, dtype = self._dtype)

    def calculate(self, 
                  reference_fingerprints: List[Fingerprint], 
                  fingerprints: List[Fingerprint], 
                  mids: List[str] = [], 
                  reference_mids = []) -> object:
        """
        Calculate similarity of fingerprints to given reference fingerprints.

        **Arguments:**

        reference_fingerprints: `List[Fingerprint]` 
            Fingerprints that correspond to columns of the matrix
        
        fingerprints: `List[Fingerprint]`
            Fingerprints that correspond to rows of the matrix
        
        **Keyword arguments:**

        mids: `List[str]` 
            Material identifier for the rows of the matrix

            default: `None`

        reference_mids: `List[str]`
            Material identifier for the columns of the matrix
        
            default: `None`

        If possible, mids and reference_mids are taken directly from the Fingerprint objects.

        **Returns:**

        self: `OverlapSimilarityMatrix`
            Calculated matrix object
        """
        try:
            with multiprocessing.Pool() as p:
                matrix = p.map(self._calc_simimilarities_multiprocess, [(fingerprints, reference_fingerprints, idx) for idx in range(len(fingerprints))])
        except TypeError as e: #Fingerprints are not parallelizable
            report_error(None, "Parallel computing not possible, calculating matrix serial.")
            report_error(None, str(e))
            matrix = []
            for fingerprint in fingerprints:
                matrix.append(fingerprint.get_similarities(reference_fingerprints))
        self.set_matrix(np.array([np.array(row) for row in matrix]))
        try:
            row_mids = [fp.mid for fp in fingerprints]
            column_mids = [fp.mid for fp in reference_fingerprints]
            self.set_mids(row_mids, column_mids)
            self.fp_type = fingerprints[0].fp_type
            self.fp_name = fingerprints[0].name
        except AttributeError as e:
            report_error(None, "An error occured, but was caught: ")
            report_error(None, str(e))
            self.set_mids(mids, reference_mids)
        return self

    def get_entries(self) -> np.ndarray:
        """
        Get all entries of the matrix as a list.

        **Returns:**

        entries: `numpy.ndarray`
            All entries of the matrix in a (N*M, 1)-dim list
        """
        return np.array(self.matrix).flatten()

    def get_row(self, mid: str | int) -> np.ndarray:
        """
        Get a row of the matrix.

        **Arguments:**

        mid: `str` or `int`
            Material id or matrix index of the requested row

        **Returns:**

        matrix_row: `numpy.ndarray`
            Row of the matrix
        """
        if isinstance(mid, int):
            mid = self.row_mids[mid]
        return np.array(self.dataframe[mid:mid])[0]

    def get_column(self, mid:  str | int) -> np.ndarray:
        """
        Get a column of the matrix.

        **Arguments:**

        mid: `str` or `int`
            Material id or matrix index of the requested column

        **Returns:**

        matrix_column: `numpy.ndarray`
            Column of the matrix
        """
        if isinstance(mid, int):
            mid = self.column_mids[mid]
        return np.array(self.dataframe[mid])

    def get_symmetric_matrix(self):
        """
        Not implemented for OverlapSimilarityMatrix().

        This function has no meaning for overlap similarity matrices.

        **Raises:**

        NotImplementedError: upon function call
        """
        raise NotImplementedError("OverlapSimilarityMatrix() does not support symmetric matrices.")

    def get_sub_matrix(self, 
                       row_mid_list: List[str], 
                       column_mid_list: List[str], 
                       copy: bool = True) -> object:
        """
        Get sub matrix.

        **Arguments:**

        column_mid_list: `List[str]` 
            list of mids of materials in matrix column to include in sub matrix
            
        row_mid_list: `List[str]` 
            list of mids of materials in matrix row to include in sub matrix
        
        **Keyword arguments:**

        copy: `bool`
            Return a new similarity matrix. If set to False, apply changes to ``self``.

            default: `True`

        **Returns:**

        matrix: `OverlapSimilarityMatrix` 
            if `copy == True`
            
        self: `OverlapSimilarityMatrix` 
            if `copy == False`, self, restricted to, and sorted by, elements in `mid_list`
        """
        frame = self.dataframe
        frame = frame[column_mid_list]
        frame = frame.transpose()
        frame = frame[row_mid_list]
        frame = frame.transpose()
        if copy:
            new_matix = OverlapSimilarityMatrix()
            new_matix.set_dataframe(frame)
            return new_matix #OverlapSimilarityMatrix(matrix = frame.values, column_mids = column_mid_list, row_mids = row_mid_list)
        else:
            self.set_matrix(frame.values)
            self.set_mids(row_mid_list, column_mid_list)
            return self

    def save(self, 
             filename: str = 'overlap_similarity_matrix.npy', 
             filepath: str = '.', 
             overwrite: bool = False) -> None:
        """
        Save SimilarityMatrix to numpy binary file(s).

        **Keyword arguments:**

        filename: `str`
            name of the matrix file

            default: 'similarity_matrix.npy'
        
        data_path: `str`
            relative path to created files

            default: '.' 
        
        overwrite: `bool`
            Overwrite matrix file if it exists.

            default: `False`
        """
        matrix_path = os.path.join(filepath, filename)
        if os.path.exists(matrix_path) and not overwrite:
            raise FileExistsError(f"Matrix at {matrix_path} exists. To overwrite set `overwrite = True`.")
        np.save(matrix_path, np.array([[self.row_mids, self.column_mids], self.matrix], dtype=object))

    def get_entry(self, 
                  row_mid: str,
                  column_mid: str) -> np.float64:
        """
        Get a single entry of the matrix.

        **Arguments:**
    
        row_mid: `str`
            material id of the material in the row of the matrix
        
        column_mid: `str`
            material id of the material in the column of the matrix
        
        **Returns:**
        
        similarity: `numpy.float64`
            similarity between both materials
        """
        return self._dataframe[column_mid][row_mid]

    def transpose(self) -> None:
        """
        Exchange rows and columns of matrix.
        """
        self._dataframe = self.dataframe.transpose()

    def _calc_simimilarities_multiprocess(self, tfp__kfp__idx):
        target_fingerprints, kernel_fingerprints, idx = tfp__kfp__idx
        fps = target_fingerprints[idx].get_similarities(kernel_fingerprints)
        return fps

    def align(self, matrices: List[SimilarityMatrix] | SimilarityMatrix) -> None:
        """
        Align the materials in this matrix and all provided matrices.

        **Arguments:**

        matrices: `OverlapSimilarityMatrix` or `List[OverlapSimilarityMatrix]`
            matrix object(s) to align with
        
        **Returns:**
        
        `None`

        Warning! Entries in both matrices will be altered, i.e. unique entries in each matrix will be dropped.
        """
        if isinstance(matrices, (list, np.ndarray)):
            shared_row_mids = [mid for mid in self.row_mids if np.array([mid in matrix.row_mids for matrix in matrices]).all()]
            shared_column_mids = [mid for mid in self.column_mids if np.array([mid in matrix.column_mids for matrix in matrices]).all()]
            for matrix in matrices:
                matrix.get_sub_matrix(shared_row_mids, shared_column_mids, copy = False) 
        else:
            shared_row_mids = [mid for mid in self.row_mids if mid in matrices.row_mids]
            shared_column_mids = [mid for mid in self.column_mids if mid in matrices.column_mids]
            matrices.get_sub_matrix(shared_row_mids, shared_column_mids, copy = False)
        self.get_sub_matrix(shared_row_mids, shared_column_mids, copy = False)

    def _check_matrix_alignment(self, simat):
        if (self.column_mids == simat.column_mids).all() and (self.row_mids == simat.row_mids).all():
            return True
        return False

    def __add__(self, simat):
        if isinstance(simat, OverlapSimilarityMatrix):
            if not self._check_matrix_alignment(simat):
                raise IndexError("Matrices not aligned. Use ``align()`` method.")
            matrix = self.matrix + simat.matrix
        elif np.isscalar(simat):
            matrix = simat + self.matrix
        else:
            raise TypeError("Unknown variable type.")
        return OverlapSimilarityMatrix(matrix, row_mids = self.row_mids, column_mids = self.column_mids)

    def __sub__(self, simat):
        if isinstance(simat, OverlapSimilarityMatrix):
            if not self._check_matrix_alignment(simat):
                raise IndexError("Matrices not aligned. Use ``align()`` method.")
            matrix = self.matrix - simat.matrix
        elif np.isscalar(simat):
            matrix = simat - self.matrix
        else:
            raise TypeError("Unknown variable type.")
        return OverlapSimilarityMatrix(matrix, row_mids = self.row_mids, column_mids = self.column_mids)

    def __mul__(self, simat):
        if isinstance(simat, OverlapSimilarityMatrix):
            if not self._check_matrix_alignment(simat):
                raise IndexError("Matrices not aligned. Use ``align()`` method.")
            matrix = self.matrix * simat.matrix
        elif np.isscalar(simat):
            matrix = simat * self.matrix
        else:
            raise TypeError("Unknown variable type.")
        return OverlapSimilarityMatrix(matrix, row_mids = self.row_mids, column_mids = self.column_mids)

    def __eq__(self, simat):
        try:
            return ((self._dataframe == simat._dataframe).all()).all()
        except ValueError:
            sorted_self = self.get_sub_matrix(simat.row_mids, simat.column_mids)
            return ((sorted_self._dataframe == simat._dataframe).all()).all()            

class BatchedSimilarityMatrix():
    """
    A similarity matrix for parallel computation distributed over different tasks.

    The calculation of pairwise similarities between fingerprints can be distributed over
    different processors. To do so, the matrix is split into different independent 
    sub-matrices (batches), which are calculated separately.

    Each `BatchedSimilarityMatrix` stores all required data and metadata in a single folder.

    To reduce memory consumption, the fingerprints are input via a separate file containing 
    serialized `Fingerprint` objects. This file can be generated via, _e.g._:

    .. code-block:: python

        import json

        long_fingerprint_list_data = [fp.serialize() for fp in long_fingerprint_list]

        with open('path/to/long/fingerprint/file', 'w') as fingerprint_file:
            json.dump(long_fingerprint_list_data, fingerprint_file)

    In a next step, a `BatchedSimilarityMatrix` object can be used to split this large file
    into separate batches for computation of the similarity matrix.

    .. code-block:: python

        batched_similarity_matrix.fingerprint_file_batches('long_fingerprint_file_name',
                                                           'path/to/long/fingerprint/file')

    Now all data is prepared and the matrix can be calculated:

    .. code-block:: python
    
        batched_similarity_matrix.calculate(similarity_function)

    Note that because the fingerprints were serialized before, the similarity function must be 
    specified for calculating similarities.

        **Keyword arguments:**

        root_path: `str`
            Path where the data folder shall be created.

        default: "."

        matrix_folder_name: `str`
            Name of the similarity matrix data folder.

            This name should be unique and descriptive!

            default: "batched_similarity_matrix"

        fingerprint_files_name: `str`
            Base name of fingerprint files to be generated.

        load_from_file: `bool`
            Load metadata from file and ignore further keyword arguments.

            default: `True`

        batch_size: `int`
            Maximal number of fingerprints in a batch. 
            A similarity matrix has a maximum of `batch_size` ** 2 entries.

            default:  `10000`

        n_tasks: `int`
            Total number of tasks that are used to compute the similarity matrix.

            default: `1`

        task_id: `int`
            Id of the current task. 
            This number specifies which batches of the similarity matrix are calculated.

            default: `0`

        size: `int`
            Total number of fingerprints. 
            This number is automatically set if the function `fingerprint_file_batches` is called.

            default: `0`

        symmetric: `bool`
            Assume that similarity matrix is symmetric and calculate only unique batches.
            Setting this option to `True` reduces the number of batches that are calculated 
            by ca. a factor of two, as off-diagonal elements are computed only once.

    **Methods:**
    """

    def __init__(self, 
                 root_path: str = ".", 
                 matrix_folder_name: str = "batched_similarity_matrix",
                 fingerprint_files_name: str = "batch_similarity_fingerprints", 
                 load_from_file: bool = True,
                 batch_size: int = 10000,
                 n_tasks: int = 1, 
                 task_id: int = 0,
                 size: int = 0,
                 symmetric: bool = True,
                 dtype = np.float32):
        folder_path = os.path.join(root_path, matrix_folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.matrix_folder_name = matrix_folder_name
        self._folder_path = os.path.abspath(folder_path)
        self._task_id = task_id
        self._dtype = dtype
        if load_from_file and os.path.exists(os.path.join(self.folder_path, self.metadata_filename)):
            metadata = self._metadata_from_file()
            self.fingerprint_files_name = metadata["fingerprint_files_name"]
            self._n_tasks = metadata["n_tasks"]
            self._batch_size = metadata["batch_size"]
            self._size = metadata["size"]
            self._symmetric = metadata["symmetric"]
        else:
            self.fingerprint_files_name = str(fingerprint_files_name)
            self._n_tasks = int(n_tasks)
            self._batch_size = int(batch_size)
            self._size = int(size)
            self._symmetric = bool(symmetric)
        
    @property
    def n_tasks(self) -> int:
        """
        Total number of tasks used to calculate similarity matrix.
        """
        return self._n_tasks
    
    @property
    def task_id(self) -> int:
        """
        Task id of the current task.
        """
        return self._task_id
    
    @property
    def batch_size(self) -> int:
        """
        Number of fingerprints in a batch.
        """
        return self._batch_size
    
    @property
    def folder_path(self) -> str:
        """
        Path to folder where all data corresponding to this matrix is stored.
        """
        return self._folder_path

    @property
    def size(self) -> int:
        """
        Total number of fingerprints that are used.
        """
        return self._size

    @property
    def symmetric(self) -> bool:
        """
        Matrix is symmetric.
        """
        return self._symmetric
    
    @property
    def batch_iterator(self) -> BatchIterator:
        """
        `BatchIterator` object that is used to iterate over batches of the similarity matrix.
        """
        return BatchIterator(self.size, self.batch_size, self.n_tasks, self.task_id, self.symmetric)
    
    @property
    def metadata(self) -> dict:
        """
        Metadata for BatchedSimilarityMatrix.
        """
        metadata = {
            "fingerprint_files_name" : self.fingerprint_files_name,
            "folder_path" : self.folder_path,
            "n_tasks" : self.n_tasks,
            "batch_size" : self.batch_size,
            "size" : self.size,
            "symmetric" : self.symmetric
        }
        return metadata
    
    @property
    def metadata_filename(self) -> str:
        """
        Name of the metadata file.
        """
        return f"{self.matrix_folder_name}_metadata.json"

    @property
    def mid_filename(self) -> str:
        """
        Name of the file where all material ids are stored.
        """
        return f"{self.matrix_folder_name}_mids.txt"
    
    @property
    def most_similar_materials_filename(self) -> str:
        """
        (Base) name of file(s) where all most similar materials are stored.
        """
        return f"{self.matrix_folder_name}_most_similar_materials_{self.task_id}.txt"
    
    @property
    def matrices_for_this_task(self):
        """
        Return the number of similarity matrices that are calculated for this task.
        """
        n_tasks = len([_batch for _batch in self.batch_iterator])
        return n_tasks

    def set_task_id(self, task_id: int) -> None:
        assert task_id < self.n_tasks, f"task id must be smaller than number of tasks {self.n_tasks}"
        self._task_id = task_id

    def set_n_tasks(self, n_tasks: int, write_to_metadata: bool = False):
        assert self.task_id < n_tasks, f"current task id is smaller than number of tasks {n_tasks}"
        self._n_tasks = n_tasks
        if write_to_metadata:
            self.save_metadata()

    def save_metadata(self, overwrite: bool = True) -> None:
        """
        Save current metadata to `self.folder_name/self.metadata_filename`.

        **Keyword arguments:**

        overwrite: `bool`
            Overwrite file if it exists.

            default: `True`
        """
        filename = os.path.join(self.folder_path, self.metadata_filename)
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError("Metadata file exists! Use `overwrite = True` to overwrite.")
        with open(filename, "w") as metadata_file:
            json.dump(self.metadata, metadata_file)
        
    def fingerprint_file_batches(self, 
                                 fingerprint_file_name: str, 
                                 fingerprint_file_path: str,
                                 overwrite: bool = False,
                                 write_mid_file: bool = True,
                                 save_metadata_updates: bool = True) -> None:
        """
        Split a single, large fingerprint file into batches that can be read by the individual
        tasks of a `BatchedSimilarityMatrix`. 

        **Arguments:**

        fingerprint_file_name: `str`
            Name of the file containing a json-encoded list of serialized `Fingerprint` objects.

        fingerprint_file_path: `str`
            Path to fingerprint file.

        **Keyword arguments:**

        overwrite: `bool`
            Overwrite batched fingerprint files if they exist.
            If this is set to `False` and (any of) the files exist(s), a `FileExistsError` is raised.

            default: `False`

        write_mid_file: `bool`
            Write a file that contains an enumeration of all material ids in the similarity matrix.

            default: `True`

        save_metadata_updates: `bool`
            Save the metadata after determining the total number of fingerprints.

            default: `True`
        """
        with open(os.path.join(fingerprint_file_path, fingerprint_file_name), "r") as fp_file:
            file_data = json.load(fp_file)
        self._size = len(file_data)
        if save_metadata_updates:
            self.save_metadata()
        fingerprint_batches = BatchIterator.linear_batch_list(self.size, self.batch_size)
        for idx, jdx in fingerprint_batches:
            filename = self.gen_fingerprint_batch_file_name(idx, jdx)
            filepath = os.path.join(self.folder_path, filename)
            if os.path.exists(filepath) and not overwrite:
                raise FileExistsError("Fingerprint file exists! Use `overwrite = True` to overwrite.")
            with open(filepath, "w") as fp_file:
                json.dump(file_data[idx:jdx], fp_file)
        if write_mid_file:
            filepath = os.path.join(self.folder_path, self.mid_filename)
            if os.path.exists(filepath) and not overwrite:
                raise FileExistsError("Mid file exists! Use `overwrite = True` to overwrite.")
            with open(filepath, "w") as mid_file:
                for idx, data_entry in enumerate(file_data):
                    mid_file.write(f"{idx} {json.loads(data_entry)['mid']}\n")
        
    def gen_fingerprint_batch_file_name(self, idx: int, jdx: int) -> str:
        """
        Generate name of a fingerprint batch file from indices.

        **Arguments:**

        idx, jdx: `int`
            Start and end index of the fingerprints file batch.

        **Returns:**

        filename: `str`
            Name of a file with given indices.
        """
        return f"{self.fingerprint_files_name}_{idx}_{jdx}.json"
    
    def gen_matrix_batch_file_name(self, batch: List[List[int]]) -> str:
        """
        Generate name of a matrix batch file from batch.

        **Arguments:**

        batch: `List[List[int]]`
            batch, as returned by `BatchIterator` objects

        **Returns:**

        filename: `str`
            Name of a file corresponding to given batch.
        """
        return f"{self.matrix_folder_name}_{batch[0][0]}_{batch[0][1]}__{batch[1][0]}_{batch[1][1]}.npy"
        
    def fingerprints_from_batch_file(self, idx: int, jdx: int) -> List[Fingerprint]:
        """
        Read fingerprints from file and deserialize. Generates a list of Fingerprints from the files written by
        the function `fingerprint_file_batches`.

        idx, jdx: `int`
            Start and end index of the fingerprints file batch.

        **Returns:**

        fingerprints: `List[Fingerprint]`
            Fingerprint objects from file
        """
        filename = self.gen_fingerprint_batch_file_name(idx, jdx)
        filepath = os.path.join(self.folder_path, filename)
        with open(filepath, "r") as fp_file:
            fps = json.load(fp_file)
        return [Fingerprint.deserialize(fp) for fp in fps]
    
    def get_row_by_mid(self, mid: str) -> List[float]:
        """
        Get a row of the (already calculated) similarity matrix from the mid of a material.

        Calling this function requires the the mid file was written by `fingerprint_file_batches`.

        **Arguments:**

        mid: `str`
            Material id of the reference entry.

        **Returns:**

        similarities: `List[float]`
            Similarities of all materials of this similarity matrix to the reference specified by `mid`

        **Raises:**

        `FileNotFoundError`: Mid file does not exist (or has a non-compatible name).

        `KeyError`: No material of given `mid` in the mid file (and thus in the matrix).
        """
        matrix_row = []
        mid_file_path = os.path.join(self.folder_path, self.mid_filename)
        if not os.path.exists(mid_file_path):
            raise FileNotFoundError("Mid file not found. Can not extract row by mid.")
        # linear search through file to find index of mid in matrix
        ref_idx = None
        with open(mid_file_path, "r") as mid_file:
            for line in mid_file:
                idx, line_mid = line.strip().split()
                if line_mid == mid:
                    ref_idx = int(idx)
                    break
        if ref_idx is None:
            raise KeyError("Mid was not found in file!")
        # open matrices that contain this entry
        for batch in self.batch_iterator.get_batches_for_index(ref_idx):
            matrix_name = self.gen_matrix_batch_file_name(batch)
            matrix = SimilarityMatrix.load(filename = matrix_name, filepath=self.folder_path)
            if batch[1][0] <= ref_idx < batch[1][1]:
                matrix_row.extend(matrix[mid])
            else:
                matrix_row.extend(matrix.get_column(mid))
        return matrix_row
            
    def calculate(self, 
                  similarity_function: Callable, 
                  overwrite: bool = False):
        """
        Calculate similarity matrix entries and write them to files.

        **Arguments:**

        similarity_function: `Callable`
            Function of two `Fingerprint` objects that returns their similarity.

        **Keyword arguments:**
        
        overwrite: `bool`
            Recalculate and overwrite existing matrix batch files. 
        """
        for batch in self.batch_iterator:
            filename = self.gen_matrix_batch_file_name(batch)
            if os.path.exists(os.path.join(self.folder_path, filename)) and not overwrite:
                print(f"Matrix {filename} exists. Skipping.")
                # ignore existing files
                continue
            fingerprints = self.fingerprints_from_batch_file(batch[0][0], batch[0][1])
            for fp in fingerprints:
                fp.set_similarity_function(similarity_function)
            if batch[0] == batch[1]: # diagonal elements
                matrix = SimilarityMatrix(dtype=self._dtype).calculate(fingerprints, symmetric=self.symmetric)
            else:
                row_fingerprints = self.fingerprints_from_batch_file(batch[1][0], batch[1][1])
                for fp in row_fingerprints:
                    fp.set_similarity_function(similarity_function)
                matrix = OverlapSimilarityMatrix(dtype=self._dtype).calculate(fingerprints, row_fingerprints)
            matrix.save(filename = filename, filepath = self.folder_path, overwrite = overwrite)
    
    def write_most_similar_materials_file(self, 
                                          k: int = 10, 
                                          remove_self: bool = True):
        """
        Write files containing all `k` most similar materials for each entry of the matrix.

        **Keyword arguments:**

        k: `int`
            Number of most similar materials to find.

            default: `10`

        remove_self: `bool`
            Find most similar materials *without* self.

            default: `True`
        """
        self.k = int(k) 
        for batch_row in self.batch_iterator.get_batch_rows():
            if remove_self:
                self.k += 1
            matrix_position = -1
            position_list = []
            for batch in batch_row:
                if batch[0] == batch[1]:
                    position_list.append(0)
                    matrix_position = 1
                    continue
                position_list.append(matrix_position)
            with multiprocessing.Pool() as pool:
                most_similar_batches = pool.map(self._get_most_similar_batch, zip(batch_row, position_list))
            if remove_self:
                self.k -= 1
            most_similar = self._merge_most_similar_batches(most_similar_batches, remove_self = remove_self)
            self._append_most_similar_materials_file(most_similar)

    def get_entry_histogram(self, bins = np.arange(0,1.01,0.01)):
        """
        Generate a histogram of entries in the matrix.

        **Keyword arguments:**

        bins: `np.ndarray`
            Bins of the histogram. Each bin contains the number of matrix entries 
            in the range [bins[i], bins[i+1]).

            default: `np.arange(0,1.01,0.01)`

        **Returns:**

        (entries, bins): `(np.ndarray, np.ndarray)`
            entries[i] contains the absolute number of matrix entries with similarity
            in the range [bins[i], bins[i+1]).
        """
        entries = np.zeros(len(bins)-1)
        for batch in self.batch_iterator.batches:
            filename = self.gen_matrix_batch_file_name(batch)
            simat = SimilarityMatrix.load(filename = filename, filepath=self.folder_path)
            if type(simat) == SimilarityMatrix:
                in_matrix_entries = simat.get_unique_entries()
            elif type(simat) == OverlapSimilarityMatrix:
                in_matrix_entries = simat.get_entries()
            hist_, _ = np.histogram(in_matrix_entries, bins = bins)
            entries += hist_
        return entries, bins
    
    def _get_most_similar_batch(self, batch_position):
        batch, position = batch_position
        most_similar_batch = []
        matrix_file_name = self.gen_matrix_batch_file_name(batch)
        matrix = SimilarityMatrix.load(filename = matrix_file_name, filepath = self.folder_path)
        # unify matrix format
        if position == 0: # diagonals
            mids = matrix.mids
            ref_mids = mids
        elif position == -1: # horizontal
            mids = matrix.column_mids
            ref_mids = matrix.row_mids
        else: # vertical
            mids = matrix.row_mids
            ref_mids = matrix.column_mids
            matrix.transpose()
        matrix = matrix.matrix.tolist()
        # add for each matrix row the most similar materials
        for row in matrix:
            most_similar_batch.append(sorted(zip(row, mids), reverse = True)[:self.k])
        return ref_mids, most_similar_batch
    
    def _merge_most_similar_batches(self, most_similar_batches, remove_self = True):
        ref_mids = []
        most_similar = []
        for batch in most_similar_batches:
            ref_mids.append(batch[0])
            most_similar.append(np.array(batch[1]))
        for idx in range(len(ref_mids)-1):
            assert all(ref_mids[idx] == ref_mids[idx+1]), "Reference mids do not coincide."
        ref_mids = ref_mids[0]
        full_list = np.concatenate(most_similar, axis = 1).tolist()
        best_per_row = []
        for ref_mid, sub_list in zip(ref_mids, full_list):
            if remove_self:
                sub_list = filter(lambda x: x[1] != ref_mid, sub_list)
            best_per_row.append(sorted(sub_list, reverse=True)[:self.k])
        return ref_mids, best_per_row

    def _append_most_similar_materials_file(self, most_similar):
        most_similar_materials_filepath = os.path.join(self.folder_path, self.most_similar_materials_filename)
        with open(most_similar_materials_filepath, "a") as outfile:
            for ref_mid, most_similar_materials in zip(*most_similar):
                data = {ref_mid : {mid: sim for sim,mid in most_similar_materials}}
                outfile.write(json.dumps(data))
                outfile.write("\n")
    
    def _metadata_from_file(self):
        filename = os.path.join(self.folder_path, self.metadata_filename)
        with open(filename, "r") as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def __repr__(self) -> str:
        return f"BatchedSimilarityMatrix(size = {self.size}) at {self.folder_path}"