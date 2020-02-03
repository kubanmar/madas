import numpy as np
import pandas as pd
import os, csv
import multiprocessing
import json
import copy
import matplotlib.pyplot as plt
from functools import partial

from fingerprint import Fingerprint
from utils import report_error, BatchIterator, Float32ToJson

class SimilarityMatrix():
    """
    A matrix, that stores all (symmetric) similarites between materials.
    Kwargs:
        * matrix: np.ndarray; default: None; Initialize matrix with precomputed similarities.
        * mids: list; default: None; Values for material ids for precomputed similarities.
    """

    def __init__(self, matrix = [], mids = []):
        self.matrix = matrix
        self.mids = mids
        self._iter_index = 0

    def calculate(self, fingerprints, mids = [], multiprocess = True, print_to_screen = True):
        self.matrix = []
        self.mids = mids
        n_matrix_rows = len(fingerprints)
        if multiprocess:
            with multiprocessing.Pool() as p:
                self.matrix = p.map(self._get_similarities_list_index, [(idx, fingerprints) for idx in range(len(fingerprints))])
        else:
            for idx, fp in enumerate(fingerprints):
                matrix_row = []
                for fp2 in fingerprints[idx:]:
                    matrix_row.append(fp.get_similarity(fp2))
                self.matrix.append(np.array(matrix_row))
                if print_to_screen:
                    print('SimilarityMatrix generated: {:6.3f} %'.format(idx/n_matrix_rows*100), end = '\r')
        self.matrix = np.array(self.matrix)
        if self.matrix.shape[0] == 0:
            raise RuntimeError('Similarity matrix could not be generated.')
        if print_to_screen:
            print('\nFinished SimilarityMatrix generation.\n')

    def get_sorted_square_matrix(self, new_mid_list):
        """
        Return an array of similarities, which is sorted by the list of mids that was given as input.
        Especially useful to visualize the results of clustering.
        Args:
            * new_mid_list: list of strings; the list of mids that is used to create a new square matrix
        Returns:
            * sorted_matrix: np.ndarray; square matrix of materials similarities
        """
        sorted_matrix = []
        for mid1 in new_mid_list:
            if mid1 in self.mids:
                new_row = []
                for mid2 in new_mid_list:
                    if mid2 in self.mids:
                        new_row.append(self.get_entry(mid1,mid2))
                sorted_matrix.append(new_row)
        return np.array(sorted_matrix)

    def sort_by_mid_list(self, mid_list): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Implement that. Like, now.
        """
        Sort matrix by a given list of mids.
        Args:
            * mid_list: list of strings; sorted list of material ids
        Returns:
            * None
        """

        raise NotImplementedError("Not implemented (yet).")

    def lookup_similarity(self, fp1, fp2):
        return self.get_entry(fp1.mid, fp2.mid)

    def align(self, second_matrix):
        """
        Align the materials in this matrix and a second matrix.
        Args:
            * second_matrix: SimilarityMatrix(); matrix object to align with
        Returns:
            * None
        Warning! Entries in both matrices will be altered, i.e. unique entries in each matrix will be dropped.
        """
        matching_self, matching_matrix, matching_mids = self.get_matching_matrices(second_matrix)
        self.matrix = matching_self
        self.mids = matching_mids
        second_matrix.matrix = matching_matrix
        second_matrix.mids = matching_mids

    def get_complement(self, maximum = 1, get_matrix_object = False):
        """
        Calculate complement of the similarity matrix.
        Kwargs:
            * maximum; float; default: 1; reference maximum, i.e. complement = maximum - similarity
            * get_matrix_object; bool; default: False; Return SimilarityMatrix() instead of np.ndarray
        Returns:
            * np.ndarray __or__ SimilarityMatrix() object
        """
        complement = []
        for row in self.matrix:
            complement.append(maximum-row)
        if get_matrix_object:
            distance_matrix = SimilarityMatrix(filename = 'distance_matrix.csv')
            distance_matrix.matrix = complement
            distance_matrix.mids = self.mids
            return distance_matrix
        return np.array(complement)

    def get_entry(self, mid1, mid2):
        """
        Get any entry of the matrix.
        Args:
            * mid1: string; material id of the first material
            * mid2: string; material id of the second material
        Returns:
            * similarity; float; similarity between both materials
        """
        idx1 = self.mids.index(mid1)
        idx2 = self.mids.index(mid2)
        if idx1 < idx2:
            return self.matrix[idx1][idx2-idx1]
        else:
            return self.matrix[idx2][idx1-idx2]

    def get_symmertric_matrix(self):
        """
        Get square matrix form. Transfers the internally stored triangular matrix to a (symmetric) square matrix.
        Returns:
            * square_matrix; np.ndarray; matrix of similarities
        """
        matrix_shape = self._get_shape()
        is_already_symmetric = np.array([x == matrix_shape[0] for x in matrix_shape])
        if is_already_symmetric.all() == True:
            return self.matrix
        symmetric_matrix = []
        for idx in range(len(self.matrix)):
            symmetric_matrix.append(self.get_row(idx, use_matrix_index = True))
        return np.array(symmetric_matrix)

    def get_square_matrix(self): #dowmward compatibility
        return self.get_symmertric_matrix()

    @staticmethod
    def triangular_from_square_matrix(matrix):
        """
        Get the upper triangular part of a square matrix.
        Args:
            * matrix: list of lists, np.ndarray; symmetric square matrix
        Returns:
            * triangular_matrix: np.ndarray; triangular form of input matrix
        """
        triangular_matrix = []
        for index, row in enumerate(matrix):
            triangular_matrix.append(row[index:])
        return triangular_matrix

    def get_row(self, mid, use_matrix_index = False):
        """
        Get a full row of the matrix.
        Args:
            * mid: string or int; Material Id oder matrix index of requested matrix row.
        Kwargs:
            * use_matrix_index: bool; default: False; use index of matrix row instead of Material Id
        Returns:
            * row; list; Similarities of material with given mid to all other materials in the matrix.
        """
        row = []
        if not use_matrix_index:
            mid_idx = np.where(np.array(self.mids) == mid)[0][0]
        else:
            if mid >= 0:
                mid_idx = mid
            else:
                mid_idx = len(self) + mid
        if self.batched:
            row = self._get_row_from_batches(mid_idx)
            return row
        if len(self.matrix.shape) == 2:
            if self.matrix.shape[0] == self.matrix.shape[1]:
                return self.matrix[mid_idx]
            else:
                raise ValueError('Matrix shape inconsistent.')
        for idx in range(len(self)):
            if idx < mid_idx:
                row.append(self.matrix[idx][mid_idx-idx])
            elif idx > mid_idx:
                row.append(self.matrix[mid_idx][idx-mid_idx])
            else:
                row.append(self.matrix[idx][0])
        return np.array(row)

    def _get_row_from_batches(self, index):
        row = []
        for batch in self.batches:
            if batch[0][0] <= index: #lower boundary
                offset = batch[0][0]
                if batch[0][1] > index: # requested row is in batch
                    matrix_batch_name = BatchIterator().make_file_name(batch, folder_name = self.batch_folder_name)
                    matrix_batch = np.load(os.path.join(self.batch_folder_name, matrix_batch_name))
                    if batch[0] == batch[1]: # diagonal elements
                        for idx, item in enumerate(matrix_batch[index - offset]):
                            row.append([idx+offset, item])
                    else: # off-diagonals
                        for idx, item in enumerate(matrix_batch[index - offset]):
                            row.append([idx+batch[1][0], item])
                elif batch[0][1] <= index and batch[1][0] <= index and batch[1][1] > index: # vertical batches
                    matrix_batch_name = BatchIterator().make_file_name(batch, folder_name = self.batch_folder_name)
                    matrix_batch = np.load(os.path.join(self.batch_folder_name, matrix_batch_name))
                    for idx, item in enumerate(np.transpose(matrix_batch)[index - offset]):
                        row.append([idx+offset, item])
        row.sort(key = lambda x: x[0])
        return np.array([x[1] for x in row])

    def get_entries(self):
        """
        Return all enries of the matrix.
        Returns:
            * entries: np.ndarray; 1-d array of all entries of the matrix
        """
        entries = []
        for row in self.matrix:
            for element in row:
                entries.append(element)
        return np.array(entries)

    def get_k_nearest_neighbors(self, ref_mid, k = 10, remove_self = True):
        """
        Get the k nearest materials and the respective similarities for a material.
        Args:
            * ref_mid: string; material id of the requested material
        Kwargs:
            * k: int; default: 10; number of next nearest neighbors to return
            * remove_self: bool; default: True; remove the requested material from the results list (is should have similarity 1)
        Returns:
            * dict: {ref_mid: {<1st_nearest_mid>:<similarity>}, {2nd_nearest_mid>:<similiarty>, ...}}
        """
        row = [(mid,entry) for mid, entry in zip(self.mids, self.get_row(ref_mid))]
        row.sort(reverse = True, key = lambda x: x[1])
        if remove_self:
            for idx, item in enumerate(row):
                if item[0] == ref_mid:
                    row.pop(idx)
                    break
        return {ref_mid : {mid: entry for mid, entry in row[:k]}}

    def save(self, matrix_filename = 'similarity_matrix.npy', mids_filename = 'similarity_matrix_mids.npy', data_path = '.'):
        """
        Save SimilarityMatrix to numpy binary file(s).
        Kwargs:
            * matrix_filename: string; default: 'similarity_matrix.npy'; name of the matrix file
            * mids_filename: string; default: 'similarity_matrix_mids.npy'; name of the mids file
            * data_path: string; default: '.'; relative path to created files
        """
        matrix_path = os.path.join(data_path, matrix_filename)
        mids_path = os.path.join(data_path, mids_filename)
        np.save(matrix_path, self.matrix)
        np.save(mids_path, self.mids)

    @staticmethod
    def load(matrix_filename = 'similarity_matrix.npy', mids_filename = 'similarity_matrix_mids.npy', data_path = '.', memory_mapped = False, batched = False, batch_size = 10000, **kwargs):
        """
        Load SimilarityMatrix from file. Static method.
        Kwargs:
            * matrix_filename: string; default: 'similarity_matrix.npy'; name of the matrix file
            * mids_filename: string; default: 'similarity_matrix_mids.npy'; name of the mids file
            * data_path: string; default: '.'; relative path to created files OR folder name for batched matrix
            * memory_mapped: bool; default: False; toogle if matrix file is a memory mapped matrix generated with ``calculate_memmap()``
            * batched: bool; default: False: toogle if matrix is calculated as batches of sub-matrices as calculated by ``calculate_batch_files()``
            * batch_size: integer; default: 10000; number of rows in a matrix batch; only used if ``batched == True``
        Addition kwargs are passed to SimilarityMatrix().__init__().
        WARNING! SimilarityMatrix() objects that are loaded with batched = True will not allow for all functionality.
        Returns:
            * SimilarityMatrix() object
        """
        self = SimilarityMatrix(**kwargs)
        matrix_path = os.path.join(data_path, matrix_filename)
        mids_path = os.path.join(data_path, mids_filename)
        self.mids = np.load(mids_path)
        if memory_mapped:
            self.matrix = np.memmap(matrix_path, mode = 'r', shape = (len(self.mids),len(self.mids)), dtype=np.float32)
        elif batched:
            self.batch_folder_name = data_path
            self.batches = BatchIterator().create_batches(len(self.mids), batch_size)
            self.batched = True
        else:
            self.matrix = np.load(matrix_path)
        return self

    def save_csv(self, filename = 'similarity_matrix.csv', data_path = '.'):
        """
        Save SimilarityMatrix to csv file.
        Kwargs:
            * filename: string; default: 'similarity_matrix.csv'; name of the created file
            * data_path: string; default: '.'; relative path to the created file
        """
        full_path = os.path.join(data_path, filename)
        with open(full_path, 'w', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(self.mids)
            for index, row in enumerate(self.matrix):
                csvwriter.writerow(row)

    @staticmethod
    def load_csv(filename = 'similarity_matrix.csv', data_path = 'data', root='.', **kwargs):
        """
        Load SimilarityMatrix from csv file. Static method.
        Kwargs:
            * filename: string; default: 'similarity_matrix.csv'; name of the file
            * data_path: string; default: '.'; relative path of the file
            * root: string; default: '.'; root path, of which the relative data_path is chosen
        Addition kwargs are passed to SimilarityMatrix().__init__().
        Returns:
            * SimilarityMatrix() object
        """
        self = SimilarityMatrix(**kwargs)
        full_path = os.path.join(data_path, filename)
        self.matrix = []
        with open(full_path, 'r', newline='') as f:
            csvreader = csv.reader(f)
            header = True
            for row in csvreader:
                if header:
                    self.mids = row
                    header = False
                else:
                    self.matrix.append(np.array([float(x) for x in row]))
        self.matrix = np.array(self.matrix)
        return self

    def get_matching_matrices(self, second_matrix):
        """
        Match matrices such, that they contain the same materials.
        Mostly inteded for the case that some materials do not support some kinds of fingerprints.
        Args:
            * second_matrix; SimilarityMatrix() object; Matrix to match materials
        Returns:
            * matching_self: np.ndarray; `self.matrix` with matched materials
            * matching_matrix: np.ndarray; `second_matrix.matrix` with matched materials
            * matching_mids: list of strings; list of mids of materials that occure in both matrices
        """
        not_in_second_matrix = []
        for mid in self.mids:
            if not (mid in second_matrix.mids):
                not_in_second_matrix.append(mid)
        not_in_self = []
        for mid in second_matrix.mids:
            if not (mid in self.mids):
                not_in_self.append(mid)
        matching_self, matching_mids = self.get_cleared_matrix(not_in_second_matrix)
        matching_matrix, matching_mids2 = second_matrix.get_cleared_matrix(not_in_self)
        if matching_mids != matching_mids2:
            print('OOOPS!', matching_mids, matching_mids2) #TODO fix that, it should report properly
        return matching_self, matching_matrix, matching_mids

    def _get_similarities_list_index(self, idx__list, square_matrix = False):
        idx, fp_list = idx__list
        if square_matrix:
            sims = np.append([-1 for x in range(idx)], fp_list[idx].get_similarities(fp_list[idx:]))
        else:
            sims = fp_list[idx].get_similarities(fp_list[idx:])
        return np.array(sims)

    def _clear_temporary_matrix(self):
        for index in range(len(self.matrix)):
            for idx in range(len(self.matrix)):
                if idx == index:
                    break
                self.matrix[index][idx] = self.matrix[idx][index]

    def _load_mids(self):
        with open(self.filename) as f:
            mids = f.readline()
        mids = mids.split(',')
        return mids

    def get_cleared_matrix(self, leave_out_mids, return_matrix_object = False):
        """
        Return a matrix where all materials with mids specified in `leave_out_mids` are excluded from the matrix.
        Args:
            * leave_out_mids: list of strings; mids of materials to leave out of the matrix
        Kwargs:
            * return_matrix_object: bool; default: False; return SimilarityMatrix() object
        Returns:
            * matrix_copy: np.ndarray; self.matrix after removing specified materials
            * mids_copy; list of strings; self.mids after removing specified materials
            or:
            * SimilarityMatrix(matrix = matrix_copy, mids = mids_copy) if return_matrix_object is True
        """
        matrix_copy = copy.deepcopy(self.matrix)
        mids_copy = copy.deepcopy(self.mids)
        for mid in leave_out_mids:
            mid_index = mids_copy.index(mid)
            mids_copy.remove(mid)
            matrix_copy = self._remove_index_from_matrix(matrix_copy, mid_index)
        if return_matrix_object:
            return SimilarityMatrix(matrix = matrix_copy, mids = mids_copy)
        return matrix_copy, mids_copy

    def _get_shape(self):
        shape_vector = []
        for row in self.matrix:
            shape_vector.append(len(row))
        return shape_vector

    def __len__(self):
        return len(self.matrix)

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index > len(self):
            self._iter_index = 0
            raise StopIteration
        else:
            self._iter_index += 1
            return self.get_row(self._iter_index - 1, use_matrix_index = True)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self.get_row(key)
            except KeyError:
                raise KeyError("No entry with mid = " + key + '.')
        elif isinstance(key, int):
            try:
                return self.get_row(key, use_matrix_index = True)
            except KeyError:
                raise KeyError('No entry with id = ' + str(key) + '.')
        else:
            raise KeyError('Key can not be interpreted as database key.')

    def __add__(self, simat):
        new_matrix = []
        if isinstance(simat, SimilarityMatrix):
            if not self._check_matrix_alignment(simat):
                raise IndexError("Matrices not aligned. Can not add.")
                return None
            for row1, row2 in zip(self.matrix, simat.matrix):
                new_matrix.append(np.array(row1) + np.array(row2))
        elif np.isreal(simat) and np.isscalar(simat):
            for row in self.matrix:
                new_matrix.append(np.array(row) + simat)
        else:
            raise NotImplementedError("Addition is not supported for chosen data type.")
        new_matrix = np.array(new_matrix)
        return SimilarityMatrix(matrix = new_matrix, mids = self.mids)

    def __sub__(self, simat):
        new_matrix = []
        if isinstance(simat, SimilarityMatrix):
            if not self._check_matrix_alignment(simat):
                raise IndexError("Matrices not aligned. Can not subtract.")
                return None
            for row1, row2 in zip(self.matrix, simat.matrix):
                new_matrix.append(np.array(row1) - np.array(row2))
        elif np.isreal(simat) and np.isscalar(simat):
            for row in self.matrix:
                new_matrix.append(np.array(row) - simat)
        else:
            raise NotImplementedError("Subtraction is not supported for chosen data type.")
        new_matrix = np.array(new_matrix)
        return SimilarityMatrix(matrix = new_matrix, mids = self.mids)

    def __mul__(self, simat):
        new_matrix = []
        if isinstance(simat, SimilarityMatrix):
            if not self._check_matrix_alignment(simat):
                raise IndexError("Matrices not aligned. Can not multiply.")
                return None
            for row1, row2 in zip(self.matrix, simat.matrix):
                new_matrix.append(np.array(row1) * np.array(row2))
        elif np.isreal(simat) and np.isscalar(simat):
            for row in self.matrix:
                new_matrix.append(np.array(row) * simat)
        else:
            raise NotImplementedError("Multiplication is not supported for chosen data type.")
        new_matrix = np.array(new_matrix)
        return SimilarityMatrix(matrix = new_matrix, mids = self.mids)

    def _check_matrix_alignment(self, simat):
        if not sorted(self.mids) == sorted(simat.mids):
            raise ValueError("Mids of matrices do not coincide. Use SimilarityMatrix().align().")
            return False
        for mid1, mid2 in zip(self.mids, simat.mids):
            if not mid1 == mid2:
                raise IndexError("Mids of matrices are not aligned. Sort one matrix using sort_by_mid_list().")
                return False
        shape_allignment = np.array([len(row1) == len(row2) for row1, row2 in zip(self.matrix, simat.matrix)])
        if not shape_allignment.all() == True:
            raise IndexError("Shapes of matrices do not coincide. Please convert to same shape.")
            return False
        return True

    @staticmethod
    def _remove_index_from_matrix(array, index):
        array_copy = copy.deepcopy(array)
        to_delete_index = index
        for index, row in enumerate(array_copy):
            if index == to_delete_index:
                break
            entry_index = to_delete_index - index
            array_copy[index] = np.delete(row,entry_index)
        array_copy = np.delete(array_copy,to_delete_index)
        return array_copy

class OverlapSimilarityMatrix(SimilarityMatrix):
    """
    A SimilarityMatrix that is used to store only similarities between different sets of fingerprints.
    """

    def __init__(self, matrix = [], row_mids = [], column_mids = []):
        self.matrix = matrix
        self.row_mids = row_mids
        self.column_mids = column_mids
        self._iter_index = 0


class BatchedSimilarityMatrix(SimilarityMatrix):
    """
    A SimilarityMatrix, with similarity values distributed in batches in different files.
    Only recommended if RAM is too small to hold full matrix and CPU cache is too small to hold fingerprints.
    """

    def __init__(self, matrix = [], mids = []):
        super().__init__(matrix = matrix, mids = mids)

    def calculate(self, fingerprints, folder_name = 'all_DOS_simat', batch_size = 10000):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        batch_iterator = BatchIterator(fingerprints, batches)
        batches = batch_iterator.create_batches(len(fingerprints), batch_size = batch_size)
        for batch in batch_iterator:
            with multiprocessing.Pool() as pool:
                matrix = pool.map(self._multiprocess_batch_files_similarity, [(fp, batch[2]) for fp in batch[1]])
            filename = batch_iterator.make_file_name(batch[0], folder_name = folder_name)
            np.save(os.path.join(folder_name, filename), np.array(matrix, dtype = np.float32))
        np.save(os.path.join(folder_name, folder_name + '_' + 'mid_list' + '.npy'), [fp.mid for fp in fingerprints])

    def _multiprocess_batch_files_similarity(self, fingerprint__fingerprint_list):
        fp, fp_list = fingerprint__fingerprint_list
        return fp.get_similarities(fp_list)


class MemoryMappedSimilarityMatrix(SimilarityMatrix):
    """
    A SimilarityMatrix, with similarity values stored on disk upon creation.
    Only recommended if RAM is too small to hold full matrix.
    """

    def __init__(self, matrix = [], mids = []):
        super().__init__(matrix = matrix, mids = mids)

    def calculate(self, fingerprints, mids = [], multiprocess = True, mapped_filename = 'data/mapped_similarity_matrix.pyc', mids_filename = 'data/mapped_similarity_matrix_mids.pyc'):
        self.matrix = np.memmap(mapped_filename, mode = 'w+', shape=(len(fingerprints),len(fingerprints)), dtype=np.float32)
        self.mids = mids
        if multiprocess:
            with multiprocessing.Pool() as p:
                self.matrix[:] = p.map(partial(self._get_similarities_list_index, square_matrix = True), [(idx, fingerprints) for idx in range(len(fingerprints))])
        else:
            for idx, fp in enumerate(fingerprints):
                matrix_row = []
                for fp2 in fingerprints[idx:]:
                    matrix_row.append(fp.get_similarity(fp2))
                self.matrix[idx] = np.array(matrix_row)
        np.save(mids_filename, self.mids)
        self._clear_temporary_matrix()
        self.matrix.flush()

    def calculate_batched(self, fingerprints, mids, mapped_filename = 'data/mapped_similarity_matrix.pyc', mids_filename = 'data/mapped_similarity_matrix_mids.pyc', batch_size = 10000):
        self.matrix = np.memmap(mapped_filename, mode = 'w+', shape=(len(fingerprints),len(fingerprints)), dtype=np.float32)
        for idx in range(len(self.matrix)):
            self.matrix[idx][:] = (-1 * np.ones(len(fingerprints)))[:]
        self.mids = mids
        self._calculate_batch(fingerprints, batch_size = batch_size)
        np.save(mids_filename, self.mids)
        self._clear_temporary_matrix()
        self.matrix.flush()

    def _calculate_batch_elements(self, batched_list):
        result = []
        batch, list1, list2 = batched_list
        for item in list1:
            result.append(item.get_similarities(list2))
        return batch, np.array(result)

    def _calculate_batch(self, value_list, batch_size = 2):
        batch_iterator = BatchIterator(value_list, batches)
        batches = batch_iterator.create_batches(len(value_list), batch_size = batch_size)
        with multiprocessing.Pool() as p:
            for result in p.imap(self._calculate_batch_elements, batch_iterator):
                batch = result[0]
                for idx, row in zip([x for x in range(batch[0][0], batch[0][1])], result[1]):
                    self.matrix[idx][batch[1][0]:batch[1][1]] = row[:]

class SimilarityFunctionScaling():
    """
    Scales a given similarity function using a given scaling function and kwargs.
    **args**:
        * scaling_function: function that maps (0,1) to (0,1)
        * similarity_function: function that returns the similiarty s in (0,1) for two fingerprints
    **kwargs**:
        * passed to scaling_function
    """
    def __init__(self, scaling_function, similarity_function, **kwargs):
        self._scaling_function = partial(scaling_function, **kwargs)
        self._similarity_function = similarity_function

    def similarity(self,fingerprint1, fingerprint2):
        return self._scaling_function(self._similarity_function(fingerprint1, fingerprint2))

def mean_shifted_scaling(x, mean = 0.5):
    return 1- np.tanh((1-x)/mean/2) / np.tanh(1/mean/2)

def get_orphans(neighbors_dict, group_member_list):
    orphans = {}
    full_list = []
    for item in group_member_list:
        for member in item:
            full_list.append(member)
    for mid in neighbors_dict.keys():
        if not mid in full_list:
            orphans[mid] = neighbors_dict[mid]
    return orphans



def batched_similarity_matrix_k_nearest_search(folder_name = 'all_DOS_simat', batch_size = 10000, k = 10, remove_self = True):
    mids = np.load(os.path.join(folder_name, folder_name + '_mid_list.npy'))
    batch_rows, index_offsets = BatchIterator().create_batch_rows(len(mids), batch_size)
    for row, index_offset in zip(batch_rows, index_offsets):
        nearest_neighbours = {}
        vertical = True # batches in batch_row start vertically
        for batch in row:
            if batch[0] == batch[1]: # and then become horizontal
                vertical = False
            file_name = BatchIterator().make_file_name(batch, folder_name = folder_name)
            batch_matrix = np.load(os.path.join(folder_name, file_name))
            if vertical:
                offset = batch[0][0]
                batch_matrix = np.transpose(batch_matrix)
            else:
                offset = batch[1][0]
            for matrix_row_index, matrix_row in enumerate(batch_matrix):
                mid = mids[matrix_row_index + index_offset]
                mid_matrix_row = [[mids[mid_idx + offset], similarity] for mid_idx, similarity in enumerate(matrix_row)]
                if remove_self:
                    if batch[0] == batch[1]:
                        mid_matrix_row.remove([mid,1.0])
                mid_matrix_row.sort(key = lambda x: x[1], reverse = True)
                matrix_row_nearest = mid_matrix_row[:k]
                if mid in nearest_neighbours:
                    for item in nearest_neighbours[mid]:
                        matrix_row_nearest.append(item)
                    matrix_row_nearest.sort(key = lambda x: x[1], reverse = True)
                    matrix_row_nearest = matrix_row_nearest[:k]
                nearest_neighbours[mid] = matrix_row_nearest
        for mid, neighbours in nearest_neighbours.items():
            print(json.dumps({mid:neighbours}, cls = Float32ToJson))

def _nearest_neighbour_search_batch_row(args):
    folder_name, k, remove_self, mids, row, index_offset = args
    nearest_neighbours = {}
    vertical = True # batches in batch_row start vertically
    for batch in row:
        if batch[0] == batch[1]: # and then become horizontal
            vertical = False
        file_name = BatchIterator().make_file_name(batch, folder_name = folder_name)
        batch_matrix = np.load(os.path.join(folder_name, file_name))
        if vertical:
            offset = batch[0][0]
            batch_matrix = np.transpose(batch_matrix)
        else:
            offset = batch[1][0]
        for matrix_row_index, matrix_row in enumerate(batch_matrix):
            mid = mids[matrix_row_index + index_offset]
            mid_matrix_row = [[mids[mid_idx + offset], similarity] for mid_idx, similarity in enumerate(matrix_row)]
            if remove_self:
                if batch[0] == batch[1]:
                    mid_matrix_row.remove([mid,1.0])
            mid_matrix_row.sort(key = lambda x: x[1], reverse = True)
            matrix_row_nearest = mid_matrix_row[:k]
            if mid in nearest_neighbours:
                for item in nearest_neighbours[mid]:
                    matrix_row_nearest.append(item)
                matrix_row_nearest.sort(key = lambda x: x[1], reverse = True)
                matrix_row_nearest = matrix_row_nearest[:k]
            nearest_neighbours[mid] = matrix_row_nearest
    is_locked = False
    while not is_locked:
        is_locked = lock.acquire(timeout=1)
        if not is_locked:
            time.sleep(5)
    for mid, neighbours in nearest_neighbours.items():
        print(json.dumps({mid:neighbours}, cls = Float32ToJson))
    lock.release()

def batched_similarity_matrix_k_nearest_search_multiprocess(folder_name = 'all_DOS_simat', batch_size = 10000, k = 10, remove_self = True):
    global lock
    lock = multiprocessing.Lock()
    mids = np.load(os.path.join(folder_name, folder_name + '_mid_list.npy'))
    batch_rows, index_offsets = BatchIterator().create_batch_rows(len(mids), batch_size)
    arg_list = [(folder_name, k, remove_self, mids, row, index_offset) for row, index_offset in zip(batch_rows, index_offsets)]
    with multiprocessing.Pool() as pool:
        pool.map(_nearest_neighbour_search_batch_row, arg_list)
    del lock

def similarity_search(db, mid, fp_type, name = None, k = 10, **kwargs):
    """
    brute-force searches an MaterialsDatabase for k most similar materials and returns them as a list
    """
    neighbors = []
    reference = db.get_fingerprint(fp_type, mid = mid)
    fingerprints = db.get_fingerprints(fp_type, name = name, log = False)
    for index, fingerprint in enumerate(fingerprints):
        similarity = reference.get_similarity(fingerprint)
        if index <= k:
            neighbors.append([similarity, fingerprint.mid])
        else:
            if similarity > neighbors[-1][0]:
                neighbors.append([similarity, fingerprint.mid])
                neighbors.sort(reverse = True)
                neighbors = neighbors[0:k]
    return neighbors

def parallel_similarity_search(db, fp_type, name = None, k = 10, debug = False, **kwargs):
    fingerprints = db.get_fingerprints(fp_type, name = name, log = False)
    if debug:
        print('loaded fingerprints')
    with multiprocessing.Pool() as p:
        p.map(get_nearest_neighbors_from_fingerprint_list,[[reference, fingerprints, k] for reference in fingerprints])
    if debug:
        print('finished')

def get_nearest_neighbors_from_fingerprint_list(ref_fp_fp_list_k_neighbors):
    """
    input:
        (<referencefingerprint>, <fingerprint list>, <nr of neighbors>)
    prints:
        mid: [mid1: similarity1, ...]
    """
    reference, fp_list, k = ref_fp_fp_list_k_neighbors
    similarity_list = []
    count = 0
    for index, item in enumerate(fp_list):
        similarity = reference.get_similarity(item)
        listlen = len(similarity_list)
        if count < k:
            count += 1
            similarity_list.append([reference.get_similarity(item), item.mid])
            continue
        if similarity > similarity_list[-1][0]:
            similarity_list.append([reference.get_similarity(item), item.mid])
            similarity_list.sort(reverse = True)
            similarity_list = similarity_list[:k]
    print(json.dumps({reference.mid:{x[1]:x[0] for x in similarity_list}}))
