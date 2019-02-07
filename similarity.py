import numpy as np
import logging
import os, csv
import multiprocessing
import json
import copy
import math
import matplotlib.pyplot as plt

from fingerprints import Fingerprint

def returnfunction(*args): #Why do I have this?
    return [*args]

def _calc_sim_multiprocess(fp1__fp2):
    fp1 = fp1__fp2[0]
    fp2 = fp1__fp2[1]
    s =fp1.get_similarity(fp2)
    return s


class SimilarityMatrix():
    """
    A matrix, that stores all (symmetric) similarites between materials in a database.
    """

    def __init__(self, root = '.', data_path = 'data', large = False, filename = 'similarity_matrix.csv', print_to_screen = True):
        self.matrix = []
        self.mids = []
        self.fp_type = None
        self.log = logging.getLogger('log')
        self.data_path = os.path.join(root, data_path)
        self.large = large
        self.filename = filename
        self.print_to_screen = print_to_screen

    def calculate(self, fp_type, db, filename = 'similarity_matrix.csv', multiprocess = True, **kwargs):
        """
        Calculates the SimilarityMatrix.
            If SimilarityMatrix.large == True: The matrix is written to file during calculation.
        """
        if self.large: #For large databases, the matrix will not fit the RAM, thus calculation gets killed.
            full_path = os.path.join(self.data_path, filename)
            outfile = open(full_path, 'w')
            csvwriter = csv.writer(outfile)
        self.fp_type = fp_type
        fingerprints = []
        for id in range(1, db.count()+1):
            row = db.get(id)
            if hasattr(row, fp_type):
                try:
                    fingerprints.append(Fingerprint(fp_type, mid = row.mid, db_row = row, log = False))
                    self.mids.append(row.mid)
                except (TypeError, json.decoder.JSONDecodeError):
                    self.log.error('"None" for fingerprint of type '+fp_type+'. Skipping material '+row.mid+ ' for similarity matrix.')
            else:
                self.log.error('No fingerprint of type '+fp_type+'. Skipping material '+row.mid+ ' for similarity matrix.')
        self.log.debug('SimilaritMatrix: All %s fingerprints loaded.' %(fp_type))
        if self.large:
            csvwriter.writerow(self.mids)
        n_matrix_rows = len(fingerprints)
        for idx, fp in enumerate(fingerprints):
            if not multiprocess:
                matrix_row = []
                for jdx, fp2 in enumerate(fingerprints[idx:]):
                    matrix_row.append(fp.get_similarity(fp2, **kwargs))
                if not self.large:
                    self.matrix.append(np.array(matrix_row))
                else:
                    csvwriter.writerow(matrix_row)
            else:
                with multiprocessing.Pool() as p:
                    self.matrix.append(np.array(p.map(_calc_sim_multiprocess,[[fp, fp2] for fp2 in fingerprints[idx:]])))
            if self.print_to_screen:#self.fp_type == "SOAP":#math.ceil(idx/n_matrix_rows*100)%10 == 0:
                print('SimilarityMatrix generated: {:6.3f} %'.format(idx/n_matrix_rows*100), end = '\r')
        if self.large:
            outfile.close()
            self.matrix = None
        else:
            self.matrix = np.array(self.matrix)
            if self.matrix.shape[0] == 0:
                self.log.error('Empty similarity matrix.')
                raise RuntimeError('Similarity matrix could not be generated.')
            print('\nFinished SimilarityMatrix generation.\n')

    def get_complement(self, maximum = 1, get_matrix_object = False):
        complement = []
        for row in self.matrix:
            complement.append(maximum-row)
        if get_matrix_object:
            distance_matrix = SimilarityMatrix(filename = 'distance_matrix.csv')
            distance_matrix.matrix = complement
            distance_matrix.mids = self.mids
            return distance_matrix
        return np.array(complement)

    def get_square_matrix(self):
        square_matrix = []
        for mid in self.mids:
            square_matrix.append(self.get_row(mid))
        return np.array(square_matrix)

    def get_row(self, mid):
        row = []
        if self.mids == []:
            self._load_mids()
        mid_idx = self.mids.index(mid)
        if self.large:
            full_path = os.path.join(self.data_path, self.filename)
            with open(full_path, 'r', newline='') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    pass #CONTINUE HERE LATER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # if index(row) < index(mid), choose entry
                    # else add row
        else:
            for idx in range(len(self.mids)):
                if idx < mid_idx:
                    row.append(self.matrix[idx][mid_idx-idx])
                elif idx > mid_idx:
                    row.append(self.matrix[mid_idx][idx-mid_idx])
                else:
                    row.append(self.matrix[idx][0])
        return row

    def get_k_nearest(self, mid, k = 10):
        neighbors = self.get_row(mid)
        neighbors_zipped = [(sim, idx) for idx, sim in enumerate(neighbors)]
        neighbors_zipped.sort(reverse = True)
        k_new = k
        for item in neighbors_zipped:
            if item[0] >= 1.0:
                k_new += 1
            else:
                break
        n_nearest = neighbors_zipped[:k_new] #last n entries without last (because is self-similiarty = 1)
        neighbors_list = [[self.mids[x[1]], x[0]] for x in n_nearest]
        return neighbors_list

    def gen_neighbors_dict(self, k = 10):
        neighbors_dict = {}
        for mid in self.mids:
            neighbors_dict[mid] = self.get_k_nearest(mid, k = k)
        return neighbors_dict

    def save(self, filename = 'similarity_matrix.csv'):
        self.filename = filename
        if self.large:
            self.log.error('Warning: Writing not implemented for large matrices. File is written during generation process.')
            return
        full_path = os.path.join(self.data_path, filename)
        with open(full_path, 'w', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(self.mids)
            for index, row in enumerate(self.matrix):
                csvwriter.writerow(row)

    def load(self, filename = 'similarity_matrix.csv'):
        self.filename = filename
        if self.large:
            self.log.error('Warning: Loading not implemented for large matrices.')
            return None
        full_path = os.path.join(self.data_path, filename)
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

    def get_matching_matrices(self, second_matrix):
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
            print('OOOPS!', matching_mids, matching_mids2)
        return matching_self, matching_matrix, matching_mids

    def get_correlation(self, second_matrix):
        matching_self, matching_matrix, matching_mids = self.get_matching_matrices(second_matrix)
        m_diff = matching_self - matching_matrix
        summe = 0
        for row in m_diff:
            summe += np.dot(row,row)
        return np.sqrt(summe)

    def plot_correlation(self, second_matrix, show = True):
        matching_self, matching_matrix, matching_mids = self.get_matching_matrices(second_matrix)
        pairs = []
        for idx in range(len(matching_self)):
            for jdx in range(len(matching_self[idx])):
                pairs.append([matching_self[idx][jdx],matching_matrix[idx][jdx]])
        xs = np.array([x[0] for x in pairs])
        ys = np.array([x[1] for x in pairs])
        fit = np.polyfit(xs, ys, 1, full = True)
        import matplotlib.pyplot as plt
        polyfunction = np.poly1d(fit[0])
        plt.scatter(xs,ys, s=1, alpha=0.1)
        plt.scatter(xs, polyfunction(xs), s=1)
        if show:
            plt.show()

    def _load_mids(self):
        with open(self.filename) as f:
            mids = f.readline()
        mids = mids.split(',')
        return mids

    def _get_index_in_matrix(self, mid1, mid2): #TODO its not finished!
        row_index = self.mids.index(mid1)
        column_index = len(self.mids)

    def get_cleared_matrix(self, leave_out_mids):
        if self.large:
            raise NotImplementedError("Cleared matrix is not implemented for large matrices.")
        matrix_copy = copy.deepcopy(self.matrix)
        mids_copy = copy.deepcopy(self.mids)
        for mid in leave_out_mids:
            mid_index = mids_copy.index(mid)
            mids_copy.remove(mid)
            matrix_copy = self._remove_index_from_matrix(matrix_copy, mid_index)
        return matrix_copy, mids_copy

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

class SimilarityCrawler():

    def __init__(self, first_members, threshold = 0.9):
        self.members = {}
        for mid in first_members.keys():
            self.members[mid] = first_members[mid]
        self.threshold = threshold

    def reduce_threshold(self, reduce_by = 0.05):
        self.threshold -= reduce_by
        return self.threshold

    def expand(self, neighbors_dict, associates):
        expanded = False
        new_member = None
        used_members = []
        for group in associates:
            used_members = used_members + group
        for mid in self.members.keys():
            for new_mid, similarity in self.members[mid]:
                if float(similarity) >= self.threshold:
                    candidate = new_mid
                    if not candidate in self.members.keys() and not candidate in used_members:
                        new_member = candidate
                        expanded = True
                        break
        if new_member != None:
            self.members[new_member] = neighbors_dict[new_member]
        return expanded

    def iterate(self, neighbors_dict, associates):
        running = True
        nsteps = 0
        while running:
            running = self.expand(neighbors_dict, associates)
            nsteps += 1
        return nsteps

    def report(self):
        return [x for x in self.members.keys()]

class DBSCANClusterer():

    def __init__(self, distance_matrix = np.array(None), mid_list = np.array(None)):
        from sklearn.cluster import DBSCAN
        self.matrix = [] if distance_matrix.all() == None else distance_matrix
        self.mids = [] if mid_list == None else mid_list
        self.dbscan = DBSCAN(eps = 0.5, metric='precomputed', n_jobs=-1)
        self.clusters = []
        self.orphans = []

    def set_threshold(self, threshold):
        self.dbscan.set_params(eps = threshold)

    def cluster(self, distance_matrix = None, mid_list = None):
        if distance_matrix == None:
            distance_matrix = self.matrix
        if mid_list == None:
            mid_list = self.mids
        self.dbscan.fit(distance_matrix)
        labels = self.dbscan.labels_
        self.clusters = []
        self.orphans = []
        for label in np.unique(labels):
            if label == -1:
                self.orphans.append([mid for index, mid in enumerate(mid_list) if labels[index] == label])
            self.clusters.append([mid for index, mid in enumerate(mid_list) if labels[index] == label])
        return len(self.clusters)

    def maximize_n_clusters(self, distance_matrix = None, mid_list = None):
        return_list = []
        for threshold in range(999,0,-1):
            self.set_threshold(threshold/1000)
            return_list.append([threshold/1000, self.cluster(distance_matrix, mid_list)])
        return return_list

    def optimize_clusters_interactive(self, distance_matrix = None, mid_list = None):
        liste = self.maximize_n_clusters(distance_matrix, mid_list)
        xs = [x[0] for x in liste]
        ys = [x[1] for x in liste]
        plt.plot(xs,ys)
        plt.show()
        threshold = float(input('Enter threshold:'))
        self.set_threshold(threshold)


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

def similarity_search(db, mid, fp_type, k = 10, **kwargs):
    """
    brute-force searches an MaterialsDatabase for k most similar materials and returns them as a list
    """
    neighbors = []
    reference = db.get_fingerprint(mid, fp_type)
    for index in range(1, db.atoms_db.count()+1):
        row = db.atoms_db.get(index)
        try:
            fingerprint = Fingerprint(fp_type, mid = row.mid, db_row = row)
        except AttributeError:
            db.log.error('No Fingerprint of type %s for db entry %s.' %(fp_type, row.mid))
            continue
        similarity = reference.get_similarity(fingerprint, **kwargs)
        if index <= k:
            neighbors.append([similarity, row.mid])
        else:
            if similarity > neighbors[-1][0]:
                neighbors.append([similarity, row.mid])
                neighbors.sort(reverse = True)
                neighbors = neighbors[0:k]
    return neighbors

def parallel_similarity_search(db, fp_type, k = 10, debug = False, **kwargs):
    fingerprints = []
    for id in range(1, db.count()+1):
        row = db.get(id)
        if hasattr(row, fp_type):
            fingerprints.append(Fingerprint(fp_type, mid = row.mid, db_row = row, log = False))
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
    output:
        mid: [mid1: similarity1, ...]
    """
    reference = ref_fp_fp_list_k_neighbors[0]
    fp_list = ref_fp_fp_list_k_neighbors[1]
    k = ref_fp_fp_list_k_neighbors[2]
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
