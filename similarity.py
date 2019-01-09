import numpy as np
import logging
from fingerprints import Fingerprint
import os, csv

def returnfunction(*args): #Why do I have this?
    return [*args]

class SimilarityMatrix():
    """
    A matrix, that stores all (symmetric) similarites between materials in a database.
    """

    def __init__(self, root = '.', data_path = 'data'):
        self.matrix = []
        self.mids = []
        self.fp_type = None
        self.log = logging.getLogger('log')
        self.data_path = os.path.join(root, data_path)

    def calculate(self, fp_type, db, **kwargs):
        self.fp_type = fp_type
        fingerprints = []
        for id in range(1, db.count()+1):
            row = db.get(id)
            if hasattr(row, fp_type):
                fingerprints.append(Fingerprint(fp_type, mid = row.mid, db_row = row))
                self.mids.append(row.mid)
            else:
                self.log.error('No fingerprint of type '+fp_type+'. Skipping material '+row.mid+ ' for similarity matrix.')
        for idx, fp in enumerate(fingerprints):
            matrix_row = []
            for jdx, fp2 in enumerate(fingerprints[idx:]):
                matrix_row.append(fp.get_similarity(fp2, **kwargs))
            self.matrix.append(np.array(matrix_row))
        self.matrix = np.array(self.matrix)
        if self.matrix.shape[0] == 0:
            self.log.error('Empty similarity matrix.')
            raise RuntimeError('Similarity matrix could not be generated.')

    def get_row(self, mid):
        row = []
        mid_idx = self.mids.index(mid)
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
        n_nearest = sorted(neighbors_zipped)[(-1 * k + 1):-1] #last n entries without last (because is self-similiarty = 1)
        neighbors_dict = {self.mids[x[1]]:x[0] for x in n_nearest}
        return neighbors_dict

    def gen_neighbors_dict(self, k = 10):
        neighbors_dict = {}
        for mid in self.mids:
            neighbors_dict[mid] = self.get_k_nearest(mid, k = k)
        return neighbors_dict

    def save(self, filename = 'similarity_matrix.csv'):
        full_path = os.path.join(self.data_path, filename)
        with open(full_path, 'w', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(self.mids)
            for index, row in enumerate(self.matrix):
                csvwriter.writerow(row)

    def load(self, filename = 'similarity_matrix.csv'):
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

class SimilarityCrawler(): #TODO needs updates for changed k-nearest-neighbor-dict

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
            for similarity in self.members[mid].keys():
                if float(similarity) >= self.threshold:
                    candidate = self.members[mid][similarity]
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

def orphans(neighbors_dict, group_member_list): #TODO needs updates for changed k-nearest-neighbor-dict
    orphans = {}
    full_list = []
    for item in group_member_list:
        for member in item:
            full_list.append(member)
    for mid in neighbors_dict.keys():
        if not mid in full_list:
            orphans[mid] = neighbors_dict[mid]
    return orphans

def similarity_search(atoms_db, mid, fp_type, k = 10):
    pass
