import random
import numpy as np
import matplotlib.pyplot as plt
import time, copy


class Cluster():
    """
    Cluster of materials that have a certain similarity among each other.
    """
    def __init__(self, threshold = 0.95, threshold_step = 0.05):
        self.threshold = threshold
        self.threshold_step = threshold_step
        self.fingerprints = []

    def add_fingerprint(self, fingerprint):
        self.fingerprints.append(fingerprint)

    def reduce_threshold(self):
        if self.threshold > 0:
            self.threshold = round(self.threshold - self.threshold_step, 5)
        else:
            raise AssertionError('Reduced cluster threshold below zero.')
        #print('new threshold', self.threshold)

    def populate(self, fingerprint_list, init = 'random'):
        """
        Populate the cluster with materials.
        Returns tuple of:
            * True: bool; clustering was successful, new materials are added
            * False: bool; clustering was not successful, no new materials are added
        and:
            * init_threshold: float; maximum threshold of similarities that can still be found in the fingerprint list
        """
        init_threshold = self.threshold
        if len(fingerprint_list) == 0:
            return False, init_threshold
            #raise AssertionError('Tried to run Cluster().populate() with empty fingerprint list.') #Might replace with return statement later.
        if len(self.fingerprints) == 0:
            if len(fingerprint_list) == 1:
                self.add_fingerprint(fingerprint_list.pop(0))
                return False, init_threshold
            if init == 'random':
                self._init_random(fingerprint_list)
            elif init == 'highest_threshold':
                #print('initializing new cluster with highest threshold method')
                found_init_value = self._init_highest_threshold(fingerprint_list)
                while not found_init_value:
                    self.reduce_threshold()
                    found_init_value = self._init_highest_threshold(fingerprint_list)
                init_threshold = self.threshold
        changed = True
        added_materials = False
        new_members = self.fingerprints
        while changed:
            #print('current number of members: ', len(self.fingerprints))
            changed, added_new_materials, new_members = self._crawl_fingerprint_list(new_members, fingerprint_list)
            added_materials = added_materials or added_new_materials
        return added_materials, init_threshold

    def _crawl_fingerprint_list(self, members_list, fingerprint_list):
        changed = False
        new_members = []
        added_materials = False
        for fp in members_list:
            sims = [fp.get_similarity(x) for x in fingerprint_list]
            indices = [idx for idx,x in enumerate(sims) if x >= self.threshold] #i.e. indices of materials that have a similarity with the member larger than the threshold
            if len(indices) > 0:
                for new_member in self._pop_list(fingerprint_list, indices):
                    new_members.append(new_member)
                changed = True
                added_materials = True
        for member in new_members:
            self.fingerprints.append(member)
        return changed, added_materials, new_members

    def _init_random(self, fingerprint_list):
        if len(self.fingerprints) == 0:
            self.add_fingerprint(fingerprint_list.pop(random.randint(0,len(fingerprint_list)-1)))

    def _init_highest_threshold(self, fingerprint_list):
        success = False
        for idx, fp1 in enumerate(fingerprint_list):
            for fp2 in fingerprint_list[idx+1:]:
                sim = fp1.get_similarity(fp2)
                if sim >= self.threshold:
                    success = True
                    break
            if success:
                break
        if success:
            self.add_fingerprint(fingerprint_list.pop(idx))
            return True
        else:
            return False

    def __len__(self):
        return len(self.fingerprints)

    def _pop_list(self, list_to_pop, popping_indices):
        popped_items = []
        popping_indices.sort(reverse = True)
        for idx in popping_indices:
            popped_items.append(list_to_pop.pop(idx))
        return popped_items

def cluster_similarity(cluster1, cluster2):
    sim = 0
    for fingerprint in cluster1.fingerprints:
        sims = [fingerprint.get_similarity(fp) for fp in cluster2.fingerprints]
        sim += sum(sims) / len(sims)
    sim /= len(cluster1)
    return sim

class DatabaseCrawler():
    """
    A crawler that goes through a database and clusters all materials that are similar up to a certain threshold into clusters.
    """
    def __init__(self, database, fingerprint_type = 'DOS', fingerprint_name = None, init_threshold = 0.95):
        self.clusters = []
        self.threshold = None
        self.orphans = []
        self.fp_name = fingerprint_name if fingerprint_name != None else fingerprint_type
        self.db_hash = self._gen_hash(database.atoms_db_path)
        self.fingerprints = database.get_fingerprints(fingerprint_type, name = fingerprint_name, log = False)
        self.init_threshold = init_threshold

    def _new_custer(self, min_size = 5, init_clusters = 'highest_threshold', **kwargs):
        new_cluster = Cluster(threshold = self.init_threshold, **kwargs)
        while len(new_cluster) < min_size and len(self.orphans) > 0:
            added_materials, self.init_threshold = new_cluster.populate(self.orphans, init = init_clusters)
            while not added_materials and len(self.orphans) > 0:
                new_cluster.reduce_threshold()
                added_materials = new_cluster.populate(self.orphans, init = init_clusters)
        self.orphans = [x for x in self.orphans if not x in new_cluster.fingerprints]
        #print('finished new cluster', new_cluster)
        return new_cluster

    def cluster_all(self, fingerprints = None, init_clusters = 'highest_threshold', **kwargs):
        if fingerprints == None:
            fingerprints = self.fingerprints
        self.orphans = fingerprints
        iterations = 0
        while len(self.orphans) > 0:
            print('iterations', iterations,'n_clusters', len(self),'n_orphans', len(self.orphans), end='\r')
            new_cluster = self._new_custer(init_clusters = init_clusters, **kwargs)
            self.clusters.append(new_cluster)
            iterations += 1

    def __len__(self):
        return len(self.clusters)

    @staticmethod
    def _gen_hash(string):
        return hash(string).to_bytes(20, 'little', signed = True).hex()

class SimilarityDictCrawler():

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
        plt.xlabel('Distance (1-similarity) threshold')
        plt.ylabel('Number of clusters')
        plt.show()
        threshold = float(input('Enter threshold:'))
        self.set_threshold(threshold)
