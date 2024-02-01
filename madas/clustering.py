from typing import List
import os

import numpy as np
from sklearn.cluster import DBSCAN

from madas import SimilarityMatrix

class SimilarityMatrixClusterer():
    """
    A wrapper for clustering methods to directly apply them to `SimilarityMatrix()` objects.

    **Arguments:**

    similarity_matrix: `SimilarityMatrix()` 
        Similarity matrix that should be clustered

    **Keyword arguments:**

    custerer: `type`
        a class that implements a `fit()` method that is used to cluster a np.ndarray matrix
        
        default: `sklearn.cluster.DBSCAN`
    
    clusterer_kwargs: `dict`
        Keyword arguments to be passed to the clusterer upon initializaion
        
        default: `{'metric':'precomputed', 'eps' : 0.15}`

    use_complement: `bool` 
        Switch if the similarity matrix (set to `False`) or distance matrix (set to `True`)
        is used for clustering. 
        
        Some algorithms explicitly require similarity (sometimes called affinity) matrices,
        other require distances (or, dissimilarities, metrics). To treat them all on the same 
        footing, this variable allows to set the behavior of the `matrix` property accordingly.
        
        default: `True`
    """

    def __init__(self, 
                 similarity_matrix: SimilarityMatrix, 
                 clusterer: type = DBSCAN, 
                 clusterer_kwargs: dict = {'metric':'precomputed', 'eps' : 0.15}, 
                 use_complement: bool = True):
        self.simat = similarity_matrix
        self.use_complement = use_complement
        self.clusterer = clusterer(**clusterer_kwargs)

    @property
    def matrix(self) -> np.ndarray:
        """
        Return values similarity matrix that is clustered.

        IF `self.use_complement == True`: return complement of similarity matrix
        
        ELSE: return similarity matrix
        """
        if self.use_complement:
            return 1 - self.simat.matrix
        else:
            return self.simat.matrix

    @property
    def mids(self) -> List[str]:
        """
        List of mids associated with the similarity matrix.
        """
        return [mid for mid in self.simat.mids]

    def cluster(self, **kwargs) -> object:
        """
        Perform clustering on similarity matrix and return self.

        Keyword arguments are passed to the `fit` function of the clusterer.

        **Returns:**

        self: `SimilarityMatrixClusterer`
            Self after calling `fit` of `self.clusterer`
        """
        self.clusterer.fit(self.matrix, **kwargs)
        return self

    def set_clusterer_params(self, **kwargs) -> None:
        """
        Set parameters of `self.clusterer`.
        """
        self.clusterer.set_params(**kwargs)

    @property
    def nclusters(self):
        """
        Get the number of clusters, i.e., the number of unique cluster labels.
        """
        return len(np.unique(self.labels))

    @property
    def labels(self):
        """
        Return labels of clusterer.
        """
        return self.clusterer.labels_

    def get_mids_sorted_by_cluster_labels(self, remove_orphans: bool = False):
        """
        Get mids of the similarity matrix, sorted, ascending, by cluster label.   

        **Keyword arguments:**

        remove_orphans: `bool`
            Remove all entries with cluster label -1.

            default: `False`
        """
        zip_list = list(zip(self.labels, self.mids))
        if remove_orphans:
            zip_list = filter(lambda x: x[0] != -1, zip_list)
        return [x[1] for x in sorted(zip_list, key = lambda x: x[0])]

    def save(self, 
             filename: str = 'SiMatClus.npy', 
             filepath: str = '.'):
        """
        Save clusterer to numpy format.

        **Keyword arguments:**

        filename: `str`
            Name of file to be written.

        filepath: `str`
            Relative path of file to be written.

        **Returns:**
        
        `None`
        """
        np.save(os.path.join(filepath, filename), np.array([self.simat, self.use_complement, self.clusterer], dtype=object))

    @staticmethod
    def load(filename = 'SiMatClus.npy', filepath = '.'):
        """
        Load clusterer from numpy format.

        WARNING: This allows for unpickling object files. Always make sure that the file you attempt to load is safe.

        **Keyword arguments:**

        filename: `str`
            Name of file to be loaded.

        filepath: `str`
            Relative path of file to be loaded.

        **Returns:**
        
        `self` : `SimilarityMatrixClusterer`
        """        
        simat, use_complement, clusterer = np.load(os.path.join(filepath, filename), allow_pickle = True)
        self = SimilarityMatrixClusterer(simat)
        self.clusterer = clusterer
        self.use_complement = use_complement
        return self

    def get_label_dict(self) -> dict:
        """
        Get a dictionary of all materials, where for each material id the cluster label is stored.

        **Returns:**

        label_dict: `dict`
            Dictionary {*mid1*:*label1*, [...]} mapping material ids to cluster labels.
        """        
        return dict(zip(self.mids, self.labels))

    def get_mids_by_cluster_label(self, cluster_label: int) -> List[str]:
        """
        Get mids of all materials that have the specified cluster label.

        **Arguments:**

        cluster_label: *int*
            Label of requested cluster.

        **Returns:**

        mid_list: `List[str]`
            Material ids of all materials in the specified cluster.
        """
        return [key for key, value in self.get_label_dict().items() if value == cluster_label]

    def get_cluster_sub_matrix(self, cluster_label : int) -> SimilarityMatrix:
        """
        Return a sub matrix of the similarity matrix that contains only the elements from the specified cluster.

        **Arguments:**

        cluster_label: `int`
            Label of cluster
        """
        return self.simat.get_sub_matrix(self.get_mids_by_cluster_label(cluster_label))

    def get_sorted_similarity_matrix(self):
        """
        Return a `SimilarityMatrix` where all entries are sorted by ascending cluster label.
        This is helpful for visualization.

        **Returns:**

        similarity_matrix: `SimilarityMatrix`
            Similarity matrix with sorted entries.
        """
        return self.simat.get_sub_matrix(self.get_mids_sorted_by_cluster_labels())

    @property
    def unique_labels(self) -> np.ndarray:
        """
        Get list of all unque cluster labels.

        **Returns:**

        unique_labels: `np.ndarray`
            Unique cluster labels
        """
        return np.unique(self.labels)

    def get_cluster_size(self, cluster_label: int) -> int:
        """
        Get the size (i.e. the number of members) of a specific cluster.

        **Arguments:**

        cluster_label: `int`
            Label of cluster to get size of.

        **Returns**

        cluster_size: `int`
            Number of members of the specified cluster.
        """
        return len(self.get_mids_by_cluster_label(cluster_label))
