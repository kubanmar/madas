import pytest
from simdatframe.clustering import SimilarityMatrixClusterer
from simdatframe import SimilarityMatrix, Fingerprint

from sklearn.datasets import make_blobs

@pytest.fixture()
def blobs_and_labels():
    return make_blobs(random_state=1)

@pytest.fixture()
def fingerprints(blobs_and_labels):
    pos, _ = blobs_and_labels
    return [Fingerprint("DUMMY").from_list(pos_) for pos_ in pos]

@pytest.fixture()
def similarity_matrix(fingerprints):
    simat = SimilarityMatrix().calculate(fingerprints)
    return simat

@pytest.fixture()
def fitted_clusterer(similarity_matrix):
    simat = similarity_matrix
    clus = SimilarityMatrixClusterer(simat)
    clus.set_clusterer_params(eps = 0.65)
    clus.cluster()
    return clus

def test_fit(similarity_matrix, blobs_and_labels):
    _, labels = blobs_and_labels
    simat = similarity_matrix
    clus = SimilarityMatrixClusterer(simat)

    assert (clus.matrix == 1 - simat.matrix).all(), "Distance matrix is not complement of similarity matrix"
    assert (clus.mids == simat.mids).all(), "Did not copy correct mids"

    clus.set_clusterer_params(eps = 0.65)
    clus.cluster()

    assert len(set(labels)) == clus.nclusters, "Did not recover correct number of clusters"
    assert (labels == clus.labels).all(), "Cluster labels are not correct"

def test_save_and_load(fitted_clusterer, tmpdir):
    fitted_clusterer.save(filepath = tmpdir)
    loaded_clusterer = SimilarityMatrixClusterer.load(filepath = tmpdir)
    
    assert (fitted_clusterer.labels == loaded_clusterer.labels).all(), "Loading did not return same cluster labels"
    assert fitted_clusterer.simat == loaded_clusterer.simat, "Loading did not return same SimilarityMatrix"
    assert fitted_clusterer.uses_distances == loaded_clusterer.uses_distances, "Loading did not return correct distance handling"

def test_get_mids_sorted_by_cluster_labels(fitted_clusterer):
    label_dict = fitted_clusterer.get_label_dict()
    current_label = min(list(label_dict.values()))
    for mid in fitted_clusterer.get_mids_sorted_by_cluster_labels():
        lab_ = label_dict[mid]
        assert lab_ >= current_label, "Later label larger than precessedor"
        if lab_ > current_label:
            current_label = lab_
