from __future__ import absolute_import
from scipy.spatial.distance import pdist as scipy_pdist

def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]

metrics = {
    'hamming': lambda a, b: pdist(a, b, "hamming"),
    'euclidean': lambda a, b: pdist(a, b, "euclidean"),
    'angular': lambda a, b: pdist(a, b, "cosine")
}
