from __future__ import absolute_import
from scipy.spatial.distance import pdist as scipy_pdist
import numpy

def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]

# Need own implementation of jaccard because numpy's implementation is different
def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(a & b)
    return intersect / (len(a) + len(b) - intersect)


metrics = {
    'hamming': lambda a, b: pdist(a, b, "hamming"),
    'jaccard': lambda a, b: jaccard(a, b),
    'euclidean': lambda a, b: pdist(a, b, "euclidean"),
    'angular': lambda a, b: pdist(a, b, "cosine")
}
